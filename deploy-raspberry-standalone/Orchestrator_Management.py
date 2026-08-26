"""
Orquestador: enruta cada turno del chat a un modo (Trivia / Búsqueda Web /
Chat libre), lleva el loop de conversación, Y contiene la lógica de qué hacer
en cada uno de esos tres modos una vez decidido — antes repartido entre
chat.py (el "recepcionista": a quién mandar cada turno) y workers.py (los
"departamentos": qué hace cada uno), fusionados acá en un solo archivo por
ser las dos mitades de la misma orquestación. Los Agents (veredictos/caras,
sin o con LLM) y los Clients (Ollama, carrito, música, voz) siguen viviendo
en sus propias carpetas — esto es el único punto que los conecta a todos.

A diferencia de un chat con personalidad fija, el ruteo es un router puro:
por cada mensaje del usuario decide a cuál de los tres modos mandarlo. Corre
en CADA turno, no solo al arrancar, para que el usuario pueda saltar de
trivia a preguntar algo actual o charlar libre sin reiniciar la sesión — el
costo es una llamada extra al LLM por mensaje.

Corre con:
    python Orchestrator_Management.py
Escribe 'salir' para terminar.
"""

import difflib
import queue
import random
import sys
import threading
import time

import httpx

import display  # display.py: carita en la LCD conectada a esta Raspberry Pi
import voz_server  # voz_server.py: página de voz+texto, corre en un hilo aparte
from Agents.Agent_Behavior import elegir_cara_por_calidad, reaccionar as reaccionar_expresion
from Agents.Agent_Corrector import evaluar_respuesta
from Agents.Agent_Router import enrutar  # router sin LLM (TF-IDF + regresión logística)
from Agents.Agent_Verificator import verificar_y_corregir
from Clients import Camara_Client
from Clients import Voice_Output_Client as voz_output  # Ereberus habla en voz alta (edge-tts + mpv)
from Clients.Llama_Client import CHAT_MODEL, TRIVIA_MODEL, VERIFICADOR_MODEL, generar_respuesta
from personalidad import construir_personalidad, obtener_system_prompt
from preguntas import SIN_CONTEXTO, recuperar_contexto
from preguntas import pregunta_aleatoria as _pregunta_aleatoria
from preguntas import pregunta_por_tema as _pregunta_por_tema
from preguntas import preguntas_por_tema as _preguntas_por_tema

# El modelo puede colar un emoji o una comilla tipográfica que la consola no
# sabe representar, y eso tiraría UnicodeEncodeError a mitad del streaming. Con
# 'replace' sale un '?' y el chat sigue. No se fuerza ningún encoding: el de por
# defecto ya coincide con el de la consola.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# ══════════════════════════════════════════════════════════════════════
# Helpers compartidos
# ══════════════════════════════════════════════════════════════════════

RESPUESTA_SIN_CONTEXTO = "No tengo información sobre eso en mi base de datos."

# Saludos/mensajes triviales: no ameritan gastar tiempo en la búsqueda BM25.
SALUDOS_TRIVIALES = {
    "hola", "buenas", "hey", "qué tal", "que tal", "hi", "hello",
    "buenos días", "buenas tardes", "buenas noches", "gracias", "ok", "vale",
}


def _es_mensaje_trivial(mensaje):
    return mensaje.strip().lower().strip("¡!¿?.,") in SALUDOS_TRIVIALES


def _mensajes_con_personalidad(persona_str, contenido_usuario, modelo=CHAT_MODEL):
    """Arma la lista de mensajes para generar_respuesta(): system (si
    obtener_system_prompt devuelve algo -- viene vacío cuando `modelo` es
    ereberus-personalidad, que ya trae la personalidad horneada, ver
    personalidad.py) + el mensaje del usuario.

    `modelo` default a CHAT_MODEL (Chat libre/Búsqueda web); comentar_resultado()
    y reaccionar_libre() (Trivia) pasan TRIVIA_MODEL explícito -- son modelos
    distintos a propósito, ver Clients/Llama_Client.py."""
    system_prompt = obtener_system_prompt(persona_str, modelo=modelo)
    mensajes = []
    if system_prompt:
        mensajes.append({"role": "system", "content": system_prompt})
    mensajes.append({"role": "user", "content": contenido_usuario})
    return mensajes


def _quitar_pregunta_final(texto):
    """Si el texto cierra con una pregunta, se la recorta.

    Garantía por código, no por prompt: aunque se le pida no preguntar, un modelo
    de 3B igual cierra con una pregunta de vez en cuando (~1 de cada 4), y en modo
    quiz esa pregunta descoloca al usuario porque nadie la va a corregir.
    """
    texto = texto.strip()
    if not texto.endswith("?"):
        return texto
    idx = texto.rfind("¿")
    return texto[:idx].strip() if idx > 0 else texto


# ══════════════════════════════════════════════════════════════════════
# Chat libre
# ══════════════════════════════════════════════════════════════════════

def generar_apertura(persona_str, on_token=None):
    """El robot abre la sesión: se presenta y anuncia que va a preguntar.

    No debe cerrar preguntando "¿quieres responder algunas preguntas?": el bucle
    arranca con la primera pregunta del dataset sin esperar confirmación, así que
    una pregunta acá quedaría sin responder y la interacción se ve rota.
    """
    mensajes = _mensajes_con_personalidad(persona_str, (
        "Instrucción: inicia tú la conversación, todavía no hay mensaje del "
        "usuario. Preséntate en una frase muy breve (con tu personalidad, sin "
        "sonar a asistente genérico) y avisa que le vas a hacer preguntas y que "
        "él responde. Es un aviso, no una pregunta: no cierres preguntando nada "
        "ni pidas permiso ni confirmación."
    ))
    # Sin streaming a propósito: hay que ver el texto completo para poder recortar
    # una pregunta final, y eso no se puede hacer si ya se imprimió token a token.
    return _quitar_pregunta_final(generar_respuesta(mensajes, on_token=on_token))


def responder(mensaje_usuario, persona_str, n_results=2, on_token=None):
    """Devuelve (texto_final, fue_corregida) — fue_corregida es el veredicto
    del verificador (True = tuvo que arreglar la respuesta), que
    Agent_Behavior.elegir_cara_por_calidad() usa para elegir happy vs
    sad/angry."""
    contexto = "" if _es_mensaje_trivial(mensaje_usuario) else recuperar_contexto(
        mensaje_usuario, n_results=n_results
    )

    if contexto is SIN_CONTEXTO:
        # Corte a nivel de código, no de prompt: un modelo chico no respeta de forma
        # confiable la instrucción de "no uses tu conocimiento propio", así que ni se
        # le manda la pregunta si no hay nada relevante en la base de datos.
        if on_token:
            on_token(RESPUESTA_SIN_CONTEXTO)
        return RESPUESTA_SIN_CONTEXTO, True  # "no tengo el dato" no es un momento feliz

    # Sin historial: cada turno es system (fijo, cacheado) + un único user con
    # el contexto recuperado por BM25 + la pregunta, nada más.
    #
    # En un saludo no hay contexto que mandar. Mandar el bloque <rag> vacío hace
    # que el modelo lo lea como "no hay datos" y conteste "no tengo el dato" a un
    # simple "hola"; por eso ahí se manda el mensaje pelado.
    if contexto:
        user = f"CONTEXTO:\n<rag>\n{contexto}\n</rag>\n\nPregunta: {mensaje_usuario}"
    else:
        user = mensaje_usuario

    mensajes = _mensajes_con_personalidad(persona_str, user)

    # Sin streaming a propósito: el agente verificador necesita el texto
    # completo para revisarlo antes de que el usuario lo vea — si se
    # imprimiera token a token, ya estaría en pantalla para cuando se
    # detectara que hay que corregirlo.
    borrador = generar_respuesta(mensajes).strip()
    texto_final, fue_corregida = verificar_y_corregir(mensaje_usuario, borrador)
    if on_token:
        on_token(texto_final)
    return texto_final, fue_corregida


# ══════════════════════════════════════════════════════════════════════
# Trivia
# ══════════════════════════════════════════════════════════════════════
#
# Es el flujo para el que está hecho el dataset (robot EVA): cada registro trae
# la pregunta, la respuesta_esperada con la que corregir, y la 'cara' que el
# robot debe poner.

# Catálogo de temas/actividades (columna "Actividad / Tema" del Excel, campo
# "tema" en preguntas.jsonl). resolver_tema() clasifica contra esta lista.
TEMAS_CATALOGO = [
    # "númerica" así, mal tildada, porque es el texto tal cual viene del Excel
    # (columna Actividad/Tema) — tiene que calzar exacto con preguntas.jsonl.
    # "Chistes" se sacó del catálogo (y del dataset) a pedido del usuario.
    "Adivinanza númerica", "Dilema", "Dilema del coche autónomo",
    "Interaccion personalizada (COMIDA)", "Juego de colores", "Juego de emociones",
    "Juego de imitación", "Juego de multiplicar nivel Alto 1",
    "Juego de multiplicar nivel Alto 2", "Juego de multiplicar nivel Simple 1",
    "Juego de multiplicar nivel Simple 2", "Prueba de multiplicar nivel Simple",
    "Prueba de reconocimiento de Color", "Reconocimiento Musical",
    "Reconocimiento visual", "Trivia",
]

# Los dos únicos temas del catálogo con veredicto real vía cámara en vez de
# texto (ver _jugar_emociones() más abajo) -- ninguna pregunta de estos temas
# trae respuesta_esperada, así que sin este caso especial caerían en
# reaccionar_libre() (conversación sin verificar nada).
TEMAS_JUEGO_EMOCIONES = {"Juego de emociones", "Juego de imitación"}


def obtener_pregunta(ya_usados):
    """Una pregunta que no se haya hecho todavía (preguntas.py, en memoria —
    ya no es un request de red)."""
    return _pregunta_aleatoria(ya_usados)


def obtener_pregunta_por_tema(tema, ya_usados):
    """Como obtener_pregunta, pero acotado a una categoría puntual (columna
    Actividad/Tema del Excel). `tema` ya viene resuelto por resolver_tema()."""
    return _pregunta_por_tema(tema, ya_usados)


def obtener_preguntas_por_tema(tema, ya_usados, cantidad=5):
    """Como obtener_pregunta_por_tema, pero trae una tanda de `cantidad` de
    una sola vez, para encadenarlas sin volver a filtrar por cada pregunta."""
    return _preguntas_por_tema(tema, ya_usados, cantidad)


def comentar_resultado(pregunta, esperada, respuesta_usuario, acerto, persona_str, on_token=None):
    """Reacción hablada del robot tras corregir, con su personalidad.

    Usa TRIVIA_MODEL, no CHAT_MODEL -- Trivia tiene su propio modelo aparte
    (ereberus-personalidad por default, ver Clients/Llama_Client.py).

    El veredicto (acerto) ya viene resuelto por evaluar_respuesta() — acá NO
    se le pide al modelo que vuelva a resolver la pregunta, solo que
    reaccione. Sin ser explícito con esto, un modelo chico tiende a
    "re-resolver" el ejercicio por su cuenta en vez de limitarse a comentar
    el resultado, y a veces lo hace mal (ej. lee "1 por 8" como "1/8" y
    contesta con eso en vez de confirmar el acierto).
    """
    if acerto:
        instruccion = (
            f"El estudiante respondió '{respuesta_usuario}' y ACERTÓ (la respuesta "
            f"correcta es {esperada}). Confírmaselo en una frase corta."
        )
    else:
        instruccion = (
            f"El estudiante respondió '{respuesta_usuario}' y SE EQUIVOCÓ. Dile en una "
            f"frase corta que no es correcto y que la respuesta correcta era {esperada}."
        )
    mensajes = _mensajes_con_personalidad(persona_str, (
        f"Pregunta que se hizo: {pregunta}\n\n"
        f"{instruccion} No vuelvas a resolver la pregunta ni expliques el cálculo: "
        "el veredicto ya está decidido, tu única tarea es reaccionar a él. "
        "No hagas otra pregunta."
    ), modelo=TRIVIA_MODEL)
    return generar_respuesta(mensajes, on_token=on_token, modelo=TRIVIA_MODEL).strip()


def comentar_resultado_emocion(objetivo, detectada, acerto, persona_str, on_token=None):
    """Como comentar_resultado(), pero para el Juego de emociones/imitación:
    el veredicto no sale de comparar texto (Agent_Corrector) sino de la
    cámara (Clients/Camara_Client.py) contra la cara que se le pidió al
    usuario que hiciera -- ver _jugar_emociones(). Usa TRIVIA_MODEL, mismo
    criterio que comentar_resultado()."""
    if acerto:
        instruccion = (
            f"Le pediste al usuario que pusiera cara de {objetivo.lower()} y la cámara "
            f"detectó justo esa cara. Celébraselo en una frase corta."
        )
    else:
        instruccion = (
            f"Le pediste al usuario que pusiera cara de {objetivo.lower()} pero la cámara "
            f"detectó cara de {detectada.lower()}. Dile en una frase corta que no era esa, "
            "con humor, sin ser pesado."
        )
    mensajes = _mensajes_con_personalidad(persona_str, (
        f"{instruccion} No vuelvas a pedir otra cara ni hagas otra pregunta."
    ), modelo=TRIVIA_MODEL)
    return generar_respuesta(mensajes, on_token=on_token, modelo=TRIVIA_MODEL).strip()


def _jugar_emociones(pregunta, persona_str, on_token=None):
    """Ronda del Juego de emociones/imitación: elige al azar una cara entre
    las que trae la columna 'cara' de ESTA pregunta (ej. entre Enojado/
    Feliz/Triste), se la pide al usuario en voz alta, y la verifica con la
    cámara real (Clients/Camara_Client.py) -- mismo patrón que la rama CON
    respuesta_esperada de manejar_trivia(), pero el veredicto sale de la
    cámara en vez de Agent_Corrector.

    Devuelve el bool acerto/error, o None si no se pudo evaluar (cámara no
    disponible o sin cara detectada) -- en ese caso no cuenta para el
    marcador, se reacciona igual que reaccionar_libre() (sin veredicto)."""
    opciones = [o.strip() for o in pregunta["cara"].split("/") if o.strip()]
    if not opciones:
        opciones = ["Feliz"]  # fallback: nunca debería pasar, pero sin esto no habría qué pedir
    objetivo = random.choice(opciones)
    pedido = f"¡Hazme una cara de {objetivo.lower()}!"
    print(f"Asistente: {pedido}")
    voz_output.hablar(pedido)
    time.sleep(1)  # un segundo para que el usuario pose antes de capturar
    detectada, confianza = Camara_Client.detectar_emocion()

    if detectada is None:
        reaccionar_expresion(pregunta)
        display.mostrar_cara("speaking")
        voz_output.hablar(reaccionar_libre(pregunta["pregunta"], pedido, persona_str, on_token=on_token))
        return None

    acerto = detectada == objetivo
    print(f"    [cámara] pedido={objetivo} detectado={detectada} (confianza {confianza:.0%}) -> {'OK' if acerto else 'MAL'}")
    expresion = reaccionar_expresion(pregunta, acerto)
    cara = expresion["cara"] or ("happy" if acerto else "sad")
    display.mostrar_cara("content")
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara(cara)
    reaccion = comentar_resultado_emocion(objetivo, detectada, acerto, persona_str, on_token=on_token)
    voz_output.hablar(reaccion)
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("speaking")
    return acerto


def reaccionar_libre(pregunta, respuesta_usuario, persona_str, on_token=None):
    """Para preguntas sin respuesta_esperada (Reconocimiento Musical, Juego
    de colores...): no hay nada que corregir, así que en vez de
    evaluar_respuesta/comentar_resultado el robot solo reacciona.

    Usa TRIVIA_MODEL, no CHAT_MODEL -- mismo criterio que comentar_resultado()."""
    mensajes = _mensajes_con_personalidad(persona_str, (
        f"Pregunta: {pregunta}\n"
        f"Respuesta del usuario: {respuesta_usuario}\n\n"
        "Reacciona en una frase corta. No hay respuesta correcta acá: no "
        "corrijas ni evalúes, solo reacciona con tu personalidad. No hagas "
        "otra pregunta."
    ), modelo=TRIVIA_MODEL)
    return generar_respuesta(mensajes, on_token=on_token, modelo=TRIVIA_MODEL).strip()


def resolver_tema(eleccion_usuario):
    """El usuario elige la sesión con texto libre ('quiero chistes', 'algo de
    colores', o directo 'Interaccion'); se mapea a EXACTAMENTE una categoría
    del catálogo por texto, sin LLM de por medio — las opciones que se le
    muestran ya son 5 strings literales sacados del propio catálogo (ver
    manejar_trivia más abajo), así que alcanza con match directo/difuso
    contra esa lista. Si no hay relación clara, cae en Trivia (la sesión con
    más datos)."""
    texto = eleccion_usuario.strip().lower()
    catalogo_low = [t.lower() for t in TEMAS_CATALOGO]

    # 1) match exacto o por substring en cualquier sentido (cubre "chistes" ->
    #    "Chistes", "algo de colores" -> "Juego de colores").
    for tema, tema_low in zip(TEMAS_CATALOGO, catalogo_low):
        if texto == tema_low or tema_low in texto or texto in tema_low:
            return tema

    # 2) match difuso (typos, palabras de más/menos) contra el catálogo completo.
    cercano = difflib.get_close_matches(texto, catalogo_low, n=1, cutoff=0.5)
    if cercano:
        return TEMAS_CATALOGO[catalogo_low.index(cercano[0])]

    return "Trivia"


# ══════════════════════════════════════════════════════════════════════
# Búsqueda web
# ══════════════════════════════════════════════════════════════════════
#
# Vía la Instant Answer API de DuckDuckGo: no necesita API key, pero solo
# suele traer algo para temas tipo enciclopedia (matches contra Wikipedia),
# no para vocabulario general en español ni para "qué pasó hoy" en sentido
# estricto. Aun así es lo único con info fuera del dataset local.

SIN_RESULTADO_WEB = "No se encontró información en la web."
BUSQUEDA_WEB_NO_DISPONIBLE = "Búsqueda web no disponible."


def tool_web_search(query):
    """None si no hubo resultado (sin distinguir "vacío" de "falló"): a
    responder_busqueda_web() le alcanza con saber que no hay nada que pasarle
    al modelo."""
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "skip_disambig": "1"},
            timeout=5.0,
        )
        data = resp.json()
        return data.get("AbstractText", "") or None
    except Exception:
        return None


def extraer_termino_busqueda(mensaje_usuario):
    """DuckDuckGo Instant Answer matchea casi textual contra un título de
    Wikipedia: "busca Peru", "qué es Peru" o "dime sobre Peru" no traen
    nada, pero "Peru" a secas sí. Se le pide al LLM que aísle el tema/
    entidad central, sin verbos ni envoltorio conversacional, antes de
    mandarlo a buscar — mismo patrón que resolver_tema() para Trivia.

    Usa VERIFICADOR_MODEL, no CHAT_MODEL: es extracción con una regla de
    formato estricta ("solo el término, nada más"), no una reacción con
    personalidad -- mismo motivo que Agent_Verificator.py (ver la nota ahí
    y en Clients/Llama_Client.py sobre el bug real que causó al mezclar
    esta clase de tarea con CHAT_MODEL una vez que pasó a ser un fine-tune
    de estilo)."""
    mensajes = [
        {"role": "system", "content": (
            "Extrae el tema o entidad principal de la pregunta del usuario, "
            "tal como aparecería como título de un artículo de enciclopedia: "
            "sin verbos ('busca', 'qué es', 'dime sobre'), sin signos de "
            "interrogación, solo el nombre del tema/persona/lugar/cosa.\n"
            "Responde EXCLUSIVAMENTE con ese término, nada más."
        )},
        {"role": "user", "content": mensaje_usuario},
    ]
    # temperature=0: es extracción, no charla — no queremos variación entre corridas.
    return generar_respuesta(mensajes, temperature=0, max_tokens=15, modelo=VERIFICADOR_MODEL).strip()


def responder_busqueda_web(mensaje_usuario, persona_str, on_token=None):
    """Busca en la web y le pide al modelo que conteste con eso.

    Si no hay resultado, el corte es por código, no por instrucción de
    prompt: pedirle al modelo "si no hay nada dilo tal cual" no es
    confiable — de un turno a otro inventa la respuesta con su propio
    conocimiento, o directo se rehúsa ("no estoy diseñado para eso"). Mismo
    criterio que SIN_CONTEXTO en responder() (chat libre).

    Devuelve (texto_final, fue_corregida) — ver la nota en responder()."""
    termino = extraer_termino_busqueda(mensaje_usuario)
    resultado = tool_web_search(termino)
    if resultado is None:
        if on_token:
            on_token(SIN_RESULTADO_WEB)
        return SIN_RESULTADO_WEB, True  # no encontrar nada tampoco es un momento feliz

    mensajes = _mensajes_con_personalidad(persona_str, (
        f"Pregunta: {mensaje_usuario}\n"
        f"Resultado de la búsqueda web: {resultado}\n\n"
        "Responde con esa información, en una frase corta."
    ))

    # Sin streaming: el verificador necesita el texto completo antes de que
    # el usuario lo vea (ver la misma nota en responder(), chat libre).
    borrador = generar_respuesta(mensajes).strip()
    texto_final, fue_corregida = verificar_y_corregir(mensaje_usuario, borrador)
    if on_token:
        on_token(texto_final)
    return texto_final, fue_corregida


# ══════════════════════════════════════════════════════════════════════
# Router + loop de conversación
# ══════════════════════════════════════════════════════════════════════

RUTAS = ["TRIVIA", "BUSQUEDA_WEB", "CHAT_LIBRE"]

# Frases que, dichas A MITAD de una tanda de trivia (respondiendo una
# pregunta o eligiendo tema), señalan que el usuario se quiere ir a otra
# cosa — no una respuesta rara, sino que quiere salir del juego. Heurística
# por palabras clave, sin LLM: costo cero, a cambio de no cubrir frases fuera
# de esta lista (si no matchea nada, el mensaje se sigue tratando como
# respuesta/elección normal). No incluye "salir" a secas -eso ya termina la
# sesión entera más abajo, antes de llegar acá- solo variantes más largas.
_SALIR_TRIVIA = [
    "otra cosa", "cambiar de tema", "cambiemos de tema",
    "ya no quiero seguir", "ya no quiero jugar", "no quiero seguir jugando",
    "quiero parar", "para de preguntar", "deja de preguntar", "detente",
    "salir de la trivia", "salir del juego", "salir de trivia",
    "pausa la trivia", "pausar la trivia", "pausemos",
    "dejemos la trivia", "dejar la trivia",
    "hablemos de otra cosa", "quiero charlar", "prefiero charlar",
    "quiero conversar",
]


def _quiere_salir_trivia(mensaje_usuario):
    texto = mensaje_usuario.strip().lower()
    return any(frase in texto for frase in _SALIR_TRIVIA)


# Cola compartida: tanto el hilo que lee la terminal (_hilo_stdin) como
# voz_server.py (voz transcrita o texto tipeado en la página web) empujan
# acá — _leer_entrada() bloquea en un solo lugar sin importar de cuál de los
# dos caminos vino el mensaje.
_entrada_queue = queue.Queue()


def _hilo_stdin():
    while True:
        try:
            linea = input()
        except EOFError:
            break
        _entrada_queue.put(linea)


def _leer_entrada(prompt="Tú: "):
    print(prompt, end="", flush=True)
    return _entrada_queue.get().strip()


def enrutar_mensaje(mensaje_usuario):
    # El clasificador (Agents/Agent_Router.py, sin LLM) vive ahí — acá solo
    # se decide el fallback por si alguna vez devolviera algo raro (hoy no
    # debería pasar: el clasificador siempre devuelve una de RUTAS_VALIDAS).
    ruta = enrutar(mensaje_usuario)
    return ruta if ruta in RUTAS else "CHAT_LIBRE"


PREGUNTAS_POR_TANDA = 5
PAUSA_CAMBIO_CARA = 4  # segundos en 'content' antes de pasar a la cara que sigue (habla o reacción)


def _preguntar_siguiente(estado):
    """Saca la siguiente pregunta de la cola de la tanda actual y la imprime
    tal cual viene del dataset, sin pasarla por el modelo (si la reformulara,
    la respuesta_esperada dejaría de servir para corregir). Devuelve False si
    la cola ya estaba vacía (se acabó la tanda)."""
    if not estado["cola_preguntas"]:
        return False
    actual = estado["cola_preguntas"].pop(0)
    estado["pregunta_pendiente"] = actual
    display.mostrar_cara("speaking")  # se lanza una pregunta: la LCD "habla"
    print(f"Asistente [{actual['cara']}]: {actual['pregunta']}\n")
    voz_output.hablar(actual["pregunta"])  # bloquea hasta que termina de decirla
    # La pregunta se imprime de una (no hay streaming que marque cuándo
    # "termina de hablar"), así que se le da un tiempo fijo en 'speaking'
    # -antes de pasar a la cara de reposo mientras espera que el usuario
    # conteste- para que de verdad se alcance a ver, no un flash instantáneo.
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")
    return True


def _iniciar_tanda(tema, estado):
    """Pide de una sola vez una tanda de PREGUNTAS_POR_TANDA preguntas de un
    tema ya elegido (un solo request al backend) y arranca con la primera;
    las demás quedan en cola_preguntas para encadenarlas sin volver a pasar
    por el menú de temas entre una y otra."""
    preguntas = obtener_preguntas_por_tema(tema, estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
    if not preguntas:
        print(f"Asistente: No quedan preguntas de {tema}.\n")
        return
    estado["ya_usados"].update(p["id"] for p in preguntas)
    estado["cola_preguntas"] = preguntas
    _preguntar_siguiente(estado)


def manejar_trivia(mensaje_usuario, estado, persona_str, on_token):
    """Tres estados posibles en modo trivia, en este orden de prioridad:
    1) se le mostraron opciones y este mensaje es la elección,
    2) hay una pregunta pendiente y este mensaje es la respuesta a corregir,
    3) recién entra a trivia: se muestran 5 opciones al azar del catálogo
       real para que elija, en vez de que el modelo le adivine el tema al
       primer mensaje."""
    if estado["esperando_tema"]:
        estado["esperando_tema"] = False
        tema = resolver_tema(mensaje_usuario)
        estado["tema_actual"] = tema
        anuncio = f"Vamos con {tema}. Van {PREGUNTAS_POR_TANDA} preguntas seguidas."
        print(f"Asistente: {anuncio}")
        voz_output.hablar(anuncio)
        _iniciar_tanda(tema, estado)
        return

    pendiente = estado["pregunta_pendiente"]
    if pendiente is not None:
        estado["pregunta_pendiente"] = None
        print("Asistente: ", end="", flush=True)
        if pendiente["respuesta_esperada"]:
            estado["total"] += 1
            acerto = evaluar_respuesta(pendiente["pregunta"], pendiente["respuesta_esperada"],
                                        mensaje_usuario)
            estado["aciertos"] += acerto
            # Agent_Behavior.reaccionar() decide la cara específica de ESTA
            # pregunta (cara_respuesta_buena/mala del dataset, sin LLM) y de
            # paso revisa música/desplazamiento de la misma fila. Fallback
            # genérico solo por si esa columna viniera vacía en algún caso raro.
            expresion = reaccionar_expresion(pendiente, acerto)
            cara = expresion["cara"] or ("happy" if acerto else "sad")
            # Se muestra ANTES de la reacción hablada, no "speaking": acá lo
            # que importa comunicar primero es el veredicto, no que está hablando.
            # Una pausa breve en 'content' antes de saltar a la emoción: mostrarla
            # de un tirón se siente demasiado instantáneo/robótico para una reacción.
            display.mostrar_cara("content")
            time.sleep(PAUSA_CAMBIO_CARA)
            display.mostrar_cara(cara)
            reaccion = comentar_resultado(pendiente["pregunta"], pendiente["respuesta_esperada"],
                                           mensaje_usuario, acerto, persona_str, on_token=on_token)
            voz_output.hablar(reaccion)
            # Se queda un rato en la emoción (acierto o error, mismo trato para
            # las dos) antes de pasar a 'speaking' — el asistente ya va a decir
            # algo más (la próxima pregunta, o si se acabó la tanda) así que
            # vuelve a "hablar" antes de eso, no queda pegado en la emoción.
            time.sleep(PAUSA_CAMBIO_CARA)
            display.mostrar_cara("speaking")
        elif estado["tema_actual"] in TEMAS_JUEGO_EMOCIONES:
            # Único caso con veredicto real fuera de respuesta_esperada: la
            # cámara reemplaza a Agent_Corrector -- ver _jugar_emociones().
            acerto = _jugar_emociones(pendiente, persona_str, on_token=on_token)
            if acerto is not None:
                estado["total"] += 1
                estado["aciertos"] += acerto
        else:
            # Temas como Chistes o Reconocimiento Musical no tienen una
            # respuesta correcta que corregir: no hay veredicto, así que
            # Agent_Behavior.reaccionar() no elige cara (queda la genérica de
            # "hablando") — pero igual revisa música/desplazamiento, que no
            # dependen de acierto/error.
            reaccionar_expresion(pendiente)
            display.mostrar_cara("speaking")
            reaccion = reaccionar_libre(pendiente["pregunta"], mensaje_usuario, persona_str, on_token=on_token)
            voz_output.hablar(reaccion)
        print("\n")

        # Sigue encadenando la tanda; recién cuando se acaba se vuelve a
        # preguntar qué hacer (el Router igual reclasifica el próximo mensaje,
        # pero preguntarlo explícito evita que el usuario se quede sin saber
        # qué esperar después de la última pregunta).
        if not _preguntar_siguiente(estado):
            # Se acabó la tanda sola (no por una salida a mitad de pregunta):
            # ya no hay nada que retomar, así que el próximo mensaje vuelve a
            # pasar por el Router como cualquier turno normal.
            estado["en_trivia"] = False
            # Ya está en 'speaking' desde la rama de arriba (o se acaba de
            # poner acá si no hubo veredicto) para el mensaje de cierre de
            # tanda; recién después de decirlo pasa a la cara de reposo.
            display.mostrar_cara("speaking")
            cierre = ("Esas eran las 5. ¿Seguimos con más trivia, "
                      "buscamos algo en la web, o prefieres charlar?")
            print(f"Asistente: {cierre}\n")
            voz_output.hablar(cierre)
            time.sleep(PAUSA_CAMBIO_CARA)
            display.mostrar_cara("content")
        return

    opciones = ", ".join(random.sample(TEMAS_CATALOGO, 5))
    estado["esperando_tema"] = True
    print(f"Asistente: Puedes elegir entre: {opciones}.\n")


def _reanudar_trivia(estado):
    """Retoma una trivia que se había pausado (a diferencia de manejar_trivia,
    no consume nada del estado ni cuenta el mensaje que disparó el retomo como
    respuesta): si había una pregunta esperando corrección la vuelve a
    mostrar tal cual; si estaba esperando que elija tema, se lo recuerda."""
    display.mostrar_cara("speaking")
    if estado["esperando_tema"]:
        print("Asistente: Retomamos. ¿Qué tema eliges?\n")
        voz_output.hablar("Retomamos. ¿Qué tema eliges?")
    else:
        pendiente = estado["pregunta_pendiente"]
        print("Asistente: Retomamos donde quedamos.")
        print(f"Asistente [{pendiente['cara']}]: {pendiente['pregunta']}\n")
        voz_output.hablar(f"Retomamos donde quedamos. {pendiente['pregunta']}")
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")


def manejar_busqueda_web(mensaje_usuario, persona_str, on_token):
    display.mostrar_cara("speaking")  # generando/revisando la respuesta (ver nota en responder())
    print("Asistente: ", end="", flush=True)
    texto_final, fue_corregida = responder_busqueda_web(mensaje_usuario, persona_str, on_token=on_token)
    print("\n")
    voz_output.hablar(texto_final)
    # happy si la respuesta ya estaba bien, sad/angry si el verificador tuvo
    # que corregirla — mismo criterio de 3 caras que usa Trivia, aplicado acá
    # al veredicto del verificador en vez del veredicto del corrector.
    display.mostrar_cara(elegir_cara_por_calidad(fue_corregida))
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")  # cara de reposo hasta el próximo turno


def manejar_chat_libre(mensaje_usuario, persona_str, on_token):
    display.mostrar_cara("speaking")  # generando/revisando la respuesta (ver nota en responder())
    print("Asistente: ", end="", flush=True)
    texto_final, fue_corregida = responder(mensaje_usuario, persona_str, on_token=on_token)
    print("\n")
    voz_output.hablar(texto_final)
    # happy si la respuesta ya estaba bien, sad/angry si el verificador tuvo
    # que corregirla — mismo criterio de 3 caras que usa Trivia, aplicado acá
    # al veredicto del verificador en vez del veredicto del corrector.
    display.mostrar_cara(elegir_cara_por_calidad(fue_corregida))
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")  # cara de reposo hasta el próximo turno


def main():
    # Carga el modelo de voz PRIMERO, en el hilo principal — cargar un
    # modelo ONNX desde un hilo secundario se cuelga silenciosamente en este
    # entorno (ver la nota histórica en Clients/Voice_Output_Client.py), por
    # eso esto va antes de arrancar cualquier threading.Thread(). Con
    # edge-tts cargar() ya no hace nada, pero se deja la llamada por si
    # algún día se vuelve a un motor local.
    voz_output.cargar()

    # Hilo que lee la terminal línea a línea y lo mete en _entrada_queue —
    # así _leer_entrada() puede bloquear en un solo lugar (la cola) sin
    # importar si el mensaje llegó tipeado acá o desde voz_server.py (voz
    # transcrita o texto de la página web).
    threading.Thread(target=_hilo_stdin, daemon=True).start()
    voz_server.iniciar(_entrada_queue)

    # try/finally alrededor de TODO lo que sigue: si algo de acá adentro tira
    # una excepción sin atrapar (ej. Ollama/un modelo caído, ver el 404 real
    # que pasó con TRIVIA_MODEL apuntando a un modelo no importado), display.
    # detener() igual se llama -- si no, el mpv/tk que abrió display.py queda
    # huérfano corriendo, y como DRM solo permite un master de pantalla a la
    # vez, la próxima corrida no puede tomar control y la cara queda pegada
    # en lo último que mostró el huérfano. Bug real encontrado 2026-08-26
    # (ver también display._mpv_huerfano_vivo(), que además evita competir
    # por la pantalla si un huérfano de éstos ya quedó vivo).
    try:
        _main_loop()
    finally:
        display.detener()


def _main_loop():
    display.mostrar_cara("content")  # arranca en reposo...
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("speaking")  # ...y recién ahora la IA "habla" (saluda, por defecto)
    saludo = "Hola, mi nombre es Ereberus, ¿cuál es tu nombre?"
    print(f"Asistente: {saludo}")
    voz_output.hablar(saludo)
    # Vuelve a 'content' ANTES de esperar el input, no después — igual que en
    # el resto del archivo (_preguntar_siguiente, manejar_chat_libre, etc.):
    # la cara de reposo tiene que estar puesta mientras se espera al usuario,
    # no quedarse en "hablando" durante todo ese rato.
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")
    nombre = _leer_entrada() or "amigo"
    bienvenida = (f"Mucho gusto en conocerte, {nombre}. Puedo hacerte trivia, buscarte algo "
                  "de actualidad, o simplemente conversar — voy cambiando de modo según lo "
                  "que me pidas.")
    print(f"Asistente: {bienvenida} Escribe 'salir' para terminar.\n")
    voz_output.hablar(bienvenida)
    display.mostrar_cara("content")  # cara de reposo mientras espera el primer mensaje

    persona_str = construir_personalidad()
    imprimir = lambda t: print(t, end="", flush=True)

    # pregunta_pendiente: mientras haya una, el próximo mensaje es la respuesta
    # a corregir. esperando_tema: mientras esté activo, el próximo mensaje es
    # la elección de una de las 5 opciones mostradas. cola_preguntas: el resto
    # de la tanda actual, para encadenarlas sin volver al menú de temas.
    # en_trivia: controla si el próximo mensaje se fuerza a manejar_trivia
    # (sin pasar por el Router) o no — se puede apagar a mitad de una pregunta
    # (_quiere_salir_trivia) SIN perder pregunta_pendiente/cola_preguntas, que
    # quedan "congeladas" para poder retomar la tanda más adelante. tema_actual:
    # el tema exacto (uno de TEMAS_CATALOGO) de la tanda en curso -- lo único
    # que usa hoy es manejar_trivia() para saber si toca veredicto por cámara
    # (TEMAS_JUEGO_EMOCIONES) en vez de reaccionar_libre().
    estado = {
        "pregunta_pendiente": None, "esperando_tema": False, "en_trivia": False,
        "cola_preguntas": [], "ya_usados": set(), "aciertos": 0, "total": 0,
        "tema_actual": None,
    }

    while True:
        entrada = _leer_entrada()
        if entrada.lower() in ("salir", "exit", "quit"):
            break
        if not entrada:
            continue

        if estado["en_trivia"]:
            if _quiere_salir_trivia(entrada):
                estado["en_trivia"] = False
                display.mostrar_cara("content")
                print("Asistente: Listo, dejamos la trivia pausada — la "
                      "retomamos cuando quieras. ¿Qué tienes en mente?\n")
                continue
            manejar_trivia(entrada, estado, persona_str, imprimir)
            continue

        ruta = enrutar_mensaje(entrada)
        if ruta == "TRIVIA":
            estado["en_trivia"] = True
            # Si queda una pregunta o un tema pendiente de una tanda pausada,
            # se retoma tal cual en vez de arrancar una tanda nueva.
            if estado["pregunta_pendiente"] is not None or estado["esperando_tema"]:
                _reanudar_trivia(estado)
            else:
                manejar_trivia(entrada, estado, persona_str, imprimir)
        elif ruta == "BUSQUEDA_WEB":
            manejar_busqueda_web(entrada, persona_str, imprimir)
        else:
            manejar_chat_libre(entrada, persona_str, imprimir)

    if estado["total"]:
        print(f"\nAsistente: Terminamos. Acertaste {estado['aciertos']} de {estado['total']}.")
    print("Asistente: ¡Hasta luego!")
    # display.detener() ya no va acá -- lo maneja el finally de main(), que
    # cubre también la salida por excepción, no solo esta salida normal.


if __name__ == "__main__":
    main()
