"""
Orquestador: enruta cada turno del chat a un modo (Trivia / Chat libre), lleva
el loop de conversación, Y contiene la lógica de qué hacer en cada uno de esos
modos una vez decidido — antes repartido entre chat.py (el "recepcionista": a
quién mandar cada turno) y workers.py (los "departamentos": qué hace cada
uno), fusionados acá en un solo archivo por ser las dos mitades de la misma
orquestación. Los Agents (veredictos/caras, sin o con LLM) y los Clients
(Ollama, carrito, música, voz) siguen viviendo en sus propias carpetas — esto
es el único punto que los conecta a todos.

A diferencia de un chat con personalidad fija, el ruteo es un router puro:
por cada mensaje del usuario decide a cuál de los dos modos mandarlo. Corre
en CADA turno mientras no está en trivia (ver Agent_Router.enrutar(), sin
LLM), para que el usuario pueda saltar de trivia a charlar libre sin
reiniciar la sesión.

Búsqueda web (DuckDuckGo Instant Answer) se sacó a pedido del usuario -- el
proyecto se queda con Trivia y Chat libre solamente. `httpx` ya no hace falta
como dependencia de este archivo (era solo para esa búsqueda).

Corre con:
    python Orchestrator_Management.py
Escribe 'salir' para terminar.
"""

import difflib
import os
import queue
import random
import sys
import threading
import time

import display  # display.py: carita en la LCD conectada a esta Raspberry Pi
import perf_monitor  # perf_monitor.py: @medir por componente + muestreo de CPU/memoria (ver perf_report.py)
import voz_server  # voz_server.py: página de voz+texto, corre en un hilo aparte
from Agents.Agent_Behavior import (
    cara_para_emocion, elegir_cara_pregunta, expresar_desplazamiento,
    expresar_musica,
)
from Agents.Agent_Corrector import evaluar_respuesta
from Agents.Agent_Router import enrutar  # router sin LLM (TF-IDF + regresión logística)
from Clients import Camara_Client
from Clients import Voice_Output_Client as voz_output  # Lora habla en voz alta (edge-tts + mpv)
from Clients.Llama_Client import (
    CHAT_MODEL, SALIDA_TRIVIA_MODEL, TRIVIA_MODEL, clasificar_salida_trivia, generar_respuesta,
)
import registro_chat  # registro_chat.py: guarda los turnos de Chat libre para el pipeline de mejora
import memoria_episodica  # BM25 sobre esos mismos turnos -- recordar Chat libre entre sesiones, ver el módulo
from personalidad import construir_personalidad, obtener_system_prompt
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

# RESPUESTA_SIN_CONTEXTO / SALUDOS_TRIVIALES / _es_mensaje_trivial se
# eliminaron junto con el RAG de Chat libre (ver responder()): existian solo
# para saltear la busqueda BM25 en saludos y para cortar cuando no habia
# contexto. Sin RAG no hay busqueda que saltear ni corte que hacer.


def _mensajes_con_personalidad(persona_str, contenido_usuario, modelo=CHAT_MODEL):
    """Arma la lista de mensajes para generar_respuesta(): system (si
    obtener_system_prompt devuelve algo -- viene vacío cuando `modelo` es
    lora-personalidad, que ya trae la personalidad horneada, ver
    personalidad.py) + el mensaje del usuario.

    `modelo` default a CHAT_MODEL (Chat libre); comentar_resultado_emocion()
    (Juego de emociones) pasa TRIVIA_MODEL explícito -- son modelos
    distintos a propósito, ver Clients/Llama_Client.py. comentar_resultado()
    (Trivia con respuesta verificable) ya no pasa por acá desde 2026-08-31,
    ver la nota ahí."""
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


def responder(mensaje_usuario, persona_str):
    """Chat libre: le pasa el mensaje del usuario a CHAT_MODEL con la
    personalidad y devuelve texto_final.

    Ya no acepta on_token: el streaming a consola (imprimir token a token)
    se sacó a pedido del usuario -- ya no se mira la consola de la Pi, se
    interactúa por la página de voz_server.py (el teléfono), y streaming o
    no da el mismo tiempo total (ver la nota en Llama_Client.generar_respuesta()),
    así que no había ninguna razón para mantenerlo. voz_output.hablar()
    necesitaba el texto completo de todos modos.

    YA NO USA RAG. Antes buscaba contexto con BM25 sobre preguntas.jsonl y,
    si no encontraba nada, cortaba con una frase fija sin llamar al modelo.
    Se saco a pedido del usuario: el corpus eran las preguntas de la Trivia,
    asi que el "chat libre" solo podia hablar de eso.

    Ademas andaba mal. El umbral era `score > 0` (preguntas.py), o sea que
    UNA palabra en comun bastaba para inyectar contexto, y palabras vacias
    enganchaban cualquier cosa: a "que tiempo hace" le metia la pregunta
    "con cuantos paises hace frontera" y el modelo contestaba "3" con total
    seguridad. Peor que no saber: inventaba.

    Sin RAG, CHAT_MODEL (lora-chat) conversa con su personalidad. Medido en
    la Pi: acierta preguntas factuales comunes (Graham Bell 1876, Paris) y
    responde bien lo social. Dos limitaciones conocidas y no resueltas:
    se equivoca en aritmetica (235x47 -> "5.109", es 11.045), y con temas
    emocionales a veces se desentiende ("hoy me fue mal en el colegio" ->
    "no puedo ayudar con eso"). Eso se arregla en el fine-tuning
    (chat_training/), no acá.

    Ya no devuelve `problema`: era True solo en el caso SIN_CONTEXTO, que ya
    no existe, y Agent_Behavior.elegir_cara_por_calidad() (la única que lo
    consumía, para variar la cara al final del turno) se sacó del todo a
    pedido del usuario (2026-08-30) -- Chat libre se queda en "speaking"
    durante toda la respuesta, sin cambiar de cara al terminar. Si más
    adelante hace falta esa señal de nuevo, tiene que salir de algo real
    (un verificador, un score), no de la ausencia de contexto.

    Memoria episódica (2026-08-31, ver memoria_episodica.py): antes de
    generar, se busca con BM25 un turno pasado de Chat libre relevante para
    ESTE mensaje. A diferencia del RAG viejo de más arriba, acá el gate no
    es "score > 0" sino overlap real de palabras de contenido, así que no
    debería repetir ese bug -- igual, si un recuerdo se cuela sin venir al
    caso, es el primer sospechoso a revisar."""
    # Sin system prompt, a pedido del usuario: se manda SOLO el mensaje.
    # El comportamiento (frases cortas, no decir que es una IA) tiene que
    # venir del fine-tuning, no de una instruccion en cada turno -- una regla
    # de prompt es un pedido que un modelo chico incumple bajo presion, como
    # se vio en la v1 ("eres inteligencia artificial" -> "Si, soy
    # inteligencia artificial", teniendo la regla que lo prohibia).
    recuerdo = memoria_episodica.buscar_relevante(mensaje_usuario)
    contenido = f"{recuerdo}\n{mensaje_usuario}" if recuerdo else mensaje_usuario
    mensajes = [{"role": "user", "content": contenido}]
    texto_final = generar_respuesta(mensajes).strip()
    # Se registra el mensaje ORIGINAL (sin el recuerdo inyectado) -- lo que
    # revisa curar.py es lo que dijo el usuario, no el prompt armado.
    # Se registra DESPUES de tener la respuesta completa, no token a token:
    # lo que se revisa en curar.py es el turno entero. Ver
    # chat_libre_training/README.md.
    registro_chat.registrar(mensaje_usuario, texto_final, CHAT_MODEL)
    return texto_final


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
    # Cada entrada es una actividad/sesión con su nivel (bloque de ~5 preguntas):
    # tiene que calzar EXACTO con el campo "tema" de preguntas.jsonl (columna
    # "Actividad / Tema" del Excel). Si agregas/renombras temas en el dataset,
    # actualiza esta lista también.
    "Arte, música y cultura - Nivel 1",
    "Arte, música y cultura - Nivel 2",
    "Arte, música y cultura - Nivel 3",
    "Arte, música y cultura - Nivel 4",
    "Arte, música y cultura - Nivel 5",
    "Arte, música y cultura - Nivel 6",
    "Arte, música y cultura - Nivel 7",
    "Ciencia y naturaleza - Nivel 1",
    "Ciencia y naturaleza - Nivel 2",
    "Ciencia y naturaleza - Nivel 3",
    "Ciencia y naturaleza - Nivel 4",
    "Ciencia y naturaleza - Nivel 5",
    "Ciencia y naturaleza - Nivel 6",
    "Ciencia y naturaleza - Nivel 7",
    "Cultura general - Nivel 1",
    "Cultura general - Nivel 2",
    "Deporte y entretenimiento - Nivel 1",
    "Deporte y entretenimiento - Nivel 2",
    "Dilema del coche autónomo - Nivel 1",
    "Dilema del coche autónomo - Nivel 2",
    "Geografía - Nivel 1",
    "Geografía - Nivel 2",
    "Geografía - Nivel 3",
    "Geografía - Nivel 4",
    "Geografía - Nivel 5",
    "Geografía - Nivel 6",
    "Geografía - Nivel 7",
    "Geografía - Nivel 8",
    "Geografía - Nivel 9",
    "Historia - Nivel 1",
    "Historia - Nivel 2",
    "Historia - Nivel 3",
    "Interacción personalizada (comida) - Nivel 1",
    "Interacción personalizada (comida) - Nivel 2",
    "Juego de emociones - Nivel 1",
    "Juego de imitación - Nivel 1",
    "Juego de imitación - Nivel 2",
    "Matemática: multiplicación - Nivel 1",
    "Matemática: multiplicación - Nivel 2",
    "Matemática: multiplicación - Nivel 3",
    "Matemática: multiplicación - Nivel 4",
    "Razonamiento matemático - Nivel 1",
    "Razonamiento matemático - Nivel 2",
    "Razonamiento matemático - Nivel 3",
    "Razonamiento matemático - Nivel 4",
    "Razonamiento matemático - Nivel 5",
    "Razonamiento matemático - Nivel 6",
    "Razonamiento matemático - Nivel 7",
    "Reconocimiento musical - Nivel 1",
    "Reconocimiento musical - Nivel 2",
    "Socialización / presentación - Nivel 1",
]

# Los dos únicos temas del catálogo con veredicto real vía cámara en vez de
# texto (ver _jugar_emociones() más abajo) -- ninguna pregunta de estos temas
# trae respuesta_esperada, así que sin este caso especial caerían en la rama
# sin veredicto de manejar_trivia() (sin reacción hablada, ver ahí).
TEMAS_JUEGO_EMOCIONES = {
    "Juego de emociones - Nivel 1",
    "Juego de imitación - Nivel 1",
    "Juego de imitación - Nivel 2",
}


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


# Reacciones fijas para comentar_resultado() -- mismo patrón que
# SALUDOS_APERTURA/DESPEDIDAS (random.choice(), sin LLM).
#
# Hasta 2026-08-31 esto lo generaba un LLM (TRIVIA_MODEL/lora-trivia) para
# que "reaccionara con personalidad" -- se sacó por completo: aun con todo
# lo optimizable ya optimizado (num_ctx, gobernador de CPU en performance,
# reducir llamados de Ollama por turno -- ver TODO-mantenimiento.md) seguía
# tardando varios segundos por reacción, y de los ~14 LLMs chicos probados
# como reemplazo ninguno bajaba de ahí de forma confiable. La reacción es
# siempre la misma idea (acertaste / no, era X) así que no hacía falta
# generarla -- una lista fija la resuelve en 0ms, sin depender de Ollama.
RESPUESTAS_TRIVIA_ACIERTO = [
    "¡Exacto, {esperada}!",
    "Sí, correcto.",
    "Así es, nada mal.",
    "Correcto, bien ahí.",
]

RESPUESTAS_TRIVIA_ERROR = [
    "No, era {esperada}.",
    "Nop, la correcta es {esperada}.",
    "Incorrecto, la respuesta era {esperada}.",
    "No es así, correcta: {esperada}.",
]


def comentar_resultado(esperada, acerto):
    """Reacción corta al veredicto de una pregunta de Trivia con respuesta
    verificable (evaluar_respuesta(), sin LLM) -- ver la nota de
    RESPUESTAS_TRIVIA_ACIERTO/ERROR más arriba sobre por qué esto ya no usa
    TRIVIA_MODEL."""
    plantilla = random.choice(RESPUESTAS_TRIVIA_ACIERTO if acerto else RESPUESTAS_TRIVIA_ERROR)
    return plantilla.format(esperada=esperada)


def comentar_resultado_emocion(objetivo, detectada, acerto, persona_str, on_token=None):
    """Como comentar_resultado(), pero para el Juego de emociones/imitación:
    el veredicto no sale de comparar texto (Agent_Corrector) sino de la
    cámara (Clients/Camara_Client.py) contra la cara que se le pidió al
    usuario que hiciera -- ver _jugar_emociones(). Usa TRIVIA_MODEL, mismo
    criterio que comentar_resultado() -- incluido max_tokens=20 y "máximo 5
    palabras" en vez de "frase corta", ver la nota ahí."""
    if acerto:
        instruccion = (
            f"Le pediste al usuario que pusiera cara de {objetivo.lower()} y la cámara "
            f"detectó justo esa cara. Celébraselo en máximo 5 palabras."
        )
    else:
        instruccion = (
            f"Le pediste al usuario que pusiera cara de {objetivo.lower()} pero la cámara "
            f"detectó cara de {detectada.lower()}. Dile en máximo 5 palabras que no era esa, "
            "con humor."
        )
    mensajes = _mensajes_con_personalidad(persona_str, (
        f"{instruccion} No vuelvas a pedir otra cara ni hagas otra pregunta."
    ), modelo=TRIVIA_MODEL)
    return generar_respuesta(mensajes, on_token=on_token, modelo=TRIVIA_MODEL, max_tokens=20).strip()


def _jugar_emociones(pregunta, persona_str, on_token=None):
    """Ronda del Juego de emociones/imitación: elige al azar una cara entre
    las que trae la columna 'cara' de ESTA pregunta (ej. entre Enojado/
    Feliz/Triste), se la pide al usuario en voz alta, y la verifica con la
    cámara real (Clients/Camara_Client.py) -- mismo patrón que la rama CON
    respuesta_esperada de manejar_trivia(), pero el veredicto sale de la
    cámara en vez de Agent_Corrector.

    Devuelve el bool acerto/error, o None si no se pudo evaluar (cámara no
    disponible o sin cara detectada) -- en ese caso no cuenta para el
    marcador y no hay reacción hablada (ver el chequeo `detectada is None`
    más abajo)."""
    # La emoción a imitar sale de respuesta_esperada: en estas preguntas esa
    # columna NO es una respuesta de texto que se corrija con
    # Agent_Corrector (el veredicto lo da la cámara), es el enunciado del
    # juego -- qué cara tiene que poner Lora para que el usuario la imite.
    #
    # Antes salía de la columna 'cara' con random.choice sobre un split("/"),
    # asumiendo valores tipo "Enojado/Feliz/Triste". Los datos reales nunca
    # tuvieron ese formato: 'cara' trae un valor único, y es "Neutral" en 7
    # de las 10 preguntas del juego -- o sea que Lora pedía "hazme una cara
    # de neutral" casi siempre y el juego no tenía variedad ninguna.
    objetivo = (pregunta.get("respuesta_esperada") or "").strip()
    if not objetivo:
        # Fallback al comportamiento viejo para filas sin la columna cargada.
        opciones = [o.strip() for o in pregunta["cara"].split("/") if o.strip()]
        objetivo = random.choice(opciones) if opciones else "Feliz"
    # El robot presenta la cara que hay que imitar ANTES de pedirla por voz --
    # referencia visual, no solo el nombre hablado. Se mantiene en pantalla
    # mientras el usuario posa (ver display.mostrar_cara() más abajo, recién
    # cambia cuando sale el veredicto).
    display.mostrar_cara(cara_para_emocion(objetivo) or "content")
    pedido = f"¡Hazme una cara de {objetivo.lower()}!"
    print(f"Asistente: {pedido}")
    voz_output.hablar(pedido)
    time.sleep(1)  # un segundo para que el usuario pose antes de capturar
    detectada, confianza = Camara_Client.detectar_emocion()

    if detectada is None:
        # Este juego (a diferencia de manejar_trivia(), que sí usa
        # comentar_resultado()) sigue sin agente de comentario a propósito:
        # el veredicto sale de la cámara, no de texto. No hay veredicto acá
        # (sin cara detectada), así que no hay una cara de veredicto que
        # mostrar (queda 'speaking').
        display.mostrar_cara("speaking")
        time.sleep(PAUSA_ANTES_ACCION_FISICA)
        expresar_musica(pregunta)
        expresar_desplazamiento(pregunta)
        return None

    acerto = detectada == objetivo
    print(f"    [cámara] pedido={objetivo} detectado={detectada} (confianza {confianza:.0%}) -> {'OK' if acerto else 'MAL'}")
    # Veredicto hablado -- sin agente de comentario a propósito, ver la nota
    # más arriba: solo la palabra fija según acerto/error, no la lista de
    # frases de comentar_resultado(). Acá la voz hace falta igual: el
    # usuario está posando frente a la cámara, no mirando la pantalla, así
    # que no ve la cara de veredicto sin este aviso por voz.
    #
    # Orden a propósito: la VOZ va primero y bloquea (tiene que terminar de
    # decirse) -- recién después se disparan cara/música/desplazamiento
    # juntos, en paralelo entre sí (ver _reaccionar_veredicto()). Si la cara
    # cambiara antes de hablar, como era antes, el usuario vería el
    # veredicto en pantalla un instante antes de escucharlo -- con la voz
    # primero, los tres le llegan justo después de la palabra.
    voz_output.hablar("¡Correcto!" if acerto else "¡Incorrecto!")
    cara = elegir_cara_pregunta(pregunta, acerto) or ("happy" if acerto else "sad")
    _reaccionar_veredicto(cara, pregunta)
    display.mostrar_cara("speaking")
    return acerto


def resolver_tema(eleccion_usuario):
    """El usuario elige la sesión con texto libre ('quiero chistes', 'algo de
    colores', o directo 'Interaccion'); se mapea a EXACTAMENTE una categoría
    del catálogo por texto, sin LLM de por medio — las opciones que se le
    muestran ya son 5 strings literales sacados del propio catálogo (ver
    manejar_trivia más abajo), así que alcanza con match directo/difuso
    contra esa lista. Si no hay relación clara, cae en un tema al azar del
    catálogo -- NO en un tema fijo tipo "Trivia": desde la reorganización por
    niveles ya no existe una categoría "más grande" que sirva de catch-all
    genérico (son 51 sesiones de ~5 preguntas cada una), así que un fallback
    fijo apuntaría a un tema que puede no tener nada que ver, o peor, ya no
    existir -- bug real encontrado 2026-08-26: con el catálogo viejo "Trivia"
    tenía la mayoría de las preguntas y funcionaba como catch-all razonable,
    pero tras la reorganización "Trivia" dejó de ser un tema válido y el
    fallback quedaba muerto (0 preguntas, el robot anunciaba la tanda y al
    toque decía que no había nada)."""
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

    return random.choice(TEMAS_CATALOGO)


# ══════════════════════════════════════════════════════════════════════
# Router + loop de conversación
# ══════════════════════════════════════════════════════════════════════

RUTAS = ["TRIVIA", "CHAT_LIBRE"]

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
    # Ampliada 2026-08-27 a pedido del usuario: la lista original no cubría
    # variantes comunes de "quiero irme del trivia" ("basta", "ya no más",
    # "terminemos", etc.) -- se probó reemplazarla por Agent_Router.py (ya
    # existe, es rápido, sin LLM), pero se descartó: ese clasificador nunca
    # vio ejemplos de respuestas cortas de trivia ("cuarenta", "cinco por
    # ocho"), así que correrlo acá arriesgaba sacar al usuario de una
    # pregunta real por una respuesta corta mal clasificada. Ampliar la
    # lista es aditivo, sin ese riesgo -- a cambio de seguir siendo un
    # catálogo finito, no cualquier forma de decirlo.
    "basta", "ya basta", "no quiero jugar más", "no quiero jugar mas",
    "no quiero más preguntas", "no quiero mas preguntas",
    "olvida la trivia", "olvida el trivia", "no quiero trivia",
    "no más trivia", "no mas trivia",
    "cansado de las preguntas", "cansada de las preguntas",
    "quiero hacer otra cosa", "vamos a otra cosa",
    "suficiente trivia", "ya fue suficiente",
    "terminemos la trivia", "terminemos el trivia",
]


# Disparadores de que el usuario se puso a hablar de algo personal/una
# experiencia propia (problemas familiares, de estudio, etc.) en vez de
# responder la pregunta.
#
# Esta lista y _SALIR_TRIVIA YA NO son el mecanismo principal: hoy decide
# lora-salida-trivia (ver _quiere_salir_trivia() acá abajo y
# salida_trivia_training/). Quedan como fallback para los dos casos en que
# no se puede consultar al modelo -- sin pregunta pendiente que darle, o
# modelo/Ollama no disponibles.
#
# Por qué durante un tiempo esto fue solo keywords: se probó primero
# qwen2.5:0.5b sin reentrenar (few-shot) y falló, se equivocó en 4 de 5
# casos de prueba -- 0.5B es demasiado chico para este juicio semántico
# abierto (a diferencia del router, que es una clasificación acotada de
# pocas etiquetas). El modelo grande (llama3.2:3b-q4s) sí acertó 5/5, pero
# reintroduce el problema de recursos que llevó a sacarlo de CHAT_MODEL.
# Lo que destrabó el problema fue fine-tunear el chico para esta única
# decisión: lora-salida-trivia acierta 15/15 en ~0.14s por consulta y pesa
# 397MB, igual que CHAT_MODEL/TRIVIA_MODEL. Como fallback las keywords
# siguen teniendo la limitación de siempre: cubren las variantes más
# comunes, no cualquier forma de decirlo.
_TEMA_PERSONAL = [
    "me paso", "me pasó", "el otro dia", "el otro día",
    "cuando era", "recuerdo cuando", "me acuerdo cuando",
    "extraño a", "extraño mi", "mi profesor", "mi profesora",
    "mi maestro", "mi maestra", "mi mama", "mi mamá", "mi papa", "mi papá",
    "mi abuela", "mi abuelo", "mi amigo", "mi amiga",
    "problema con", "tuve un problema", "tengo un problema",
    "me regaño", "me regañó", "se pelearon", "se estan peleando",
    "se están peleando", "estoy triste", "estoy preocupado",
    "estoy preocupada", "me siento mal", "quiero contarte",
    "te quiero contar", "queria contarte", "quería contarte",
]


# Por debajo de esta cantidad de palabras, un mensaje sin ninguna keyword de
# salida se trata como respuesta directa sin consultar al LLM -- ver la nota
# de rendimiento en _quiere_salir_trivia(). La mayoría de las respuestas
# reales de Trivia son cortas (un número, un nombre, "no se"), así que este
# umbral cubre el caso común sin perder el juicio fino del LLM en mensajes
# largos (donde SÍ vale la pena, es donde más se equivocan las keywords).
_PALABRAS_RESPUESTA_CORTA = 4


def _quiere_salir_trivia(mensaje_usuario, pregunta_pendiente=None):
    """True si el usuario quiere dejar la trivia a mitad de una pregunta o
    elección de tema.

    Lo decide lora-salida-trivia (clasificar_salida_trivia()), que ve la
    pregunta pendiente y el mensaje juntos: eso le permite distinguir cosas
    que ninguna lista de palabras cubre -- una opinión larga a un dilema es
    una RESPUESTA legítima, mientras que contar un problema propio con las
    mismas palabras es SALIR.

    Optimización (2026-08-30): si el mensaje es corto (<=
    _PALABRAS_RESPUESTA_CORTA palabras) y no matchea ninguna keyword de las
    listas de abajo, es casi seguro una respuesta directa -- no vale la pena
    pagar otra llamada a Ollama (con su propio costo de contención medido
    con perf_report.py: Trivia hacía 2 llamadas por turno contra 1 de Chat
    libre) para confirmar algo tan obvio. El LLM se reserva para los casos
    donde de verdad hace falta el juicio fino: mensajes largos, o que sí
    matchean alguna keyword pero podrían ser ambiguos (ver el docstring
    original de esta función, arriba, sobre por qué las keywords solas no
    alcanzaban). Costo: en el caso raro de un mensaje CORTO que sea salida
    real sin ninguna keyword conocida, esto puede perderlo -- antes el LLM
    lo hubiera podido cachar, ahora no se lo consulta. Trade-off aceptado a
    pedido del usuario, a cambio de la mitad de las llamadas a Ollama en la
    mayoría de los turnos.

    Cae a las listas de palabras clave (_SALIR_TRIVIA/_TEMA_PERSONAL) en
    tres casos ahora: sin pregunta pendiente (eligiendo tema, fuera del
    dominio del modelo), clasificar_salida_trivia() devolvió None (Ollama
    caído/modelo sin importar), o el mensaje es corto y sin keywords (arriba).
    Así ningún problema de infraestructura corta el turno ni deja al usuario
    trabado en la trivia."""
    texto = mensaje_usuario.strip().lower()
    tiene_keyword = any(frase in texto for frase in _SALIR_TRIVIA) or any(frase in texto for frase in _TEMA_PERSONAL)

    es_corto_y_sin_keyword = len(texto.split()) <= _PALABRAS_RESPUESTA_CORTA and not tiene_keyword
    if pregunta_pendiente and not es_corto_y_sin_keyword:
        veredicto = clasificar_salida_trivia(pregunta_pendiente, mensaje_usuario)
        if veredicto is not None:
            return veredicto
    return tiene_keyword


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
# Reacción a una respuesta: manejar_trivia() usa comentar_resultado() (frase
# fija, sin LLM desde 2026-08-31) ANTES de esta pausa -- la voz bloquea hasta
# terminar de decirse, y recién ahí se mueve la cara al veredicto, se espera
# esta pausa, y se disparan música/desplazamiento (en paralelo entre sí,
# hilo/Popen, no bloquean). _jugar_emociones() sigue sin comentario generado
# a propósito (comentar_resultado_emocion() sigue sin uso, ver la nota ahí)
# -- el veredicto ahí sale de la cámara, con palabra fija en vez de frase de
# la lista de comentar_resultado().
PAUSA_ANTES_ACCION_FISICA = 1

# Cuánto esperar, como máximo, a que un teléfono se conecte y active la voz
# antes de la primera frase de la sesión (el saludo) -- ver
# _esperar_telefono_si_corresponde() más abajo.
ESPERAR_TELEFONO_SEG = int(os.environ.get("ESPERAR_TELEFONO_SEG", "60"))


def _esperar_telefono_si_corresponde(timeout=ESPERAR_TELEFONO_SEG):
    """Si VOZ_MOTOR=telefono, espera hasta `timeout` segundos a que
    voz_server.hay_cliente_conectado() sea True antes de dejar que el
    caller siga (y hable).

    Por qué hace falta: sin esto, el saludo de _main_loop() sale ~4s
    después de arrancar el proceso (ver PAUSA_CAMBIO_CARA ahí) -- nadie
    llega a abrir la página y tocar 'Activar voz' tan rápido, así que la
    primera frase de CADA sesión caía siempre al respaldo (edge-tts, nube),
    exactamente lo que VOZ_MOTOR=telefono busca evitar. El resto de la
    sesión no tiene este problema: para cuando el usuario ya escribió su
    nombre, tuvo de sobra para conectarse.

    Si nadie se conecta en `timeout` segundos, sigue igual que antes (la
    frase cae al respaldo) -- esto no cambia ese comportamiento, solo le da
    una chance real de no necesitarlo."""
    if voz_output.MOTOR != "telefono":
        return
    print(f"[voz] esperando hasta {timeout}s a que se conecte un teléfono con la voz activada...")
    esperado = 0
    while esperado < timeout and not voz_server.hay_cliente_conectado():
        time.sleep(1)
        esperado += 1
    if voz_server.hay_cliente_conectado():
        print(f"[voz] teléfono conectado a los {esperado}s, listo para saludar")
    else:
        print(f"[voz] nadie se conectó en {timeout}s -- el saludo cae al respaldo (edge-tts)")


def _reaccionar_veredicto(cara, pregunta, musica_ya_sonada=False):
    """Cara + música + desplazamiento del veredicto, EN PARALELO entre sí --
    se disparan los tres seguidos, sin esperas entre uno y otro: música
    (Popen) y desplazamiento (escritura serial ~instantánea, o hilo aparte
    para 'Girar 360°') ya son fire-and-forget, y mostrar_cara() es un
    request IPC corto a mpv, así que ninguno bloquea a los demás.

    El único bloqueo real de esta función es el sleep del final -- no es
    para "esperar" a música/desplazamiento (que ya corren solos), sino para
    darle tiempo a la cara de terminar de cargar en pantalla antes de que el
    próximo mostrar_cara("speaking") la tape (ver la nota de PAUSA_CAMBIO_CARA
    más abajo en manejar_trivia()/_jugar_emociones(), bug real encontrado
    2026-08-26: con solo 1s la cara de veredicto casi no se llegaba a ver).

    La voz (si hay reacción hablada) va SIEMPRE ANTES de llamar a esto, no
    acá adentro -- a pedido del usuario: primero la voz (bloqueante, tiene
    que terminar de decirse), y recién después cara/música/desplazamiento
    juntos. Ver _jugar_emociones(), el único caller con voz."""
    display.mostrar_cara(cara)
    # Si la canción ya sonó al lanzar la pregunta (Reconocimiento Musical,
    # ver _preguntar_siguiente), no se repite acá.
    if not musica_ya_sonada:
        expresar_musica(pregunta)
    expresar_desplazamiento(pregunta)
    time.sleep(PAUSA_CAMBIO_CARA)


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

    # Reconocimiento Musical: la canción ES el enunciado, no un adorno. Va
    # DESPUÉS de decir la pregunta ("te voy a poner una canción, ¿sabes cuál
    # es?") y ANTES de habilitar la respuesta, bloqueando -- si no, se le
    # pediría al usuario que adivine encima de la música recién arrancada.
    #
    # Bug que esto arregla: expresar_musica() solo se llamaba desde las
    # reacciones (_reaccionar_veredicto y la rama sin respuesta_esperada de
    # manejar_trivia), o sea DESPUÉS de que el usuario contestaba. Como las
    # 10 preguntas musicales del dataset tienen respuesta_esperada, la
    # canción sonaba recién al dar el veredicto: se le pedía adivinar algo
    # que todavía no había escuchado.
    #
    # Se recuerda en el estado para que la reacción de después no la repita
    # (serían 20s de la misma canción dos veces en el mismo turno).
    #
    # Mientras suena, la LCD muestra un countdown visual (faces/countdown.*,
    # ver display.py) en vez de quedarse en "speaking" sin moverse -- a
    # pedido del usuario (2026-08-30), le da al usuario una señal de que
    # el tiempo sigue corriendo mientras escucha. Vuelve a "speaking" apenas
    # termina, antes de la pausa/`content` de más abajo (mismo criterio que
    # el resto de la función: no se toca nada si la pregunta no trae música).
    es_musical = bool(actual.get("musical"))
    if es_musical:
        display.mostrar_cara("countdown")
    estado["musica_ya_sonada"] = bool(expresar_musica(actual, esperar=True))
    if es_musical:
        display.mostrar_cara("speaking")

    # La pregunta se imprime de una (no hay streaming que marque cuándo
    # "termina de hablar"), así que se le da un tiempo fijo en 'speaking'
    # -antes de pasar a la cara de reposo mientras espera que el usuario
    # conteste- para que de verdad se alcance a ver, no un flash instantáneo.
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")
    return True


def _cerrar_tanda(estado):
    """Mensaje de cierre cuando se acaba la tanda sola (cola_preguntas vacía,
    no por una salida a mitad de pregunta): ya no hay nada que retomar, así
    que el próximo mensaje vuelve a pasar por el Router como cualquier turno
    normal."""
    estado["en_trivia"] = False
    display.mostrar_cara("speaking")
    cierre = ("Esas eran las 5. ¿Seguimos con más trivia, "
              "buscamos algo en la web, o prefieres charlar?")
    print(f"Asistente: {cierre}\n")
    voz_output.hablar(cierre)
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")


def _correr_tanda_emociones(estado, persona_str, on_token):
    """Para TEMAS_JUEGO_EMOCIONES el veredicto sale de la cámara, no de una
    respuesta de texto -- a diferencia de Trivia normal, no tiene sentido
    esperar un mensaje del usuario entre pregunta y pregunta (no hay nada
    que responder por texto). Por eso esta función corre la tanda entera
    (hasta PREGUNTAS_POR_TANDA rondas) de un tirón: pregunta -> cara a
    imitar -> captura -> veredicto -> siguiente pregunta, sin devolver el
    control a _main_loop()/_leer_entrada() hasta que se acaba la tanda.

    Trade-off a propósito: mientras corre, no hay forma de que el usuario
    la interrumpa a mitad (_quiere_salir_trivia() solo se chequea entre
    turnos en _main_loop()) -- aceptable para un juego con cámara en vivo,
    igual que el resto del turno bloquea mientras habla/captura."""
    while _preguntar_siguiente(estado):
        pendiente = estado["pregunta_pendiente"]
        estado["pregunta_pendiente"] = None
        acerto = _jugar_emociones(pendiente, persona_str, on_token=on_token)
        if acerto is not None:
            estado["total"] += 1
            estado["aciertos"] += acerto
        print("\n")
    _cerrar_tanda(estado)


def _iniciar_tanda(tema, estado, persona_str=None, on_token=None):
    """Pide de una sola vez una tanda de PREGUNTAS_POR_TANDA preguntas de un
    tema ya elegido (un solo request al backend). Para TEMAS_JUEGO_EMOCIONES
    corre la tanda entera de una (ver _correr_tanda_emociones()); para el
    resto arranca solo con la primera pregunta -- las demás quedan en
    cola_preguntas para encadenarlas sin volver a pasar por el menú de temas,
    pero esperando la respuesta de texto de cada una antes de seguir."""
    preguntas = obtener_preguntas_por_tema(tema, estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
    if not preguntas:
        print(f"Asistente: No quedan preguntas de {tema}.\n")
        return
    estado["ya_usados"].update(p["id"] for p in preguntas)
    estado["cola_preguntas"] = preguntas
    if tema in TEMAS_JUEGO_EMOCIONES:
        _correr_tanda_emociones(estado, persona_str, on_token)
    else:
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
        _iniciar_tanda(tema, estado, persona_str, on_token)
        return

    pendiente = estado["pregunta_pendiente"]
    if pendiente is not None:
        # Nunca queda pendiente una pregunta de TEMAS_JUEGO_EMOCIONES acá --
        # esas se resuelven enteras, tanda completa, dentro de
        # _correr_tanda_emociones() (ver _iniciar_tanda()), sin devolver el
        # control hasta que se acaban. Lo que llega a esta rama es siempre
        # una respuesta de texto a corregir.
        estado["pregunta_pendiente"] = None
        display.mostrar_cara("speaking")  # generando el comentario -- ver la nota de orden más abajo
        if pendiente["respuesta_esperada"]:
            estado["total"] += 1
            acerto = evaluar_respuesta(pendiente["respuesta_esperada"], mensaje_usuario)
            estado["aciertos"] += acerto
            # Agent_Behavior.elegir_cara_pregunta() decide la cara específica
            # de ESTA pregunta (cara_respuesta_buena/mala del dataset, sin
            # LLM). Fallback genérico solo por si esa columna viniera vacía.
            cara = elegir_cara_pregunta(pendiente, acerto) or ("happy" if acerto else "sad")
            # Reacción hablada de vuelta con comentar_resultado() -- desde
            # 2026-08-31 es una frase fija (ver la nota ahí), no un LLM, pero
            # el orden se mantiene: voz PRIMERO (bloquea, tiene que terminar
            # de decirse), y recién después cara/música/desplazamiento juntos
            # (_reaccionar_veredicto()), mismo criterio que _jugar_emociones().
            comentario = comentar_resultado(pendiente["respuesta_esperada"], acerto)
            print(f"Asistente: {comentario}\n")
            voz_output.hablar(comentario)
            _reaccionar_veredicto(cara, pendiente, estado.get("musica_ya_sonada", False))
            display.mostrar_cara("speaking")
        else:
            # Solo 11 preguntas del dataset caen acá (verificado 2026-08-30):
            # "Dilema del coche autónomo" (Niveles 1 y 2, 10 preguntas de
            # opinión, sin respuesta correcta) + 1 de "Socialización/
            # presentación". Reconocimiento musical NO es de este grupo --
            # tiene respuesta_esperada (el nombre de la canción) y ya cae en
            # la rama de arriba (comentar_resultado()). Acá no hay veredicto
            # ni cara específica (queda la genérica de "hablando"), y a
            # pedido del usuario (2026-08-31) tampoco hay reacción hablada:
            # sin nada que corregir, se pasa directo a música/desplazamiento
            # (antes esto llamaba a reaccionar_libre(), un LLM que reaccionaba
            # sin evaluar nada -- se eliminó junto con la función).
            time.sleep(PAUSA_ANTES_ACCION_FISICA)
            if not estado.get("musica_ya_sonada", False):
                expresar_musica(pendiente)
            expresar_desplazamiento(pendiente)

        # Sigue encadenando la tanda; recién cuando se acaba se vuelve a
        # preguntar qué hacer (el Router igual reclasifica el próximo mensaje,
        # pero preguntarlo explícito evita que el usuario se quede sin saber
        # qué esperar después de la última pregunta).
        if not _preguntar_siguiente(estado):
            _cerrar_tanda(estado)
        return

    opciones = ", ".join(random.sample(TEMAS_CATALOGO, 5))
    estado["esperando_tema"] = True
    anuncio = f"Puedes elegir entre: {opciones}."
    print(f"Asistente: {anuncio}\n")
    voz_output.hablar(anuncio)


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


def manejar_chat_libre(mensaje_usuario, persona_str):
    display.mostrar_cara("speaking")  # generando la respuesta -- ya no se imprime token a token, ver responder()
    texto_final = responder(mensaje_usuario, persona_str)
    print(f"Asistente: {texto_final}\n")
    voz_output.hablar(texto_final)
    # Se queda en "speaking" durante toda la respuesta, sin cambiar de cara
    # al final -- a pedido del usuario (2026-08-30). Antes elegía happy/
    # sad-o-angry con Agent_Behavior.elegir_cara_por_calidad(), pero esa
    # elección dependía de una señal (`problema`) que quedó hardcodeada en
    # False desde que se sacó el RAG de Chat libre, así que en la práctica
    # siempre terminaba en "happy" -- se sacó del todo en vez de mantener
    # una elección que nunca elegía nada distinto.
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")  # cara de reposo hasta el próximo turno


def _precargar_uno(modelo):
    try:
        generar_respuesta([{"role": "user", "content": "hola"}], max_tokens=1, modelo=modelo)
    except Exception as e:
        print(f"[warmup] no se pudo precargar {modelo}: {e}")


def _precargar_salida_trivia():
    """Como _precargar_uno(), pero para SALIDA_TRIVIA_MODEL -- ese modelo no
    se usa con generar_respuesta() sino con clasificar_salida_trivia()
    (system prompt propio, ver Llama_Client.py), así que se precalienta con
    una llamada dummy a esa misma función en vez de reusar _precargar_uno.

    Antes este modelo NO se precargaba (solo CHAT_MODEL/TRIVIA_MODEL, ver
    _precargar_modelos()) pese a que el comentario de más abajo en
    main() ya decía "Ollama carga los 3 modelos" -- bug real encontrado
    2026-08-30 con perf_report.py: el primer _quiere_salir_trivia() de
    cada sesión (primera vez que el usuario responde algo raro a mitad de
    una trivia) pagaba la carga completa desde disco (~15-18s) a mitad de
    conversación, en vez de durante el warmup como los otros dos."""
    try:
        clasificar_salida_trivia("pregunta de prueba", "respuesta de prueba")
    except Exception as e:
        print(f"[warmup] no se pudo precargar {SALIDA_TRIVIA_MODEL}: {e}")


def _precargar_modelos():
    """Dispara un mensaje mínimo a CHAT_MODEL, TRIVIA_MODEL y
    SALIDA_TRIVIA_MODEL para que Ollama los cargue en RAM ANTES de que
    llegue el primer mensaje real del usuario. Sin esto, la primera
    respuesta de la sesión (o, para SALIDA_TRIVIA_MODEL, el primer intento
    de salir de una trivia) paga la carga completa desde disco (~15-18s
    medidos en esta Pi, contra ~1-5s ya en caliente -- misma causa que el
    benchmark de OLLAMA_KEEP_ALIVE en TODO-mantenimiento.md, pero acá
    aplica siempre, no solo tras inactividad).

    Un hilo por modelo, en paralelo: ya no hay motivo para encadenarlos
    (VERIFICADOR_MODEL se sacó del todo -- Chat libre ahora es una sola
    llamada a CHAT_MODEL, sin una segunda pasada de verificación detrás que
    compitiera por CPU con la primera).

    Hilos daemon: no se esperan entre sí ni bloquean el saludo/lectura del
    nombre de _main_loop(), que corre en paralelo y le da a esto varios
    segundos de ventana antes de la primera pregunta real. Si Ollama no
    responde (apagado, sin red), se loguea y se sigue -- la primera
    respuesta real simplemente paga el costo de cargar como si no hubiera
    warmup, no rompe nada."""
    threading.Thread(target=_precargar_uno, args=(CHAT_MODEL,), daemon=True).start()
    threading.Thread(target=_precargar_uno, args=(TRIVIA_MODEL,), daemon=True).start()
    threading.Thread(target=_precargar_salida_trivia, daemon=True).start()


def main():
    # Carga el modelo de voz PRIMERO, en el hilo principal — cargar un
    # modelo ONNX desde un hilo secundario se cuelga silenciosamente en este
    # entorno (ver la nota histórica en Clients/Voice_Output_Client.py), por
    # eso esto va antes de arrancar cualquier threading.Thread(). Con
    # edge-tts cargar() ya no hace nada, pero se deja la llamada por si
    # algún día se vuelve a un motor local.
    voz_output.cargar()

    perf_monitor.iniciar_muestreo_recursos()  # CPU/memoria del proceso cada pocos segundos, ver perf_report.py
    _precargar_modelos()  # Ollama carga los 3 modelos en background mientras suena el saludo

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


# 4 variantes del saludo de apertura, a pedido del usuario (2026-08-30) --
# antes era un único texto fijo, siempre igual sesión tras sesión. Las 4
# terminan pidiendo el nombre (necesario para el resto de _main_loop()), y
# se elige una al azar por sesión con random.choice(), no rotando en orden.
SALUDOS_APERTURA = [
    "Hola, mi nombre es Lora, ¿cuál es tu nombre?",
    "¡Hola! Soy Lora. ¿Y vos cómo te llamás?",
    "Hola, hola. Soy Lora, tu robot. ¿Quién sos vos?",
    "¡Buenas! Me llamo Lora. Contame tu nombre.",
]

# 4 variantes de despedida, mismo criterio que SALUDOS_APERTURA -- se eligen
# al azar al cerrar la sesión (ver el final de _main_loop()). "{nombre}" se
# completa con estado["nombre"] (el que contestó al arrancar, o "amigo" si
# no dijo nada).
DESPEDIDAS = [
    "¡Hasta luego, {nombre}! Que te vaya bien.",
    "Nos vemos, {nombre}. ¡Fue un gusto!",
    "Listo por hoy, {nombre}. ¡Cuidate!",
    "¡Chau, {nombre}! Volvé cuando quieras.",
]


def _main_loop():
    _esperar_telefono_si_corresponde()  # ver la nota ahí -- evita que el saludo caiga siempre a edge-tts
    display.mostrar_cara("content")  # arranca en reposo...
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("speaking")  # ...y recién ahora la IA "habla" (saluda, por defecto)
    saludo = random.choice(SALUDOS_APERTURA)
    print(f"Asistente: {saludo}")
    voz_output.hablar(saludo)
    # Vuelve a 'content' ANTES de esperar el input, no después — igual que en
    # el resto del archivo (_preguntar_siguiente, manejar_chat_libre, etc.):
    # la cara de reposo tiene que estar puesta mientras se espera al usuario,
    # no quedarse en "hablando" durante todo ese rato.
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")
    nombre = _leer_entrada() or "amigo"
    # "buscarte algo de actualidad" se sacó (2026-08-30): mencionaba la
    # búsqueda web (DuckDuckGo), eliminada del proyecto hace tiempo -- este
    # texto había quedado desactualizado. Ahora presenta explícitamente las
    # dos opciones reales (Trivia / Chat libre) en vez de una mención vaga
    # a "cambiar de modo según lo que pidas".
    bienvenida = (f"Mucho gusto, {nombre}. Podemos jugar Trivia o simplemente charlar "
                  "-- vos decidís, decime qué querés hacer.")
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
    # el tema exacto (uno de TEMAS_CATALOGO) de la tanda en curso -- lo usan
    # _iniciar_tanda() (para saber si corre la tanda entera de un tirón vía
    # _correr_tanda_emociones(), TEMAS_JUEGO_EMOCIONES) y elegir_cara_pregunta()
    # indirectamente a través de cada pregunta. nombre: lo que contestó al
    # arrancar la sesión (o "amigo" si no dijo nada) -- se guarda en el
    # estado para poder despedirse por su nombre al final, no solo usarlo en
    # el saludo inicial.
    estado = {
        "pregunta_pendiente": None, "esperando_tema": False, "en_trivia": False,
        "cola_preguntas": [], "ya_usados": set(), "aciertos": 0, "total": 0,
        "tema_actual": None, "nombre": nombre, "musica_ya_sonada": False,
    }

    while True:
        entrada = _leer_entrada()
        if entrada.lower() in ("salir", "exit", "quit"):
            break
        if not entrada:
            continue

        if estado["en_trivia"]:
            # La pregunta pendiente es el contexto que necesita el modelo
            # para juzgar si `entrada` la está respondiendo. Puede no haber
            # (el usuario está eligiendo tema): ahí _quiere_salir_trivia()
            # se arregla con las keywords.
            _pendiente = estado["pregunta_pendiente"]
            if _quiere_salir_trivia(entrada, _pendiente["pregunta"] if _pendiente else None):
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
        else:
            manejar_chat_libre(entrada, persona_str)

    # Despedida real -- con voz y cara, como cualquier otro mensaje del
    # robot (antes era un print() plano, sin hablar ni cambiar de cara, la
    # única respuesta de toda la sesión que no lo hacía). Usa el nombre
    # guardado en estado["nombre"] al arrancar, no un genérico "hasta luego".
    display.mostrar_cara("speaking")
    if estado["total"]:
        resumen = f"Terminamos. Acertaste {estado['aciertos']} de {estado['total']}."
        print(f"\nAsistente: {resumen}")
        voz_output.hablar(resumen)
        time.sleep(PAUSA_CAMBIO_CARA)
        display.mostrar_cara("speaking")
    despedida = random.choice(DESPEDIDAS).format(nombre=estado["nombre"])
    print(f"Asistente: {despedida}")
    voz_output.hablar(despedida)
    display.mostrar_cara("happy")
    # display.detener() ya no va acá -- lo maneja el finally de main(), que
    # cubre también la salida por excepción, no solo esta salida normal.


if __name__ == "__main__":
    main()
