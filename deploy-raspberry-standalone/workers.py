"""
Workers: la lógica de cada modo (Trivia, Búsqueda Web, Chat libre).

El manager (chat.py) solo decide a cuál mandar cada turno; toda la
lógica de qué hacer una vez decidido vive acá.
"""

import difflib

import httpx

from llama_client import generar_respuesta
from personalidad import obtener_system_prompt
from preguntas import SIN_CONTEXTO, recuperar_contexto
from preguntas import pregunta_aleatoria as _pregunta_aleatoria
from preguntas import pregunta_por_tema as _pregunta_por_tema
from preguntas import preguntas_por_tema as _preguntas_por_tema
from verificador import verificar_y_corregir

RESPUESTA_SIN_CONTEXTO = "No tengo información sobre eso en mi base de datos."

# Saludos/mensajes triviales: no ameritan gastar tiempo en la búsqueda BM25.
SALUDOS_TRIVIALES = {
    "hola", "buenas", "hey", "qué tal", "que tal", "hi", "hello",
    "buenos días", "buenas tardes", "buenas noches", "gracias", "ok", "vale",
}


def _es_mensaje_trivial(mensaje):
    return mensaje.strip().lower().strip("¡!¿?.,") in SALUDOS_TRIVIALES


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


# ─── Chat libre ────────────────────────────────────────────────

def generar_apertura(persona_str, on_token=None):
    """El robot abre la sesión: se presenta y anuncia que va a preguntar.

    No debe cerrar preguntando "¿quieres responder algunas preguntas?": el bucle
    arranca con la primera pregunta del dataset sin esperar confirmación, así que
    una pregunta acá quedaría sin responder y la interacción se ve rota.
    """
    mensajes = [
        {"role": "system", "content": obtener_system_prompt(persona_str)},
        {"role": "user", "content": (
            "Instrucción: inicia tú la conversación, todavía no hay mensaje del "
            "usuario. Preséntate en una frase muy breve (con tu personalidad, sin "
            "sonar a asistente genérico) y avisa que le vas a hacer preguntas y que "
            "él responde. Es un aviso, no una pregunta: no cierres preguntando nada "
            "ni pidas permiso ni confirmación."
        )},
    ]
    # Sin streaming a propósito: hay que ver el texto completo para poder recortar
    # una pregunta final, y eso no se puede hacer si ya se imprimió token a token.
    return _quitar_pregunta_final(generar_respuesta(mensajes, on_token=on_token))


def responder(mensaje_usuario, persona_str, n_results=2, on_token=None):
    """Devuelve (texto_final, fue_corregida) — fue_corregida es el veredicto
    del verificador (True = tuvo que arreglar la respuesta), que cara_agente
    usa para elegir happy vs sad/angry."""
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

    mensajes = [
        {"role": "system", "content": obtener_system_prompt(persona_str)},
        {"role": "user", "content": user},
    ]

    # Sin streaming a propósito: el agente verificador necesita el texto
    # completo para revisarlo antes de que el usuario lo vea — si se
    # imprimiera token a token, ya estaría en pantalla para cuando se
    # detectara que hay que corregirlo.
    borrador = generar_respuesta(mensajes).strip()
    texto_final, fue_corregida = verificar_y_corregir(mensaje_usuario, borrador)
    if on_token:
        on_token(texto_final)
    return texto_final, fue_corregida


# ─── Trivia ────────────────────────────────────────────────────
#
# Es el flujo para el que está hecho el dataset (robot EVA): cada registro trae
# la pregunta, la respuesta_esperada con la que corregir, y la 'cara' que el
# robot debe poner.

# Catálogo de temas/actividades (columna "Actividad / Tema" del Excel, campo
# "tema" en preguntas.jsonl). resolver_tema() clasifica contra esta lista.
TEMAS_CATALOGO = [
    # "númerica" así, mal tildada, porque es el texto tal cual viene del Excel
    # (columna Actividad/Tema) — tiene que calzar exacto con preguntas.jsonl.
    "Adivinanza númerica", "Chistes", "Dilema", "Dilema del coche autónomo",
    "Interaccion personalizada (COMIDA)", "Juego de colores", "Juego de emociones",
    "Juego de imitación", "Juego de multiplicar nivel Alto 1",
    "Juego de multiplicar nivel Alto 2", "Juego de multiplicar nivel Simple 1",
    "Juego de multiplicar nivel Simple 2", "Prueba de multiplicar nivel Simple",
    "Prueba de reconocimiento de Color", "Reconocimiento Musical",
    "Reconocimiento visual", "Trivia",
]


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
    mensajes = [
        {"role": "system", "content": obtener_system_prompt(persona_str)},
        {"role": "user", "content": (
            f"Pregunta que se hizo: {pregunta}\n\n"
            f"{instruccion} No vuelvas a resolver la pregunta ni expliques el cálculo: "
            "el veredicto ya está decidido, tu única tarea es reaccionar a él. "
            "No hagas otra pregunta."
        )},
    ]
    return generar_respuesta(mensajes, on_token=on_token).strip()


def reaccionar_libre(pregunta, respuesta_usuario, persona_str, on_token=None):
    """Para preguntas sin respuesta_esperada (Chistes, Reconocimiento Musical,
    Juego de colores...): no hay nada que corregir, así que en vez de
    evaluar_respuesta/comentar_resultado el robot solo reacciona."""
    mensajes = [
        {"role": "system", "content": obtener_system_prompt(persona_str)},
        {"role": "user", "content": (
            f"Pregunta: {pregunta}\n"
            f"Respuesta del usuario: {respuesta_usuario}\n\n"
            "Reacciona en una frase corta. No hay respuesta correcta acá: no "
            "corrijas ni evalúes, solo reacciona con tu personalidad. No hagas "
            "otra pregunta."
        )},
    ]
    return generar_respuesta(mensajes, on_token=on_token).strip()


def resolver_tema(eleccion_usuario):
    """El usuario elige la sesión con texto libre ('quiero chistes', 'algo de
    colores', o directo 'Interaccion'); se mapea a EXACTAMENTE una categoría
    del catálogo por texto, sin LLM de por medio — las opciones que se le
    muestran ya son 5 strings literales sacados del propio catálogo (ver
    chat.py), así que alcanza con match directo/difuso contra esa lista. Si no
    hay relación clara, cae en Trivia (la sesión con más datos)."""
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


# ─── Búsqueda web ──────────────────────────────────────────────
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
    mandarlo a buscar — mismo patrón que resolver_tema() para Trivia."""
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
    return generar_respuesta(mensajes, temperature=0, max_tokens=15).strip()


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

    mensajes = [
        {"role": "system", "content": obtener_system_prompt(persona_str)},
        {"role": "user", "content": (
            f"Pregunta: {mensaje_usuario}\n"
            f"Resultado de la búsqueda web: {resultado}\n\n"
            "Responde con esa información, en una frase corta."
        )},
    ]

    # Sin streaming: el verificador necesita el texto completo antes de que
    # el usuario lo vea (ver la misma nota en responder(), chat libre).
    borrador = generar_respuesta(mensajes).strip()
    texto_final, fue_corregida = verificar_y_corregir(mensaje_usuario, borrador)
    if on_token:
        on_token(texto_final)
    return texto_final, fue_corregida
