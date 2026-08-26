"""
Agente de comportamiento: decide qué cara mostrar (y, para Trivia, qué
acciones de música/desplazamiento disparar) — sin LLM, 100% determinista.
Fusiona lo que antes eran reactor.py + cara_agente.py, porque las dos cosas
son la misma idea (elegir una cara a partir de un veredicto ya resuelto) con
dos fuentes de datos distintas:

- elegir_cara_pregunta()/expresar_musica()/expresar_desplazamiento()/
  reaccionar(): para Trivia, a partir de las columnas del dataset
  (cara_respuesta_buena, cara_respuesta_mala, musical, desplazamiento en
  preguntas.jsonl) — se invoca para CUALQUIER pregunta de Trivia, tenga o no
  respuesta_esperada (Reconocimiento Musical, Chistes, etc. son temas del
  mismo catálogo, no un modo aparte del router).
- elegir_cara_por_calidad(): para Chat libre y Búsqueda Web, donde no hay
  fila de dataset asociada — el veredicto es "¿la respuesta del asistente
  estaba bien o hubo que corregirla?" (Agent_Verificator.verificar_y_corregir),
  y la cara sale de una elección genérica happy / sad-o-angry al azar.

Desplazamiento y música SÍ están conectados a hardware real: el primero manda
el comando al carrito mecanum vía Clients/Carrito_Client.py, el segundo
reproduce el audio (recortado a Musica_Client.REPRODUCCION_MAX_SEG) vía
Clients/Musica_Client.py (mpv). En los dos casos, si el hardware no responde
(carrito apagado, mpv no instalado, archivo inexistente), la llamada falla
gracioso — log, no excepción — y el resto del turno de trivia sigue normal.
Si una columna viene vacía, no se hace nada con esa parte — ni log, ni acción.
"""

import random

from Clients import Carrito_Client, Musica_Client

# El dataset usa nombres en español (columna 'cara' original); display.py
# espera las claves en inglés de los archivos en faces/*.gif. "Neutral" mapea
# a "content" porque es la misma cara de reposo que ya usa
# Orchestrator_Management.py en el resto del flujo — no hay un .gif de
# "neutral" aparte.
_CARA_A_DISPLAY = {
    "Feliz": "happy",
    "Triste": "sad",
    "Enojado": "angry",
    "Neutral": "content",
}

# ─── Trivia: cara/música/desplazamiento a partir del dataset ────────────

def elegir_cara_pregunta(pregunta, acerto):
    """Cara para el veredicto de una pregunta CON respuesta_esperada:
    acerto=True -> cara_respuesta_buena, acerto=False -> cara_respuesta_mala,
    traducida a la clave que entiende display.mostrar_cara(). None si la
    columna viene vacía o con un valor que no se pudo mapear."""
    campo = "cara_respuesta_buena" if acerto else "cara_respuesta_mala"
    valor = (pregunta.get(campo) or "").strip()
    return _CARA_A_DISPLAY.get(valor)


def expresar_musica(pregunta):
    """Si la pregunta trae algo en 'musical', reproduce ese archivo de
    musica/ de verdad (Musica_Client.py, recortado a 20s). Vacío/ausente ->
    no hace nada, devuelve None."""
    musica = (pregunta.get("musical") or "").strip()
    if not musica:
        return None
    print(f"    [música: {musica}]")
    Musica_Client.reproducir(musica)
    return musica


def expresar_desplazamiento(pregunta):
    """Si la pregunta trae algo en 'desplazamiento', se lo manda de verdad al
    carrito mecanum (Carrito_Client.py) — 'Girar 360°' usa mover_360()
    (varios pulsos de rotación seguidos, sin comando único en el firmware);
    el resto (Adelante/Atrás/Izquierda/Derecha) es un solo mover(). Vacío/
    ausente -> no hace nada, devuelve None. Si el carrito no responde (no
    configurado, apagado, fuera de red), no corta el turno de trivia — solo
    queda logueado por Carrito_Client."""
    desplazamiento = (pregunta.get("desplazamiento") or "").strip()
    if not desplazamiento:
        return None
    print(f"    [desplazamiento: {desplazamiento}]")
    if desplazamiento.lower().startswith("girar"):
        Carrito_Client.mover_360()
    else:
        Carrito_Client.mover(desplazamiento)
    return desplazamiento


def reaccionar(pregunta, acerto=None):
    """Punto de entrada único de Trivia: decide (y por ahora loguea) las
    señales que correspondan a esta pregunta.

    acerto=None es para preguntas SIN respuesta_esperada (Chistes,
    Reconocimiento Musical...) — no hay veredicto, así que no se elige cara
    acá (esas siguen con la cara genérica "speaking" que ya pone
    Orchestrator_Management.py); solo se revisan música/desplazamiento, que
    no dependen de un veredicto.
    """
    cara = elegir_cara_pregunta(pregunta, acerto) if acerto is not None else None
    musica = expresar_musica(pregunta)
    desplazamiento = expresar_desplazamiento(pregunta)
    return {"cara": cara, "musica": musica, "desplazamiento": desplazamiento}


# ─── Chat libre / Búsqueda web: cara genérica por calidad ────────────────

CARA_BUENA = "happy"
CARAS_MALA = ["sad", "angry"]


def elegir_cara_por_calidad(fue_corregida):
    """fue_corregida viene de Agent_Verificator.verificar_y_corregir(): True
    si la respuesta estaba mal y hubo que arreglarla, False si ya estaba
    bien."""
    return random.choice(CARAS_MALA) if fue_corregida else CARA_BUENA
