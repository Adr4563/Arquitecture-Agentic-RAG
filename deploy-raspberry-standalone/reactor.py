"""
Agente de expresión: decide qué cara mostrar y qué acciones de música/
desplazamiento corresponden a la pregunta de trivia actual — sin LLM, 100%
determinista a partir de las columnas del dataset (cara_respuesta_buena,
cara_respuesta_mala, musical, desplazamiento en preguntas.jsonl).

Reemplaza a corrector.elegir_cara() (que elegía sad/angry al azar, igual
para cualquier pregunta) por la cara específica que trae cada fila del
dataset. Se invoca para CUALQUIER pregunta de Trivia, tenga o no
respuesta_esperada — Reconocimiento Musical, Chistes, etc. son temas del
mismo catálogo (TEMAS_CATALOGO), no un modo aparte del router.

Desplazamiento SÍ está conectado a hardware real: manda el comando al
carrito mecanum vía carrito_client.py (ver ese archivo para el protocolo y
la variable CARRITO_HOST). Si el carrito no está prendido/conectado, la
llamada falla gracioso (log, no excepción) y el resto del turno de trivia
sigue normal. Música todavía no tiene reproducción real (no hay bocina en
este despliegue) — por ahora solo se loguea. Si una columna viene vacía, no
se hace nada con esa parte — ni log, ni acción.
"""

import carrito_client

# El dataset usa nombres en español (columna 'cara' original); display.py
# espera las claves en inglés de los archivos en faces/*.gif. "Neutral" mapea
# a "content" porque es la misma cara de reposo que ya usa chat.py en el
# resto del flujo — no hay un .gif de "neutral" aparte.
_CARA_A_DISPLAY = {
    "Feliz": "happy",
    "Triste": "sad",
    "Enojado": "angry",
    "Neutral": "content",
}


def elegir_cara(pregunta, acerto):
    """Cara para el veredicto de una pregunta CON respuesta_esperada:
    acerto=True -> cara_respuesta_buena, acerto=False -> cara_respuesta_mala,
    traducida a la clave que entiende display.mostrar_cara(). None si la
    columna viene vacía o con un valor que no se pudo mapear."""
    campo = "cara_respuesta_buena" if acerto else "cara_respuesta_mala"
    valor = (pregunta.get(campo) or "").strip()
    return _CARA_A_DISPLAY.get(valor)


def expresar_musica(pregunta):
    """Si la pregunta trae algo en 'musical', lo expresa (por ahora, log por
    consola). Vacío/ausente -> no hace nada, devuelve None."""
    musica = (pregunta.get("musical") or "").strip()
    if not musica:
        return None
    print(f"    [música: {musica}]")
    return musica


def expresar_desplazamiento(pregunta):
    """Si la pregunta trae algo en 'desplazamiento', se lo manda de verdad al
    carrito mecanum (carrito_client.py) — 'Girar 360°' usa mover_360()
    (varios pulsos de rotación seguidos, sin comando único en el firmware);
    el resto (Adelante/Atrás/Izquierda/Derecha) es un solo mover(). Vacío/
    ausente -> no hace nada, devuelve None. Si el carrito no responde (no
    configurado, apagado, fuera de red), no corta el turno de trivia — solo
    queda logueado por carrito_client."""
    desplazamiento = (pregunta.get("desplazamiento") or "").strip()
    if not desplazamiento:
        return None
    print(f"    [desplazamiento: {desplazamiento}]")
    if desplazamiento.lower().startswith("girar"):
        carrito_client.mover_360()
    else:
        carrito_client.mover(desplazamiento)
    return desplazamiento


def reaccionar(pregunta, acerto=None):
    """Punto de entrada único: decide (y por ahora loguea) las señales que
    correspondan a esta pregunta.

    acerto=None es para preguntas SIN respuesta_esperada (Chistes,
    Reconocimiento Musical...) — no hay veredicto, así que no se elige cara
    acá (esas siguen con la cara genérica "speaking" que ya pone chat.py);
    solo se revisan música/desplazamiento, que no dependen de un veredicto.
    """
    cara = elegir_cara(pregunta, acerto) if acerto is not None else None
    musica = expresar_musica(pregunta)
    desplazamiento = expresar_desplazamiento(pregunta)
    return {"cara": cara, "musica": musica, "desplazamiento": desplazamiento}
