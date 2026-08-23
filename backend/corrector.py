"""
Agente corrector: decide si la respuesta del usuario a una pregunta de Trivia
es correcta o no, y qué cara le corresponde a ese veredicto en la pantalla
LCD (frontend/faces/*.gif).

Separado del resto de los workers a propósito: es el único punto del sistema
que emite un veredicto objetivo (CORRECTO/INCORRECTO) en vez de una reacción
con personalidad, y corre con temperature=0 porque calificar debe ser
determinista, no creativo.
"""

import random

from llama_client import generar_respuesta

# Ante un acierto siempre la misma cara (no hay ambigüedad en "bien hecho").
# Ante un error se alterna entre dos para que la reacción no se sienta
# repetitiva pregunta tras pregunta.
CARA_ACIERTO = "happy"
CARAS_ERROR = ["sad", "angry"]


def elegir_cara(acerto):
    """Cara a mostrar en la LCD según el veredicto de evaluar_respuesta():
    'happy' si acertó, 'sad' o 'angry' (al azar) si no."""
    return CARA_ACIERTO if acerto else random.choice(CARAS_ERROR)


def evaluar_respuesta(pregunta, esperada, respuesta_usuario):
    """Compara la respuesta del usuario con la esperada. Devuelve True/False.

    Se evalúa con el modelo y no comparando strings porque el usuario escribe
    en libre: "paris", "es París", "la capital es paris" son todas correctas
    para una esperada de "París", y un == las daría por malas.
    """
    mensajes = [
        {"role": "system", "content": (
            "Eres un corrector. Te dan una pregunta, la respuesta correcta y la que "
            "dio un estudiante. Decide si el estudiante acertó.\n"
            "Ignora mayúsculas, tildes, artículos y redacción: solo importa si el "
            "contenido coincide. Si la respuesta tiene varios elementos, acierta si "
            "menciona los principales.\n"
            "Responde EXACTAMENTE con una palabra: CORRECTO o INCORRECTO."
        )},
        {"role": "user", "content": (
            f"Pregunta: {pregunta}\n"
            f"Respuesta correcta: {esperada}\n"
            f"Respuesta del estudiante: {respuesta_usuario}"
        )},
    ]
    # temperature=0: corregir es determinista, no queremos variación entre corridas.
    veredicto = generar_respuesta(mensajes, temperature=0, max_tokens=5).upper()
    # Ojo: "CORRECTO" es subcadena de "INCORRECTO", así que un `in` a secas da
    # True para ambos y nadie reprueba nunca. Hay que descartar el negativo antes.
    return "INCORRECTO" not in veredicto and "CORRECTO" in veredicto
