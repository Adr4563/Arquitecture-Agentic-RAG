"""
Agente corrector: decide si la respuesta del usuario a una pregunta de Trivia
es correcta o no. Solo eso — un veredicto objetivo (CORRECTO/INCORRECTO),
no una reacción con personalidad. Corre con temperature=0 porque calificar
debe ser determinista, no creativo.

Qué cara le corresponde a ese veredicto ya NO se decide acá: eso lo resuelve
Agent_Behavior.py (sin LLM, a partir de las columnas cara_respuesta_buena/
cara_respuesta_mala de la pregunta) — separado a propósito, porque elegir
una cara no necesita al modelo, y este archivo es el único punto del sistema
que sí lo necesita para emitir el veredicto.
"""

from Clients.Llama_Client import generar_respuesta


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
