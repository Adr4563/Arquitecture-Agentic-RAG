"""
Agente de caras: decide qué cara (happy/sad/angry) le corresponde a una
respuesta de Chat libre o Búsqueda Web, según si el verificador la encontró
buena o tuvo que corregirla.

Mismo criterio de 3 opciones que corrector.elegir_cara() usa para Trivia
(acierto -> happy, error -> sad/angry al azar), pero acá el veredicto no es
"¿la respuesta del usuario es correcta?" sino "¿la respuesta del asistente
estaba bien o hubo que corregirla?" (ver verificador.verificar_y_corregir).
No hace falta una llamada aparte al modelo para esto: el veredicto ya lo
tiene el verificador, esta función solo lo traduce a una cara.
"""

import random

CARA_BUENA = "happy"
CARAS_MALA = ["sad", "angry"]


def elegir_cara(fue_corregida):
    """fue_corregida viene de verificador.verificar_y_corregir(): True si la
    respuesta estaba mal y hubo que arreglarla, False si ya estaba bien."""
    return random.choice(CARAS_MALA) if fue_corregida else CARA_BUENA
