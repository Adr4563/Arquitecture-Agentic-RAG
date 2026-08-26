"""
Personalidad Big Five (OCEAN) y el system prompt que de ella depende.
"""

import os

# Modelos que ya tienen la personalidad y las reglas de estilo entrenadas en
# los pesos (fine-tuning LoRA, ver README "Personalidad horneada en el
# modelo"): "ereberus-personalidad" (general, las 4 categorías),
# "ereberus-trivia" (solo trivia_training/, especializado en Trivia) y
# "ereberus-chat" (solo chat_training/, especializado en Chat libre/
# Búsqueda web) -- mandarles este system prompt de nuevo sería redundante
# (y les costaría tokens de más en cada turno, justo lo que existen para
# evitar). Con cualquier otro modelo (qwen2.5:0.5b, el de 3B, etc.) el
# prompt se manda como siempre.
_MODELOS_CON_PERSONALIDAD_HORNEADA = {"ereberus-personalidad", "ereberus-trivia", "ereberus-chat"}

# ─── Personalidad Big Five (OCEAN) ───────────────────────────
BIG_FIVE_TRAITS = [
    ("Extraversion", "extroverted", "introverted"),
    ("Agreeableness", "agreeable", "antagonistic"),
    ("Conscientiousness", "conscientious", "unconscientious"),
    ("Neuroticism", "neurotic", "emotionally stable"),
    ("Openness", "open to experience", "closed to experience"),
]


_INTENSIDAD = {1: "very", 2: "somewhat"}  # matiz para valores alejados del centro (5 = sin matiz)

# Valor por rasgo, 1=muy bajo ... 5=muy alto (3 = sin inclinación). Fijo, no se
# pregunta interactivo.
#
# Está por rasgo y no como un único valor global a propósito: bajar todo junto
# arrastra Conscientiousness, y con un modelo de 3B eso lo vuelve descuidado —
# llega a negar datos que sí tiene en el contexto ("no tengo información sobre
# la capital de Francia" teniendo "París" delante). Se deja alto para que use
# los datos, y el tono seco se consigue con Extraversion/Agreeableness bajos.
VALORES = {
    "Extraversion": 2,        # poco efusivo, va al grano
    "Agreeableness": 2,       # seco, no complaciente
    "Conscientiousness": 5,   # alto a propósito: se apega al contexto y no inventa
    "Neuroticism": 2,         # estable, no dramático
    "Openness": 3,
}


def construir_personalidad():
    rasgos = []
    for nombre, alto, bajo in BIG_FIVE_TRAITS:
        valor = VALORES[nombre]

        if valor == 3:
            rasgos.append(f"balanced between {alto} and {bajo}")
        elif valor > 3:
            matiz = _INTENSIDAD.get(5 - valor + 1, "")  # 5->"very", 4->"somewhat"
            rasgos.append(f"{matiz} {alto}".strip())
        else:
            matiz = _INTENSIDAD.get(valor, "")  # 1->"very", 2->"somewhat"
            rasgos.append(f"{matiz} {bajo}".strip())
    rasgos[-1] = "and " + rasgos[-1]
    return ", ".join(rasgos)


_system_prompt_cache = None  # se arma una sola vez y se reutiliza en cada turno


def obtener_system_prompt(persona_str, modelo=None):
    """Instrucciones fijas (personalidad + reglas de estilo): no cambian turno a
    turno, así que se construyen una sola vez por sesión, no en cada mensaje.

    `modelo`: qué modelo va a recibir este prompt -- si está en
    _MODELOS_CON_PERSONALIDAD_HORNEADA devuelve "" (ese modelo ya trae la
    personalidad horneada, mandarla de nuevo sería redundante). Si no se
    pasa nada, cae en CHAT_MODEL (compatibilidad con callers viejos que no
    distinguen modelo).
    El caller decide qué hacer con un system prompt vacío (ver
    Orchestrator_Management._mensajes_con_personalidad)."""
    if modelo is None:
        modelo = os.environ.get("CHAT_MODEL", "qwen2.5:0.5b")
    if modelo in _MODELOS_CON_PERSONALIDAD_HORNEADA:
        return ""
    global _system_prompt_cache
    if _system_prompt_cache is None:
        _system_prompt_cache = (
            f"Eres un chatbot con personalidad {persona_str}. Hablas español natural, "
            "como una persona real, nunca como asistente genérico.\n\n"

            "REGLAS:\n"
            "- Máx. 25 palabras. Una idea puntual, sin relleno ni repetir la pregunta.\n"
            "- Sigue la perspectiva del usuario salvo que diga algo objetivamente falso.\n"
            # Solo la regla en positivo. Ofrecerle además una salida ("si no
            # corresponde, di que no tienes el dato") hace que un modelo de 3B se
            # agarre de ella y niegue datos que tiene delante: con la cláusula de
            # escape responde "no tengo el dato" teniendo "París" en el CONTEXTO,
            # y sin ella responde "París". El caso de no haber contexto relevante
            # ya está cortado por código antes de llegar acá (ver SIN_CONTEXTO).
            "- El CONTEXTO trae preguntas de la base de datos con su 'Respuesta'. Si "
            "alguna corresponde a lo que se te pregunta, contesta con esa Respuesta: "
            "es el dato correcto, no lo pongas en duda ni digas que no lo tienes.\n"
            # En modo quiz las preguntas las pone la base de datos, no el modelo.
            # Si el modelo cierra con una pregunta propia, el usuario la contesta
            # y esa respuesta se corrige contra la pregunta del dataset, que era
            # otra: el marcador queda mal y la interacción se desincroniza.
            "- No hagas preguntas propias ni cierres preguntando: las preguntas las "
            "pone la base de datos. Limítate a lo que se te pide en cada turno.\n"
            "- Nunca digas: 'como IA/asistente', 'entiendo', 'lamento', '¿en qué más te "
            "ayudo?', 'lo siento', 'tienes razón', 'en resumen', 'me encanta'."
        )
    return _system_prompt_cache
