# -*- coding: utf-8 -*-
"""Arma dataset_salida.jsonl para entrenar_salida.py: pares
{"instruccion": "Pregunta: ...\\nUsuario: ...\\n¿RESPUESTA o SALIR?",
 "objetivo": "RESPUESTA" | "SALIR"}.

Dos clases:
- RESPUESTA: intentos reales de responder la pregunta pendiente -- respuestas
  cortas (sacadas de respuesta_esperada real del dataset, con variantes tipo
  "es {r}"/"creo que es {r}"/con typo, mismo criterio que el benchmark de
  Agent_Corrector.py) + opiniones largas legítimas para dilemas abiertos.
- SALIR: el usuario se quiere ir del trivia -- frases explícitas
  (_SALIR_TRIVIA de Orchestrator_Management.py, parafraseadas en oraciones
  completas) + temas personales/experiencias (_TEMA_PERSONAL, también
  parafraseadas y ampliadas) + charla genérica sin relación (reusa los 247
  ejemplos CHAT_LIBRE de router_training/dataset_router.jsonl).

Cada ejemplo de SALIR se empareja con una pregunta AL AZAR del dataset real
(preguntas.jsonl) -- lo off-topic que es no depende de cuál pregunta estaba
pendiente. Cada ejemplo de RESPUESTA se empareja con SU PROPIA pregunta.

Uso:
    python generar_dataset.py
"""
import json
import os
import random

random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
PREGUNTAS_PATH = os.path.join(HERE, "..", "base_datos", "preguntas.jsonl")
ROUTER_DATASET_PATH = os.path.join(HERE, "..", "router_training", "dataset_router.jsonl")
SALIDA_PATH = os.path.join(HERE, "dataset_salida.jsonl")

PLANTILLA = "Pregunta: {pregunta}\nUsuario: {mensaje}\n¿RESPUESTA o SALIR?"


def cargar_preguntas():
    with open(PREGUNTAS_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cargar_chat_libre():
    with open(ROUTER_DATASET_PATH, encoding="utf-8") as f:
        filas = [json.loads(l) for l in f if l.strip()]
    return [f["frase"] for f in filas if f["ruta"] == "CHAT_LIBRE"]


# ─── Clase RESPUESTA ───────────────────────────────────────────────────

ENVOLTORIOS_RESPUESTA = [
    "{r}", "es {r}", "la respuesta es {r}", "creo que es {r}",
    "{r}, obvio", "yo creo que {r}", "pienso que es {r}", "{r}!",
]

# Opiniones largas legítimas para preguntas SIN respuesta_esperada (dilemas,
# reconocimiento musical, interacción personalizada...) -- el usuario SIGUE
# respondiendo, solo que con una frase más elaborada. Mezcla de tono según
# el tipo de pregunta para que el modelo generalice, no memorice un patrón.
OPINIONES_LARGAS = [
    "yo elegiría girar porque prefiero perder un objeto material antes que dañar a alguien más",
    "prefiero seguir adelante, creo que es la decisión más razonable en ese caso",
    "no estoy seguro pero me suena que podría ser esa opción",
    "me parece que la primera alternativa es la más justa para todos",
    "la verdad no sabría decidir, pero si tuviera que elegir iría por la segunda",
    "sí, me gustó bastante esa canción, me trajo buenos recuerdos",
    "no la reconozco pero me pareció alegre y con buen ritmo",
    "prefiero el rojo, siempre me gustó ese color",
    "tengo hambre, sí, ya me lo pueden calentar cuando quieran",
    "no, todavía no tengo hambre, gracias por preguntar",
    "me llamo Roberto, mucho gusto",
    "sí, hoy me siento bastante bien",
]


def _typo(s):
    if len(s) < 4:
        return s
    i = random.randint(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def generar_respuestas(preguntas):
    ejemplos = []
    con_resp = [p for p in preguntas if (p.get("respuesta_esperada") or "").strip()]
    for p in con_resp:
        r = p["respuesta_esperada"].strip().lower()
        envoltorio = random.choice(ENVOLTORIOS_RESPUESTA)
        mensaje = envoltorio.format(r=r)
        if random.random() < 0.25:
            mensaje = _typo(mensaje)
        ejemplos.append({"pregunta": p["pregunta"], "mensaje": mensaje, "objetivo": "RESPUESTA"})

    sin_resp = [p for p in preguntas if not (p.get("respuesta_esperada") or "").strip()]
    for p in sin_resp:
        mensaje = random.choice(OPINIONES_LARGAS)
        ejemplos.append({"pregunta": p["pregunta"], "mensaje": mensaje, "objetivo": "RESPUESTA"})
    return ejemplos


# ─── Clase SALIR ───────────────────────────────────────────────────────

FRASES_SALIR_EXPLICITO = [
    "quiero cambiar de tema", "cambiemos de tema mejor",
    "ya no quiero seguir jugando trivia", "ya no quiero jugar más a esto",
    "quiero parar un rato", "para de preguntarme cosas",
    "detente por favor", "quiero salir de la trivia",
    "pausemos la trivia un momento", "dejemos la trivia por ahora",
    "prefiero hablar de otra cosa ahora", "quiero charlar un rato nomás",
    "prefiero charlar contigo", "quiero conversar de otra cosa",
    "basta de preguntas por hoy", "ya basta, quiero descansar",
    "no quiero jugar más por ahora", "no quiero más preguntas, gracias",
    "olvida la trivia un rato", "no quiero seguir con la trivia",
    "no más trivia por hoy", "estoy cansado de las preguntas",
    "estoy cansada de tantas preguntas", "quiero hacer otra cosa un rato",
    "vamos a otra cosa mejor", "ya fue suficiente trivia por hoy",
    "terminemos la trivia acá", "hablemos de otra cosa un momento",
]

FRASES_TEMA_PERSONAL = [
    "hoy tuve un problema con mi profesor de matemáticas, me regañó injustamente delante de toda la clase",
    "mis papás se están peleando mucho en casa últimamente y no sé qué hacer al respecto",
    "extraño mucho a mi abuela, hoy es su cumpleaños y no puedo verla",
    "quiero contarte algo que me pasó ayer en el colegio con mis compañeros",
    "el otro día tuve una discusión fuerte con mi mejor amigo y todavía estoy triste",
    "recuerdo cuando era chico y jugaba en el patio de mi abuelo",
    "tuve un problema con mi mamá esta mañana antes de venir",
    "mi maestra me dijo algo que me hizo sentir mal en clase de historia",
    "estoy preocupado porque tengo un examen difícil la próxima semana",
    "me siento un poco mal hoy, no sé bien por qué",
    "te quiero contar que mi amigo se mudó de ciudad la semana pasada",
    "queria contarte que mi papá consiguió un trabajo nuevo",
    "mi abuelo está enfermo y estamos todos preocupados en casa",
    "se estan peleando mis papas otra vez y me pone muy triste",
    "hoy en el colegio me sentí muy solo todo el día",
]

FRASES_SALIR_TEMPLATES = [
    "{f}", "{f}, ¿está bien?", "oye, {f}", "eh, {f}",
    "disculpa, {f}", "{f} un rato",
]


def generar_salir_explicito(preguntas):
    ejemplos = []
    for f in FRASES_SALIR_EXPLICITO:
        pregunta = random.choice(preguntas)
        plantilla = random.choice(FRASES_SALIR_TEMPLATES)
        mensaje = plantilla.format(f=f)
        ejemplos.append({"pregunta": pregunta["pregunta"], "mensaje": mensaje, "objetivo": "SALIR"})
    return ejemplos


def generar_tema_personal(preguntas):
    ejemplos = []
    # cada frase de tema personal se usa contra 3 preguntas distintas al azar --
    # lo off-topic que es no depende de la pregunta puntual.
    for f in FRASES_TEMA_PERSONAL:
        for _ in range(3):
            pregunta = random.choice(preguntas)
            ejemplos.append({"pregunta": pregunta["pregunta"], "mensaje": f, "objetivo": "SALIR"})
    return ejemplos


def generar_chat_libre_generico(preguntas, chat_libre):
    ejemplos = []
    for frase in chat_libre:
        pregunta = random.choice(preguntas)
        ejemplos.append({"pregunta": pregunta["pregunta"], "mensaje": frase, "objetivo": "SALIR"})
    return ejemplos


def main():
    preguntas = cargar_preguntas()
    chat_libre = cargar_chat_libre()
    print(f"{len(preguntas)} preguntas reales, {len(chat_libre)} frases CHAT_LIBRE reusadas")

    ejemplos = []
    ejemplos += generar_respuestas(preguntas)
    ejemplos += generar_salir_explicito(preguntas)
    ejemplos += generar_tema_personal(preguntas)
    ejemplos += generar_chat_libre_generico(preguntas, chat_libre)

    random.shuffle(ejemplos)

    n_resp = sum(1 for e in ejemplos if e["objetivo"] == "RESPUESTA")
    n_salir = sum(1 for e in ejemplos if e["objetivo"] == "SALIR")
    print(f"Total: {len(ejemplos)} (RESPUESTA={n_resp}, SALIR={n_salir})")

    with open(SALIDA_PATH, "w", encoding="utf-8") as f:
        for e in ejemplos:
            instruccion = PLANTILLA.format(pregunta=e["pregunta"], mensaje=e["mensaje"])
            f.write(json.dumps({"instruccion": instruccion, "objetivo": e["objetivo"]}, ensure_ascii=False) + "\n")
    print(f"Guardado {SALIDA_PATH}")


if __name__ == "__main__":
    main()
