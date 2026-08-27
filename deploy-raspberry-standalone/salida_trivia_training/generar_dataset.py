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


# ─── Clase RESPUESTA: los 3 huecos que tenía el dataset v1 ─────────────
#
# El dataset original solo tenía dos fuentes: respuestas literales sacadas
# de preguntas.jsonl (RESPUESTA) y frases CHAT_LIBRE del router (SALIR). Con
# eso, la regla más simple que explica los 542 ejemplos es "si no es
# literalmente la respuesta -> SALIR", y eso fue exactamente lo que aprendió
# el modelo: en 26 casos adversarios dio 16/26, con los 10 fallos todos en
# la misma dirección (falsos SALIR). Peor todavía, 7 de esos 10 eran peores
# que las listas de palabras clave que el modelo venía a reemplazar.
#
# Las tres categorías de acá abajo son esos huecos. Todas son RESPUESTA: el
# usuario SIGUE en la trivia, aunque no esté diciendo la respuesta.
#
# Ojo al editar: las frases de acá son a propósito DISTINTAS de las que usa
# probar_salida.py. Si se copian las de test al entrenamiento, el test deja
# de medir generalización y vuelve a dar 100% sin significar nada.

# 1. Rendirse. Decir "no sé" es participar, no irse -- y es la respuesta más
# común de un chico en una trivia. Con el dataset v1 el modelo lo mandaba a
# SALIR, o sea que lo expulsaba de la trivia justo por no saber.
FRASES_RENDIRSE = [
    "no lo sé", "no tengo idea", "no se me ocurre nada", "paso",
    "me rindo", "no la sé", "esa no la sé", "ni la más mínima idea",
    "no tengo ni la menor idea", "difícil, no me sale",
    "no me la sé, lo siento", "uf, esa no", "creo que no la sé",
    "no sabría decirte", "no estoy seguro de esa", "no caigo",
]

# 2. Meta-trivia: habla DE la trivia (pide pista, pide repetir, pregunta por
# la mecánica, comenta la dificultad). Sigue enganchado con el juego.
FRASES_META_TRIVIA = [
    "¿me puedes dar una ayudita?", "dame una ayuda por favor",
    "¿la puedes repetir?", "no escuché bien, ¿otra vez?",
    "no entendí la pregunta", "¿puedes explicarla de nuevo?",
    "¿cuántas van?", "¿esta es la última?", "¿me quedan muchas?",
    "esta está complicada", "qué difícil esta", "esa estuvo sencilla",
    "¿me la puedes decir más despacio?", "¿vale si respondo aproximado?",
    "¿cuánto llevo bien hasta ahora?", "¿me das más tiempo para pensar?",
]

# 3. Respuesta envuelta en vocabulario personal. Acá el modelo v1 copiaba
# literalmente la debilidad de _TEMA_PERSONAL: ve "mi mamá"/"mi profe" y
# dispara SALIR, aunque la frase termine en la respuesta correcta.
ENVOLTORIOS_RESPUESTA_PERSONAL = [
    "mi mamá me enseñó que es {r}", "mi papá me dijo que era {r}",
    "mi profe nos explicó que es {r}", "en el colegio me enseñaron que es {r}",
    "mi hermano me contó que era {r}", "me acuerdo de esto de clase, es {r}",
    "mi abuela me lo enseñó, es {r}", "lo vi con mi maestra, creo que {r}",
    "estudié esto la semana pasada, es {r}", "mi amigo me dijo que {r}",
]

# 4. Opinión larga con anclaje personal, para las preguntas sin
# respuesta_esperada (dilemas). Es el caso más difícil de los cuatro: habla
# de la abuela Y responde el dilema al mismo tiempo.
OPINIONES_LARGAS_PERSONALES = [
    "yo giraría, en mi casa siempre me enseñaron que una persona vale más que un objeto",
    "seguiría de largo, mi papá dice que uno no debe arriesgar a los que van adentro",
    "elegiría la segunda, me acuerdo que en el colegio hablamos de algo parecido",
    "creo que giraría, mi profe nos hizo pensar en esto una vez y me quedó eso",
    "la primera me parece mejor, mi hermano opina lo mismo cuando lo charlamos",
    "no sé bien, pero mi mamá diría que hay que proteger a las personas primero",
]


def generar_rendirse(preguntas):
    """Cada frase contra 3 preguntas al azar: rendirse no depende de cuál
    pregunta era."""
    ejemplos = []
    for f in FRASES_RENDIRSE:
        for _ in range(3):
            p = random.choice(preguntas)
            ejemplos.append({"pregunta": p["pregunta"], "mensaje": f, "objetivo": "RESPUESTA"})
    return ejemplos


def generar_meta_trivia(preguntas):
    ejemplos = []
    for f in FRASES_META_TRIVIA:
        for _ in range(3):
            p = random.choice(preguntas)
            ejemplos.append({"pregunta": p["pregunta"], "mensaje": f, "objetivo": "RESPUESTA"})
    return ejemplos


def generar_respuesta_personal(preguntas):
    """La respuesta REAL de la pregunta, envuelta en lenguaje personal. Solo
    para preguntas con respuesta_esperada, si no no hay {r} que envolver."""
    ejemplos = []
    con_resp = [p for p in preguntas if (p.get("respuesta_esperada") or "").strip()]
    # 45 de las 222: suficiente para enseñar el patrón sin inundar el dataset
    # con una sola forma de decir las cosas.
    for p in random.sample(con_resp, min(45, len(con_resp))):
        r = p["respuesta_esperada"].strip().lower()
        mensaje = random.choice(ENVOLTORIOS_RESPUESTA_PERSONAL).format(r=r)
        ejemplos.append({"pregunta": p["pregunta"], "mensaje": mensaje, "objetivo": "RESPUESTA"})

    sin_resp = [p for p in preguntas if not (p.get("respuesta_esperada") or "").strip()]
    for p in sin_resp:
        mensaje = random.choice(OPINIONES_LARGAS_PERSONALES)
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
    ejemplos += generar_rendirse(preguntas)
    ejemplos += generar_meta_trivia(preguntas)
    ejemplos += generar_respuesta_personal(preguntas)
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
