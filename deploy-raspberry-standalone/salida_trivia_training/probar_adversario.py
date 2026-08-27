# -*- coding: utf-8 -*-
"""Casos ADVERSARIOS para lora-salida-trivia -- el complemento de
probar_salida.py.

Por qué existe: probar_salida.py da 15/15 tanto con el modelo v1 (que en
producción expulsaba de la trivia a cualquiera que dijera "no sé") como con
el v2. Sus 15 casos cubren las mismas categorías con las que se generó
dataset_salida.jsonl, así que solo pueden confirmar lo que el modelo ya
aprendió -- no detectan los huecos del dataset. Estos 26 casos atacan
justamente los bordes: cada grupo es una hipótesis distinta sobre cómo
podría estar fallando.

Las frases de acá son a propósito DISTINTAS de las de generar_dataset.py.
Si se copian al entrenamiento, este test deja de medir generalización y
vuelve a dar 100% sin significar nada -- que es exactamente el error que
hizo parecer bueno al v1. Al agregar casos nuevos, mantener esa separación.

Ver RESULTADOS.md para los números de cada versión.

Uso:
    python probar_adversario.py
    python probar_adversario.py --modelo qwen2.5:0.5b
    python probar_adversario.py --host http://192.168.1.50:11434
"""
import argparse
import collections
import time

import requests

SYSTEM_PROMPT = (
    "Sos un clasificador. Te dan la pregunta de una trivia y el mensaje del "
    "usuario. Respondé EXACTAMENTE una palabra: RESPUESTA si el usuario "
    "intenta responder esa pregunta (aunque esté mal, o sea una opinión "
    "larga si la pregunta la pide), o SALIR si se puso a hablar de otra "
    "cosa (un tema personal, un problema, charla sin relación, o pide "
    "explícitamente parar/cambiar de tema)."
)

MAT = "Multiplica cinco por ocho, ¿qué resultado te da?"
FRA = "¿Cuál es la capital de Francia?"
DIL = ("Un coche autónomo puede seguir adelante y aplastar un robot, o girar y "
       "chocar contra una pared. ¿Qué opción elegirías?")

# (grupo, pregunta, mensaje, esperado)
CASOS = [
    # A. La respuesta correcta envuelta en lenguaje personal. Es la trampa que
    # heredaron las keywords: ven "mi mamá" y disparan SALIR aunque la frase
    # termine en la respuesta.
    ("A personal-pero-es-respuesta", MAT, "mi mama me enseño que da cuarenta", "RESPUESTA"),
    ("A personal-pero-es-respuesta", FRA, "mi profesor de geografia dijo que era paris", "RESPUESTA"),
    ("A personal-pero-es-respuesta", MAT, "me acuerdo cuando aprendi las tablas, es cuarenta", "RESPUESTA"),
    ("A personal-pero-es-respuesta", DIL, "yo giraria, mi abuela siempre decia que la vida vale mas que las cosas", "RESPUESTA"),
    # B. Rendirse. Decir "no sé" es participar, no irse -- y es lo más común
    # que dice un chico en una trivia.
    ("B rendirse", MAT, "no se", "RESPUESTA"),
    ("B rendirse", FRA, "ni idea la verdad", "RESPUESTA"),
    ("B rendirse", MAT, "no me acuerdo", "RESPUESTA"),
    # C. Meta-trivia: habla DE la trivia (pista, repetir, mecánica), sigue
    # enganchado con el juego.
    ("C meta-trivia", MAT, "me das una pista?", "RESPUESTA"),
    ("C meta-trivia", FRA, "puedes repetir la pregunta?", "RESPUESTA"),
    ("C meta-trivia", MAT, "cuantas preguntas faltan?", "RESPUESTA"),
    ("C meta-trivia", MAT, "esta pregunta esta muy facil", "RESPUESTA"),
    # D. Respuesta equivocada -> sigue siendo un intento de responder.
    ("D respuesta-mala", MAT, "cincuenta", "RESPUESTA"),
    ("D respuesta-mala", FRA, "madrid", "RESPUESTA"),
    ("D respuesta-mala", MAT, "creo que treinta y cinco", "RESPUESTA"),
    # E. Tema personal SIN ninguna frase de _TEMA_PERSONAL. El caso que
    # motivó todo el pipeline: acá es donde las keywords no llegan.
    ("E personal-sin-keyword", MAT, "ayer se me murio el perro", "SALIR"),
    ("E personal-sin-keyword", FRA, "me duele mucho la cabeza desde temprano", "SALIR"),
    ("E personal-sin-keyword", MAT, "estoy cansado, dormi como tres horas", "SALIR"),
    ("E personal-sin-keyword", FRA, "no me fue bien en el examen y estoy bajoneado", "SALIR"),
    # F. Se va de tema sin decir "cambiar de tema" en ningún momento.
    ("F off-topic", MAT, "sabes jugar ajedrez?", "SALIR"),
    ("F off-topic", FRA, "que dia es hoy", "SALIR"),
    ("F off-topic", MAT, "ponme una cancion", "SALIR"),
    ("F off-topic", FRA, "cuentame algo de robots", "SALIR"),
    # G. Robustez a cómo escribe de verdad un chico apurado en una tablet.
    ("G ruido", MAT, "CUARENTA", "RESPUESTA"),
    ("G ruido", FRA, "paaris", "RESPUESTA"),
    ("G ruido", MAT, "kuarenta", "RESPUESTA"),
    ("G ruido", FRA, "QUIERO PARAR YA", "SALIR"),
]


def preguntar(host, modelo, pregunta, mensaje):
    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Pregunta: {pregunta}\nUsuario: {mensaje}\n¿RESPUESTA o SALIR?"},
    ]
    t0 = time.time()
    resp = requests.post(
        f"{host}/v1/chat/completions",
        json={"model": modelo, "messages": mensajes, "temperature": 0,
              "max_tokens": 5, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.json()["choices"][0]["message"]["content"].strip().upper(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    # Mismo default que Clients/Llama_Client.SALIDA_TRIVIA_MODEL: si se
    # desalinean, estos scripts miden un modelo distinto del que corre el
    # robot. Para comparar contra el v1 o el base: --modelo lora-salida-trivia
    # / --modelo qwen2.5:0.5b
    ap.add_argument("--modelo", default="lora-salida-trivia-v2")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    print(f"Probando modelo={args.modelo!r} contra {len(CASOS)} casos adversarios...\n")
    por_grupo = collections.OrderedDict()
    fallos = []
    for grupo, pregunta, mensaje, esperado in CASOS:
        obtenido, seg = preguntar(args.host, args.modelo, pregunta, mensaje)
        ok = obtenido.startswith(esperado)
        g = por_grupo.setdefault(grupo, [0, 0])
        g[1] += 1
        g[0] += ok
        if not ok:
            fallos.append((grupo, mensaje, esperado, obtenido))
        marca = "OK   " if ok else "FALLA"
        print(f"  [{marca}] {seg:5.2f}s  esperado={esperado:9} obtuvo={obtenido:10} {mensaje[:52]!r}")

    print("\n--- por grupo ---")
    for grupo, (ok, total) in por_grupo.items():
        print(f"  {grupo:28} {ok}/{total}")
    ok_tot = sum(v[0] for v in por_grupo.values())
    total = sum(v[1] for v in por_grupo.values())
    print(f"\n{ok_tot}/{total} correctos ({100.0 * ok_tot / total:.1f}%)")

    if fallos:
        print("\n--- fallos ---")
        for grupo, mensaje, esperado, obtenido in fallos:
            print(f"  [{grupo}] {mensaje!r}")
            print(f"      esperado={esperado} obtuvo={obtenido}")


if __name__ == "__main__":
    main()
