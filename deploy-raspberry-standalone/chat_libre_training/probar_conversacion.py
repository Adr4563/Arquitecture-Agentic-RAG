# -*- coding: utf-8 -*-
"""Evaluacion de una version -- paso 6 del pipeline, el que NO se saltea.

Por que existe: en salida_trivia_training la suite "normal" daba 15/15 con
un modelo que en produccion expulsaba de la trivia a cualquiera que dijera
"no se". Un test escrito junto con el dataset solo confirma lo que el modelo
ya aprendio. Estos casos son distintos a proposito: NO tienen que estar en
dataset_chat_libre.jsonl.

No puntua "que tan linda" es la respuesta: eso NO se automatiza de forma
confiable, y fingir que si es peor que no medir. Solo cuentan como FALLA las
propiedades objetivamente verificables, que son justo donde vimos fallar a
los modelos chicos:

  1. RESPONDE      no vacio (qwen3:0.6b devolvia "")
  2. NO_REPITE     sin bucles ("me gusta el futbol, me gusta el futbol...")
  3. LARGO         <= 45 palabras (el robot habla en voz alta y bloquea)
  4. NO_ASISTENTE  sin "como IA", "no puedo ayudarte con eso"
  5. NO_INVENTA    ante lo que NO puede saber (la hora, el clima, una cuenta
                   grande) lo admite en vez de inventar. Medido en
                   llama3.2:1b: "que hora es" -> "Hoy, el 15 de marzo, a las
                   10:00 AM", y "918 por 447" -> "918 / 447 = 2.080".

Hubo un sexto chequeo, ENGANCHA, que exigia un signo de pregunta en los
temas personales. Se saco: marcaba como falla "Me encanta también, hay algo
de magia en resolverlos", que es una buena respuesta. Enganchar no requiere
repreguntar, y ningun heuristico simple distingue eso. Los casos personales
siguen en la suite y sus respuestas se imprimen, pero se juzgan a ojo -- esa
dimension la decide una persona, no el script.

Uso:
    python probar_conversacion.py --modelo lora-chat-libre-v1
    python probar_conversacion.py --modelo qwen2.5:0.5b      # base, para comparar
    python probar_conversacion.py --modelo llama3.2:1b --anotar-version 1
"""
import argparse
import io
import json
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONES_PATH = os.path.join(HERE, "versiones.json")
SYSTEM_PROMPT = (
    "Hablas español natural, como una persona real. Responde en máximo 25 "
    "palabras, una idea puntual, sin relleno ni repetir la pregunta. Nunca "
    "digas que eres una IA o un asistente."
)

# (mensaje, comprobaciones extra). Frases DISTINTAS de las del dataset.
CASOS = [
    # Personales: sin chequeo automatico de "engancha" (ver el docstring).
    # Se imprimen para revisar a ojo.
    ("me quede sin amigos en la escuela",        []),
    ("mi abuelo esta internado",                 []),
    ("me da verguenza hablar en publico",        []),
    ("saque la nota mas alta del curso",         []),
    ("me encanta armar rompecabezas",            []),
    ("mi gato se escapo de casa",                []),
    ("que hora es en este momento",              ["NO_INVENTA"]),
    ("va a llover esta tarde",                   ["NO_INVENTA"]),
    ("cuanto es 918 por 447",                    ["NO_INVENTA"]),
    # Identidad: el chequeo NO_ASISTENTE ya existia, pero ningun caso
    # preguntaba por ella -- v1 fallaba 3 de 6 y la suite daba 11/12 igual.
    ("que eres tu",                              []),
    ("eres inteligencia artificial",             []),
    ("eres real",                                []),
    ("sos un programa",                          []),
    ("hola que tal",                             []),
    ("chau nos vemos",                           []),
    ("no entiendo nada",                         []),
]
_ASISTENTE = ["como ia", "como asistente", "soy una ia", "no puedo ayudarte con eso",
              "no puedo asistir", "lo siento, pero no puedo"]
_ADMITE = ["no puedo", "no tengo", "no se", "no sé", "no sabria", "no sabría",
           "calculadora", "no manejo", "no dispongo"]


def preguntar(host, modelo, mensaje):
    r = requests.post(host + "/v1/chat/completions", json={
        "model": modelo,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": mensaje}],
        "temperature": 0.3, "max_tokens": 60, "stream": False}, timeout=180)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.json()["choices"][0]["message"]["content"].strip()


def hay_bucle(texto):
    """Detecta repeticion patologica: la misma frase de 3+ palabras dos veces."""
    palabras = re.findall(r"\w+", texto.lower())
    for n in (3, 4, 5):
        vistos = set()
        for i in range(len(palabras) - n + 1):
            gram = tuple(palabras[i:i + n])
            if gram in vistos:
                return True
            vistos.add(gram)
    return False


def evaluar(mensaje, respuesta, extras):
    fallos = []
    if not respuesta.strip():
        return ["RESPONDE"]
    if hay_bucle(respuesta):
        fallos.append("NO_REPITE")
    if len(respuesta.split()) > 45:
        fallos.append("LARGO")
    bajo = respuesta.lower()
    if any(f in bajo for f in _ASISTENTE):
        fallos.append("NO_ASISTENTE")
    if "NO_INVENTA" in extras and not any(a in bajo for a in _ADMITE):
        fallos.append("NO_INVENTA")
    return fallos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--anotar-version", type=int, default=None,
                    help="guarda el resultado en versiones.json para esa version")
    args = ap.parse_args()

    print("Evaluando " + args.modelo + " contra " + str(len(CASOS)) + " casos...\n")
    ok = 0
    for mensaje, extras in CASOS:
        t0 = time.time()
        try:
            r = preguntar(args.host, args.modelo, mensaje)
        except Exception as e:
            print("  [ERROR] " + mensaje + " -> " + type(e).__name__)
            continue
        fallos = evaluar(mensaje, r, extras)
        ok += not fallos
        marca = "OK   " if not fallos else "FALLA"
        print("  [%s %4.1fs] %s" % (marca, time.time() - t0, mensaje))
        print("      -> " + r.replace(chr(10), " ")[:96])
        if fallos:
            print("      x " + ", ".join(fallos))

    total = len(CASOS)
    pct = 100.0 * ok / total
    print("\n%d/%d (%.0f%%)" % (ok, total, pct))

    if args.anotar_version is not None and os.path.isfile(VERSIONES_PATH):
        with io.open(VERSIONES_PATH, encoding="utf-8") as f:
            hist = json.load(f)
        for v in hist:
            if v["version"] == args.anotar_version:
                v["eval_adversario"] = "%d/%d" % (ok, total)
        with io.open(VERSIONES_PATH, "w", encoding="utf-8", newline="\n") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
        print("Anotado en versiones.json para v" + str(args.anotar_version))


if __name__ == "__main__":
    main()
