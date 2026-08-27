# -*- coding: utf-8 -*-
"""Prueba `lora-salida-trivia` (o cualquier otro modelo, ver --modelo) contra
un set de casos reales -- los mismos que se usaron a mano para comparar
qwen2.5:0.5b (falló) vs llama3.2:3b-q4s (5/5) antes de decidir entrenar este
modelo, más algunos extra para cubrir más variantes.

Standalone a propósito (no importa nada de Clients/Llama_Client.py): así se
puede copiar esta carpeta sola a otra máquina y probarla ahí sin depender
del resto del repo -- solo necesita `requests` y un Ollama corriendo con el
modelo ya importado (ver README.md, pasos 1-5).

Uso:
    python probar_salida.py
    python probar_salida.py --modelo qwen2.5:0.5b     # comparar contra otro modelo
    python probar_salida.py --host http://192.168.1.50:11434
"""
import argparse
import json
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

# (pregunta, mensaje, esperado) -- mezcla de los 5 casos originales del
# benchmark manual + variantes nuevas para no probar solo lo que ya sabíamos
# que funcionaba.
CASOS = [
    # Respuestas cortas típicas -- deben dar RESPUESTA
    ("Multiplica cinco por ocho, ¿qué resultado te da?", "cuarenta", "RESPUESTA"),
    ("¿Cuál es la capital de Francia?", "paris", "RESPUESTA"),
    ("¿Cuánto es nueve por seis?", "54", "RESPUESTA"),
    ("Te voy a poner una canción, ¿sabes cómo se llama?", "Danza Kuduro", "RESPUESTA"),
    ("¿Qué decisión tomarías vos?", "giraría a la derecha", "RESPUESTA"),
    # Respuestas largas legítimas (dilemas/opiniones) -- deben dar RESPUESTA
    ("Multiplica cinco por ocho, ¿qué resultado te da?",
     "creo que la respuesta correcta sería cuarenta porque cinco por ocho es esa multiplicación",
     "RESPUESTA"),
    ("Un coche autónomo puede seguir adelante y aplastar un robot, o girar y chocar contra una pared. ¿Qué opción elegirías?",
     "yo elegiría girar porque prefiero perder un objeto material antes que dañar a alguien más",
     "RESPUESTA"),
    # Frases explícitas de salir -- deben dar SALIR
    ("Multiplica cinco por ocho, ¿qué resultado te da?", "quiero cambiar de tema", "SALIR"),
    ("¿Cuál es la capital de Francia?", "basta, ya no quiero jugar más", "SALIR"),
    ("¿Cuánto es nueve por seis?", "prefiero hablar de otra cosa ahora", "SALIR"),
    # Temas personales -- el caso que motivó todo este pipeline
    ("Multiplica cinco por ocho, ¿qué resultado te da?",
     "hoy tuve un problema con mi profesor de matemáticas, me regañó injustamente delante de toda la clase",
     "SALIR"),
    ("Multiplica cinco por ocho, ¿qué resultado te da?",
     "mis papás se están peleando mucho en casa últimamente y no sé qué hacer al respecto",
     "SALIR"),
    ("¿Cuál es la capital de Francia?",
     "extraño mucho a mi abuela, hoy es su cumpleaños y no puedo verla",
     "SALIR"),
    # Charla genérica sin relación -- deben dar SALIR
    ("¿Cuánto es nueve por seis?", "cuéntame un chiste", "SALIR"),
    ("¿Cuál es la capital de Francia?", "qué te parece la música clásica", "SALIR"),
]


def preguntar(host, modelo, pregunta, mensaje):
    mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta: {pregunta}\nUsuario: {mensaje}\n¿RESPUESTA o SALIR?"},
    ]
    t0 = time.time()
    resp = requests.post(
        f"{host}/v1/chat/completions",
        json={"model": modelo, "messages": mensajes, "temperature": 0, "max_tokens": 5, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"
    contenido = resp.json()["choices"][0]["message"]["content"].strip().upper()
    return contenido, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    # Mismo default que Clients/Llama_Client.SALIDA_TRIVIA_MODEL: si se
    # desalinean, estos scripts miden un modelo distinto del que corre el
    # robot. Para comparar contra el v1 o el base: --modelo lora-salida-trivia
    # / --modelo qwen2.5:0.5b
    ap.add_argument("--modelo", default="lora-salida-trivia-v2")
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    print(f"Probando modelo={args.modelo!r} contra {len(CASOS)} casos...\n")
    ok = 0
    for pregunta, mensaje, esperado in CASOS:
        try:
            obtenido, dt = preguntar(args.host, args.modelo, pregunta, mensaje)
        except requests.RequestException as e:
            print(f"[ERROR] no se pudo consultar Ollama: {e}")
            return
        acerto = esperado in obtenido and not ("RESPUESTA" in obtenido and "SALIR" in obtenido)
        # ambiguo si contiene las dos o ninguna
        if "RESPUESTA" in obtenido and "SALIR" not in obtenido:
            veredicto = "RESPUESTA"
        elif "SALIR" in obtenido and "RESPUESTA" not in obtenido:
            veredicto = "SALIR"
        else:
            veredicto = f"AMBIGUO({obtenido!r})"
        marca = "OK" if veredicto == esperado else "FALLO"
        ok += veredicto == esperado
        print(f"  [{marca:5}] {dt:5.2f}s  esperado={esperado:10} obtuvo={veredicto:18} {mensaje[:55]!r}")

    print(f"\n{ok}/{len(CASOS)} correctos ({ok / len(CASOS) * 100:.1f}%)")


if __name__ == "__main__":
    main()
