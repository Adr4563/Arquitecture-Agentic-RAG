"""
Cliente HTTP hacia Ollama (generar_respuesta() -- la única llamada a un LLM
que le queda a este archivo).

El router YA NO llama a Ollama: enrutar() se mudó a Agents/Agent_Router.py,
un clasificador clásico (TF-IDF + regresión logística, sin LLM) -- ver ese
archivo y router_training/ para el porqué y el benchmark. La recuperación de
preguntas/contexto tampoco es HTTP: vive en preguntas.py, un módulo que
corre en este mismo proceso (ver la nota en ese archivo). Orchestrator_
Management.py y los Agents importan generar_respuesta() de acá,
recuperar_contexto()/pregunta_aleatoria()/etc. de preguntas.py, y enrutar()
de Agent_Router.
"""

import json
import os

import requests

# ─── Ollama ──────────────────────────────────────────────────
# Por default apunta a localhost (todo corre en esta misma Raspberry Pi). Si
# alguna vez hiciera falta correr Ollama en otra máquina, se puede
# sobreescribir con la IP correspondiente:
#   export CHAT_SERVER_HOST=http://192.168.1.44:11434
CHAT_SERVER_HOST = os.environ.get("CHAT_SERVER_HOST", "http://localhost:11434")
# Configurable por env var (no solo hardcodeado) para poder subir a un modelo más
# grande si el hardware lo banca, sin tocar código — ver
# deploy-raspberry-standalone/README.md.
#
# qwen2.5:0.5b (0.5B, Alibaba) por default — no llama3.2:3b-q4s (3B, Meta), que
# era el default anterior. Benchmark propio (bench_modelos.py, 7 candidatos x 5
# prompts representativos de este proyecto, con el system prompt real de
# personalidad.py, corridos varias veces para chequear consistencia):
#   - qwen2.5:0.5b: 484MB residente en Ollama (`ollama ps`) vs 2.5GB de
#     llama3.2:3b-q4s -- ~5x menos RAM. Respuestas 1-3s en caliente vs 7-12s.
#     Calidad: 5/5 corridas correctas, coherentes, en español, sin fugas de
#     formato ni frases prohibidas.
#   - llama3.2:1b (1.5GB) se probó como alternativa "liviana" obvia y perdió:
#     en 2/3 corridas se negó a confirmar una respuesta CORRECTA ("¡Lo siento!
#     No puedo confirmar..."), y en 3/3 corridas filtró las etiquetas <rag>
#     crudas al texto que ve el usuario. Bug reproducible, no un accidente.
#   - tinyllama, qwen3:0.6b y smollm2:360m: no siguen el system prompt (lo
#     repiten textual en vez de responder, o devuelven texto vacío).
#   - smollm2:1.7b: calidad aceptable pero 2.7GB residente -- más pesado que
#     el default ANTERIOR, no tiene sentido como reemplazo "liviano".
# Bonus: como el router (Agent_Router.py) ya no llama a Ollama para nada,
# este es el ÚNICO modelo que Ollama mantiene cargado en RAM -- antes eran 2
# (este + ROUTER_MODEL). Si hace falta más calidad de prosa y el hardware lo
# permite, sobreescribir con `export CHAT_MODEL=llama3.2:3b-q4s`.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:0.5b")  # nombre del modelo en `ollama list` — genera las respuestas reales
# (no usar una variante -fp16: corre 100% en CPU sin VRAM y es extremadamente lenta;
# los modelos cuantizados -q4_K_*/-q4s son los viables en CPU)


def generar_respuesta(mensajes, temperature=0.3, max_tokens=50, on_token=None):
    """Llama al modelo con streaming. Si se pasa on_token(chunk), se invoca por cada
    pedazo de texto a medida que llega (para imprimirlo en vivo); igual devuelve el
    texto completo al final. No baja el tiempo total, pero se percibe mucho más
    rápido porque el usuario ve la respuesta aparecer en vez de esperar en blanco.

    max_tokens=50 (antes 100): el system prompt (personalidad.py) ya limita toda
    respuesta a ~25 palabras, así que 50 tokens da margen de sobra sin recortar
    nada — solo corta el peor caso en que el modelo no respeta esa regla y sigue
    generando de más, que es puro costo de CPU desperdiciado en esta Pi."""
    resp = requests.post(
        f"{CHAT_SERVER_HOST}/v1/chat/completions",
        json={
            "model": CHAT_MODEL,
            "messages": mensajes,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()
    # Ollama manda 'Content-Type: text/event-stream' sin charset. Ante un text/*
    # sin charset, requests asume ISO-8859-1 (regla legacy de HTTP) y con
    # decode_unicode=True decodifica los bytes UTF-8 como latin-1: "triángulo"
    # llega como "triÃ¡ngulo". El cuerpo es UTF-8, así que hay que decirlo.
    resp.encoding = "utf-8"
    texto = []
    for linea in resp.iter_lines(decode_unicode=True):
        if not linea or not linea.startswith("data: "):
            continue
        payload = linea[len("data: "):]
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
        if delta:
            texto.append(delta)
            if on_token:
                on_token(delta)
    return "".join(texto)
