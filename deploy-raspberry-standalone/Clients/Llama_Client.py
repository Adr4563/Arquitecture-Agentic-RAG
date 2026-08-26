"""
Cliente HTTP hacia Ollama (generar_respuesta(), enrutar()).

La recuperación de preguntas/contexto YA NO es HTTP: vive en preguntas.py,
un módulo que corre en este mismo proceso (ver la nota en ese archivo sobre
por qué acá no hace falta un servidor separado como en el despliegue
dividido). Orchestrator_Management.py y los Agents importan generar_respuesta()/
enrutar() de acá, y recuperar_contexto()/pregunta_aleatoria()/etc. de preguntas.py.
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
# Configurable por env var (no solo hardcodeado) para poder bajar a un modelo más
# liviano en hardware limitado (ej. Raspberry Pi) sin tocar código — ver
# deploy-raspberry-standalone/README.md.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:3b-q4s")  # nombre del modelo en `ollama list` — genera las respuestas reales
# (no usar la variante -fp16: corre 100% en CPU sin VRAM y es extremadamente lenta;
# -q4s está cuantizado y es viable en CPU)

# El router no necesita el modelo grande: es una clasificación de 3 etiquetas,
# no generación de texto. qwen2.5:0.5b con few-shot + salida JSON forzada
# clasifica igual de bien que llama3.2:1b en este caso y corre ~3x más rápido
# (benchmark en rag-model-bench/: 5/5 aciertos, ~760ms promedio vs ~2.6s).
ROUTER_MODEL = "qwen2.5:0.5b"


def generar_respuesta(mensajes, temperature=0.3, max_tokens=100, on_token=None):
    """Llama al modelo con streaming. Si se pasa on_token(chunk), se invoca por cada
    pedazo de texto a medida que llega (para imprimirlo en vivo); igual devuelve el
    texto completo al final. No baja el tiempo total, pero se percibe mucho más
    rápido porque el usuario ve la respuesta aparecer en vez de esperar en blanco."""
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


RUTAS_VALIDAS = {"TRIVIA", "BUSQUEDA_WEB", "CHAT_LIBRE"}

# Few-shot: sin ejemplos, qwen2.5:0.5b clasificaba con ~50-60% de acierto
# (confundía preguntas de precio/contrato con otras rutas). Con estos 6
# ejemplos sube a 5/5 en el set de prueba — ver rag-model-bench/bench.mjs.
_ROUTER_SYSTEM_PROMPT = """Eres un router. Clasifica el mensaje del usuario en UNA de estas rutas y responde SOLO con un JSON de la forma {"ruta": "..."}.
- TRIVIA: quiere jugar, que le hagan preguntas, trivia, matemática, chistes, o cualquier minijuego — incluye pedir seguir/retomar/continuar un juego de preguntas que ya había empezado.
- BUSQUEDA_WEB: pregunta algo actual o reciente que no se puede saber de memoria (noticias, el clima, quién ganó algo hace poco, la fecha de hoy).
- CHAT_LIBRE: charla general o cualquier otra pregunta que no sea ninguna de las anteriores.
No agregues explicación ni texto fuera del JSON.

Ejemplos:
Usuario: "quiero jugar trivia"
{"ruta": "TRIVIA"}
Usuario: "hazme una pregunta de matematica"
{"ruta": "TRIVIA"}
Usuario: "sigamos con la trivia"
{"ruta": "TRIVIA"}
Usuario: "continuemos con las preguntas"
{"ruta": "TRIVIA"}
Usuario: "volvamos al juego de preguntas"
{"ruta": "TRIVIA"}
Usuario: "que clima hace hoy en lima"
{"ruta": "BUSQUEDA_WEB"}
Usuario: "quien gano el partido de anoche"
{"ruta": "BUSQUEDA_WEB"}
Usuario: "hola como estas"
{"ruta": "CHAT_LIBRE"}
Usuario: "cuentame un chiste"
{"ruta": "CHAT_LIBRE"}"""


def enrutar(mensaje_usuario):
    """Clasifica el mensaje en TRIVIA / BUSQUEDA_WEB / CHAT_LIBRE con el modelo
    liviano del router (ROUTER_MODEL), no con el modelo grande de generación.
    Usa /api/chat (nativo de Ollama, no el compatible con OpenAI) porque es el
    que soporta format="json" para forzar una salida parseable sin post-procesar
    texto libre. Sin streaming: la respuesta son unos pocos tokens, no vale la
    pena el streaming para esto.

    Devuelve "" si el modelo no respondió una ruta válida (el caller decide el
    fallback, hoy CHAT_LIBRE) — nunca lanza por una clasificación rara, solo
    por un problema real de red/servidor.
    """
    mensajes = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": mensaje_usuario},
    ]
    resp = requests.post(
        f"{CHAT_SERVER_HOST}/api/chat",
        json={
            "model": ROUTER_MODEL,
            "messages": mensajes,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 20},
        },
        timeout=30,
    )
    resp.raise_for_status()
    contenido = resp.json()["message"]["content"]
    try:
        ruta = json.loads(contenido).get("ruta", "").strip().upper()
    except (json.JSONDecodeError, AttributeError):
        ruta = ""
    return ruta if ruta in RUTAS_VALIDAS else ""
