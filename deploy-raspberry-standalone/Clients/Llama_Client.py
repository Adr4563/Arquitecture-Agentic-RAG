"""
Cliente HTTP hacia Ollama: generar_respuesta() (las respuestas que ve el
usuario) y clasificar_salida_trivia() (una decisión binaria interna, ver
más abajo) -- las dos únicas llamadas a un LLM que le quedan a este
archivo.

El router YA NO llama a Ollama: enrutar() se mudó a Agents/Agent_Router.py,
un clasificador clásico (TF-IDF + regresión logística, sin LLM) -- ver ese
archivo y router_training/ para el porqué y el benchmark. La recuperación de
preguntas/contexto tampoco es HTTP: vive en preguntas.py, un módulo que
corre en este mismo proceso (ver la nota en ese archivo). Orchestrator_
Management.py y los Agents importan generar_respuesta() de acá,
recuperar_contexto()/pregunta_aleatoria()/etc. de preguntas.py, y enrutar()
de Agent_Router.

Tres modelos, uno por rol (ver TRIVIA_MODEL y SALIDA_TRIVIA_MODEL más
abajo). Para generar_respuesta(), cuál se usa lo decide el caller pasando
`modelo=`; si no se pasa nada, usa CHAT_MODEL. clasificar_salida_trivia()
usa siempre SALIDA_TRIVIA_MODEL, no es configurable por llamada: es un
modelo entrenado para esa única tarea y no sirve para otra cosa.
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
# lora-chat (ver chat_training/) por default: fine-tuning LoRA de
# qwen2.5:0.5b entrenado SOLO con los ejemplos de Chat libre (mismo criterio
# que TRIVIA_MODEL/lora-trivia más abajo -- separar
# el fine-tuning general lora-personalidad, que mezclaba las 4
# categorías, en un modelo por rol). Historia de benchmarks que llevaron
# hasta acá (bench_modelos.py, 7 candidatos x 5 prompts, con el system
# prompt real de personalidad.py):
#   - qwen2.5:0.5b (default antes de lora-chat): 484MB residente en
#     Ollama (`ollama ps`) vs 2.5GB de llama3.2:3b -- ~5x menos RAM.
#     Respuestas 1-3s en caliente vs 7-12s. 5/5 correctas, sin fugas.
#   - llama3.2:1b (1.5GB): perdió -- en 2/3 corridas se negó a confirmar una
#     respuesta CORRECTA ("¡Lo siento! No puedo confirmar..."), y en 3/3
#     filtró las etiquetas <rag> crudas al texto que ve el usuario.
#   - tinyllama, qwen3:0.6b, smollm2:360m: no siguen el system prompt.
#   - smollm2:1.7b: 2.7GB residente -- más pesado que qwen2.5:0.5b, sin
#     sentido como reemplazo "liviano".
# Bonus: como el router (Agent_Router.py) ya no llama a Ollama para nada, y
# Agent_Verificator.py se sacó del todo (Chat libre ya no tiene una segunda
# pasada de revisión -- Llama_Client responde directo), este y TRIVIA_MODEL
# son los únicos dos modelos que Ollama mantiene cargados en RAM. Si hace
# falta más calidad de prosa y el hardware lo permite, sobreescribir con
# `export CHAT_MODEL=llama3.2:3b`.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:1b")  # nombre del modelo en `ollama list` — genera las respuestas reales
# (no usar una variante -fp16: corre 100% en CPU sin VRAM y es extremadamente lenta;
# los modelos cuantizados -q4_K_*/-q4s son los viables en CPU)

# Modelo separado SOLO para las reacciones de Trivia (comentar_resultado()/
# reaccionar_libre() en Orchestrator_Management.py -- hoy definidas pero sin
# uso, ver la nota "Trivia sin agente de comentario") -- a pedido del
# usuario, distinto del que atiende Chat libre (CHAT_MODEL, sin tocar).
# Por default lora-trivia (ver trivia_training/): fine-tuning LoRA
# entrenado SOLO con los 121 ejemplos de trivia (a diferencia de
# lora-personalidad, que entrena sobre las 4 categorías mezcladas) --
# más especializado en este rol puntual. Antecesor (lora-personalidad)
# ya había resultado más confiable que qwen2.5:0.5b + prompt completo (0/8
# vs 3/8 en "repite la instrucción en vez de reaccionar", ver README
# "Personalidad horneada en el modelo").
TRIVIA_MODEL = os.environ.get("TRIVIA_MODEL", "lora-trivia")

# Tercer modelo, para una sola decisión binaria de Trivia: dado el turno
# pendiente, ¿el usuario está intentando responder la pregunta (RESPUESTA) o
# se fue a hablar de otra cosa (SALIR)? Ver salida_trivia_training/ para el
# pipeline de entrenamiento completo.
#
# Antes esto se resolvía solo con listas de palabras clave
# (_SALIR_TRIVIA/_TEMA_PERSONAL en Orchestrator_Management.py, que siguen
# ahí como fallback). Por qué se pudo cambiar recién ahora: qwen2.5:0.5b sin
# reentrenar, con few-shot, falló 4 de 5 casos -- 0.5B es demasiado chico
# para este juicio semántico abierto. llama3.2:3b-q4s acertó 5/5 pero
# reintroduce el problema de RAM/latencia que llevó a sacarlo de CHAT_MODEL.
# El fine-tune LoRA del chico para esta única tarea (lora-salida-trivia)
# acierta 15/15 en ~0.14s por consulta y pesa 397MB, lo mismo que
# CHAT_MODEL/TRIVIA_MODEL -- eso es lo que hizo viable un tercer modelo
# residente en la Pi.
# El sufijo -v2 es a proposito: el v1 (dataset de 542 ejemplos) quedaba en
# 16/26 en probar_adversario.py y mandaba a SALIR a quien decia "no se" o
# pedia una pista -- peor que las keywords que reemplazaba. El v2 (710
# ejemplos, con las categorias rendirse/meta-trivia/respuesta-personal que
# faltaban) da 25/26. Dejar los dos nombres en Ollama permite volver al v1
# con un export si algo sale mal. Ver salida_trivia_training/RESULTADOS.md.
SALIDA_TRIVIA_MODEL = os.environ.get("SALIDA_TRIVIA_MODEL", "lora-salida-trivia-v2")

# El Modelfile de lora-salida-trivia ya trae este mismo SYSTEM horneado,
# pero se manda igual explícito: así la llamada es idéntica a la que validó
# el modelo (salida_trivia_training/probar_salida.py, 15/15) y no depende de
# que quien lo importe a Ollama haya usado ese Modelfile.
_SALIDA_TRIVIA_SYSTEM = (
    "Sos un clasificador. Te dan la pregunta de una trivia y el mensaje del "
    "usuario. Respondé EXACTAMENTE una palabra: RESPUESTA si el usuario "
    "intenta responder esa pregunta (aunque esté mal, o sea una opinión "
    "larga si la pregunta la pide), o SALIR si se puso a hablar de otra "
    "cosa (un tema personal, un problema, charla sin relación, o pide "
    "explícitamente parar/cambiar de tema)."
)


def clasificar_salida_trivia(pregunta, mensaje_usuario):
    """True si el usuario se está yendo de la trivia, False si está
    intentando responder la pregunta, None si no se pudo decidir (Ollama
    caído, modelo sin importar, respuesta inesperada).

    El None es importante: el caller (_quiere_salir_trivia() en
    Orchestrator_Management.py) cae a las listas de palabras clave cuando
    pasa eso, así un problema de infraestructura nunca corta el turno del
    usuario ni lo deja trabado en la trivia.

    Sin streaming y con max_tokens=5 porque la salida es una sola palabra:
    no hay nada que mostrar en vivo. temperature=0 porque es una
    clasificación, no generación."""
    mensajes = [
        {"role": "system", "content": _SALIDA_TRIVIA_SYSTEM},
        {"role": "user",
         "content": f"Pregunta: {pregunta}\nUsuario: {mensaje_usuario}\n¿RESPUESTA o SALIR?"},
    ]
    try:
        resp = requests.post(
            f"{CHAT_SERVER_HOST}/v1/chat/completions",
            json={
                "model": SALIDA_TRIVIA_MODEL,
                "messages": mensajes,
                "temperature": 0,
                "max_tokens": 5,
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"  # mismo motivo que en generar_respuesta()
        veredicto = resp.json()["choices"][0]["message"]["content"].strip().upper()
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None
    # startswith y no ==: con max_tokens=5 el modelo puede colar un punto o
    # un salto de línea detrás de la palabra.
    if veredicto.startswith("SALIR"):
        return True
    if veredicto.startswith("RESPUESTA"):
        return False
    return None


def generar_respuesta(mensajes, temperature=0.3, max_tokens=50, on_token=None, modelo=None):
    """Llama al modelo con streaming. Si se pasa on_token(chunk), se invoca por cada
    pedazo de texto a medida que llega (para imprimirlo en vivo); igual devuelve el
    texto completo al final. No baja el tiempo total, pero se percibe mucho más
    rápido porque el usuario ve la respuesta aparecer en vez de esperar en blanco.

    max_tokens=50 (antes 100): el system prompt (personalidad.py) ya limita toda
    respuesta a ~25 palabras, así que 50 tokens da margen de sobra sin recortar
    nada — solo corta el peor caso en que el modelo no respeta esa regla y sigue
    generando de más, que es puro costo de CPU desperdiciado en esta Pi.

    `modelo`: qué modelo de Ollama usar para ESTA llamada puntual -- si no se
    pasa, CHAT_MODEL (el default general). Los callers de Trivia pasan
    TRIVIA_MODEL explícito (ver Orchestrator_Management.py)."""
    resp = requests.post(
        f"{CHAT_SERVER_HOST}/v1/chat/completions",
        json={
            "model": modelo or CHAT_MODEL,
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
