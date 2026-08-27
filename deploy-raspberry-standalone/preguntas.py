"""
Acceso a las preguntas de trivia/RAG: BM25 (keyword) sobre preguntas.jsonl,
en memoria, sin embeddings ni HTTP.

chat.py/workers.py y esta lógica corren en el mismo proceso: no hay ninguna
razón para exponer esto por red y pagar un round-trip HTTP a localhost en
cada pregunta de trivia o mensaje de chat libre. Por eso es un módulo
normal, importado directo por workers.py — no un servidor aparte.
"""

import json
import os
import random
import re

from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
PREGUNTAS_FILE = os.path.join(HERE, "base_datos", "preguntas.jsonl")

SIN_CONTEXTO = None  # sentinel: no hubo resultados relevantes para recuperar_contexto()

# Índice BM25 (búsqueda por keyword) en memoria — toda la "base vectorial" de
# este módulo. Se arma una sola vez, al importar (ver el final del archivo).
_bm25 = None
_bm25_ids = []
_preguntas_cache = {}  # id -> {"pregunta", "cara", "respuesta_esperada", "tema"}


# Palabras genéricas que aparecen en casi cualquier pregunta ("qué", "cuándo",
# "es"...) — si no se filtran, BM25 "rescata" preguntas sin relación real solo
# por compartir estas palabras, en vez de un término realmente distintivo.
_STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en",
    "y", "o", "a", "que", "qué", "es", "son", "fue", "fueron", "se", "su",
    "sus", "por", "para", "con", "cual", "cuál", "cuales", "cuáles", "quien",
    "quién", "quienes", "quiénes", "como", "cómo", "cuando", "cuándo",
    "donde", "dónde", "porque", "por qué", "cuanto", "cuánto", "cuanta",
    "cuánta", "cuantos", "cuántos", "cuantas", "cuántas",
}


def _tokenizar(texto):
    # \w cubre letras acentuadas gracias a re.UNICODE (default en Python 3), así
    # que esto separa palabras y descarta signos de puntuación pegados (¿?¡!.,).
    palabras = re.findall(r"\w+", texto.lower())
    return [p for p in palabras if p not in _STOPWORDS_ES]


def _cargar_preguntas():
    """Lee preguntas.jsonl y arma _preguntas_cache + el índice BM25."""
    global _bm25, _bm25_ids, _preguntas_cache

    with open(PREGUNTAS_FILE, encoding="utf-8") as f:
        preguntas = [json.loads(l) for l in f if l.strip()]

    if not preguntas:
        print("preguntas.jsonl está vacío, nada que cargar.")
        return

    _bm25_ids = [p["id"] for p in preguntas]
    _preguntas_cache = {p["id"]: p for p in preguntas}
    _bm25 = BM25Okapi([_tokenizar(p["pregunta"]) for p in preguntas])
    print(f"[preguntas.py] {len(preguntas)} preguntas cargadas en memoria.")


def _formatear(p):
    """Los campos que le llegan a chat.py/reactor.py por cada pregunta —
    incluye cara_respuesta_buena/mala y musical/desplazamiento (reactor.py
    las necesita para decidir cara/música/desplazamiento sin LLM); vacío
    ("") si la fila no las trae, nunca falta la clave."""
    return {
        "id": p["id"], "pregunta": p["pregunta"],
        "respuesta_esperada": p.get("respuesta_esperada", ""), "cara": p.get("cara", "Neutral"),
        "cara_respuesta_buena": p.get("cara_respuesta_buena", ""),
        "cara_respuesta_mala": p.get("cara_respuesta_mala", ""),
        "musical": p.get("musical", ""),
        "desplazamiento": p.get("desplazamiento", ""),
    }


def pregunta_aleatoria(excluir=()):
    """Una pregunta al azar para que el robot se la haga al usuario.

    Solo entrega las que tienen respuesta_esperada: sin ella no hay forma de
    corregir al usuario, y son ~77% del dataset (187 de 243). El
    caller manda los ids ya preguntados para no repetir dentro de la sesión.
    Devuelve None si no queda ninguna disponible."""
    ya_usados = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados and p.get("respuesta_esperada", "").strip()
    ]
    if not disponibles:
        return None
    return _formatear(random.choice(disponibles))


def pregunta_por_tema(tema, excluir=()):
    """Una pregunta al azar de una categoría puntual (columna Actividad/Tema
    del Excel, campo 'tema' en preguntas.jsonl). El caller ya resolvió la
    categoría exacta antes de llamar acá.

    El tema es una categoría exacta, no algo a lo que haya que "parecerse":
    ~11 preguntas quedan etiquetadas con dos temas juntos ("A / B", cuando el
    mismo ítem se reusa en dos minijuegos), por eso se separa por "/".

    A diferencia de pregunta_aleatoria(), no exige respuesta_esperada: hay
    temas enteros (Chistes, Reconocimiento Musical...) donde ninguna pregunta
    la tiene, porque no hay una "respuesta correcta" que corregir. Devuelve
    None si no queda ninguna disponible."""
    tema = (tema or "").strip()
    if not tema:
        return None
    ya_usados = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados
        and tema in [t.strip() for t in p.get("tema", "").split("/")]
    ]
    if not disponibles:
        return None
    return _formatear(random.choice(disponibles))


def preguntas_por_tema(tema, excluir=(), cantidad=5):
    """Como pregunta_por_tema(), pero devuelve una tanda de varias de una vez
    en vez de una por una — para armar un bloque de N preguntas seguidas de
    un mismo tema sin repetir el sorteo cada vez."""
    tema = (tema or "").strip()
    if not tema:
        return []
    ya_usados = set(excluir)
    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados
        and tema in [t.strip() for t in p.get("tema", "").split("/")]
    ]
    elegidas = random.sample(disponibles, min(cantidad, len(disponibles)))
    return [_formatear(p) for p in elegidas]


def recuperar_contexto(query, n_results=2):
    """Búsqueda por keyword (BM25) para Chat libre — sin embeddings: un
    ranking sobre las mismas palabras tokenizadas al cargar preguntas.jsonl."""
    relevantes = []
    if _bm25 is not None:
        scores = _bm25.get_scores(_tokenizar(query))
        top = sorted(zip(_bm25_ids, scores), key=lambda x: x[1], reverse=True)
        relevantes = [pid for pid, score in top if score > 0][:n_results]

    if not relevantes:
        return SIN_CONTEXTO

    # Solo Pregunta/Respuesta -- "Cara" es metadata de animación para Trivia
    # (Agent_Behavior.py), no un dato relevante para responder algo en Chat
    # libre. Se sacó de acá porque lora-chat (fine-tuned SOLO con
    # contextos de 2 líneas, ver chat_training/) a veces la repetía tal cual
    # como si fuera parte de la respuesta -- no seguía un formato que nunca
    # vio en el entrenamiento.
    return "\n\n".join(
        f"Pregunta: {_preguntas_cache[pid]['pregunta']}\n"
        f"Respuesta: {_preguntas_cache[pid].get('respuesta_esperada', '')}"
        for pid in relevantes
    )


_cargar_preguntas()  # se arma una sola vez, al importar el módulo
