"""
Servidor de embeddings + RAG (Flask + Ollama para embeddings + Chroma).

Al arrancar: lee base_datos/preguntas.jsonl, calcula los vectores (vía Ollama,
modelo qwen3-embedding:0.6b) e indexa todo en ChromaDB (colección "preguntas").

Requiere Ollama corriendo con el modelo bajado:
    ollama pull qwen3-embedding:0.6b

Uso:
    python embed_server.py

Endpoints:
    POST /v1/embeddings  {"input": [...]}                  -> vectores crudos
    POST /pregunta       {"query": "...", "n_results": 2}  -> preguntas + cara + respuesta
"""

import os
import json
import re
import random
import time
import requests
from flask import Flask, request, jsonify
from rank_bm25 import BM25Okapi
import chromadb

HERE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(HERE)
PREGUNTAS_FILE = os.path.join(RAIZ, "base_datos", "preguntas.jsonl")
CHROMA_PATH = os.path.join(RAIZ, "base_datos", "chroma_data")
PORT = 8081

OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"

app = Flask(__name__)

db = chromadb.PersistentClient(path=CHROMA_PATH)
# Colección propia por modelo de embedding: cada modelo produce vectores de
# distinta dimensión (nomic=768, qwen3-0.6b=1024, qwen3-8b=4096) y Chroma fija
# la dimensión al crear la colección. Compartir nombre entre modelos revienta
# con un error de dimensiones al indexar.
coll = db.get_or_create_collection("preguntas_qwen3_06b")

# Índice BM25 (búsqueda por keyword exacto) en memoria, en paralelo al vector
# store de Chroma. Se reconstruye cada vez que se sincroniza preguntas.jsonl.
_bm25 = None
_bm25_ids = []
_preguntas_cache = {}  # id -> {"pregunta", "cara", "respuesta_esperada"}


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


def _reconstruir_bm25(preguntas):
    global _bm25, _bm25_ids, _preguntas_cache
    _bm25_ids = [p["id"] for p in preguntas]
    _preguntas_cache = {p["id"]: p for p in preguntas}
    _bm25 = BM25Okapi([_tokenizar(p["pregunta"]) for p in preguntas])


TAMANIO_LOTE_EMBED = 32  # mandar los 246+ de una sola vez a Ollama a veces tira 400 (flaky, no confiable)
REINTENTOS_EMBED = 3


def _embed_lote(textos, intento=1):
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": textos},
        timeout=120,
    )
    if resp.status_code == 400 and intento < REINTENTOS_EMBED:
        # Se vio fallar de forma intermitente con lotes grandes (Ollama
        # cargando/liberando el modelo, contención con otro request en
        # simultáneo) — el mismo lote exacto suele funcionar en el reintento.
        time.sleep(1.5 * intento)
        return _embed_lote(textos, intento + 1)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _embed(textos):
    """Manda los textos a Ollama en lotes chicos en vez de todos de una vez:
    un solo request con 246+ textos falla con 400 de forma intermitente."""
    embeddings = []
    for i in range(0, len(textos), TAMANIO_LOTE_EMBED):
        embeddings.extend(_embed_lote(textos[i:i + TAMANIO_LOTE_EMBED]))
    return embeddings


def _indexar_preguntas():
    """Sincroniza ChromaDB con preguntas.jsonl: agrega ids nuevos, actualiza los que
    cambiaron de contenido y borra los que ya no están en el archivo."""
    with open(PREGUNTAS_FILE, encoding="utf-8") as f:
        preguntas = [json.loads(l) for l in f if l.strip()]

    if not preguntas:
        print("preguntas.jsonl está vacío, nada que indexar.")
        return

    ids_archivo = {p["id"] for p in preguntas}
    ids_chroma = set(coll.get(include=[])["ids"])

    ids_a_borrar = ids_chroma - ids_archivo
    if ids_a_borrar:
        coll.delete(ids=list(ids_a_borrar))

    # upsert reemplaza documento+metadata si el id ya existe, o lo crea si no.
    # Así una pregunta editada en el jsonl se re-embebe con su contenido actual.
    coll.upsert(
        documents=[p["pregunta"] for p in preguntas],
        embeddings=_embed([p["pregunta"] for p in preguntas]),
        ids=[p["id"] for p in preguntas],
        metadatas=[{"cara": p.get("cara", "Neutral"),
                    "respuesta_esperada": p.get("respuesta_esperada", ""),
                    "tema": p.get("tema", "")} for p in preguntas],
    )
    print(f"Preguntas sincronizadas: {len(preguntas)} vigentes, "
          f"{len(ids_a_borrar)} eliminadas.")

    _reconstruir_bm25(preguntas)


@app.route("/v1/embeddings", methods=["POST"])
def embeddings_endpoint():
    body = request.get_json()
    textos = body["input"]
    if isinstance(textos, str):
        textos = [textos]
    vectores = _embed(textos)
    data = [{"embedding": v, "index": i} for i, v in enumerate(vectores)]
    return jsonify({"data": data, "model": EMBED_MODEL})


@app.route("/pregunta_aleatoria", methods=["POST"])
def pregunta_aleatoria_endpoint():
    """Devuelve una pregunta al azar para que el robot se la haga al usuario.

    Solo entrega las que tienen respuesta_esperada: sin ella no hay forma de
    corregir al usuario, y son casi la mitad del dataset (111 de 246).
    El cliente manda los ids ya preguntados para no repetir dentro de la sesión.
    """
    body = request.get_json(silent=True) or {}
    ya_usados = set(body.get("excluir", []))

    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados and p.get("respuesta_esperada", "").strip()
    ]
    if not disponibles:
        return jsonify({"pregunta": None})  # se acabaron las preguntas

    p = random.choice(disponibles)
    return jsonify({"pregunta": {
        "id": p["id"],
        "pregunta": p["pregunta"],
        "respuesta_esperada": p["respuesta_esperada"],
        "cara": p.get("cara", "Neutral"),
    }})


@app.route("/pregunta_por_tema", methods=["POST"])
def pregunta_por_tema_endpoint():
    """Devuelve una pregunta al azar de una categoría puntual (columna
    Actividad/Tema del Excel, campo 'tema' en preguntas.jsonl). El cliente ya
    resolvió la categoría exacta con el LLM antes de llamar acá — el usuario
    elige la sesión con texto libre, no con el nombre exacto de la categoría.

    A diferencia de /pregunta, esto no es una búsqueda semántica: el tema es
    una categoría exacta, no algo a lo que haya que "parecerse". Se filtra en
    memoria (no con el `where` de Chroma) porque ~11 preguntas quedan
    etiquetadas con dos temas juntos ("A / B", cuando el mismo ítem se reusa en
    dos minijuegos) y Chroma no filtra por substring en metadata.

    A diferencia de /pregunta_aleatoria, no exige respuesta_esperada: hay temas
    enteros (Chistes, Reconocimiento Musical...) donde ninguna pregunta la
    tiene, porque no hay una "respuesta correcta" que corregir.
    """
    body = request.get_json(silent=True) or {}
    tema = (body.get("tema") or "").strip()
    ya_usados = set(body.get("excluir", []))
    if not tema:
        return jsonify({"pregunta": None})

    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados
        and tema in [t.strip() for t in p.get("tema", "").split("/")]
    ]
    if not disponibles:
        return jsonify({"pregunta": None})

    p = random.choice(disponibles)
    return jsonify({"pregunta": {
        "id": p["id"],
        "pregunta": p["pregunta"],
        "respuesta_esperada": p.get("respuesta_esperada", ""),
        "cara": p.get("cara", "Neutral"),
    }})


@app.route("/preguntas_por_tema", methods=["POST"])
def preguntas_por_tema_endpoint():
    """Como /pregunta_por_tema, pero devuelve una tanda de varias de una vez
    (un solo request) en vez de una por una — para armar un bloque de N
    preguntas seguidas de un mismo tema."""
    body = request.get_json(silent=True) or {}
    tema = (body.get("tema") or "").strip()
    ya_usados = set(body.get("excluir", []))
    cantidad = body.get("cantidad", 5)
    if not tema:
        return jsonify({"preguntas": []})

    disponibles = [
        p for pid, p in _preguntas_cache.items()
        if pid not in ya_usados
        and tema in [t.strip() for t in p.get("tema", "").split("/")]
    ]
    elegidas = random.sample(disponibles, min(cantidad, len(disponibles)))
    return jsonify({"preguntas": [
        {"id": p["id"],
         "pregunta": p["pregunta"],
         "respuesta_esperada": p.get("respuesta_esperada", ""),
         "cara": p.get("cara", "Neutral")}
        for p in elegidas
    ]})


DISTANCIA_MAXIMA = 1.0  # más allá de esto, se considera que no hay match relevante (distancia l2)


@app.route("/pregunta", methods=["POST"])
def pregunta_endpoint():
    body = request.get_json()
    query = body["query"]
    n_results = body.get("n_results", 2)

    # 1) Búsqueda semántica (vector, Chroma) — capta similitud de significado.
    resultado = coll.query(query_embeddings=_embed([query]), n_results=n_results)
    vector_dist = dict(zip(resultado["ids"][0], resultado["distances"][0]))
    relevantes = [pid for pid, dist in vector_dist.items() if dist <= DISTANCIA_MAXIMA]

    # 2) Búsqueda por keyword (BM25) — rescata términos técnicos exactos que el
    # embedding semántico puede no priorizar bien. Solo llena huecos que dejó
    # el vector, sin desplazar los matches semánticos ya encontrados.
    if _bm25 is not None and len(relevantes) < n_results:
        scores = _bm25.get_scores(_tokenizar(query))
        top_bm25 = sorted(zip(_bm25_ids, scores), key=lambda x: x[1], reverse=True)
        for pid, score in top_bm25:
            if score <= 0 or pid in relevantes:
                continue
            relevantes.append(pid)
            if len(relevantes) >= n_results:
                break

    preguntas = [
        {"pregunta": _preguntas_cache[pid]["pregunta"],
         "cara": _preguntas_cache[pid].get("cara", "Neutral"),
         "respuesta_esperada": _preguntas_cache[pid].get("respuesta_esperada", ""),
         "distancia": vector_dist.get(pid)}
        for pid in relevantes
    ]
    return jsonify({"preguntas": preguntas})


if __name__ == "__main__":
    _indexar_preguntas()
    print(f"Servidor escuchando en 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
