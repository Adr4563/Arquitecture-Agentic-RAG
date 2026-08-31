# -*- coding: utf-8 -*-
"""Memoria episódica de Chat libre -- v1, sin embeddings (2026-08-31).

Reusa el JSONL que ya escribe registro_chat.py (chat_libre_training/
conversaciones.jsonl) en vez de guardar un archivo aparte: cada turno ya
queda ahí con mensaje del usuario + respuesta + timestamp. Este módulo solo
LEE ese archivo (nunca lo trunca ni lo modifica) e indexa los últimos
_MAXIMO_EPISODIOS con BM25 puro (rank_bm25, ya en requirements.txt de un
RAG anterior) para poder recordar un intercambio pasado cuando viene al
caso.

Por qué NO embeddings/reranker: la Pi ya tiene lora-chat-libre-v4/
lora-trivia/lora-salida-trivia-v2 residentes en RAM vía Ollama -- otro
modelo más (el encoder de embeddings) compite por el mismo recurso que ya
está justo. BM25 es búsqueda léxica pura, sin inferencia: para unos cientos
de episodios la consulta es sub-10ms en CPU, no le agrega nada a Ollama.

Por qué el umbral de relevancia NO es "score > 0": la primera versión del
RAG de Chat libre (ver la nota en Orchestrator_Management.responder(), "YA
NO USA RAG") usaba exactamente ese umbral y anduvo mal -- una sola palabra
vacía en común bastaba para inyectar contexto de otra pregunta, y el modelo
lo usaba con total seguridad para inventar una respuesta. Acá el gate es
"al menos 2 palabras DE CONTENIDO en común" (stopwords no cuentan), no el
score en sí -- el score de BM25 no está normalizado entre consultas, así
que un umbral numérico fijo es frágil (mismo problema, otra forma).
Deliberadamente v1: sin el refinamiento de "top1 tiene que superar a top2
por X" -- se mide primero cómo funciona el gate de overlap solo, y si
aparecen falsos recuerdos en la práctica ahí se ajusta con esa señal
adicional, no antes.
"""
import json
import os
import re

import registro_chat

_MAXIMO_EPISODIOS = 300  # últimos N turnos considerados, no todo el historial
_OVERLAP_MINIMO = 2  # palabras de contenido en común mínimas para "recordar"
# ...salvo que la palabra compartida sea RARA. "Fano se escapo ayer" comparte
# una sola palabra de contenido con "tengo un perrito que se llama Fano"
# ("fano"), asi que con el umbral de 2 no recordaba nada -- el caso exacto
# que reporto el usuario. Pero "fano" aparece en 3 de 3 episodios sobre el
# perro y en ninguno mas: es un nombre propio, mucho mas informativo que dos
# palabras comunes en comun.
#
# Se cuenta en cuantos episodios aparece cada palabra: si la compartida es
# lo bastante rara, alcanza con UNA para recordar.
#
# Umbral ABSOLUTO, no relativo al corpus. Se probo relativo (25% de los
# episodios) y fallaba justo en el caso que importa: el chico habla varias
# veces de su perro, asi que "fano" aparecia en 3 de 8 episodios (37%) y
# dejaba de contar como raro -- perdiendo exactamente el recuerdo mas util.
# En una memoria personal, lo que se repite es lo que MAS importa.
#
# Lo que evita los falsos positivos no es el umbral sino las stopwords: los
# que aparecian ("quiero jugar" -> "jugar", "tengo hambre" -> "tengo") eran
# verbos comunes, ahora filtrados antes de contar.
_TOPE_RARA = 8
_MAX_CHARS_SNIPPET = 140  # tope aprox. para que el recuerdo inyectado quede en ~40 tokens

# Lista corta y práctica, no exhaustiva -- alcanza para filtrar las palabras
# que enganchan cualquier cosa (de, que, el, la...) sin pesar un diccionario.
_STOPWORDS_ES = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "a", "en", "que", "es", "por", "con", "no", "se", "su", "sus", "para",
    "como", "más", "mas", "pero", "le", "les", "ya", "este", "esta",
    "estos", "estas", "ese", "esa", "esos", "esas", "sí", "si", "porque",
    "entre", "cuando", "muy", "sin", "sobre", "también", "tambien", "me",
    "hasta", "hay", "donde", "quien", "quién", "desde", "todo", "toda",
    # Verbos y muletillas que aparecen en casi cualquier frase de un chico y
    # no identifican un tema. Sin esto "tengo hambre" recordaba "tengo un
    # perrito" por compartir "tengo".
    "tengo", "tienes", "tenes", "tenés", "tiene", "quiero", "quieres",
    "queres", "querés", "gusta", "gustan", "gusto", "puedo", "puedes",
    "podes", "podés", "hacer", "hago", "haces", "estoy", "estas", "estás",
    "soy", "eres", "sos", "voy", "vas", "fui", "era", "ser", "estar",
    "mucho", "mucha", "poco", "algo", "nada", "bien", "mal", "hoy", "ayer",
    # Verbos de actividad genericos: aparecen en cualquier frase y no
    # identifican de QUE se hablaba. "quiero jugar" no deberia recordar
    # "me gusta jugar con Fano".
    "jugar", "juego", "juegas", "jugas", "jugás", "juegan", "jugamos",
    "ver", "veo", "ves", "mirar", "miro", "decir", "digo", "dice",
    "contar", "cuento", "cuentas", "saber", "sabes", "sabe", "ir", "voy",
    "venir", "vengo", "dar", "doy", "poner", "pongo", "llamar", "llama",
    "todos", "todas", "nos", "durante", "uno", "ni", "contra", "otro",
    "otra", "otros", "otras", "ante", "ellos", "ellas", "e", "esto", "eso",
    "mi", "mis", "tu", "tus", "él", "ella", "nosotros", "nosotras",
    "vosotros", "vosotras", "ustedes", "usted", "yo", "tú", "qué", "que",
    "cuál", "cual", "cómo", "como", "cuándo", "cuando", "dónde", "donde",
    "quién", "quien", "algo", "nada", "algunos", "algunas", "mucho",
    "muchos", "mucha", "muchas", "poco", "pocos", "está", "esta", "están",
    "estar", "ser", "soy", "eres", "somos", "son", "fue", "fui", "era",
    "han", "he", "has", "hemos", "vos", "te", "tal",
}

_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)

# Cache en memoria del índice -- se reconstruye solo cuando cambia el mtime
# del JSONL (o sea, cuando registro_chat.registrar() agregó un turno nuevo),
# no en cada mensaje.
_cache = {"mtime": None, "bm25": None, "episodios": None, "corpus_tok": None}


def _tokenizar(texto):
    """minúsculas, sin puntuación, sin stopwords -- lo que queda son
    palabras "de contenido" (nombres, modelos, temas)."""
    crudo = _TOKEN_RE.findall(texto.lower())
    return [_singular(t) for t in crudo
            if t not in _STOPWORDS_ES and len(t) > 2]


def _singular(palabra):
    """Normaliza plurales simples para que "dinosaurio" matchee
    "dinosaurios". Sin esto el recuerdo se perdia por una sola letra.

    Deliberadamente tosco (no es un lematizador): recorta "es"/"s" finales
    en palabras largas. Alcanza para nombres y sustantivos, que es lo que
    queda despues de sacar stopwords."""
    if len(palabra) > 4 and palabra.endswith("es"):
        return palabra[:-2]
    if len(palabra) > 3 and palabra.endswith("s"):
        return palabra[:-1]
    return palabra


def _cargar_episodios():
    """Últimos _MAXIMO_EPISODIOS turnos válidos del JSONL de registro_chat,
    más nuevo al final. Nunca escribe ni trunca ese archivo -- lo sigue
    necesitando curar.py completo (ver chat_libre_training/)."""
    ruta = registro_chat.REGISTRO
    if not os.path.exists(ruta):
        return []
    episodios = []
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    fila = json.loads(linea)
                except ValueError:
                    continue  # línea corrupta -- se ignora, no rompe la memoria
                usuario = (fila.get("usuario") or "").strip()
                respuesta = (fila.get("respuesta") or "").strip()
                if usuario and respuesta:
                    episodios.append({"usuario": usuario, "respuesta": respuesta})
    except OSError:
        return []
    return episodios[-_MAXIMO_EPISODIOS:]


def _obtener_indice():
    """BM25 en RAM, reconstruido solo si el JSONL cambió desde la última
    vez (mtime) -- no en cada llamado a buscar_relevante()."""
    from rank_bm25 import BM25Okapi  # import perezoso: nada de esto pesa si Chat libre no se usa

    ruta = registro_chat.REGISTRO
    mtime = os.path.getmtime(ruta) if os.path.exists(ruta) else None
    if mtime is not None and mtime == _cache["mtime"] and _cache["bm25"] is not None:
        return _cache["bm25"], _cache["episodios"], _cache["corpus_tok"]

    episodios = _cargar_episodios()
    corpus_tok = [
        _tokenizar(f"{ep['usuario']} {ep['respuesta']}") for ep in episodios
    ]
    bm25 = BM25Okapi(corpus_tok) if corpus_tok else None

    _cache["mtime"] = mtime
    _cache["bm25"] = bm25
    _cache["episodios"] = episodios
    _cache["corpus_tok"] = corpus_tok
    return bm25, episodios, corpus_tok


def buscar_relevante(mensaje_usuario):
    """Si hay un episodio pasado que comparte al menos _OVERLAP_MINIMO
    palabras de contenido con `mensaje_usuario`, devuelve un snippet corto
    (<=140 caracteres, ~40 tokens) para inyectar en el prompt de
    CHAT_MODEL. Si no hay nada suficientemente relevante, devuelve None --
    "no recordar nada" es preferible a un falso recuerdo (ver la nota del
    módulo).

    El candidato se elige por OVERLAP real (intersección de tokens), no por
    el score de BM25 -- probado en vivo: con pocos episodios el IDF clásico
    de BM25 puede colapsar a 0 para una palabra que aparece en la mitad
    exacta del corpus, aunque el overlap real sea total. El score de BM25
    se usa solo como desempate entre candidatos con el mismo overlap."""
    query_tok = set(_tokenizar(mensaje_usuario))
    if not query_tok:
        return None

    bm25, episodios, corpus_tok = _obtener_indice()
    if bm25 is None or not episodios:
        return None

    # Documento-frecuencia: en cuantos episodios aparece cada palabra. Una
    # palabra que sale en 1-2 episodios es distintiva (un nombre propio);
    # una que sale en 20 no dice nada.
    df = {}
    for tokens in corpus_tok:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    scores = bm25.get_scores(list(query_tok))
    # Un episodio cuyo mensaje es (casi) el mismo que el actual no es un
    # recuerdo: es el eco del propio mensaje. Pasa siempre, porque
    # registro_chat.py guarda cada turno y el usuario repite frases. Se
    # descartan los que comparten TODAS las palabras de contenido -- de esos
    # no hay nada nuevo que aportarle al modelo.
    # Se compara contra lo que dijo EL USUARIO, no contra corpus_tok: ese
    # indexa mensaje + respuesta juntos, asi que un episodio nunca es
    # exactamente igual a la consulta aunque el mensaje si lo sea (los
    # tokens de la respuesta de Lora sobran).
    ecos = {i for i, ep in enumerate(episodios)
            if query_tok and query_tok == set(_tokenizar(ep.get("usuario", "")))}
    mejor_idx, mejor_overlap, mejor_score, mejor_raras = None, 0, None, 0
    for i, tokens in enumerate(corpus_tok):
        if i in ecos:
            continue
        comunes = query_tok & set(tokens)
        overlap = len(comunes)
        raras = sum(1 for t in comunes if df.get(t, 0) <= _TOPE_RARA)
        # Gana el que tenga mas palabras RARAS en comun; a igualdad, mas
        # overlap total; a igualdad, mejor score de BM25.
        clave = (raras, overlap, scores[i])
        if mejor_idx is None or clave > (mejor_raras, mejor_overlap, mejor_score):
            mejor_idx, mejor_overlap, mejor_score, mejor_raras = i, overlap, scores[i], raras

    # Alcanza con UNA palabra rara, o con _OVERLAP_MINIMO comunes.
    if mejor_idx is None or (mejor_raras < 1 and mejor_overlap < _OVERLAP_MINIMO):
        return None

    # Se inyecta lo que dijo EL USUARIO, no lo que contesto Lora. El dato
    # esta del lado del usuario: ante "que raza es Fano" el recuerdo util es
    # "Fano es un caniche chiquito", no la pregunta "de que raza es?" que
    # Lora habia hecho -- ese era el segundo bug reportado.
    ep = episodios[mejor_idx]
    recuerdo = ep.get("usuario", "").strip()
    if len(recuerdo) > _MAX_CHARS_SNIPPET:
        recuerdo = recuerdo[:_MAX_CHARS_SNIPPET].rsplit(" ", 1)[0] + "..."
    return f"(Antes me contaste: {recuerdo})"
