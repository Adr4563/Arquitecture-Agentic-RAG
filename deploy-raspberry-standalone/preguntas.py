"""
Acceso a las preguntas de Trivia: preguntas.jsonl cargado en memoria, sin
HTTP.

Antes tenia ademas un indice BM25 para el RAG de Chat libre. Se elimino
cuando Chat libre dejo de usar RAG (ver Orchestrator_Management.responder()):
el corpus eran estas mismas preguntas de Trivia, que es justo de lo que Chat
libre no tiene que hablar.

chat.py/workers.py y esta lógica corren en el mismo proceso: no hay ninguna
razón para exponer esto por red y pagar un round-trip HTTP a localhost en
cada pregunta de trivia o mensaje de chat libre. Por eso es un módulo
normal, importado directo por workers.py — no un servidor aparte.
"""

import json
import os
import random


HERE = os.path.dirname(os.path.abspath(__file__))
PREGUNTAS_FILE = os.path.join(HERE, "base_datos", "preguntas.jsonl")


_preguntas_cache = {}  # id -> {"pregunta", "cara", "respuesta_esperada", "tema"}




def _cargar_preguntas():
    """Lee preguntas.jsonl y arma _preguntas_cache."""
    global _preguntas_cache

    with open(PREGUNTAS_FILE, encoding="utf-8") as f:
        preguntas = [json.loads(l) for l in f if l.strip()]

    if not preguntas:
        print("preguntas.jsonl está vacío, nada que cargar.")
        return

    _preguntas_cache = {p["id"]: p for p in preguntas}
    print(f"[preguntas.py] {len(preguntas)} preguntas cargadas en memoria.")


def _formatear(p):
    """Los campos que le llegan a chat.py/reactor.py por cada pregunta —
    incluye cara_respuesta_buena/mala y musical/desplazamiento (reactor.py
    las necesita para decidir cara/música/desplazamiento sin LLM); vacío
    ("") si la fila no las trae, nunca falta la clave."""
    return {
        "id": p["id"], "pregunta": p["pregunta"],
        "respuesta_esperada": p.get("respuesta_esperada", ""),
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


# recuperar_contexto() se elimino junto con el RAG de Chat libre (ver
# Orchestrator_Management.responder()). Buscaba con BM25 sobre estas mismas
# preguntas y devolvia el contexto para el prompt; el usuario pidio que Chat
# libre no hable de las preguntas de Trivia, asi que quedo sin llamadores.
# Con ella se fueron el indice BM25 y _tokenizar().


_cargar_preguntas()  # se arma una sola vez, al importar el módulo
