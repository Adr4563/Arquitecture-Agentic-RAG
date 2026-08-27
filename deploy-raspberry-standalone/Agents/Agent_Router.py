"""
Agente router: clasifica cada mensaje del usuario en TRIVIA / CHAT_LIBRE --
SIN LLM. Reemplaza el router anterior (Clients.Llama_Client.enrutar(),
few-shot sobre qwen2.5:0.5b) por un clasificador clásico (TF-IDF de n-gramas
de caracteres + regresión logística, scikit-learn) entrenado en
router_training/ sobre un dataset sintético.

Binario, no de 3 etiquetas: Búsqueda web (DuckDuckGo) se sacó del proyecto a
pedido del usuario -- se reentrenó sin esos ~238 ejemplos (ver
router_training/dataset_router.jsonl) para que el clasificador no siga
"sabiendo" de una ruta que ya no existe.

Por qué: enrutar es una clasificación de etiquetas fijas, no generación de
texto -- no hace falta un modelo de lenguaje para eso. El archivo entrenado
(router_modelo.joblib) pesa ~68KB y corre en microsegundos en CPU, contra
los ~400MB y 1-3s de una llamada a Ollama. Ver router_training/entrenar_router.py
para cómo se armó y benchmark de precisión (~96% held-out, 6/6 en frases
nunca vistas por el dataset).

Si algún día hace falta reentrenar (ej. TEMAS_CATALOGO gana un tema nuevo y
el router lo confunde con otra ruta), editar
router_training/dataset_router.jsonl y correr router_training/entrenar_router.py
de nuevo -- no hace falta tocar este archivo.
"""

import os

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
_MODELO_PATH = os.path.join(HERE, "router_modelo.joblib")

_modelo = joblib.load(_MODELO_PATH)
_vectorizador = _modelo["vectorizador"]
_clasificador = _modelo["clasificador"]

RUTAS_VALIDAS = {"TRIVIA", "CHAT_LIBRE"}


def enrutar(mensaje_usuario):
    """Clasifica mensaje_usuario en una de RUTAS_VALIDAS. Misma firma que el
    enrutar() anterior de Llama_Client, así que Orchestrator_Management.py no
    tuvo que cambiar su lógica, solo el import -- nunca devuelve "" (a
    diferencia del viejo enrutar() basado en LLM, que podía fallar por un
    problema de red/servidor): esto es una cuenta local, siempre responde
    algo de RUTAS_VALIDAS."""
    vector = _vectorizador.transform([mensaje_usuario])
    return _clasificador.predict(vector)[0]
