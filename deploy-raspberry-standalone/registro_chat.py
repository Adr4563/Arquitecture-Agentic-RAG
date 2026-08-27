# -*- coding: utf-8 -*-
"""Registro de conversaciones de Chat libre -- paso 1 del pipeline de mejora
(ver chat_libre_training/README.md).

Cada turno se guarda en un JSONL con el mensaje del usuario, lo que
contesto el modelo, y QUE modelo fue. Ese ultimo dato es el que permite
comparar versiones despues: si la v3 empeoro respecto de la v2, hace falta
saber cual respondio cada cosa.

Escribe y listo: no bloquea el turno ni rompe la conversacion si falla el
disco -- mismo criterio que el resto de los clientes (ver
Clients/Musica_Client.py). Un robot que se cae por no poder loguear es peor
que uno que pierde un registro.

NO se entrena directo con esto. Las respuestas de acá son las del modelo, y
entrenar a un modelo con su propia salida refuerza sus errores (colapso).
Tienen que pasar por curar.py, donde se aprueban, corrigen o descartan.
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRO = os.environ.get(
    "CHAT_LIBRE_REGISTRO",
    os.path.join(HERE, "chat_libre_training", "conversaciones.jsonl"),
)
# Interruptor por si se quiere correr el robot sin dejar rastro (una demo,
# una prueba). Por default registra: sin datos no hay pipeline.
ACTIVO = os.environ.get("CHAT_LIBRE_REGISTRAR", "1") not in ("0", "false", "False")


def registrar(mensaje_usuario, respuesta, modelo):
    """Agrega un turno de Chat libre al registro. Silencioso ante cualquier
    error: perder una linea del log nunca puede cortar la conversacion."""
    if not ACTIVO or not mensaje_usuario or not respuesta:
        return
    fila = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modelo": modelo,
        "usuario": mensaje_usuario.strip(),
        "respuesta": respuesta.strip(),
        # Lo completa curar.py: "buena" | "corregida" | "descartada".
        # Sin revisar todavia -> None.
        "veredicto": None,
        # Si el veredicto es "corregida", la respuesta que deberia haber dado.
        "correccion": None,
    }
    try:
        os.makedirs(os.path.dirname(REGISTRO), exist_ok=True)
        with open(REGISTRO, "a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[registro] no se pudo guardar el turno ({e})")
