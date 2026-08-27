# -*- coding: utf-8 -*-
"""Regenera preguntas.jsonl desde Preguntas_Robot_EVA_EvaSIM.xlsx.

El .xlsx es la FUENTE: es donde se editan las preguntas a mano. El .jsonl
es lo que lee el robot en runtime (preguntas.py). Sin este script los dos
se desincronizan en silencio -- paso: se borraron 7 preguntas y se
completaron 9 respuestas esperadas en el Excel, y el robot siguio usando
las viejas porque ningun .py lee el .xlsx.

Correr esto cada vez que se toque el Excel:

    pip install openpyxl          # no esta en requirements.txt: es solo
                                  # para esto, no lo necesita el runtime
    python excel_a_jsonl.py

Revisa el diff antes de commitear: este script SOBREESCRIBE preguntas.jsonl.
"""
import io
import json
import os

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "Preguntas_Robot_EVA_EvaSIM.xlsx")
JSONL = os.path.join(HERE, "preguntas.jsonl")
HOJA = "Preguntas Ereberus"

# Indice de columna en el Excel -> campo del jsonl. Si se agrega o mueve una
# columna en el Excel, actualizar esto (y verificar con --dry-run).
COLUMNAS = {
    "pregunta": 4,
    "respuesta_esperada": 5,
    "cara": 6,
    "tema": 9,
    "cara_respuesta_buena": 11,
    "cara_respuesta_mala": 12,
    "musical": 13,
    "desplazamiento": 14,
}
# Orden de las claves en cada linea del jsonl (para que el diff de git quede
# legible y estable entre corridas).
ORDEN = ["id", "pregunta", "cara", "respuesta_esperada", "tema",
         "cara_respuesta_buena", "cara_respuesta_mala", "musical",
         "desplazamiento"]


def leer_excel():
    hoja = openpyxl.load_workbook(XLSX, data_only=True)[HOJA]
    filas = [tuple("" if c is None else str(c).strip() for c in fila)
             for fila in hoja.iter_rows(values_only=True)]
    return [f for f in filas if any(f)]


def construir(filas):
    preguntas = []
    for i, fila in enumerate(filas[1:], 1):   # [0] es la cabecera
        if not fila[COLUMNAS["pregunta"]]:
            continue                          # fila vacia o separador
        preguntas.append({campo: (fila[idx] if idx < len(fila) else "")
                          for campo, idx in COLUMNAS.items()})
        preguntas[-1]["id"] = i
    return preguntas


def main():
    filas = leer_excel()
    preguntas = construir(filas)
    print(f"{len(preguntas)} preguntas leidas de {os.path.basename(XLSX)}")

    if os.path.isfile(JSONL):
        antes = [json.loads(l) for l in io.open(JSONL, encoding="utf-8") if l.strip()]
        viejas = {p["pregunta"] for p in antes}
        nuevas = {p["pregunta"] for p in preguntas}
        print(f"  {len(antes)} -> {len(preguntas)}"
              f"  (se van {len(viejas - nuevas)}, entran {len(nuevas - viejas)})")

    with io.open(JSONL, "w", encoding="utf-8", newline="\n") as f:
        for p in preguntas:
            ordenada = {k: p.get(k, "") for k in ORDEN}
            f.write(json.dumps(ordenada, ensure_ascii=False) + "\n")
    print(f"Escrito {JSONL}")


if __name__ == "__main__":
    main()
