# -*- coding: utf-8 -*-
"""Curacion de conversaciones -- paso 2 del pipeline (ver README.md).

Recorre los turnos que registro_chat.py fue guardando y te deja marcar cada
uno. Es el paso que hace que el pipeline mejore al modelo en vez de
empeorarlo: entrenar con las respuestas crudas del modelo le ensena a
imitar sus propios errores.

Tres veredictos:

    b  buena       -> la respuesta sirve tal cual, se usa como ejemplo
    c  corregida   -> escribis vos la respuesta que deberia haber dado
    d  descartada  -> no sirve ni corregida (fuera de dominio, sin sentido)

Solo "buena" y "corregida" llegan al dataset. Las descartadas se guardan
igual, para no volver a revisarlas.

Uso:
    python curar.py                  # revisa lo que falta
    python curar.py --revisar-todo   # incluye lo ya revisado
    python curar.py --resumen        # solo estadisticas, no toca nada
"""
import argparse
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRO = os.path.join(HERE, "conversaciones.jsonl")


def cargar():
    if not os.path.isfile(REGISTRO):
        return []
    with io.open(REGISTRO, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def guardar(filas):
    """Reescribe el archivo entero. Son miles de lineas como mucho: no vale
    la pena complicarse con edicion en el lugar, y reescribir es atomico de
    entender si algo sale mal."""
    tmp = REGISTRO + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    os.replace(tmp, REGISTRO)


def resumen(filas):
    from collections import Counter
    c = Counter(f.get("veredicto") for f in filas)
    print(f"  {len(filas)} turnos registrados")
    print(f"    sin revisar : {c.get(None, 0)}")
    print(f"    buenas      : {c.get('buena', 0)}")
    print(f"    corregidas  : {c.get('corregida', 0)}")
    print(f"    descartadas : {c.get('descartada', 0)}")
    utiles = c.get("buena", 0) + c.get("corregida", 0)
    print(f"    -> {utiles} ejemplos utiles para entrenar")
    if utiles < 100:
        print(f"       (pocos todavia: apunta a 200+ antes de reentrenar)")
    por_modelo = Counter(f.get("modelo") for f in filas)
    if len(por_modelo) > 1:
        print("    por modelo:")
        for m, n in por_modelo.most_common():
            print(f"      {m}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revisar-todo", action="store_true",
                    help="incluye los turnos ya revisados")
    ap.add_argument("--resumen", action="store_true", help="solo estadisticas")
    args = ap.parse_args()

    filas = cargar()
    if not filas:
        print(f"No hay nada en {REGISTRO}.")
        print("Conversa con el robot (Chat libre) y volve a correr esto.")
        return

    if args.resumen:
        resumen(filas)
        return

    pendientes = [f for f in filas if args.revisar_todo or f.get("veredicto") is None]
    if not pendientes:
        print("No hay turnos sin revisar.")
        resumen(filas)
        return

    print(f"{len(pendientes)} turnos por revisar.  [b]uena  [c]orregir  "
          f"[d]escartar  [s]altar  [q]uitar y guardar\n")
    revisados = 0
    for i, fila in enumerate(pendientes, 1):
        print("-" * 70)
        print(f"({i}/{len(pendientes)})  {fila['ts']}  modelo={fila['modelo']}")
        print(f"  Usuario  : {fila['usuario']}")
        print(f"  Respondio: {fila['respuesta']}")
        if fila.get("veredicto"):
            print(f"  (ya estaba marcado como {fila['veredicto']})")
        try:
            op = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCortado.")
            break
        if op == "q":
            break
        if op == "s" or op == "":
            continue
        if op == "b":
            fila["veredicto"] = "buena"
            fila["correccion"] = None
        elif op == "d":
            fila["veredicto"] = "descartada"
            fila["correccion"] = None
        elif op == "c":
            try:
                mejor = input("  respuesta correcta: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not mejor:
                print("  (vacia, se salta)")
                continue
            fila["veredicto"] = "corregida"
            fila["correccion"] = mejor
        else:
            print("  opcion desconocida, se salta")
            continue
        revisados += 1

    guardar(filas)
    print(f"\n{revisados} turnos marcados. Guardado en {os.path.basename(REGISTRO)}.\n")
    resumen(filas)


if __name__ == "__main__":
    main()
