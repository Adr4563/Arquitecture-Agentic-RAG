# -*- coding: utf-8 -*-
"""Arma dataset_chat_libre.jsonl con lo curado -- paso 3 del pipeline.

Toma de conversaciones.jsonl solo los turnos marcados "buena" o "corregida"
(en las corregidas usa TU respuesta, no la del modelo) y los deja en el
formato que espera entrenar_chat_libre.py.

Las descartadas y las sin revisar no entran. Ese filtro es lo que evita el
colapso: sin el, el modelo se entrenaria con su propia salida sin filtrar.

Uso:
    python generar_dataset.py
    python generar_dataset.py --semilla   # agrega ejemplos base de arranque
"""
import argparse
import io
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRO = os.path.join(HERE, "conversaciones.jsonl")
SALIDA = os.path.join(HERE, "dataset_chat_libre.jsonl")

# Ejemplos de arranque para la primera version, cuando todavia no hay
# conversaciones curadas suficientes. Cubren lo que medimos que el modelo
# hace mal: se desentiende de temas emocionales, no mantiene el hilo, e
# inventa datos en vez de admitir que no sabe.
#
# Son a proposito de ESTILO, no de conocimiento: un 0.5B no va a aprender
# quien pinto la Mona Lisa por mas ejemplos que se le den (medido: su base
# ya cree que fue Frida Kahlo). Lo que si aprende es a acompanar, repreguntar
# y no inventar.
SEMILLA = [
    # -- acompanar en vez de desentenderse --
    ("hoy me fue mal en el colegio", "Uf, qué feo. ¿Qué fue lo que pasó?"),
    ("me siento triste", "Lamento que estés así. ¿Querés contarme qué te pasó?"),
    ("estoy aburrido", "¿Y qué te gustaría hacer? Puedo proponerte una trivia."),
    ("nadie quiere jugar conmigo", "Eso duele. ¿Pasó hoy en el recreo?"),
    ("mi mejor amiga se mudó de ciudad", "Vas a extrañarla. ¿Pudiste despedirte?"),
    ("tengo miedo a la oscuridad", "Es normal. ¿Te pasa sobre todo al ir a dormir?"),
    ("me peleé con mi hermano", "Pasa. ¿Por qué fue la pelea?"),
    ("mañana tengo un examen", "¿De qué materia? ¿Pudiste estudiar?"),
    # -- mantener el hilo: repreguntar sobre lo que dijo --
    ("me gusta el futbol", "¿De qué equipo sos?"),
    ("me gusta dibujar", "¿Y qué te gusta dibujar más?"),
    ("tengo un perro", "¡Qué bueno! ¿Cómo se llama?"),
    ("fui a la playa el fin de semana", "¡Qué lindo! ¿Te metiste al agua?"),
    # -- admitir que no sabe en vez de inventar --
    ("cuanto es 847 por 293", "Uf, esa cuenta es difícil para mí. Mejor usá una calculadora."),
    ("que va a pasar manana", "No tengo forma de saberlo, no puedo ver el futuro."),
    ("que hora es", "No tengo reloj, no puedo saberlo."),
    ("como esta el clima hoy", "No puedo consultar el clima, no tengo internet para eso."),
    # -- cortito y natural, sin lenguaje de asistente --
    ("hola", "¡Hola! ¿Cómo andás?"),
    ("gracias", "De nada."),
    ("como te llamas", "Me llamo Lora."),
    ("adios", "¡Chau! Nos vemos."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semilla", action="store_true",
                    help="agrega los ejemplos base (para la primera version)")
    args = ap.parse_args()

    ejemplos = []
    origen = Counter()

    if args.semilla:
        for u, r in SEMILLA:
            ejemplos.append({"instruccion": u, "objetivo": r})
        origen["semilla"] = len(SEMILLA)

    if os.path.isfile(REGISTRO):
        with io.open(REGISTRO, encoding="utf-8") as f:
            for linea in f:
                if not linea.strip():
                    continue
                fila = json.loads(linea)
                v = fila.get("veredicto")
                if v == "buena":
                    objetivo = fila["respuesta"]
                elif v == "corregida":
                    objetivo = fila.get("correccion") or ""
                else:
                    continue          # descartada o sin revisar
                if not objetivo.strip():
                    continue
                ejemplos.append({"instruccion": fila["usuario"], "objetivo": objetivo.strip()})
                origen[v] += 1

    if not ejemplos:
        print("No hay ejemplos. Curá conversaciones (curar.py) o corré con --semilla.")
        return

    # Duplicados exactos: si el mismo mensaje se curo dos veces con la misma
    # respuesta, no aporta y sobrerrepresenta ese caso.
    vistos = set()
    unicos = []
    for e in ejemplos:
        clave = (e["instruccion"].lower(), e["objetivo"].lower())
        if clave in vistos:
            continue
        vistos.add(clave)
        unicos.append(e)

    with io.open(SALIDA, "w", encoding="utf-8", newline="\n") as f:
        for e in unicos:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"{len(unicos)} ejemplos escritos en {os.path.basename(SALIDA)}"
          f" ({len(ejemplos) - len(unicos)} duplicados descartados)")
    for k, n in origen.most_common():
        print(f"    {k}: {n}")
    if len(unicos) < 100:
        print("\n  Ojo: con menos de ~100 ejemplos el fine-tune tiende a")
        print("  sobreajustar y empeora el modelo. Junta mas conversaciones curadas.")


if __name__ == "__main__":
    main()
