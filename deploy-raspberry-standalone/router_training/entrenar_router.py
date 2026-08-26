# -*- coding: utf-8 -*-
"""Entrena el clasificador del router (Agents/Agent_Router.py) a partir de
dataset_router.jsonl y guarda el resultado en Agents/router_modelo.joblib.

Correr de nuevo esto (después de editar dataset_router.jsonl a mano, o
correr generar_dataset.py para pedirle más ejemplos al LLM) cada vez que se
quiera reentrenar — ej. si TEMAS_CATALOGO gana un tema nuevo y el router
tiene que aprender a reconocerlo.

Uso:
    cd deploy-raspberry-standalone/router_training
    python entrenar_router.py
"""
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset_router.jsonl")
SALIDA = os.path.join(HERE, "..", "Agents", "router_modelo.joblib")

# char_wb (n-gramas de caracteres, no de palabras): robusto a errores de
# tipeo/tildes/mayúsculas de alguien escribiendo rápido en una tablet -- una
# palabra mal tipeada no tira todo el vector a cero como pasaría con un
# vectorizador por palabras completas.
_PARAMS_VECTORIZADOR = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
_PARAMS_CLASIFICADOR = dict(max_iter=2000, C=5.0, class_weight="balanced")

# Frases de sanity-check que NO deberían estar en dataset_router.jsonl --
# sirven para chequear generalización real, no solo el held-out split (que
# tiene fugas: variantes casuales de la misma frase pueden caer una en train
# y otra en test, e inflar la métrica artificialmente).
_PRUEBAS_MANUALES = [
    ("quien fue el ultimo presidente electo", "BUSQUEDA_WEB"),
    ("hazme jugar algo de imitar animales", "TRIVIA"),
    ("esta lloviendo afuera ahora mismo", "BUSQUEDA_WEB"),
    ("contame un cuento", "CHAT_LIBRE"),
    ("quiero retomar donde estabamos", "TRIVIA"),
    ("a que hora sale el sol mañana", "BUSQUEDA_WEB"),
    ("sabes tocar algun instrumento", "CHAT_LIBRE"),
]


def main():
    datos = [json.loads(l) for l in open(DATASET, encoding="utf-8")]
    X = [d["frase"] for d in datos]
    y = [d["ruta"] for d in datos]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizador = TfidfVectorizer(**_PARAMS_VECTORIZADOR)
    Xv_train = vectorizador.fit_transform(X_train)
    Xv_test = vectorizador.transform(X_test)
    clasificador = LogisticRegression(**_PARAMS_CLASIFICADOR)
    clasificador.fit(Xv_train, y_train)

    print("=== Held-out test (20%) -- ojo, puede tener fuga leve por variantes casuales ===")
    y_pred = clasificador.predict(Xv_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=sorted(set(y))))

    print("\n=== Sanity-check con frases que NO están en el dataset ===")
    ok = 0
    for frase, esperado in _PRUEBAS_MANUALES:
        v = vectorizador.transform([frase])
        pred = clasificador.predict(v)[0]
        marca = "OK" if pred == esperado else "MAL"
        ok += pred == esperado
        print(f"  [{marca}] {frase!r:45s} esperado={esperado:14s} predicho={pred}")
    print(f"{ok}/{len(_PRUEBAS_MANUALES)} correctas")
    if ok < len(_PRUEBAS_MANUALES):
        print("!! Alguna prueba manual falló -- revisar dataset_router.jsonl antes de confiar en el modelo.")

    # Modelo final para producción: entrenado con TODO el dataset (no solo train).
    vectorizador_final = TfidfVectorizer(**_PARAMS_VECTORIZADOR)
    Xv_final = vectorizador_final.fit_transform(X)
    clasificador_final = LogisticRegression(**_PARAMS_CLASIFICADOR)
    clasificador_final.fit(Xv_final, y)

    joblib.dump({"vectorizador": vectorizador_final, "clasificador": clasificador_final}, SALIDA)
    print(f"\nGuardado {SALIDA} ({os.path.getsize(SALIDA) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
