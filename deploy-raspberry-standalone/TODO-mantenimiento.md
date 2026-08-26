# TODO de mantenimiento

Pendientes detectados en una revisión (2026-08-26) que no se resolvieron
solos por ser cambios de comportamiento, no de documentación:

## `router_modelo.joblib` serializado con una versión vieja de scikit-learn

`Agents/router_modelo.joblib` fue guardado con scikit-learn 1.8.0. El
entorno de esta Pi ya tiene 1.9.0 (fijado en `requirements.txt`), así que al
cargarlo aparece:

```
InconsistentVersionWarning: Trying to unpickle estimator TfidfTransformer
from version 1.8.0 when using version 1.9.0.
```

Funciona bien hoy, pero es frágil ante un futuro upgrade de scikit-learn.
Para limpiarlo del todo (no solo fijar la versión en `requirements.txt`,
que ya está hecho):

```bash
cd router_training
python entrenar_router.py
```

Esto reentrena y sobreescribe `Agents/router_modelo.joblib` con la versión
de scikit-learn actualmente instalada. Es determinístico a partir de
`dataset_router.jsonl` (que sí está en git), así que es seguro repetirlo —
pero se dejó pendiente en vez de aplicarlo directo por tratarse del modelo
que corre en producción en el robot.

## Ya corregido en esta misma revisión

- `llama3.2:3b-q4s` (tag inexistente en Ollama) → `llama3.2:3b` en
  `README.md` y `Clients/Llama_Client.py`. El tag real y ya descargado en
  esta Pi es `llama3.2:3b` (`ollama list`).
- `musica/README.md` actualizado: `Agent_Behavior.py` (`expresar_musica()`)
  sí reproduce audio real (ya no es solo un log), y la tabla de valores
  ahora lista los archivos reales del dataset (`danza-kuduro.mp3`,
  `more-than-words-heaven.mp3`) en vez del ejemplo obsoleto `musica mario`.
- Comentario en `preguntas.py::pregunta_aleatoria()` corregido: el dataset
  hoy tiene 248 preguntas, 187 con `respuesta_esperada` (~75%, no "casi la
  mitad / 111 de 246").
