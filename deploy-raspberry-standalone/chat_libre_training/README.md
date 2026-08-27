# Pipeline de mejora continua de Chat libre

Ciclo para que las conversaciones del robot vayan mejorando: se registran,
las revisás vos, se reentrena, se evalúa y se versiona. Cada vuelta produce
una versión nueva que queda guardada, con sus métricas, y se puede revertir.

## ⚠️ Lo que este pipeline puede y lo que no

**Sí puede** mejorar el **estilo y el comportamiento**: acompañar en vez de
desentenderse, mantener el hilo, admitir que no sabe en vez de inventar.

**No puede** darle conocimiento del mundo. `qwen2.5:0.5b` tiene 494 M de
parámetros y, medido en este proyecto, su base ya cree que **el cielo es
blanco** y que **la Mona Lisa la pintó Frida Kahlo**. El conocimiento viene
del preentrenamiento — miles de millones de tokens —, no del fine-tuning.
Ningún dataset de 200 ejemplos cambia eso.

Peor: **el fine-tuning puede empeorar el conocimiento**. Medido acá,
comparando `qwen2.5:0.5b` con `lora-chat`, que es su propio fine-tune:

| Pregunta | base | su fine-tune |
|---|---|---|
| ¿Cuántas patas tiene una araña? | 8 ✅ | **una sola** ❌ |

Por eso existe el paso 6 (evaluar). No es opcional.

## ⚠️ Por qué hay un paso manual

**Nunca se entrena con las respuestas crudas del modelo.** Hacerlo le enseña
a imitar sus propios errores: el modelo genera, se entrena con lo que
generó, y refuerza sus fallas en cada vuelta (colapso). Por eso `curar.py`
existe y no se puede saltear.

---

## El ciclo

```
  [1] conversar         registro_chat.py guarda cada turno automáticamente
        ↓
  [2] curar.py          marcás cada turno: buena / corregida / descartada
        ↓
  [3] generar_dataset.py  arma el dataset solo con lo aprobado
        ↓
  [4] entrenar_chat_libre.py   LoRA → versión nueva (v1, v2, v3...)
        ↓
  [5] GGUF → cuantizar → ollama create
        ↓
  [6] probar_conversacion.py   ← SI EMPEORÓ, NO SE DESPLIEGA
        ↓
  [7] export CHAT_MODEL=lora-chat-libre-vN
        ↓
      (volver a 1)
```

## 1. Conversar

No hay que hacer nada: `registro_chat.py` guarda cada turno de Chat libre en
`conversaciones.jsonl` (mensaje, respuesta, qué modelo respondió, fecha).

Para correr el robot sin registrar (una demo, una prueba):

```bash
export CHAT_LIBRE_REGISTRAR=0
```

## 2. Curar — el paso que no se saltea

```bash
python curar.py
python curar.py --resumen        # solo estadísticas
```

Por cada turno:

| Tecla | Qué hace |
|---|---|
| `b` | **buena** — la respuesta sirve tal cual |
| `c` | **corregida** — escribís vos la respuesta que debería haber dado |
| `d` | **descartada** — no sirve ni corregida |
| `s` | saltar, decidir después |
| `q` | guardar y salir |

Solo `buena` y `corregida` llegan al dataset.

**Las corregidas son las más valiosas.** Una respuesta "buena" solo confirma
lo que el modelo ya hacía bien; una corregida le enseña algo nuevo.

## 3. Generar el dataset

```bash
python generar_dataset.py            # solo lo curado
python generar_dataset.py --semilla  # + 20 ejemplos base (primera vez)
```

La semilla cubre lo que medimos que el modelo hace mal: se desentiende de
temas emocionales, no mantiene el hilo, e inventa en vez de admitir que no
sabe. Para la v1 alcanza; después conviene que manden tus conversaciones
reales.

**Con menos de ~100 ejemplos no reentrenes**: sobreajusta y empeora.

## 4. Entrenar

```bash
pip install torch transformers peft datasets
python entrenar_chat_libre.py
```

Corre en la PC (CPU alcanza), **no en la Pi**. Numera la versión solo, sin
pisar las anteriores, y la anota en `versiones.json` con fecha, cantidad de
ejemplos y `eval_loss`.

Usa `load_best_model_at_end`: en las dos corridas de
`salida_trivia_training` la última época quedó peor que la anterior, y sin
esto se exporta la última.

## 5. Convertir e importar

```bash
python <ruta>/llama.cpp/convert_hf_to_gguf.py D:/lora-chat-libre-v1-merged     --outfile lora-chat-libre-v1-f16.gguf --outtype f16
llama-quantize lora-chat-libre-v1-f16.gguf lora-chat-libre-v1-q4_k_m.gguf Q4_K_M
ollama create lora-chat-libre-v1 -f Modelfile   # ajustar el FROM
```

El `.gguf` f16 es intermedio: se puede borrar después de cuantizar.

## 6. Evaluar — antes de desplegar

```bash
python probar_conversacion.py --modelo lora-chat-libre-v1 --anotar-version 1
python probar_conversacion.py --modelo qwen2.5:0.5b     # el base, para comparar
```

**Si la versión nueva no supera a la anterior Y al base, no se despliega.**

Esto existe porque en `salida_trivia_training` la suite "normal" daba 15/15
con un modelo que en producción expulsaba de la trivia a cualquiera que
dijera "no sé". Un test escrito junto con el dataset solo confirma lo que el
modelo ya aprendió. **Las frases de `probar_conversacion.py` no tienen que
estar en el dataset**: si se copian, el test deja de medir generalización.

Qué chequea automáticamente (solo lo objetivamente verificable):

| Chequeo | Qué detecta |
|---|---|
| `RESPONDE` | respuesta vacía (le pasa a `qwen3:0.6b`) |
| `NO_REPITE` | bucles (`"me gusta el futbol, me gusta el futbol..."`) |
| `LARGO` | más de 45 palabras — el robot habla en voz alta y bloquea |
| `NO_ASISTENTE` | "como IA", "no puedo ayudarte con eso" |
| `NO_INVENTA` | inventa la hora, el clima o una cuenta que no sabe |

Los casos personales se imprimen pero **no se puntúan**: si una respuesta
acompaña bien no lo decide un heurístico. Esa parte la mirás vos.

Referencias medidas en la Pi: `qwen2.5:0.5b` **8/12**, `llama3.2:1b`
**9/12**. Una versión nueva tiene que superar a su base.

## 7. Desplegar

```bash
export CHAT_MODEL=lora-chat-libre-v1
./start-all.sh
```

Para volver atrás, apuntá a la versión anterior o a `llama3.2:1b`. Por eso
cada versión se guarda con nombre propio.

## Archivos

| Archivo | Qué es |
|---|---|
| `../registro_chat.py` | Guarda cada turno (paso 1). Vive afuera porque lo importa el orquestador. |
| `conversaciones.jsonl` | El registro crudo. **No se commitea**: son conversaciones reales. |
| `curar.py` | Revisión manual (paso 2). |
| `generar_dataset.py` | Arma el dataset con lo aprobado (paso 3). |
| `dataset_chat_libre.jsonl` | Lo que se entrena. Regenerable. |
| `entrenar_chat_libre.py` | Fine-tuning LoRA versionado (paso 4). |
| `Modelfile` | Para el `ollama create` (paso 5). |
| `probar_conversacion.py` | Evaluación (paso 6). |
| `versiones.json` | Historial: fecha, ejemplos, `eval_loss`, resultado de la evaluación. |

## Expectativas realistas

Una vuelta del ciclo con ~200 ejemplos curados mejora el tono y reduce las
invenciones. **No** va a convertir un modelo de 494 M en uno que converse de
cualquier tema: para eso está `llama3.2:1b` (1.2 B), que es lo que el robot
usa hoy por defecto.

El caso de uso real de este pipeline es tener un modelo **chico y rápido**
(0.5B, ~2-6s por respuesta contra los ~11s de `llama3.2:1b`) que acompañe
bien en los temas que de verdad aparecen con chicos, aceptando que no sabe
cultura general.
