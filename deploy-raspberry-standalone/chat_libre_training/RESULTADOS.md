# Resultados de `lora-chat-libre`

Historial medido de cada versión. Ver `README.md` para el pipeline.

## Cómo se mide

`probar_conversacion.py`, 12 casos con frases **ausentes del dataset** a
propósito. Solo puntúa lo objetivamente verificable (respuesta vacía,
bucles, largo, lenguaje de asistente, inventar la hora / el clima / una
cuenta). Si acompaña bien o no, lo juzga una persona leyendo las respuestas.

**El entrenamiento corre en la PC; la evaluación tiene que correr en la
Raspberry Pi**, que es donde el modelo va a vivir. Medir latencia en la PC da
números que no aplican (0.2s en PC vs ~2s en la Pi para el mismo modelo).

---

## v1 — 2026-08-27

**Dataset**: 90 ejemplos, todos de la semilla de `generar_dataset.py` (aún
sin conversaciones reales curadas). Cubren lo que medimos que falla:
acompañar, celebrar lo bueno, mantener el hilo, admitir que no sabe, charla
corta, y conocimiento básico que el base erraba.

**Entrenamiento**: 3 épocas. `eval_loss` 1.938 → 1.782 → **1.724**. Bajó en
las tres, así que acá la última SÍ fue la mejor (a diferencia de las dos
corridas de `salida_trivia_training`, donde la última empeoró).

Ese `eval_loss` no es comparable con el de `lora-salida-trivia` (0.03):
aquel clasificaba una palabra, este genera frases libres, donde hay muchas
formas válidas de decir lo mismo.

### Resultados en la Raspberry Pi

| Modelo | Suite | Latencia |
|---|---|---|
| **`lora-chat-libre-v1`** | **11/12 (92%)** | **~2 s** |
| `llama3.2:1b` (en uso hoy) | 9/12 (75%) | ~11 s |
| `qwen2.5:0.5b` (su base) | 8/12 (67%) | ~3 s |
| `lora-chat` (el viejo, RAG) | 0/12 (0%) | ~6 s |

Le gana a su propio base y a `llama3.2:1b`, siendo **5× más rápido** que
este último. El 0/12 de `lora-chat` confirma que quedó inservible al
sacarle el RAG con el que se entrenó.

### Por qué NO se desplegó

El puntaje dice 92%; las respuestas dicen otra cosa:

```
"saqué la nota más alta del curso" -> "No puedo saber cuál es la nota."
"me encanta armar rompecabezas"    -> "¡Qué bueno! ¿Qué te pasa con eso?"
"mi gato se escapó de casa"        -> "¿De qué parte? ¿De qué lado?"
"918 por 447"                      -> "El resultado es 2.071."   (FALLA)
```

Festeja mal un logro, repregunta de forma rara, y **sigue inventando en la
multiplicación pese a que ese caso exacto estaba en el dataset**.

Es la limitación declarada de la suite: mide lo verificable, no si acompaña
bien. Por eso el paso 6 del README dice que los casos personales se leen a
ojo — y leyéndolos, esta versión no está para producción.

**Causa**: 90 ejemplos escritos a mano, sin una sola conversación real. La
v1 sirvió para validar el pipeline de punta a punta (las 7 etapas, el
versionado, el despliegue a la Pi), que era su objetivo.

### Qué haría falta para una v2

Conversaciones reales pasadas por `curar.py`. Las **corregidas** son las que
más aportan: le enseñan algo que el modelo no hacía. Apuntar a 200+
ejemplos, con foco en los tres defectos de arriba.

---

## Referencias

Cuando entrenes una versión nueva, comparala contra estas mismas cifras **en
la Pi**. Si no supera a `qwen2.5:0.5b` (8/12), el fine-tuning empeoró al
modelo en vez de mejorarlo — que pasa, y es justo lo que este paso existe
para detectar.
