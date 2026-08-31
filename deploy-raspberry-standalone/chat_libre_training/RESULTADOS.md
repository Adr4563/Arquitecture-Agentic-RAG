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

---

## v5 — 2026-08-31 — cambio de modelo base

**Base**: `llama3.2:1b` (1.2B) en vez de `qwen2.5:0.5b` (494M).
Mismos 105 ejemplos, mismo entrenamiento. `eval_loss` 2.058 → 1.673 → 1.633
(no comparable con las versiones anteriores: otro tokenizador).

### Memoria semántica

30 preguntas de conocimiento general (naturaleza, geografía, ciencia,
historia, básico, matemática), medidas en la PC:

| Modelo | Aciertos | |
|---|---|---|
| `lora-chat-libre-v4` (0.5B) | **57%** | el desplegado |
| `qwen2.5:0.5b` | 53% | base del v4 |
| `llama3.2:1b` | **83%** | base del v5 |
| `llama3.2:3b` | 87% | referencia |
| **`lora-chat-libre-v5`** | **77%** | |

Dos conclusiones:

1. **El fine-tuning no agrega conocimiento.** Sobre 0.5B sumó 4 puntos
   (53 → 57). Lo que mueve el número es el tamaño del base.
2. **El fine-tuning lo degrada un poco.** 83 → 77 al entrenar sobre el 1b.
   Es la misma degradación que llevó a `lora-chat` de "la araña tiene 8
   patas" a "una sola pata", acá mucho más leve.

⚠️ **Cuidado al reproducir esto**: la primera versión del test comparaba sin
normalizar acentos y marcaba como incorrecta *"La capital de Francia es
París"*. Con ese bug `llama3.2:3b` daba 70% en vez de 87%. Normalizar antes
de comparar.

### Largo y velocidad

| Modelo | Palabras (mediana) | Total |
|---|---|---|
| `llama3.2:1b` (base) | **72** | 0.5 s |
| `lora-chat-libre-v5` | **4** | 0.2 s |
| `lora-chat-libre-v4` | 6 | 0.2 s |

Ese es el aporte real del fine-tuning acá: el 1b **ya sabía**, pero escribía
párrafos. En la Pi eso eran 11-19 s por respuesta. Con 4 palabras de
mediana, el conocimiento del 1b entra en el presupuesto de tiempo.

### Conversación

**Arreglado** — el peor error, que arrastraban todas las versiones:

```
"mi amigo se rio de mi"   v4 -> "¡Qué bueno!"          (festeja una burla)
                          v5 -> "¿Y cómo reaccionaste?"
```

**Sin resolver**, dos defectos nuevos por sobreajuste con 105 ejemplos:

- Dice **"¡Qué lora!"**, usando su propio nombre como exclamación
- Repite **"¿Y cómo te...?"** para casi cualquier mensaje

Se arreglan con más ejemplos curados (`curar.py`), no con otro modelo.

### Estado

**No desplegada.** `CHAT_MODEL` sigue en v4 hasta poder probar la v5 en la
Raspberry Pi. Pesa 770 MB contra 379 MB del v4.
