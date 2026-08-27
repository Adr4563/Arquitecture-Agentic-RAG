# Resultados: `lora-salida-trivia` v1 y v2

Historial medido de las dos versiones entrenadas del clasificador de salida
de Trivia (RESPUESTA vs SALIR). Ver `README.md` para el pipeline de
entrenamiento y `entrenar_salida.py` para la configuracion.

## Como se mide

Dos suites, y la diferencia entre ellas es el punto de todo este documento:

| Suite | Casos | Que mide |
|---|---|---|
| `probar_salida.py` | 15 | Las mismas categorias con las que se genero el dataset. **Solo confirma**, no descubre. |
| `probar_adversario.py` | 26 | Los bordes: 7 grupos, cada uno una hipotesis de como podria fallar. Frases deliberadamente ausentes del dataset. |

La leccion del v1 es que la primera suite sola no sirve: dio 15/15 con un
modelo que en produccion expulsaba de la trivia a cualquier chico que
dijera "no se".

---

## Version 1

**Dataset**: 542 ejemplos (222 RESPUESTA / 320 SALIR), generados desde
`base_datos/preguntas.jsonl` (respuestas literales) y los ejemplos
CHAT_LIBRE de `router_training/dataset_router.jsonl` (off-topic).

**Entrenamiento**: 183 pasos, 27m52s en CPU. `eval_loss` por epoca:

| Epoca | eval_loss |
|---|---|
| 1 | 0.0787 |
| 2 | **0.001585** |
| 3 | 0.04199 |

Se exporto la **epoca 3**, no la mejor: sin `load_best_model_at_end`,
`merge_and_unload()` fusiona el estado final en memoria, no el mejor
checkpoint.

### Resultados v1

| Suite | Resultado |
|---|---|
| `probar_salida.py` | 15/15 (100%) |
| `probar_adversario.py` | **16/26 (61.5%)** |

Por grupo:

| Grupo | v1 |
|---|---|
| A. Respuesta con vocabulario personal | 1/4 |
| B. Rendirse (`no se`, `ni idea`) | 0/3 |
| C. Meta-trivia (pista, repetir) | 0/4 |
| D. Respuesta incorrecta | 3/3 |
| E. Tema personal sin keyword | 4/4 |
| F. Off-topic encubierto | 4/4 |
| G. Ruido de tipeo | 4/4 |

### Diagnostico

Los 10 fallos fueron **todos en la misma direccion: falsos SALIR**. Cero
falsos RESPUESTA. Y comparados contra las listas de palabras clave que el
modelo venia a reemplazar, **7 de los 10 eran regresiones**:

| Caso | Keywords | v1 | |
|---|---|---|---|
| `no se` / `ni idea` / `no me acuerdo` | RESPUESTA | SALIR | regresion |
| `me das una pista?` / `puedes repetir la pregunta?` | RESPUESTA | SALIR | regresion |
| `cuantas preguntas faltan?` / `esta pregunta esta muy facil` | RESPUESTA | SALIR | regresion |
| `mi mama me enseno que da cuarenta` | SALIR | SALIR | ya fallaba |

Causa: con solo dos fuentes en el dataset (respuestas literales vs
off-topic), la regla mas simple que explica los 542 ejemplos es *"si no es
literalmente la respuesta -> SALIR"*. El modelo aprendio esa regla y la
aplico a todo lo que no habia visto. El `eval_loss` de 5.9e-05 no
contradice nada de esto: significa que aprendio perfectamente los datos que
se le dieron, blind spots incluidos.

Comprobacion adicional: el `qwen2.5:0.5b` **sin entrenar** saca 18/26 en la
misma suite -- mejor que el fine-tune. El fine-tuning no agrego
razonamiento, reemplazo el razonamiento general del modelo base por la
regla del dataset.

---

## Version 2

**Que cambio en el dataset**: 542 -> 710 ejemplos, y el balance paso de
222/320 (sesgado a SALIR) a **390 RESPUESTA / 320 SALIR**. Se agregaron a
`generar_dataset.py` las tres categorias que faltaban, todas RESPUESTA:

1. **Rendirse** (16 frases x 3 preguntas) -- `no lo se`, `paso`, `me rindo`
2. **Meta-trivia** (16 x 3) -- `me puedes dar una ayudita?`, `cuantas van?`
3. **Respuesta con vocabulario personal** (45 + dilemas) -- `mi mama me
   enseno que es {r}`, y opiniones largas que hablan de la familia *y*
   responden el dilema a la vez

Las frases usadas son distintas de las de `probar_adversario.py`, a
proposito.

**Que cambio en el entrenamiento**: `load_best_model_at_end=True` +
`metric_for_best_model="eval_loss"`, para que se exporte el mejor
checkpoint y no el ultimo.

**Entrenamiento**: 240 pasos, ~35 min en CPU. `eval_loss` por epoca:

| Epoca | v1 | v2 |
|---|---|---|
| 1 | 0.0787 | 0.03212 |
| 2 | **0.001585** | **7.149e-05** |
| 3 | 0.04199 | 8.963e-05 |
| Exportada | epoca 3 | **epoca 2** |

### Resultados v2

| Suite | v1 | v2 |
|---|---|---|
| `probar_salida.py` | 15/15 | 15/15 |
| `probar_adversario.py` | 16/26 (61.5%) | **25/26 (96.2%)** |

---

## Comparativa de los tres modelos

`probar_adversario.py`, 26 casos:

| Grupo | v1 | v2 | `qwen2.5:0.5b` (base) |
|---|---|---|---|
| A. Respuesta con vocabulario personal | 1/4 | **4/4** | 4/4 |
| B. Rendirse | 0/3 | **3/3** | 2/3 |
| C. Meta-trivia | 0/4 | **3/4** | 4/4 |
| D. Respuesta incorrecta | 3/3 | 3/3 | 3/3 |
| E. Tema personal sin keyword | 4/4 | **4/4** | 0/4 |
| F. Off-topic encubierto | 4/4 | **4/4** | 1/4 |
| G. Ruido de tipeo | 4/4 | **4/4** | 4/4 |
| **Total** | **16/26 (61.5%)** | **25/26 (96.2%)** | **18/26 (69.2%)** |

Lo importante no es solo el total. Los tres modelos tienen perfiles de
error distintos:

- **v1** se pasa de SALIR: expulsa de la trivia a quien no sabe la
  respuesta o pide una pista.
- **`qwen2.5:0.5b`** se pasa de RESPUESTA: casi nunca deja salir a nadie
  (0/4 en E, 1/4 en F). Gana en total pero **desactiva de hecho la
  funcionalidad** -- el caso que motivo el pipeline lo falla entero.
- **v2** es el unico que cubre las dos direcciones a la vez.

### Unico fallo del v2

`cuantas preguntas faltan?` -> SALIR (deberia ser RESPUESTA). Es
meta-trivia: el usuario pregunta por la mecanica del juego, no se esta
yendo. Impacto bajo (no es una frase frecuente) y el fallback de keywords
no lo cubriria mejor.

## Latencia medida

| | PC (x86, CPU) | Raspberry Pi (aarch64) |
|---|---|---|
| Por clasificacion | 0.14 s | 1.4 - 3.0 s |
| Primera llamada (modelo frio) | 3.3 s | 18.9 s |

En la Pi se suman 1.5-3 s a cada turno de trivia, antes de evaluar la
respuesta. Es el costo real de la funcionalidad y hay que tenerlo en cuenta
al decidir si vale la pena.

## Como reproducir

```bash
python probar_salida.py      --modelo lora-salida-trivia-v2
python probar_adversario.py  --modelo lora-salida-trivia-v2
python probar_adversario.py  --modelo qwen2.5:0.5b            # comparar contra el base
python probar_adversario.py  --host http://<ip-de-la-pi>:11434
```

## Que aprender de esto

1. Un test escrito junto con el dataset solo confirma lo que el modelo ya
   aprendio. Hace falta una suite con frases que el entrenamiento no vio.
2. `eval_loss` cercano a cero no dice "modelo bueno", dice "aprendio los
   datos que le di". El v1 tenia loss 5.9e-05 y fallaba 10 de 26 casos.
3. Fine-tunear un modelo chico en una tarea angosta puede **borrar** el
   razonamiento general del modelo base. Conviene medir siempre contra el
   base sin entrenar.
4. Los huecos del dataset se vuelven reglas: si una categoria no aparece,
   el modelo la absorbe en la etiqueta mayoritaria.
