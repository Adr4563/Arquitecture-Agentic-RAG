# TODO de mantenimiento

Pendientes detectados en una revisión (2026-08-26) que no se resolvieron
solos por ser cambios de comportamiento, no de documentación:

## `num_ctx` de `lora-trivia` bajado de 4096 a 512 (2026-08-31)

Búsqueda de un LLM más rápido para reemplazar a `lora-trivia` (~14
candidatos, incluyendo Granite, Qwen2.5-0.5B fine-tuneado, SmolLM2-135M
reentrenado con el dataset propio, y offload a ESP32-S3): ninguno superó a
`lora-trivia` en velocidad+calidad combinadas. De ahí quedaron dos palancas
de bajo riesgo sobre el propio `lora-trivia`, ver
`trivia_training/Modelfile`:

1. **`num_ctx` 4096 → 512** (este cambio): el prompt real de
   `comentar_resultado()`/`reaccionar_libre()` mide ~120-140 tokens
   (`prompt_eval_count` de `/api/chat`), 512 alcanza de sobra. Probado antes
   de aplicar con 4 casos estándar (acierto, error, respuesta vaga,
   musical) contra el modelo viejo (ctx 4096) y el nuevo (ctx 512) en la
   Raspberry, secuencial, uno a la vez: misma calidad de respuesta en los 4,
   sin regresión. La velocidad no cambió de forma medible (el cuello de
   botella real es el `prompt_eval` bajo contención de CPU, no el tamaño
   del contexto reservado), pero baja el uso de RAM del KV-cache, así que
   se dejó aplicado en producción (`ollama create lora-trivia -f
   Modelfile`).
2. **Reordenar el prompt** (pendiente): mover el texto fijo/boilerplate al
   principio y el contenido variable (pregunta, respuesta del usuario) al
   final, para que Ollama pueda reusar el KV-cache entre turnos (LCP —
   longest common prefix — visible en el log nativo como `checking sim =
   X (N/M) > 0.100`). Requiere retestear con los mismos 4 casos antes de
   aplicar: `lora-trivia` se entrenó con el orden actual, así que es
   sensible a cambios de forma del prompt (ver la nota de
   `comentar_resultado()` sobre el intento fallido de pedirle comparar
   `esperada` vs `respuesta_usuario` en vez de pasarle el veredicto ya
   resuelto).

## Gobernador de CPU fijado en `performance` (2026-08-30)

Medido con `perf_report.py`: el comentario de Trivia (`comentar_resultado()`,
`llama_generar:lora-trivia`) tardaba ~10.5s con el gobernador default
(`ondemand`), corriendo a 1.2GHz de un máximo de 1.8GHz. Cambiando a
`performance` (siempre al máximo, sin rampa de subida) bajó a ~8.4s
(~20%) en la misma prueba, misma pregunta, Pi por lo demás en reposo.

Se probó primero reentrenar/recuantizar el modelo esperando una mejora de
velocidad -- no sirvió para eso (ver más abajo esta misma sección de
decisiones de 2026-08-30 sobre `lora-trivia`): la CPU, no el tamaño del
modelo, era el cuello de botella real.

Aplicado con un servicio systemd (no viene con el repo, hay que
reaplicarlo si se reinstala la Pi):

```bash
sudo tee /etc/systemd/system/cpu-performance.service <<'EOF'
[Unit]
Description=Fijar el gobernador de CPU en 'performance' (menos latencia para Ollama)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now cpu-performance.service
```

Costo: más consumo/calor en reposo (la CPU ya no baja de frecuencia sola).
Temperatura medida tras el cambio: 63°C, lejos del umbral de throttling —
no es un problema hoy, pero vale la pena revisarla si la Pi queda en un
lugar con poca ventilación.

Verificar: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
(debería decir `performance` en las 4 líneas).

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

## `OLLAMA_KEEP_ALIVE`: probado en 2 min (revertido), ahora en `-1` (para siempre)

Con `CHAT_MODEL`=lora-chat-libre-v4 y `TRIVIA_MODEL`=lora-trivia
precargados al arrancar (`_precargar_modelos()`) + `SALIDA_TRIVIA_MODEL`
que se carga solo, bajo demanda, la primera vez que hace falta, Ollama
puede mantener los 3 cargados en RAM a la vez (~1.5GB juntos). No hay forma
de pasar `keep_alive` por request en el endpoint `/v1/chat/completions` que
usa `Clients/Llama_Client.py` (es compatible OpenAI, no la API nativa de
Ollama) -- solo se puede fijar a nivel de servicio systemd, con un drop-in
en `/etc/systemd/system/ollama.service.d/override.conf`.

**2026-08-26 — se probó bajarlo a 2 min, se revirtió al default (5 min):**
sí funcionaba (el modelo se descargaba a los ~2 min), pero el turno
siguiente a una pausa >2 min tardaba ~28s en vez de los ~10-12s normales
(recarga desde disco antes de generar la reacción) -- notorio en uso real.
Como la Pi no tenía problema de RAM, no valía la pena esa latencia por
liberar memoria que no hacía falta liberar -- se revirtió al default de
Ollama (sin override).

**2026-08-30 — cambiado a `-1` (nunca se descargan):** con el default de 5
min, `perf_report.py` mostró picos de ~20-30s en `llama_generar:*` a mitad
de sesión (media 7-8s) -- el modelo se había descargado por inactividad
(ej. una tanda larga de Trivia sin pasar por Chat libre, o viceversa) y el
siguiente turno de ESE modo pagó la recarga desde disco. Mismo diagnóstico
que el de 2026-08-26, pero en la dirección contraria: acá no hace falta
liberar memoria (la Pi sigue con >5GB libres con los 2-3 modelos cargados a
la vez), así que no tiene sentido pagar esa latencia nunca. Aplicado:

```
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Verificado con `ollama ps`: los 2 modelos precargados quedan con
`UNTIL: Forever`. Si en algún momento la RAM se pone justa (más modelos, u
otro proceso pesado corriendo a la vez), volver a un valor finito (`2m`,
`5m`, etc.) revierte esto.

Ojo: esto es configuración de sistema, no de código -- **no se sube a git
ni se replica sola** si se reinstala esta Pi desde cero.

## `preguntas.jsonl` tiene 13 preguntas con texto duplicado (`id` distinto)

Encontrado probando una tanda real de "Juego de multiplicar nivel Simple 1"
(2026-08-26): la ronda 4 y la ronda 5 salieron con la pregunta exacta
"¿Cuánto es cinco por ocho?" -- dos filas distintas del dataset (`p-24` y
`p-79`) con el mismo texto y la misma `respuesta_esperada`. `_iniciar_tanda`
filtra por `id` en `estado["ya_usados"]`, no por texto, así que no detecta
que son "la misma" pregunta.

13 de las 243 preguntas del dataset tienen esta duplicación, casi todas bajo
el tema combinado "Juego de multiplicar nivel Simple 1 / Juego de
multiplicar nivel Simple 2" (`p-24`/`p-79`, `p-25`/`p-80`, `p-26`/`p-81`,
`p-27`/`p-82`, `p-28`/`p-83`, `p-29`/`p-84`, `p-30`/`p-85`, `p-31`/`p-86`),
más "¿Cuál es tu nombre?" bajo "... / Juego de colores" (`p-22`/`p-94`).

No se aplicó ningún fix porque implica decidir qué hacer con el dataset de
producción (¿borrar la fila duplicada? ¿son a propósito, una por cada
combinación de tema doble?) -- eso lo tiene que decidir quien mantiene el
Excel/dataset, no es un cambio de código seguro de aplicar solo.

## Ya corregido: `hilo_voz.join()` sin definir en `manejar_trivia()`

Encontrado leyendo el código antes de probar la tanda real de "Juego de
emociones" a través de `manejar_trivia()` (2026-08-26): en la rama de temas
SIN veredicto (`Chistes`, `Reconocimiento Musical`, `Juego de colores`,
`Interaccion personalizada (COMIDA)`... cualquier tema sin
`respuesta_esperada` y fuera de `TEMAS_JUEGO_EMOCIONES`) quedó un
`hilo_voz.join()` colgado -- `hilo_voz` no está definido en ningún lado del
archivo, resto de una versión con threading que se probó y revirtió más
temprano el mismo día (ver el diff sin commitear). Habría tirado
`NameError: name 'hilo_voz' is not defined` y cortado el turno la primera
vez que un usuario real llegara a una pregunta de esos temas -- no se había
detectado porque las pruebas reales del día se hicieron con temas de
matemática (con `respuesta_esperada`, otra rama) o de emociones (rama
`TEMAS_JUEGO_EMOCIONES`, otra rama también), nunca con un tema de la rama
"sin veredicto".

Fix: se sacó esa línea (no queda ningún hilo que joinear en la versión
síncrona actual).

## Ya corregido: cámara nunca se liberaba entre rondas del Juego de emociones

Encontrado probando una tanda real de "Juego de emociones"/"Juego de
imitación" (2026-08-26, tras agregar que el robot muestre la cara a imitar
en pantalla ANTES de pedirla por voz): la ronda 1 conseguía veredicto de
cámara bien, pero TODAS las rondas siguientes de ese mismo proceso fallaban
con `Camera in Configured state trying acquire() requiring state
Available` -- `Camara_Client.detectar_emocion()` devolvía `(None, None)`
como si no hubiera cámara conectada.

Causa: `ai-camera/reconocer_emocion.py::_capturar_y_detectar()` crea una
`Picamera2()` nueva en cada llamada y hace `picam2.stop()` al final, pero
nunca `picam2.close()` -- sin eso la cámara queda en estado "Configured", no
"Available", y la siguiente `Picamera2()` de ese mismo proceso no puede
`acquire()`la. Como `Orchestrator_Management.py` corre como un solo proceso
de larga vida, esto significaba que **solo la primera ronda del Juego de
emociones de toda la vida del robot conseguía veredicto de cámara real** --
todas las siguientes, hasta reiniciar el proceso, quedaban sin evaluar
(silencioso: se loguea y sigue sin cortar el turno, así que no se notaba
sin mirar los logs).

Fix: se agregó `picam2.close()` después de `picam2.stop()`. Verificado con
3 detecciones consecutivas en el mismo proceso (antes fallaba desde la
segunda) y con una tanda real de 5 rondas del Juego de emociones (antes
fallaban las rondas 2-5, ahora las 5 consiguieron veredicto de cámara).

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
