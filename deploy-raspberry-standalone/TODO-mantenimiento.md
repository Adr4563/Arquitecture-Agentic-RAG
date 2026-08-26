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

## `OLLAMA_KEEP_ALIVE`: se probó bajarlo a 2 min, se revirtió al default (5 min)

Con 3 modelos separados por rol (`CHAT_MODEL`=ereberus-chat,
`TRIVIA_MODEL`=ereberus-trivia, `VERIFICADOR_MODEL`=qwen2.5:0.5b), Ollama
puede mantener los 3 cargados en RAM a la vez (~1.6GB juntos) con su
`keep_alive` default de 5 minutos. Se probó bajarlo a 2 minutos a nivel de
servicio systemd (no hay forma de pasar `keep_alive` por request en el
endpoint `/v1/chat/completions` que usa `Clients/Llama_Client.py`, es
compatible OpenAI, no la API nativa de Ollama) con un drop-in en
`/etc/systemd/system/ollama.service.d/override.conf`.

Validado en vivo: SÍ funcionaba (el modelo se descargaba a los ~2 min), pero
el turno siguiente a una pausa >2 min tardaba ~28s en vez de los ~10-12s
normales (recarga desde disco antes de generar la reacción) -- notorio en
uso real. Como esta Pi no tiene problema de RAM hoy (5.9GB libres de sobra
incluso con los 3 modelos cargados), no valía la pena pagar esa latencia
por liberar memoria que no hacía falta liberar -- **se revirtió** al
default de Ollama (sin override, sin `OLLAMA_KEEP_ALIVE` seteada).

Si en algún momento SÍ hace falta (memoria más ajustada, u otro proceso
pesado corriendo a la vez), el override queda documentado acá para
reaplicarlo:

```
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_KEEP_ALIVE=2m"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

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
