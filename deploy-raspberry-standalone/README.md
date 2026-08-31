# Despliegue: todo en la Raspberry Pi

Todo el proyecto vive en esta carpeta: código, dataset y modelos corren en
la propia Raspberry Pi, en un solo proceso Python (`Orchestrator_Management.py`,
que arma todo junto con `preguntas.py`, `Agents/`, `Clients/`, `display.py` y
`voz_server.py`) más Ollama como único servicio externo — sin depender de
ninguna otra máquina.

Probado pensando en una **Raspberry Pi 4 (8GB RAM, 128GB de almacenamiento)**.

## ⚠️ Rendimiento esperado

Una Pi 4 no tiene GPU para inferencia — todo corre en CPU (ARM Cortex-A72).
Ollama mantiene cargados **3 modelos**, todos fine-tunes LoRA de
`qwen2.5:0.5b` (397 MB cada uno) y todos importados a mano con
`ollama create`, no bajados con `ollama pull`:

| Variable | Default | Rol | Entrenamiento |
|---|---|---|---|
| `CHAT_MODEL` | `lora-chat` | Genera las respuestas que ve el usuario (chat libre, búsqueda web) | `chat_training/` |
| `TRIVIA_MODEL` | `lora-trivia` | Reacciones de trivia | `trivia_training/` |
| `SALIDA_TRIVIA_MODEL` | `lora-salida-trivia` | Decide si el usuario se fue de la trivia (RESPUESTA vs SALIR) | `salida_trivia_training/` |

Los tres pesan lo mismo porque salen del mismo modelo base. El tercero es
opcional: si no está importado, la salida de trivia se decide con listas de
palabras clave (menos preciso, pero no rompe nada) — ver
`_quiere_salir_trivia()` en `Orchestrator_Management.py`.

El router YA NO usa Ollama para nada — ver "Router sin LLM" más abajo.

Se eligió `qwen2.5:0.5b` sobre `llama3.2:3b` (el default anterior) tras
benchmarquear 7 modelos livianos (Meta/Facebook y HuggingFace, ver
`Clients/Llama_Client.py` para el detalle y los números) con los prompts
reales de este proyecto:

| Modelo | RAM residente (`ollama ps`) | Respuesta en caliente | Calidad en las 5 corridas |
|---|---|---|---|
| **qwen2.5:0.5b (default)** | **484 MB** | **1–3s** | 5/5 correctas, sin fugas ni frases prohibidas |
| llama3.2:3b (default anterior) | 2.5 GB | 7–12s | 5/5 correctas, prosa más elaborada |
| llama3.2:1b (Meta) | 1.5 GB | 4–7s | Falló: se negó a confirmar una respuesta correcta en 2/3 corridas, filtró `<rag>` crudo en 3/3 |
| smollm2:1.7b (HuggingFace) | 2.7 GB | 10–14s | Más pesado que el default anterior — sin sentido como "liviano" |
| tinyllama, qwen3:0.6b, smollm2:360m | 0.7–1 GB | — | No siguen el system prompt (lo repiten, o texto vacío) |

Con `qwen2.5:0.5b` la Pi debería sentirse fluida. Si en algún momento se
quiere prosa más rica y el hardware lo banca, se puede volver al modelo
grande sin tocar código (ver "Subir a un modelo más grande" más abajo).

`preguntas.py` no usa ningún modelo — busca sobre `preguntas.jsonl` con BM25
(keyword) en memoria dentro del mismo proceso de `Orchestrator_Management.py`,
así que no consume Ollama, RAM de embeddings, ni un puerto/proceso aparte.

### Router sin LLM

Clasificar cada turno en TRIVIA / CHAT_LIBRE es una
clasificación de 3 etiquetas fijas, no generación de texto — no hace falta
un modelo de lenguaje para eso. `Agents/Agent_Router.py` reemplaza al viejo
router basado en LLM (few-shot sobre Ollama) por un clasificador clásico
(TF-IDF de n-gramas de caracteres + regresión logística, scikit-learn),
entrenado en `router_training/` sobre ~700 frases (generadas con LLM +
curadas a mano + augmentadas):

| | Router con LLM (antes) | Router sin LLM (ahora) |
|---|---|---|
| Peso del modelo | 397 MB (qwen2.5:0.5b) | **~185 KB** |
| Tiempo por turno | 1–3s (llamada a Ollama) | **microsegundos** (cuenta local) |
| RAM extra | Compartida con `CHAT_MODEL` | Ninguna — ni siquiera toca Ollama |
| Precisión | No medida formalmente | 95% held-out, 7/7 en frases nunca vistas |

Si algún día `TEMAS_CATALOGO` gana un tema nuevo y el router lo confunde con
otra ruta, se reentrena editando `router_training/dataset_router.jsonl` y
corriendo `python router_training/entrenar_router.py` — no hace falta tocar
`Agent_Router.py`.

### Corrector de Trivia sin LLM

Corregir una respuesta de Trivia contra `respuesta_esperada` tampoco
necesita un LLM: el 74% de las `respuesta_esperada` del dataset son UNA sola
palabra, y otro ~27% son puramente numéricas (los juegos de multiplicar).
`Agents/Agent_Corrector.py` compara directo: si `respuesta_esperada` es un
número, extrae el/los número(s) de la respuesta del usuario y los compara
matemáticamente (nada de ambigüedad de texto); si es palabra/frase, normaliza
(sin tildes/mayúsculas/artículos) y compara por substring + solapamiento de
palabras clave + fuzzy para typos.

| | Corrector con LLM (antes) | Corrector sin LLM (ahora) |
|---|---|---|
| Tiempo por corrección | 1–3s (llamada a Ollama) | microsegundos |
| Accuracy (187 preguntas reales del dataset, variantes correctas parafraseadas/con typo + incorrectas) | No medida formalmente | **98.4%** |
| Recall en incorrectas (nunca acredita una mal) | No medida | **100%** |
| Bug real encontrado | Llegó a leer "7 por 8" como "7/8" y decir que la respuesta era 0.875 | No aplica — comparación matemática directa, sin ese riesgo |

`Agent_Verificator.py` (la revisión de coherencia de Chat libre) se sacó
del proyecto por completo: Chat libre ahora es una sola llamada a
`CHAT_MODEL`, sin una segunda pasada de corrección detrás -- Llama_Client
responde directo. De paso esto habilitó streaming real en Chat libre (antes
estaba apagado porque el verificador necesitaba el texto completo antes de
mostrarlo).

### Personalidad horneada en el modelo (opcional)

La generación de texto en sí (`responder()`, `comentar_resultado_emocion()`,
etc. -- `comentar_resultado()` de Trivia ya no genera nada, son frases
fijas desde 2026-08-31, ver TODO-mantenimiento.md) manda el system prompt
completo de la personalidad (~200-350 tokens) en CADA llamada — a diferencia del router y el corrector, acá sí hace falta un
modelo generativo, no hay forma de sacarlo por completo. Pero sí se puede
"hornear" la personalidad y las reglas de estilo DENTRO de los pesos con un
fine-tuning LoRA, para no tener que repetirlas en texto cada vez.

`lora-personalidad` es un fine-tuning LoRA de `qwen2.5:0.5b` entrenado
sobre 251 ejemplos (generados por destilación: `llama3.2:3b` con el
prompt completo genera la respuesta objetivo, y se entrena el modelo chico
a reproducirla con un system prompt mucho más corto). Pipeline completo,
reproducible, en `personalidad_training/`.

| | qwen2.5:0.5b + prompt completo (default) | lora-personalidad (opcional) |
|---|---|---|
| Tokens de prompt por turno | ~324–362 | **~99–137** (~65% menos) |
| Velocidad | 1–6s | **0.9–1.7s** (igual o más rápido) |
| Calidad (5 casos comparados) | Correcta, pero violó su propia regla "no cierres preguntando" en 1/5 casos | Correcta, no violó ninguna regla |

No es el default todavía a propósito: se entrenó una sola vez sobre un
dataset chico (251 ejemplos) y solo se probó contra 5 casos — el router y
el corrector pasaron por bastante más validación (cientos de casos reales
del dataset) antes de convertirse en default. Para probarlo:

```bash
export CHAT_MODEL=lora-personalidad
./start-all.sh
```

`Clients/Llama_Client.py` y `personalidad.py` detectan el nombre solos:
con `lora-personalidad` activo, `obtener_system_prompt()` devuelve `""`
y el system prompt largo deja de mandarse (ver
`Orchestrator_Management._mensajes_con_personalidad`) — no hace falta tocar
código para usarlo.

⚠️ El modelo (`.gguf`, unos 370MB) no vive en este repo (muy pesado para
git) — hay que entrenarlo/convertirlo en una PC de desarrollo siguiendo
`personalidad_training/entrenar_personalidad.py` y copiar el resultado a la
Pi, o correr `ollama create` directo ahí con el `.gguf` ya transferido.

## 1. Instalar Ollama en la Pi

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

El instalador deja Ollama corriendo como servicio de systemd (arranca solo
al bootear la Pi — no hace falta iniciarlo a mano). Verificalo con:

```bash
systemctl status ollama
```

## 2. Descargar el modelo

```bash
ollama pull qwen2.5:0.5b   # router de intención Y generación de respuestas por default
```

`llama3.2:3b` NO hace falta descargarlo salvo que se quiera pasar a él
después (ver "Subir a un modelo más grande" más abajo) — con el default no
se usa.

## 3. Instalar dependencias de Python

Desde **esta carpeta**:

```bash
pip install -r requirements.txt
```

## 4. Levantar todo

```bash
chmod +x start-all.sh   # solo la primera vez
./start-all.sh
```

El script verifica que Ollama esté arriba y que `CHAT_MODEL`/`TRIVIA_MODEL`
estén importados (si falta alguno, aborta); avisa sin abortar si falta
`SALIDA_TRIVIA_MODEL`, que tiene fallback. Después corre
`Orchestrator_Management.py` en primer plano, que es donde interactúas por
teclado — `preguntas.py` se carga en memoria al arrancar
`Orchestrator_Management.py` mismo, y `voz_server.py` levanta automáticamente
una página web (ver abajo), sin nada más que levantar a mano.

No hace falta exportar `CHAT_SERVER_HOST` — apunta a `localhost` por defecto
en `Clients/Llama_Client.py`, que es justo lo que se necesita cuando Ollama
corre en la misma máquina.

## Subir a un modelo más grande (más prosa, más RAM/tiempo)

`CHAT_MODEL` se puede sobreescribir por variable de entorno sin tocar código.
Si el hardware tiene margen y se prefiere una prosa más elaborada que la de
`lora-chat` (fine-tune de `qwen2.5:0.5b`), `llama3.2:3b` (el default de este
proyecto hasta el benchmark de más arriba) sigue siendo una opción válida —
ya viene probado. Ojo que pierde la personalidad horneada del fine-tune, así
que `personalidad.py` vuelve a mandar el system prompt completo:

```bash
ollama pull llama3.2:3b   # si no está descargado todavía
export CHAT_MODEL=llama3.2:3b
./start-all.sh
```

Ojo: respuestas de 7-12s en vez de 1-3s (ver la tabla de la sección de
rendimiento). El router (`Agent_Router.py`) no se ve afectado por esta
variable — no usa Ollama, así que sigue corriendo en microsegundos esté
`CHAT_MODEL` en lo que esté.

## Escribirle a Lora desde el celular/otra PC (página web)

Además de la terminal, `Orchestrator_Management.py` levanta automáticamente
una página en `http://<ip-de-la-pi>:8081/` con un campo de texto — lo que
escribas ahí llega al mismo lugar que si lo tipearas en la terminal (misma
cola interna), así que podés tener la terminal en la Pi y mandar mensajes
desde el celular en la misma red. `VOZ_PORT` cambia el puerto si hace falta.

⚠️ **Todavía NO hay entrada por voz** (a propósito, por ahora): se probó con
`faster-whisper` (transcripción 100% local, sin nube) y funcionaba bien,
pero `WhisperModel(...)` se cuelga silenciosamente si se construye fuera del
hilo principal — que es como corre hoy. Falta resolver eso antes de
reactivarla; mientras tanto la página es solo texto (ver la nota al
principio de `voz_server.py`).

La voz de SALIDA (lo que dice Lora) sí está activa: usa `edge-tts`
(`Clients/Voice_Output_Client.py`, voz `es-AR-ElenaNeural`) — a diferencia de
la entrada, esto necesita internet (servicio en la nube de Microsoft, gratis
y sin API key). Si la Pi se queda sin WiFi, se loguea el error y el chat
sigue funcionando solo con texto.

### Que hable el teléfono en vez de la Pi

```bash
export VOZ_MOTOR=telefono
./start-all.sh
```

Con esta variable, cuando Lora contesta la Pi no sintetiza nada: le manda el
texto a la misma página de arriba (`http://<ip-de-la-pi>:8081/`), y el
teléfono lo dice con la síntesis de voz nativa del navegador (Web Speech
API) — usa el procesador y el parlante del teléfono, no los de la Pi. Hay
que abrir la página y tocar **"🔊 Activar voz de Lora"** una vez (los
navegadores móviles exigen un toque del usuario antes de dejar sonar
síntesis de voz por primera vez); de ahí en más queda escuchando sola.

Si nadie tiene la página abierta con la voz activada, o el teléfono no
avisa que terminó de hablar en 20s (pestaña cerrada, se durmió la
pantalla), se degrada a `edge-tts` automáticamente — mismo criterio que ya
usa Piper cuando falta el `.onnx`.

No se evaluó correr el modelo de síntesis (Piper) directamente en hardware
embebido tipo ESP32-S3: el `.onnx` (20-100MB) no entra en su RAM (512KB
SRAM + unos MB de PSRAM), muy por debajo de lo que necesita una red VITS —
por eso la síntesis siempre corre en la Pi o en el teléfono, nunca en el
ESP32 del carrito.

### Hablarle a Lora en vez de escribir (entrada por voz)

El botón **"🎤 Hablar"** de la misma página transcribe lo que decís con la
Web Speech API del navegador (`SpeechRecognition`) y lo manda como si lo
hubieras tipeado — sin instalar nada en la Pi. Solo funciona bien en
**Chrome/Android**; Safari/iOS no la soporta (ningún navegador ahí, todos
usan el motor de Safari por dentro).

`SpeechRecognition` exige "contexto seguro" (HTTPS o `localhost`) — la
página de esta Pi se sirve en HTTP plano sobre una IP de LAN
(`http://192.168.x.x:8081/`), así que Chrome bloquea el micrófono ahí por
default. Para habilitarlo en el teléfono, **una sola vez**:

1. Abrí `chrome://flags/#unsafely-treat-insecure-origin-as-secure` en Chrome.
2. Pegá la URL completa de la página (ej. `http://192.168.1.38:8081`) en el
   campo de texto que aparece.
3. Cambiá el flag de "Default" a "Enabled".
4. Reiniciá Chrome (el botón "Relaunch" que aparece abajo).

Después de eso, el botón de micrófono pide permiso de audio la primera vez
(como cualquier sitio) y ya queda andando en ese teléfono.

## Carrito mecanum (movimiento)

`Clients/Carrito_Client.py` habla con el ESP32 del carrito (ver
`../carrito-mecanum-esp32/2-l298n-mecanum/`) por **Serial (USB)**, no por
WiFi/HTTP — el ESP32 va conectado por cable directo a esta misma Pi.

Por default usa `/dev/ttyACM0`; si el puerto es otro (`ls /dev/ttyACM*
/dev/ttyUSB*` para verlo), sobreescribilo:

```bash
export CARRITO_PORT=/dev/ttyUSB0
```

Sin el ESP32 conectado (cable desconectado, apagado, etc.) no hace falta
configurar nada — `mover()`/`mover_360()` loguean `[carrito] no se pudo
abrir ...` y siguen sin cortar el flujo de trivia, igual que antes cuando
faltaba `CARRITO_HOST`.

## Carita en pantalla

`display.py` detecta solo si hay sesión gráfica: con la Pi headless y una
pantalla LCD/HDMI conectada directo, usa `mpv --vo=drm` (sin necesitar
X11/Wayland); con sesión gráfica (o en Windows, para probar en una PC) usa
un visor propio en Tkinter (`face_viewer.py`). Requiere `mpv` instalado
(`sudo apt install mpv`) y estar en el grupo `video`
(`sudo usermod -aG video,render $USER`, reiniciar sesión para que aplique).

## Archivos

```
deploy-raspberry-standalone/
├── Orchestrator_Management.py   ← entrypoint: python Orchestrator_Management.py
├── preguntas.py
├── personalidad.py
├── display.py, face_viewer.py, faces/
├── voz_server.py
│
├── Agents/            ← veredictos y decisiones (con o sin LLM)
│   ├── Agent_Router.py       (+ router_modelo.joblib)
│   ├── Agent_Corrector.py
│   └── Agent_Behavior.py
│
├── Clients/           ← wrappers hacia servicios/hardware externos
│   ├── Llama_Client.py
│   ├── Carrito_Client.py
│   ├── Musica_Client.py
│   └── Voice_Output_Client.py
│
├── router_training/   ← dataset + script para reentrenar Agent_Router
│   ├── dataset_router.jsonl
│   └── entrenar_router.py
│
├── personalidad_training/   ← pipeline de fine-tuning (opcional, correr en PC de desarrollo)
│   ├── dataset_personalidad.jsonl
│   ├── entrenar_personalidad.py
│   └── Modelfile
│
├── base_datos/, musica/, requirements.txt, start-all.sh
```

| Archivo/carpeta | Qué es |
|---|---|
| `Orchestrator_Management.py` | Router + loop de conversación (Trivia/Chat libre) y la lógica de cada modo — fusión de lo que antes eran `chat.py` (manager) y `workers.py` (lógica por modo). |
| `preguntas.py` | Búsqueda BM25 sobre `preguntas.jsonl` en memoria — sin HTTP, sin servidor aparte. |
| `personalidad.py` | Arma el system prompt con la personalidad del robot — vacío si `CHAT_MODEL=lora-personalidad` (ver "Personalidad horneada en el modelo" arriba). |
| `display.py`, `face_viewer.py`, `faces/` | Carita en pantalla (LCD/HDMI en la Pi, o Tkinter en PC). |
| `voz_server.py` | Página web: texto o voz (Web Speech API) como entrada, voz de salida al teléfono (ver arriba). |
| `Agents/Agent_Router.py` | Sin LLM: clasifica cada turno en TRIVIA/CHAT_LIBRE (TF-IDF + regresión logística sobre `router_modelo.joblib`, ver "Router sin LLM" arriba). |
| `Agents/Agent_Corrector.py` | Sin LLM: veredicto CORRECTO/INCORRECTO de una respuesta de Trivia (número exacto o normalización+substring+fuzzy contra `respuesta_esperada`, ver "Corrector de Trivia sin LLM" arriba). |
| `Agents/Agent_Behavior.py` | Sin LLM: elige la cara de acierto/error de cada pregunta de Trivia (de las columnas del dataset) y dispara música/desplazamiento; también la cara genérica de Chat libre. Fusión de lo que antes eran `reactor.py` + `cara_agente.py`. |
| `Clients/Llama_Client.py` | Cliente HTTP hacia Ollama (`generar_respuesta()` — genera texto, ya no enruta). |
| `Clients/Carrito_Client.py` | Cliente Serial (USB) hacia el carrito mecanum (ver `../carrito-mecanum-esp32/`) para los comandos de `desplazamiento`. |
| `Clients/Musica_Client.py` | Reproduce (con `mpv`) el archivo de `musica/` que indique la columna `musical`, recortado a 20s. |
| `Clients/Voice_Output_Client.py` | Lora habla en voz alta con `edge-tts` (voz `es-AR-ElenaNeural`) + `mpv`. |
| `router_training/` | Dataset + script para reentrenar `Agent_Router.py` si hace falta (ver "Router sin LLM" arriba). |
| `personalidad_training/` | Pipeline de fine-tuning LoRA para `lora-personalidad`, opcional -- correr en una PC de desarrollo, no en la Pi (ver "Personalidad horneada en el modelo" arriba). |
| `musica/` | Archivos de audio para la columna `musical` del dataset (ver su propio README). |
| `base_datos/` | `preguntas.jsonl` + el Excel fuente (sin índice de ChromaDB — `preguntas.py` arma el índice BM25 en memoria al importarse). |
| `requirements.txt` | Dependencias de Python. |
| `start-all.sh` | Verifica Ollama+modelos y corre `Orchestrator_Management.py`. |
