# Despliegue: todo en la Raspberry Pi

Todo el proyecto vive en esta carpeta: código, dataset y modelos corren en
la propia Raspberry Pi, en un solo proceso Python (`Orchestrator_Management.py`,
que arma todo junto con `preguntas.py`, `Agents/`, `Clients/`, `display.py` y
`voz_server.py`) más Ollama como único servicio externo — sin depender de
ninguna otra máquina.

Probado pensando en una **Raspberry Pi 4 (8GB RAM, 128GB de almacenamiento)**.

## ⚠️ Rendimiento esperado

Una Pi 4 no tiene GPU para inferencia — todo corre en CPU (ARM Cortex-A72).
`CHAT_MODEL` (`qwen2.5:0.5b` por default) es el ÚNICO modelo que Ollama
mantiene cargado en RAM: genera todas las respuestas (chat libre, búsqueda
web, reacciones de trivia, corrector, verificador). El router YA NO usa
Ollama para nada — ver "Router sin LLM" más abajo.

Se eligió `qwen2.5:0.5b` sobre `llama3.2:3b-q4s` (el default anterior) tras
benchmarquear 7 modelos livianos (Meta/Facebook y HuggingFace, ver
`Clients/Llama_Client.py` para el detalle y los números) con los prompts
reales de este proyecto:

| Modelo | RAM residente (`ollama ps`) | Respuesta en caliente | Calidad en las 5 corridas |
|---|---|---|---|
| **qwen2.5:0.5b (default)** | **484 MB** | **1–3s** | 5/5 correctas, sin fugas ni frases prohibidas |
| llama3.2:3b-q4s (default anterior) | 2.5 GB | 7–12s | 5/5 correctas, prosa más elaborada |
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

Clasificar cada turno en TRIVIA / BUSQUEDA_WEB / CHAT_LIBRE es una
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

`Agent_Verificator.py` (Chat libre/Búsqueda Web) sigue usando LLM: a
diferencia del router y el corrector, a veces tiene que REESCRIBIR la
respuesta cuando la encuentra incoherente, no solo clasificarla — eso es
generación de texto, no clasificación, así que no tiene el mismo reemplazo
directo. Ya se beneficia igual del modelo más liviano (`qwen2.5:0.5b`, ver
arriba).

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

`llama3.2:3b-q4s` NO hace falta descargarlo salvo que se quiera pasar a él
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

El script verifica que Ollama y los 2 modelos estén listos y corre
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
`qwen2.5:0.5b`, `llama3.2:3b-q4s` (el default de este proyecto hasta el
benchmark de más arriba) sigue siendo una opción válida — ya viene probado:

```bash
ollama pull llama3.2:3b-q4s   # si no está descargado todavía
export CHAT_MODEL=llama3.2:3b-q4s
./start-all.sh
```

Ojo: respuestas de 7-12s en vez de 1-3s (ver la tabla de la sección de
rendimiento). El router (`Agent_Router.py`) no se ve afectado por esta
variable — no usa Ollama, así que sigue corriendo en microsegundos esté
`CHAT_MODEL` en lo que esté.

## Escribirle a Ereberus desde el celular/otra PC (página web)

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

La voz de SALIDA (lo que dice Ereberus) sí está activa: usa `edge-tts`
(`Clients/Voice_Output_Client.py`, voz `es-AR-ElenaNeural`) — a diferencia de
la entrada, esto necesita internet (servicio en la nube de Microsoft, gratis
y sin API key). Si la Pi se queda sin WiFi, se loguea el error y el chat
sigue funcionando solo con texto.

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
│   ├── Agent_Verificator.py
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
├── base_datos/, musica/, requirements.txt, start-all.sh
```

| Archivo/carpeta | Qué es |
|---|---|
| `Orchestrator_Management.py` | Router + loop de conversación (Trivia/Búsqueda Web/Chat libre) y la lógica de cada modo — fusión de lo que antes eran `chat.py` (manager) y `workers.py` (lógica por modo). |
| `preguntas.py` | Búsqueda BM25 sobre `preguntas.jsonl` en memoria — sin HTTP, sin servidor aparte. |
| `personalidad.py` | Arma el system prompt con la personalidad del robot. |
| `display.py`, `face_viewer.py`, `faces/` | Carita en pantalla (LCD/HDMI en la Pi, o Tkinter en PC). |
| `voz_server.py` | Página web de texto (ver arriba) — entrada por voz todavía deshabilitada. |
| `Agents/Agent_Router.py` | Sin LLM: clasifica cada turno en TRIVIA/BUSQUEDA_WEB/CHAT_LIBRE (TF-IDF + regresión logística sobre `router_modelo.joblib`, ver "Router sin LLM" arriba). |
| `Agents/Agent_Corrector.py` | Sin LLM: veredicto CORRECTO/INCORRECTO de una respuesta de Trivia (número exacto o normalización+substring+fuzzy contra `respuesta_esperada`, ver "Corrector de Trivia sin LLM" arriba). |
| `Agents/Agent_Verificator.py` | Único Agent que sigue usando LLM (`CHAT_MODEL`): revisa coherencia de las respuestas de Chat libre/Búsqueda Web y las reescribe si hace falta -- eso es generación de texto, no clasificación, ver la nota en "Corrector de Trivia sin LLM" arriba. |
| `Agents/Agent_Behavior.py` | Sin LLM: elige la cara de acierto/error de cada pregunta de Trivia (de las columnas del dataset) y dispara música/desplazamiento; también la cara genérica de Chat libre/Búsqueda Web. Fusión de lo que antes eran `reactor.py` + `cara_agente.py`. |
| `Clients/Llama_Client.py` | Cliente HTTP hacia Ollama (`generar_respuesta()` — genera texto, ya no enruta). |
| `Clients/Carrito_Client.py` | Cliente HTTP hacia el carrito mecanum (ver `../carrito-mecanum-esp32/`) para los comandos de `desplazamiento`. |
| `Clients/Musica_Client.py` | Reproduce (con `mpv`) el archivo de `musica/` que indique la columna `musical`, recortado a 20s. |
| `Clients/Voice_Output_Client.py` | Ereberus habla en voz alta con `edge-tts` (voz `es-AR-ElenaNeural`) + `mpv`. |
| `router_training/` | Dataset + script para reentrenar `Agent_Router.py` si hace falta (ver "Router sin LLM" arriba). |
| `musica/` | Archivos de audio para la columna `musical` del dataset (ver su propio README). |
| `base_datos/` | `preguntas.jsonl` + el Excel fuente (sin índice de ChromaDB — `preguntas.py` arma el índice BM25 en memoria al importarse). |
| `requirements.txt` | Dependencias de Python. |
| `start-all.sh` | Verifica Ollama+modelos y corre `Orchestrator_Management.py`. |
