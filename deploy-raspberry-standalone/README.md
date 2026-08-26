# Despliegue: todo en la Raspberry Pi

Todo el proyecto vive en esta carpeta: código, dataset y modelos corren en
la propia Raspberry Pi, en un solo proceso Python (`Orchestrator_Management.py`,
que arma todo junto con `preguntas.py`, `Agents/`, `Clients/`, `display.py` y
`voz_server.py`) más Ollama como único servicio externo — sin depender de
ninguna otra máquina.

Probado pensando en una **Raspberry Pi 4 (8GB RAM, 128GB de almacenamiento)**.

## ⚠️ Rendimiento esperado

Una Pi 4 no tiene GPU para inferencia — todo corre en CPU (ARM Cortex-A72).
El modelo por default de `CHAT_MODEL` (que genera TODAS las respuestas:
chat libre, búsqueda web, reacciones de trivia, corrector, verificador) y
`ROUTER_MODEL` (que clasifica cada turno) son el MISMO modelo —
`qwen2.5:0.5b` — así que Ollama solo mantiene una copia en RAM para las dos
cosas, no dos modelos separados.

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

Ojo: esto vuelve a levantar DOS modelos en RAM (`qwen2.5:0.5b` para el router
+ el que pongas acá para generar), no uno solo como con el default — y
respuestas de 7-12s en vez de 1-3s (ver la tabla de la sección de
rendimiento). `ROUTER_MODEL` no se toca con esta variable: el router siempre
corre en `qwen2.5:0.5b`, esté `CHAT_MODEL` en lo que esté.

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
├── base_datos/, musica/, requirements.txt, start-all.sh
```

| Archivo/carpeta | Qué es |
|---|---|
| `Orchestrator_Management.py` | Router + loop de conversación (Trivia/Búsqueda Web/Chat libre) y la lógica de cada modo — fusión de lo que antes eran `chat.py` (manager) y `workers.py` (lógica por modo). |
| `preguntas.py` | Búsqueda BM25 sobre `preguntas.jsonl` en memoria — sin HTTP, sin servidor aparte. |
| `personalidad.py` | Arma el system prompt con la personalidad del robot. |
| `display.py`, `face_viewer.py`, `faces/` | Carita en pantalla (LCD/HDMI en la Pi, o Tkinter en PC). |
| `voz_server.py` | Página web de texto (ver arriba) — entrada por voz todavía deshabilitada. |
| `Agents/Agent_Corrector.py` | Veredicto CORRECTO/INCORRECTO de una respuesta de Trivia (único punto que usa el LLM para eso). |
| `Agents/Agent_Verificator.py` | Revisa coherencia de las respuestas de Chat libre/Búsqueda Web antes de mostrarlas. |
| `Agents/Agent_Behavior.py` | Sin LLM: elige la cara de acierto/error de cada pregunta de Trivia (de las columnas del dataset) y dispara música/desplazamiento; también la cara genérica de Chat libre/Búsqueda Web. Fusión de lo que antes eran `reactor.py` + `cara_agente.py`. |
| `Clients/Llama_Client.py` | Cliente HTTP hacia Ollama (`generar_respuesta()`, `enrutar()`). |
| `Clients/Carrito_Client.py` | Cliente Serial (USB) hacia el carrito mecanum (ver `../carrito-mecanum-esp32/`) para los comandos de `desplazamiento`. |
| `Clients/Musica_Client.py` | Reproduce (con `mpv`) el archivo de `musica/` que indique la columna `musical`, recortado a 20s. |
| `Clients/Voice_Output_Client.py` | Ereberus habla en voz alta con `edge-tts` (voz `es-AR-ElenaNeural`) + `mpv`. |
| `musica/` | Archivos de audio para la columna `musical` del dataset (ver su propio README). |
| `base_datos/` | `preguntas.jsonl` + el Excel fuente (sin índice de ChromaDB — `preguntas.py` arma el índice BM25 en memoria al importarse). |
| `requirements.txt` | Dependencias de Python. |
| `start-all.sh` | Verifica Ollama+modelos y corre `Orchestrator_Management.py`. |
