# Despliegue: todo en la Raspberry Pi

Todo el proyecto vive en esta carpeta: código, dataset y modelos corren en
la propia Raspberry Pi, en un solo proceso Python (`Orchestrator_Management.py`,
que arma todo junto con `preguntas.py`, `Agents/`, `Clients/`, `display.py` y
`voz_server.py`) más Ollama como único servicio externo — sin depender de
ninguna otra máquina.

Probado pensando en una **Raspberry Pi 4 (8GB RAM, 128GB de almacenamiento)**.

## ⚠️ Rendimiento esperado

Una Pi 4 no tiene GPU para inferencia — todo corre en CPU (ARM Cortex-A72).
Con 8GB de RAM entran los 2 modelos sin problema, pero van a ser más lentos
que en una PC:

- `qwen2.5:0.5b` (router) es chico, debería sentirse razonablemente rápido.
- `llama3.2:3b-q4s` (el modelo que genera las respuestas) es el más pesado —
  probablemente notes varios segundos de espera por respuesta. Pruébalo
  primero; si se siente demasiado lento para una conversación fluida, hay un
  fallback más liviano (ver "Si `llama3.2:3b` va muy lento" más abajo).

`preguntas.py` no usa ningún modelo — busca sobre `preguntas.jsonl` con BM25
(keyword) en memoria dentro del mismo proceso de `chat.py`, así que no
consume Ollama, RAM de embeddings, ni un puerto/proceso aparte.

## 1. Instalar Ollama en la Pi

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

El instalador deja Ollama corriendo como servicio de systemd (arranca solo
al bootear la Pi — no hace falta iniciarlo a mano). Verificalo con:

```bash
systemctl status ollama
```

## 2. Descargar los 2 modelos

```bash
ollama pull llama3.2:3b-q4s        # genera las respuestas (chat.py)
ollama pull qwen2.5:0.5b           # router de intención (chat.py)
```

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

## Si `llama3.2:3b` va muy lento

`CHAT_MODEL` se puede sobreescribir por variable de entorno. Para probar con
el modelo liviano en vez del de 3B:

```bash
export CHAT_MODEL=qwen2.5:0.5b
./start-all.sh
```

Ojo: `qwen2.5:0.5b` está afinado en `Clients/Llama_Client.py` como
clasificador de rutas (few-shot para TRIVIA/BUSQUEDA_WEB/CHAT_LIBRE), no
como modelo de charla — úsalo como prueba de velocidad, pero esperá
respuestas de menor calidad conversacional que con `llama3.2:3b`.

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
| `Clients/Carrito_Client.py` | Cliente HTTP hacia el carrito mecanum (ver `../carrito-mecanum-esp32/`) para los comandos de `desplazamiento`. |
| `Clients/Musica_Client.py` | Reproduce (con `mpv`) el archivo de `musica/` que indique la columna `musical`, recortado a 20s. |
| `Clients/Voice_Output_Client.py` | Ereberus habla en voz alta con `edge-tts` (voz `es-AR-ElenaNeural`) + `mpv`. |
| `musica/` | Archivos de audio para la columna `musical` del dataset (ver su propio README). |
| `base_datos/` | `preguntas.jsonl` + el Excel fuente (sin índice de ChromaDB — `preguntas.py` arma el índice BM25 en memoria al importarse). |
| `requirements.txt` | Dependencias de Python. |
| `start-all.sh` | Verifica Ollama+modelos y corre `Orchestrator_Management.py`. |
