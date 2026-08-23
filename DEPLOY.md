# Despliegue: PC (backend) + Raspberry Pi (frontend)

El backend (Ollama + `embed_server.py`) corre en la PC; `chat.py` corre en la
Raspberry Pi. Las dos máquinas tienen que estar en la misma red.

## 1. Acceso por SSH a la Raspberry Pi

La tarjeta está grabada con **Ubuntu Server for Raspberry Pi** (Raspberry Pi
Imager), con SSH habilitado desde el primer arranque.

| | |
|---|---|
| Hostname | `esp-cloud` (con `avahi-daemon`, responde como `esp-cloud.local`) |
| Usuario | `user` |
| Autenticación | contraseña (`ssh_pwauth` activo) |
| WiFi configurado | `ulwifiI1` — red **oculta y abierta**, sin clave |
| Zona horaria | `America/Lima` |

```bash
ssh user@esp-cloud.local
```

Si mDNS no resuelve (pasa en algunas redes), hay que ir por IP. Se averigua
en el router, o desde la propia Pi con `hostname -I`:

```bash
ssh user@192.168.1.XX
```

### Cambiar la red WiFi sin reflashear

La Pi solo se conecta a las redes que estén en `network-config`, en la
partición `system-boot` de la microSD. Esa partición es FAT32, así que se
edita metiendo la tarjeta en cualquier PC (en Windows aparece como una unidad
más). Para agregar la red de casa junto a la de la universidad:

```yaml
network:
  version: 2
  wifis:
    wlan0:
      dhcp4: true
      regulatory-domain: "PE"
      access-points:
        "ulwifiI1":
          hidden: true
          auth:
            key-management: none
        "NOMBRE_DE_TU_RED":
          password: "TU_CLAVE"
      optional: true
```

Se conecta a la que encuentre. Esto importa: si la Pi arranca buscando el
WiFi de la universidad y la PC está en la red de casa, no se ven entre sí.

## 2. Preparar la PC (backend)

Modelos que hacen falta en Ollama:

```bash
ollama pull llama3.2:3b            # generación de respuestas
ollama pull qwen2.5:0.5b           # router de intención
ollama pull qwen3-embedding:0.6b   # embeddings del RAG
```

Los tres nombres tienen que coincidir con los del código
(`backend/llama_client.py` y `backend/embed_server.py`). Verificar con
`ollama list`.

Dependencias y arranque:

```bash
pip install -r backend/requirements-embed-server.txt
ollama serve
python backend/embed_server.py
```

`embed_server.py` sincroniza `base_datos/preguntas.jsonl` contra ChromaDB al
arrancar (agrega, actualiza y borra) y reconstruye el índice BM25 en memoria.
Escucha en el puerto **8081**; Ollama en el **11434**.

## 3. Preparar la Raspberry Pi (frontend)

```bash
sudo apt install -y mpv python3-pip
sudo usermod -aG video,render $USER
pip install -r frontend/requirements.txt
```

El grupo `video` recién aplica en una sesión nueva — cerrá y volvé a
conectarte por SSH (o `newgrp video` para probarlo sin salir).

Antes de correr `chat.py`, exportar la IP de la PC:

```bash
export EMBED_SERVER_HOST=http://192.168.1.44:8081
export CHAT_SERVER_HOST=http://192.168.1.44:11434
python frontend/chat.py
```

Metelas en un `.env` o en el perfil de shell si querés que persistan entre
sesiones. Sin esas variables apunta a `localhost`, que sirve para probar todo
junto en una sola máquina.

La IP puede cambiar si el router reasigna DHCP — si deja de conectar,
revisar `ipconfig` en la PC para la IP actual.

## 4. Verificar la conexión antes de correr `chat.py`

```bash
curl -m 5 http://192.168.1.44:11434/api/tags
curl -m 5 -X POST http://192.168.1.44:8081/pregunta \
  -H "Content-Type: application/json" \
  -d '{"query":"test","n_results":1}'
```

Si ambos responden 200, seguí con `python chat.py`.

Si `curl` no conecta, antes de tocar código revisá que la Raspberry Pi esté en
la misma red que la PC — eso no lo arregla el código, es de red.

## 5. Carita en la pantalla HDMI (`frontend/display.py`)

Con la Pi headless (sin sesión gráfica) y una pantalla LCD/HDMI conectada
directo, `display.py` detecta eso automáticamente y muestra la carita vía
`mpv --vo=drm` (dibuja directo por DRM/KMS, sin necesitar X11/Wayland) en vez
del visor Tkinter que usa la demo en PC.

## Dependencias por componente

| Archivo | Dónde corre | Paquetes |
|---|---|---|
| `backend/requirements-embed-server.txt` | PC | `flask`, `requests`, `httpx`, `chromadb`, `rank_bm25` |
| `frontend/requirements.txt` | Raspberry Pi | `requests`, `httpx` |
| `ai-camera/requirements.txt` | Raspberry Pi (AI Camera) | `opencv-python`, `numpy`, `picamera2` |

La Pi no necesita `chromadb` ni `flask`: no indexa nada, solo consulta el
servidor de la PC por HTTP.

## Nota sobre `setup_llamacpp_ubuntu.sh`

Es de una etapa anterior del proyecto, cuando los modelos corrían con
llama.cpp y ficheros `.gguf` en `models/`. **La arquitectura actual usa
Ollama** — ese script queda solo como referencia histórica y no hay que
correrlo.
