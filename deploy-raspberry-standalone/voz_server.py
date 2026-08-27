"""
Servidor de entrada web: una página con un input de texto (+ botón Enviar)
que manda el texto a la MISMA cola que lee chat.py para la entrada de la
terminal — tipear en la terminal o tipear en esta página son 2 caminos al
mismo lugar.

Corre DENTRO del proceso de chat.py, en un hilo de fondo (Flask con
threaded=True) — no es un servicio aparte que haya que levantar con
start-all.sh, arranca solo al correr chat.py. Puerto reusado: 8081 (el
mismo que usaba el viejo embed_server.py, que ya no existe en este
despliegue — ver preguntas.py).

⚠️ TODAVÍA NO HAY VOZ (a propósito, por ahora): se había armado y probado
/voz con faster-whisper (transcripción 100% local, sin nube — funcionaba
bien llamado desde el hilo principal), pero WhisperModel(...) se cuelga
silenciosamente al construirse desde un hilo NO principal — que es
justo el caso acá, porque chat.main() ya corre en su propio hilo de fondo.
No tiró excepción, ni con salida sin buffer: se quedó esperando para
siempre. Hay que resolver eso (¿cargar el modelo en el hilo principal antes
de lanzar el resto?) antes de reactivar la transcripción — mientras tanto,
esta página es solo texto.
"""

import os
import threading

from flask import Flask, Response, jsonify, request

PORT = int(os.environ.get("VOZ_PORT", "8081"))

app = Flask(__name__)
_entrada_queue = None  # se inyecta desde chat.py al arrancar (ver iniciar())


_PAGINA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lora — texto</title>
<style>
  body { font-family: system-ui, sans-serif; text-align: center; margin-top: 3rem; padding: 0 1rem; }
  #estado { color: #666; min-height: 1.4em; }
  #transcripcion { font-style: italic; min-height: 1.4em; }
  form { margin-top: 1.5rem; display: flex; gap: 0.5rem; justify-content: center; }
  input[type=text] { font-size: 1rem; padding: 0.6rem; flex: 1; max-width: 20rem; }
  button[type=submit] { font-size: 1rem; padding: 0.6rem 1.2rem; }
</style>
</head>
<body>
  <h1>Habla con Lora</h1>
  <p id="estado">Escribí tu mensaje.</p>
  <p id="transcripcion"></p>

  <form id="form-texto">
    <input type="text" id="texto" placeholder="Escribí acá..." autocomplete="off" autofocus>
    <button type="submit">Enviar</button>
  </form>

<script>
  document.getElementById('form-texto').addEventListener('submit', async (e) => {
    e.preventDefault();
    const campo = document.getElementById('texto');
    const texto = campo.value.trim();
    if (!texto) return;
    document.getElementById('estado').textContent = 'Enviando...';
    await fetch('/texto', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texto })
    });
    document.getElementById('transcripcion').textContent = '"' + texto + '"';
    document.getElementById('estado').textContent = 'Mandado a Lora.';
    campo.value = '';
    campo.focus();
  });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(_PAGINA, mimetype="text/html")


@app.route("/texto", methods=["POST"])
def texto_directo():
    body = request.get_json(silent=True) or {}
    texto = (body.get("texto") or "").strip()
    if texto and _entrada_queue is not None:
        _entrada_queue.put(texto)
    return jsonify({"ok": bool(texto)})


def iniciar(entrada_queue):
    """Levanta Flask en un hilo de fondo — no bloquea a quien lo llama,
    chat.py sigue con su propio loop enseguida (sin modelo que cargar por
    ahora, arranca instantáneo)."""
    global _entrada_queue
    _entrada_queue = entrada_queue

    hilo = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=PORT, debug=False,
                                use_reloader=False, threaded=True),
        daemon=True,
    )
    hilo.start()
    print(f"[voz] Página de texto en http://0.0.0.0:{PORT}/ (o http://<ip-de-esta-máquina>:{PORT}/ desde otro dispositivo en la LAN)")
