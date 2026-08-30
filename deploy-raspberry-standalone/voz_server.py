"""
Servidor de entrada/salida web: una página con un input de texto (+ botón
Enviar) que manda el texto a la MISMA cola que lee chat.py para la entrada
de la terminal — tipear en la terminal o tipear en esta página son 2
caminos al mismo lugar. Además, la página puede HABLAR: si Lora contesta
mientras hay un teléfono con la página abierta (VOZ_MOTOR=telefono, ver
Clients/Voice_Output_Client.py), esta misma página recibe el texto por SSE
y lo dice con la síntesis de voz nativa del navegador (Web Speech API) --
usa el procesador y el parlante del teléfono, no los de la Pi.

Corre DENTRO del proceso de chat.py, en un hilo de fondo (Flask con
threaded=True) — no es un servicio aparte que haya que levantar con
start-all.sh, arranca solo al correr chat.py. Puerto reusado: 8081 (el
mismo que usaba el viejo embed_server.py, que ya no existe en este
despliegue — ver preguntas.py).

⚠️ TODAVÍA NO HAY ENTRADA POR VOZ (a propósito, por ahora): se había armado
y probado /voz con faster-whisper (transcripción 100% local, sin nube —
funcionaba bien llamado desde el hilo principal), pero WhisperModel(...) se
cuelga silenciosamente al construirse desde un hilo NO principal — que es
justo el caso acá, porque chat.main() ya corre en su propio hilo de fondo.
No tiró excepción, ni con salida sin buffer: se quedó esperando para
siempre. Hay que resolver eso (¿cargar el modelo en el hilo principal antes
de lanzar el resto?) antes de reactivar la transcripción — mientras tanto,
la entrada de esta página es solo texto. La SALIDA de voz (esta sección) es
un mecanismo distinto: no transcribe nada, solo hace hablar al navegador.

Por qué SSE (Server-Sent Events) y no WebSockets: la página ya usa
`fetch()` normal para /texto, y Flask sirve SSE con un generator + Response
normal, sin agregar ninguna dependencia nueva (websockets, flask-sock).
Alcanza porque el flujo es un solo sentido a la vez por turno: la Pi manda
texto a hablar (push del servidor) o el teléfono manda texto tipeado (POST
del cliente) -- nunca los dos simultaneos en el mismo turno.
"""

import json
import os
import queue
import threading

from flask import Flask, Response, jsonify, request

PORT = int(os.environ.get("VOZ_PORT", "8081"))

app = Flask(__name__)
_entrada_queue = None  # se inyecta desde chat.py al arrancar (ver iniciar())

# ── Salida de voz hacia el teléfono (VOZ_MOTOR=telefono) ────────────────
# Un Queue por página conectada (si hay más de una pestaña abierta, todas
# hablan) + un Event para que hablar_telefono() sepa cuándo el teléfono
# terminó de decir la frase (Voice_Output_Client.hablar() bloquea hasta
# entonces, igual que con edge-tts/Piper).
_clientes_voz = set()
_clientes_voz_lock = threading.Lock()
_listo_evento = threading.Event()


def hay_cliente_conectado():
    """True si al menos un teléfono tiene la página abierta con la voz
    activada (conectado a /eventos). La usa Orchestrator_Management.py para
    esperar antes de la primera frase de la sesión (el saludo) en vez de
    perderla siempre al respaldo de edge-tts -- ver la nota en
    _esperar_telefono_si_corresponde()."""
    with _clientes_voz_lock:
        return bool(_clientes_voz)


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

  <p style="margin-top: 2rem;">
    <button id="btn-voz" type="button">🔊 Activar voz de Lora</button>
  </p>
  <p id="estado-voz" style="color: #666;"></p>

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

  // ── Salida de voz (VOZ_MOTOR=telefono en Voice_Output_Client.py) ──────
  // iOS/Android exigen un gesto del usuario (tocar un botón) antes de
  // dejar sonar speechSynthesis por primera vez -- por eso el botón, no
  // arranca solo. Un solo toque alcanza para toda la sesión.
  const estadoVoz = document.getElementById('estado-voz');

  function elegirVoz() {
    const voces = speechSynthesis.getVoices();
    return voces.find(v => v.lang === 'es-AR')
        || voces.find(v => v.lang && v.lang.startsWith('es'))
        || null;
  }

  function hablar(texto) {
    const u = new SpeechSynthesisUtterance(texto);
    const voz = elegirVoz();
    if (voz) u.voice = voz;
    u.lang = 'es-AR';
    u.onstart = () => { estadoVoz.textContent = 'Lora está hablando...'; };
    u.onend = u.onerror = () => {
      estadoVoz.textContent = 'Escuchando a Lora...';
      fetch('/listo', { method: 'POST' }).catch(() => {});
    };
    speechSynthesis.speak(u);
  }

  document.getElementById('btn-voz').addEventListener('click', () => {
    // Desbloquea el motor de voz del navegador con un toque real del
    // usuario, y a partir de ahí se conecta a escuchar lo que diga Lora.
    speechSynthesis.speak(new SpeechSynthesisUtterance(''));
    estadoVoz.textContent = 'Escuchando a Lora...';
    document.getElementById('btn-voz').disabled = true;

    const fuente = new EventSource('/eventos');
    fuente.onmessage = (e) => {
      const datos = JSON.parse(e.data);
      hablar(datos.texto);
    };
    fuente.onerror = () => {
      estadoVoz.textContent = 'Se cortó la conexión con Lora, reintentando...';
    };
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


@app.route("/eventos")
def eventos():
    """SSE: cada pestaña que toca 'Activar voz' abre una conexión acá y
    recibe cada frase que hablar_telefono() le mande, una por evento."""
    cola = queue.Queue()
    with _clientes_voz_lock:
        _clientes_voz.add(cola)

    def generar():
        try:
            while True:
                texto = cola.get()
                yield f"data: {json.dumps({'texto': texto})}\n\n"
        finally:
            # Se ejecuta cuando el navegador cierra la pestaña/conexión.
            with _clientes_voz_lock:
                _clientes_voz.discard(cola)

    return Response(generar(), mimetype="text/event-stream")


@app.route("/listo", methods=["POST"])
def listo():
    """El teléfono avisa que terminó de decir la frase (onend/onerror del
    SpeechSynthesisUtterance) -- desbloquea a hablar_telefono()."""
    _listo_evento.set()
    return jsonify({"ok": True})


def hablar_telefono(texto, timeout=20):
    """Le manda `texto` a todos los teléfonos con la página abierta y con
    la voz activada, y bloquea hasta que alguno avise que terminó de
    hablarlo (o hasta `timeout` segundos). Devuelve False si no hay ningún
    teléfono conectado, o si nadie avisa a tiempo -- en los dos casos
    Voice_Output_Client.py cae al motor de respaldo (edge-tts), igual que
    hace hoy si falta el .onnx de Piper.

    timeout=20s: de sobra para frases de hasta ~25 palabras (el límite que
    ya impone personalidad.py) incluso con una voz lenta del navegador;
    si se cumple es señal de que la pestaña se cerró o el teléfono se
    durmió, no de que la frase sea larga."""
    with _clientes_voz_lock:
        destinatarios = list(_clientes_voz)
    if not destinatarios:
        return False
    _listo_evento.clear()
    for cola in destinatarios:
        cola.put(texto)
    return _listo_evento.wait(timeout)


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
    print(f"[voz] Página de texto/voz en http://0.0.0.0:{PORT}/ (o http://<ip-de-esta-máquina>:{PORT}/ desde otro dispositivo en la LAN)")
    print("[voz] Para que el teléfono hable las respuestas de Lora: abrí la página, tocá 'Activar voz de Lora' y export VOZ_MOTOR=telefono")
