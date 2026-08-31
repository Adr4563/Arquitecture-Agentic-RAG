"""
Servidor de entrada/salida web: una página con un input de texto (+ botón
Enviar, + botón de micrófono) que manda el texto a la MISMA cola que lee
chat.py para la entrada de la terminal — tipear en la terminal, tipear en
esta página, o hablarle al botón de micrófono son 3 caminos al mismo lugar.
Además, la página puede HABLAR: si Lora contesta mientras hay un teléfono
con la página abierta (VOZ_MOTOR=telefono, ver Clients/Voice_Output_Client.py),
esta misma página recibe el texto por SSE y lo dice con la síntesis de voz
nativa del navegador (Web Speech API) -- usa el procesador y el parlante
del teléfono, no los de la Pi.

Corre DENTRO del proceso de chat.py, en un hilo de fondo (Flask con
threaded=True) — no es un servicio aparte que haya que levantar con
start-all.sh, arranca solo al correr chat.py. Puerto reusado: 8081 (el
mismo que usaba el viejo embed_server.py, que ya no existe en este
despliegue — ver preguntas.py).

ENTRADA POR VOZ (2026-08-30, a pedido del usuario): se había armado y
probado antes /voz con faster-whisper (transcripción 100% local, sin nube
— funcionaba bien llamado desde el hilo principal), pero WhisperModel(...)
se colgaba silenciosamente al construirse desde un hilo NO principal — que
era justo el caso acá, porque chat.main() ya corre en su propio hilo de
fondo. Se descartó ese camino en vez de arreglarlo: en lugar de pelear ese
bug de threading y sumar un cuarto modelo pesado compitiendo por los 4
núcleos de la Pi (que ya comparten CHAT_MODEL/TRIVIA_MODEL/
SALIDA_TRIVIA_MODEL), la transcripción se hace en el propio navegador con
`SpeechRecognition` (Web Speech API, botón redondo "Hablar" en la página) --
mismo mecanismo que la síntesis de salida, pero al revés, y sin agregar
nada de carga a la Pi. El costo: manda tu audio a un servicio en la nube
(Google en Chrome/Android) para reconocerlo -- mismo trade-off ya aceptado
para VOZ_MOTOR=edge, pero acá solo afecta lo que decís vos, no lo que
contesta Lora. Solo Chrome/Android la soporta bien; Safari/iOS nunca
implementó SpeechRecognition, en ningún navegador ahí. También exige
"contexto seguro" (HTTPS o localhost) -- en una IP de LAN por HTTP plano
hay que agregar la URL a chrome://flags/#unsafely-treat-insecure-origin-as-secure
en el teléfono, ver README.

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

# Certificado propio para servir HTTPS. Hace falta porque SpeechRecognition
# (el botón "Hablar") exige "contexto seguro": HTTPS o localhost. Sobre una
# IP de LAN por HTTP plano, Chrome bloquea el micrófono y el boton no
# funciona -- que es el sintoma que se venia arrastrando.
#
# Antes la unica salida documentada era agregar la URL a
# chrome://flags/#unsafely-treat-insecure-origin-as-secure en CADA telefono.
# Con el certificado alcanza con aceptar el aviso una vez por dispositivo.
#
# Default APAGADO, a pedido del usuario: solo se quiere VER la pagina, no
# usar el microfono. Sin HTTPS la pagina carga directo, sin el aviso de
# "conexion no privada" que Chrome muestra por el certificado propio -- y que
# en Android a veces ni siquiera deja continuar (no aparece "Avanzado").
#
# VOZ_HTTPS=1 lo enciende. Solo hace falta para el boton "Hablar":
# SpeechRecognition exige contexto seguro (HTTPS o localhost).
CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".certs")
HTTPS = os.environ.get("VOZ_HTTPS", "0") not in ("0", "false", "False")

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

  /* Botones redondos con icono SVG en vez de emoji -- a pedido del
     usuario (2026-08-31): los emoji se ven distinto (o directamente
     "cuadradito sin glifo") según el teléfono/fuente instalada; un SVG
     dibujado a mano se ve igual en cualquier lado, sin depender de fuente
     de emoji del sistema. */
  .fila-botones-circulares { display: flex; flex-direction: column; align-items: center; gap: 0.4rem; margin-top: 1.5rem; }
  .btn-circular {
    width: 4rem; height: 4rem; border-radius: 50%;
    border: none; background: #2563eb; color: #fff;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    transition: background 0.15s, transform 0.1s;
  }
  .btn-circular:active { transform: scale(0.94); }
  .btn-circular:disabled { background: #aaa; cursor: not-allowed; }
  .btn-circular.escuchando { background: #dc2626; animation: pulso 1s infinite; }
  .btn-circular svg { width: 1.8rem; height: 1.8rem; fill: currentColor; }
  .etiqueta-boton { font-size: 0.85rem; color: #666; }
  @keyframes pulso {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
    50% { box-shadow: 0 0 0 10px rgba(220,38,38,0); }
  }
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

  <div class="fila-botones-circulares">
    <button id="btn-mic" class="btn-circular" type="button" aria-label="Hablar" title="Hablar">
      <svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>
    </button>
    <span class="etiqueta-boton" id="etiqueta-mic">Hablar</span>
  </div>

  <div class="fila-botones-circulares" style="margin-top: 1.5rem;">
    <button id="btn-voz" class="btn-circular" type="button" aria-label="Activar voz de Lora" title="Activar voz de Lora">
      <svg viewBox="0 0 24 24"><path d="M4 9v6h4l5 5V4L8 9H4zm11.5 3a4.5 4.5 0 0 0-2.5-4.03v8.06A4.5 4.5 0 0 0 15.5 12z"/></svg>
    </button>
    <span class="etiqueta-boton">Activar voz de Lora</span>
  </div>
  <p id="estado-voz" style="color: #666;"></p>

<script>
  async function enviarTexto(texto) {
    texto = texto.trim();
    if (!texto) return;
    document.getElementById('estado').textContent = 'Enviando...';
    await fetch('/texto', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texto })
    });
    document.getElementById('transcripcion').textContent = '"' + texto + '"';
    document.getElementById('estado').textContent = 'Mandado a Lora.';
  }

  document.getElementById('form-texto').addEventListener('submit', async (e) => {
    e.preventDefault();
    const campo = document.getElementById('texto');
    await enviarTexto(campo.value);
    campo.value = '';
    campo.focus();
  });

  // ── Entrada por voz (audio → texto), Web Speech API ──────────────────
  // A pedido del usuario (2026-08-30): escribir a mano cansa, mejor hablar.
  // Se probó antes con faster-whisper corriendo EN LA PI (transcripción
  // 100% local) y se descartó por un bug de threading sin resolver (ver la
  // nota histórica más arriba) -- esta versión transcribe en el propio
  // navegador del teléfono con SpeechRecognition, sin tocar la Pi para
  // nada. El costo: manda el audio a un servicio en la nube (Google en
  // Chrome/Android) para reconocerlo -- mismo trade-off que ya se aceptó
  // para VOZ_MOTOR=edge, pero acá solo para lo que decís vos, no lo que
  // contesta Lora.
  //
  // Solo Chrome/Android la soporta bien (webkitSpeechRecognition) -- Safari/
  // iOS nunca la implementó, en ningún navegador ahí (todos usan el motor
  // de Safari por dentro). Si no está disponible, el botón se deshabilita
  // en vez de fallar silencioso al tocarlo.
  //
  // OJO despliegue: SpeechRecognition exige "contexto seguro" (HTTPS o
  // localhost) -- en una IP de LAN por HTTP plano como esta, Chrome bloquea
  // el micrófono. Hay que agregar esta URL a
  // chrome://flags/#unsafely-treat-insecure-origin-as-secure en el
  // teléfono (ver README) para que funcione.
  const ReconocedorVoz = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btnMic = document.getElementById('btn-mic');
  const etiquetaMic = document.getElementById('etiqueta-mic');

  if (!ReconocedorVoz) {
    btnMic.disabled = true;
    etiquetaMic.textContent = 'No disponible en este navegador';
  } else {
    const reconocedor = new ReconocedorVoz();
    reconocedor.lang = 'es-AR';       // mismo idioma que la voz de salida (ver elegirVoz() más abajo)
    reconocedor.continuous = false;   // una sola frase por toque, no queda escuchando de fondo
    reconocedor.interimResults = false;

    let escuchando = false;

    reconocedor.onresult = (e) => {
      const texto = e.results[0][0].transcript;
      enviarTexto(texto);
    };
    reconocedor.onerror = (e) => {
      document.getElementById('estado').textContent =
        e.error === 'not-allowed'
          ? 'Sin permiso de micrófono (ver chrome://flags en el teléfono).'
          : 'No se pudo escuchar (' + e.error + ').';
    };
    reconocedor.onend = () => {
      escuchando = false;
      btnMic.classList.remove('escuchando');
      etiquetaMic.textContent = 'Hablar';
    };

    btnMic.addEventListener('click', () => {
      if (escuchando) return;  // evita doble-toque mientras ya está escuchando
      escuchando = true;
      btnMic.classList.add('escuchando');
      etiquetaMic.textContent = 'Escuchando...';
      document.getElementById('estado').textContent = 'Hablá ahora...';
      reconocedor.start();
    });
  }

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


def _contexto_ssl():
    """Devuelve (cert, key) para app.run(ssl_context=...), generando un
    certificado autofirmado la primera vez. None si HTTPS esta apagado o si
    falta `cryptography` -- en ese caso se sirve HTTP plano y el micrófono no
    va a andar desde el teléfono, pero el resto de la página sí."""
    if not HTTPS:
        return None
    cert = os.path.join(CERT_DIR, "voz.crt")
    key = os.path.join(CERT_DIR, "voz.key")
    if os.path.isfile(cert) and os.path.isfile(key):
        return (cert, key)
    try:
        import datetime
        import ipaddress
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print("[voz] falta `cryptography` (pip install cryptography): se sirve HTTP plano")
        return None

    # El certificado tiene que cubrir la IP con la que se entra desde el
    # telefono, no solo el hostname: se entra por IP en la LAN.
    ips = set()
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no manda nada, solo resuelve la IP local
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    ips.add("127.0.0.1")

    clave = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nombre = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "lora-voz")])
    alt = [x509.DNSName("localhost")] + [
        x509.IPAddress(ipaddress.ip_address(i)) for i in sorted(ips)
    ]
    ahora = datetime.datetime.now(datetime.timezone.utc)
    certificado = (
        x509.CertificateBuilder()
        .subject_name(nombre).issuer_name(nombre)
        .public_key(clave.public_key())
        .serial_number(x509.random_serial_number())
        # Un dia antes por si el reloj de la Pi esta atrasado -- paso: sin
        # NTP la fecha quedaba un mes atras y los certificados se rechazaban
        # por "not yet valid".
        .not_valid_before(ahora - datetime.timedelta(days=1))
        .not_valid_after(ahora + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .sign(clave, hashes.SHA256())
    )
    os.makedirs(CERT_DIR, exist_ok=True)
    with open(key, "wb") as f:
        f.write(clave.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
    with open(cert, "wb") as f:
        f.write(certificado.public_bytes(serialization.Encoding.PEM))
    print(f"[voz] certificado generado para {', '.join(sorted(ips))} en {CERT_DIR}")
    return (cert, key)


def iniciar(entrada_queue):
    """Levanta Flask en un hilo de fondo — no bloquea a quien lo llama,
    chat.py sigue con su propio loop enseguida (sin modelo que cargar por
    ahora, arranca instantáneo)."""
    global _entrada_queue
    _entrada_queue = entrada_queue

    contexto = _contexto_ssl()
    esquema = "https" if contexto else "http"
    hilo = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=PORT, debug=False,
                                use_reloader=False, threaded=True,
                                ssl_context=contexto),
        daemon=True,
    )
    hilo.start()
    print(f"[voz] Página de texto/voz en {esquema}://0.0.0.0:{PORT}/ (o {esquema}://<ip-de-esta-máquina>:{PORT}/ desde otro dispositivo en la LAN)")
    if contexto:
        print("[voz] HTTPS con certificado propio: el teléfono va a avisar que no es de confianza.")
        print("[voz] Aceptalo una vez ('Avanzado' -> 'Continuar') y el micrófono queda habilitado.")
    else:
        print("[voz] Sin HTTPS (VOZ_HTTPS=1 para activarlo). El botón 'Hablar' no")
        print("[voz] va a funcionar desde el teléfono; ver la página sí.")
    print("[voz] Para que el teléfono hable las respuestas de Lora: abrí la página, tocá 'Activar voz de Lora' y export VOZ_MOTOR=telefono")
