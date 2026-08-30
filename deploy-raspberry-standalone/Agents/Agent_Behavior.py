"""
Agente de comportamiento: decide qué cara mostrar (y, para Trivia, qué
acciones de música/desplazamiento disparar) — sin LLM, 100% determinista.
Fusiona lo que antes eran reactor.py + cara_agente.py, porque las dos cosas
son la misma idea (elegir una cara a partir de un veredicto ya resuelto) con
dos fuentes de datos distintas:

- elegir_cara_pregunta()/expresar_musica()/expresar_desplazamiento(): para
  Trivia, a partir de las columnas del dataset (cara_respuesta_buena,
  cara_respuesta_mala, musical, desplazamiento en preguntas.jsonl) — se
  invocan para CUALQUIER pregunta de Trivia, tenga o no respuesta_esperada
  (Reconocimiento Musical, Chistes, etc. son temas del mismo catálogo, no un
  modo aparte del router). Orchestrator_Management.py las llama por
  separado, no todas juntas -- el orden pedido es voz primero, después la
  cara, y recién después música/desplazamiento (ver la nota en
  manejar_trivia()), así que no hay un solo "punto de entrada" que las
  agrupe como antes.
- elegir_cara_por_calidad(): para Chat libre, donde no hay fila de dataset
  asociada — el veredicto es "¿se encontró contexto para responder?" (ver
  Orchestrator_Management.responder()), y la cara sale de una elección
  genérica happy / sad-o-angry al azar.

Desplazamiento y música SÍ están conectados a hardware real: el primero manda
el comando al carrito mecanum vía Clients/Carrito_Client.py, el segundo
reproduce el audio (recortado a Musica_Client.REPRODUCCION_MAX_SEG) vía
Clients/Musica_Client.py (mpv). En los dos casos, si el hardware no responde
(carrito apagado, mpv no instalado, archivo inexistente), la llamada falla
gracioso — log, no excepción — y el resto del turno de trivia sigue normal.
Si una columna viene vacía, no se hace nada con esa parte — ni log, ni acción.
"""

import random
import threading

import perf_monitor
from Clients import Carrito_Client, Musica_Client

# El dataset usa nombres en español (columna 'cara' original); display.py
# espera las claves en inglés de los archivos en faces/*.gif. "Neutral" mapea
# a "content" porque es la misma cara de reposo que ya usa
# Orchestrator_Management.py en el resto del flujo — no hay un .gif de
# "neutral" aparte.
_CARA_A_DISPLAY = {
    "Feliz": "happy",
    "Triste": "sad",
    "Enojado": "angry",
    "Neutral": "content",
}

# ─── Trivia: cara/música/desplazamiento a partir del dataset ────────────

def elegir_cara_pregunta(pregunta, acerto):
    """Cara para el veredicto de una pregunta CON respuesta_esperada:
    acerto=True -> cara_respuesta_buena, acerto=False -> cara_respuesta_mala,
    traducida a la clave que entiende display.mostrar_cara(). None si la
    columna viene vacía o con un valor que no se pudo mapear."""
    campo = "cara_respuesta_buena" if acerto else "cara_respuesta_mala"
    valor = (pregunta.get(campo) or "").strip()
    return _CARA_A_DISPLAY.get(valor)


def cara_para_emocion(nombre):
    """Traduce un nombre de emoción en español tal como viene en la columna
    'cara' del dataset (Feliz/Triste/Enojado/Neutral) a la clave que entiende
    display.mostrar_cara() -- usa el mismo mapeo que elegir_cara_pregunta().
    Pensada para el Juego de emociones/imitación (Orchestrator_Management.py
    ::_jugar_emociones()): el robot muestra esta cara ANTES de pedirle al
    usuario que la imite, como referencia visual (antes solo se pedía por
    voz). None si el nombre no matchea ninguna de las 4 conocidas."""
    return _CARA_A_DISPLAY.get((nombre or "").strip())


@perf_monitor.medir("musica")
def expresar_musica(pregunta, esperar=False):
    """Si la pregunta trae algo en 'musical', reproduce ese archivo de
    musica/ de verdad (Musica_Client.py, recortado a 20s). Vacío/ausente ->
    no hace nada, devuelve None.

    `esperar=True` bloquea hasta que termina la canción, para las preguntas
    de Reconocimiento Musical: ahí la música es el enunciado, no un adorno,
    y el usuario no puede responder hasta haberla escuchado. Ver
    _preguntar_siguiente() en Orchestrator_Management.py."""
    musica = (pregunta.get("musical") or "").strip()
    if not musica:
        return None
    print(f"    [música: {musica}]")
    Musica_Client.reproducir(musica, esperar=esperar)
    return musica


@perf_monitor.medir("desplazamiento")
def expresar_desplazamiento(pregunta):
    """Si la pregunta trae algo en 'desplazamiento', se lo manda de verdad al
    carrito mecanum (Carrito_Client.py) — 'Girar 360°' usa mover_360()
    (varios pulsos de rotación seguidos, sin comando único en el firmware);
    el resto (Adelante/Atrás/Izquierda/Derecha) es un solo mover(). Vacío/
    ausente -> no hace nada, devuelve None. Si el carrito no responde (no
    configurado, apagado, fuera de red), no corta el turno de trivia — solo
    queda logueado por Carrito_Client."""
    desplazamiento = (pregunta.get("desplazamiento") or "").strip()
    if not desplazamiento:
        return None
    print(f"    [desplazamiento: {desplazamiento}]")
    if desplazamiento.lower().startswith("girar"):
        # mover_360() bloquea ~2.4s (6 pulsos con pausa entre cada uno, ver
        # Carrito_Client.py) -- se lanza en un hilo aparte, fire-and-forget,
        # igual que Musica_Client.reproducir() ya hace con mpv (Popen), para
        # no sumarle ese tiempo al turno de trivia.
        threading.Thread(target=Carrito_Client.mover_360, daemon=True).start()
    else:
        Carrito_Client.mover(desplazamiento)  # un solo write serial, ya casi instantáneo
    return desplazamiento


# ─── Chat libre: cara genérica por resultado ──────────────────────────────

CARA_BUENA = "happy"
CARAS_MALA = ["sad", "angry"]


def elegir_cara_por_calidad(problema):
    """problema viene de Orchestrator_Management.responder(): True solo
    cuando no hubo contexto relevante para contestar ("no tengo el dato").
    Ya no mide si el Agent_Verificator tuvo que corregir la redacción --
    ese agente se sacó del proyecto, Llama_Client responde directo."""
    return random.choice(CARAS_MALA) if problema else CARA_BUENA
