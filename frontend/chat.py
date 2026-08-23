"""
Manager: enruta cada turno del chat a un worker (Trivia / Búsqueda Web / Chat
libre) y lleva el loop de conversación. La lógica de cada worker, los clientes
HTTP y la personalidad viven en backend/ — acá solo se orquesta.

A diferencia de un chat con personalidad fija, esto es un router puro: por
cada mensaje del usuario decide a cuál de los tres workers mandarlo. Corre en
CADA turno, no solo al arrancar, para que el usuario pueda saltar de trivia a
preguntar algo actual o charlar libre sin reiniciar la sesión — el costo es
una llamada extra al LLM por mensaje.

Corre con:
    python chat.py
Escribe 'salir' para terminar.
"""

import os
import random
import sys
import time

import display  # frontend/display.py: carita en la LCD conectada a esta Raspberry Pi

# Habilita importar los módulos de backend/ desde este script hermano.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from cara_agente import elegir_cara as elegir_cara_por_calidad  # noqa: E402
from corrector import elegir_cara, evaluar_respuesta  # noqa: E402
from llama_client import enrutar, generar_respuesta  # noqa: E402
from personalidad import construir_personalidad  # noqa: E402
from workers import (  # noqa: E402
    TEMAS_CATALOGO,
    comentar_resultado,
    obtener_preguntas_por_tema,
    reaccionar_libre,
    responder,
    responder_busqueda_web,
    resolver_tema,
)

# El modelo puede colar un emoji o una comilla tipográfica que la consola no
# sabe representar, y eso tiraría UnicodeEncodeError a mitad del streaming. Con
# 'replace' sale un '?' y el chat sigue. No se fuerza ningún encoding: el de por
# defecto ya coincide con el de la consola.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


RUTAS = ["TRIVIA", "BUSQUEDA_WEB", "CHAT_LIBRE"]


def enrutar_mensaje(mensaje_usuario):
    # El prompt, el few-shot y el modelo liviano (qwen2.5:0.5b) viven en
    # llama_client.enrutar() — acá solo se decide el fallback si no clasificó.
    ruta = enrutar(mensaje_usuario)
    return ruta if ruta in RUTAS else "CHAT_LIBRE"


PREGUNTAS_POR_TANDA = 5
PAUSA_CAMBIO_CARA = 4  # segundos en 'content' antes de pasar a la cara que sigue (habla o reacción)


def _preguntar_siguiente(estado):
    """Saca la siguiente pregunta de la cola de la tanda actual y la imprime
    tal cual viene del dataset, sin pasarla por el modelo (si la reformulara,
    la respuesta_esperada dejaría de servir para corregir). Devuelve False si
    la cola ya estaba vacía (se acabó la tanda)."""
    if not estado["cola_preguntas"]:
        return False
    actual = estado["cola_preguntas"].pop(0)
    estado["pregunta_pendiente"] = actual
    display.mostrar_cara("speaking")  # se lanza una pregunta: la LCD "habla"
    print(f"Asistente [{actual['cara']}]: {actual['pregunta']}\n")
    # La pregunta se imprime de una (no hay streaming que marque cuándo
    # "termina de hablar"), así que se le da un tiempo fijo en 'speaking'
    # -antes de pasar a la cara de reposo mientras espera que el usuario
    # conteste- para que de verdad se alcance a ver, no un flash instantáneo.
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")
    return True


def _iniciar_tanda(tema, estado):
    """Pide de una sola vez una tanda de PREGUNTAS_POR_TANDA preguntas de un
    tema ya elegido (un solo request al backend) y arranca con la primera;
    las demás quedan en cola_preguntas para encadenarlas sin volver a pasar
    por el menú de temas entre una y otra."""
    preguntas = obtener_preguntas_por_tema(tema, estado["ya_usados"], cantidad=PREGUNTAS_POR_TANDA)
    if not preguntas:
        print(f"Asistente: No quedan preguntas de {tema}.\n")
        return
    estado["ya_usados"].update(p["id"] for p in preguntas)
    estado["cola_preguntas"] = preguntas
    _preguntar_siguiente(estado)


def manejar_trivia(mensaje_usuario, estado, persona_str, on_token):
    """Tres estados posibles en modo trivia, en este orden de prioridad:
    1) se le mostraron opciones y este mensaje es la elección,
    2) hay una pregunta pendiente y este mensaje es la respuesta a corregir,
    3) recién entra a trivia: se muestran 5 opciones al azar del catálogo
       real para que elija, en vez de que el modelo le adivine el tema al
       primer mensaje."""
    if estado["esperando_tema"]:
        estado["esperando_tema"] = False
        tema = resolver_tema(mensaje_usuario)
        print(f"Asistente: Vamos con {tema}. Van {PREGUNTAS_POR_TANDA} preguntas seguidas.")
        _iniciar_tanda(tema, estado)
        return

    pendiente = estado["pregunta_pendiente"]
    if pendiente is not None:
        estado["pregunta_pendiente"] = None
        print("Asistente: ", end="", flush=True)
        if pendiente["respuesta_esperada"]:
            estado["total"] += 1
            acerto = evaluar_respuesta(pendiente["pregunta"], pendiente["respuesta_esperada"],
                                        mensaje_usuario)
            estado["aciertos"] += acerto
            # El corrector decide la cara (happy si acertó, sad/angry si no) —
            # se muestra ANTES de la reacción hablada, no "speaking": acá lo
            # que importa comunicar primero es el veredicto, no que está hablando.
            # Una pausa breve en 'content' antes de saltar a la emoción: mostrarla
            # de un tirón se siente demasiado instantáneo/robótico para una reacción.
            display.mostrar_cara("content")
            time.sleep(PAUSA_CAMBIO_CARA)
            display.mostrar_cara(elegir_cara(acerto))
            comentar_resultado(pendiente["pregunta"], pendiente["respuesta_esperada"],
                               mensaje_usuario, acerto, persona_str, on_token=on_token)
            # Se queda un rato en la emoción (acierto o error, mismo trato para
            # las dos) antes de pasar a 'speaking' — el asistente ya va a decir
            # algo más (la próxima pregunta, o si se acabó la tanda) así que
            # vuelve a "hablar" antes de eso, no queda pegado en la emoción.
            time.sleep(PAUSA_CAMBIO_CARA)
            display.mostrar_cara("speaking")
        else:
            # Temas como Chistes o Reconocimiento Musical no tienen una
            # respuesta correcta que corregir: no hay veredicto del corrector,
            # así que la reacción va con la cara genérica de "hablando".
            display.mostrar_cara("speaking")
            reaccionar_libre(pendiente["pregunta"], mensaje_usuario, persona_str, on_token=on_token)
        print("\n")

        # Sigue encadenando la tanda; recién cuando se acaba se vuelve a
        # preguntar qué hacer (el Router igual reclasifica el próximo mensaje,
        # pero preguntarlo explícito evita que el usuario se quede sin saber
        # qué esperar después de la última pregunta).
        if not _preguntar_siguiente(estado):
            # Ya está en 'speaking' desde la rama de arriba (o se acaba de
            # poner acá si no hubo veredicto) para el mensaje de cierre de
            # tanda; recién después de decirlo pasa a la cara de reposo.
            display.mostrar_cara("speaking")
            print("Asistente: Esas eran las 5. ¿Seguimos con más trivia, "
                  "buscamos algo en la web, o prefieres charlar?\n")
            time.sleep(PAUSA_CAMBIO_CARA)
            display.mostrar_cara("content")
        return

    opciones = ", ".join(random.sample(TEMAS_CATALOGO, 5))
    estado["esperando_tema"] = True
    print(f"Asistente: Puedes elegir entre: {opciones}.\n")


def manejar_busqueda_web(mensaje_usuario, persona_str, on_token):
    display.mostrar_cara("speaking")  # generando/revisando la respuesta (ver nota en workers.responder)
    print("Asistente: ", end="", flush=True)
    _, fue_corregida = responder_busqueda_web(mensaje_usuario, persona_str, on_token=on_token)
    print("\n")
    # happy si la respuesta ya estaba bien, sad/angry si el verificador tuvo
    # que corregirla — mismo criterio de 3 caras que usa Trivia, aplicado acá
    # al veredicto del verificador en vez del veredicto del corrector.
    display.mostrar_cara(elegir_cara_por_calidad(fue_corregida))
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")  # cara de reposo hasta el próximo turno


def manejar_chat_libre(mensaje_usuario, persona_str, on_token):
    display.mostrar_cara("speaking")  # generando/revisando la respuesta (ver nota en workers.responder)
    print("Asistente: ", end="", flush=True)
    _, fue_corregida = responder(mensaje_usuario, persona_str, on_token=on_token)
    print("\n")
    # happy si la respuesta ya estaba bien, sad/angry si el verificador tuvo
    # que corregirla — mismo criterio de 3 caras que usa Trivia, aplicado acá
    # al veredicto del verificador en vez del veredicto del corrector.
    display.mostrar_cara(elegir_cara_por_calidad(fue_corregida))
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("content")  # cara de reposo hasta el próximo turno


def main():
    display.mostrar_cara("content")  # arranca en reposo...
    time.sleep(PAUSA_CAMBIO_CARA)
    display.mostrar_cara("speaking")  # ...y recién ahora la IA "habla" (saluda, por defecto)
    print("Asistente: Hola, mi nombre es Ereberus, ¿cuál es tu nombre?")
    nombre = input("Tú: ").strip() or "amigo"
    print(f"Asistente: Mucho gusto en conocerte, {nombre}.")
    print("Asistente: Puedo hacerte trivia, buscarte algo de actualidad, o simplemente "
          "conversar — voy cambiando de modo según lo que me pidas. Escribe 'salir' "
          "para terminar.\n")
    display.mostrar_cara("content")  # cara de reposo mientras espera el primer mensaje

    persona_str = construir_personalidad()
    imprimir = lambda t: print(t, end="", flush=True)

    # pregunta_pendiente: mientras haya una, el próximo mensaje es la respuesta
    # a corregir. esperando_tema: mientras esté activo, el próximo mensaje es
    # la elección de una de las 5 opciones mostradas. cola_preguntas: el resto
    # de la tanda actual, para encadenarlas sin volver al menú de temas. En
    # los tres casos no se pasa por el Router — ya se sabe a qué worker va.
    estado = {
        "pregunta_pendiente": None, "esperando_tema": False,
        "cola_preguntas": [], "ya_usados": set(), "aciertos": 0, "total": 0,
    }

    while True:
        entrada = input("Tú: ").strip()
        if entrada.lower() in ("salir", "exit", "quit"):
            break
        if not entrada:
            continue

        if estado["pregunta_pendiente"] is not None or estado["esperando_tema"]:
            manejar_trivia(entrada, estado, persona_str, imprimir)
            continue

        ruta = enrutar_mensaje(entrada)
        if ruta == "TRIVIA":
            manejar_trivia(entrada, estado, persona_str, imprimir)
        elif ruta == "BUSQUEDA_WEB":
            manejar_busqueda_web(entrada, persona_str, imprimir)
        else:
            manejar_chat_libre(entrada, persona_str, imprimir)

    if estado["total"]:
        print(f"\nAsistente: Terminamos. Acertaste {estado['aciertos']} de {estado['total']}.")
    print("Asistente: ¡Hasta luego!")
    display.detener()


if __name__ == "__main__":
    main()
