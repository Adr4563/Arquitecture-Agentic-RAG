"""
Agente verificador: revisa el texto que generó otro worker antes de
mostrárselo al usuario — para Chat libre y Búsqueda Web.

Trivia no lo necesita: ya tiene su propio veredicto objetivo en
Agent_Corrector.py (evaluar_respuesta, acierto/error contra una
respuesta_esperada conocida). Acá no hay una "respuesta correcta" contra la
cual comparar — solo coherencia: que la respuesta generada de verdad
conteste lo que se le preguntó, sin desviarse a algo incoherente o fuera de
tema.

Devuelve también si tuvo que corregir algo (fue_corregida): ese veredicto
es lo único que necesita Agent_Behavior.elegir_cara_por_calidad() para
elegir happy vs sad/angry — no hace falta una llamada aparte al modelo para
eso, el verificador ya sabe si la respuesta estaba bien o mal.

Usa VERIFICADOR_MODEL (no CHAT_MODEL): CHAT_MODEL es un fine-tune
especializado en "reaccionar como Ereberus" (ereberus-chat), y ese estilo
entrenado se imponía por sobre el formato VEREDICTO/TEXTO que este agente
necesita -- devolvía su propio texto de juicio como si fuera la respuesta
final al usuario. Ver la nota en Clients/Llama_Client.py.
"""

import re

from Clients.Llama_Client import VERIFICADOR_MODEL, generar_respuesta


def verificar_y_corregir(pregunta_original, respuesta_generada):
    """Si la respuesta es coherente y responde lo que se pidió, se devuelve
    tal cual (sin reescribirla de más) y fue_corregida=False. Si está
    incoherente, fuera de tema, o inventa algo que no viene al caso, se pide
    una versión corregida y fue_corregida=True."""
    mensajes = [
        {"role": "system", "content": (
            "Eres un revisor de calidad. Te dan una pregunta y la respuesta "
            "que generó otro asistente. Decide si la respuesta es BUENA "
            "(coherente, responde lo que se pidió, tiene sentido) o MALA "
            "(incoherente, fuera de tema, o inventa algo que no viene al caso).\n\n"
            "Responde en dos líneas exactas:\n"
            "VEREDICTO: BUENA o MALA\n"
            "TEXTO: si es BUENA, repite la respuesta tal cual, sin cambiar "
            "nada. Si es MALA, escribí una versión corregida, breve y "
            "coherente, en su lugar."
        )},
        {"role": "user", "content": (
            f"Pregunta: {pregunta_original}\n"
            f"Respuesta generada: {respuesta_generada}"
        )},
    ]
    # temperature baja: es una revisión, no una charla — poca variación entre corridas.
    salida = generar_respuesta(mensajes, temperature=0.1, max_tokens=150, modelo=VERIFICADOR_MODEL).strip()

    # El modelo no siempre usa la etiqueta "TEXTO:" tal cual se le pidió (a
    # veces pone "CORRECCIÓN:", "RESPUESTA:", etc.) — en vez de buscar esa
    # palabra exacta, se toma la línea del VEREDICTO por un lado, y CUALQUIER
    # otra línea no vacía (pelándole el "etiqueta:" que traiga adelante) por
    # el otro, así no se pierde la corrección solo por el nombre de la etiqueta.
    fue_corregida = True
    lineas_texto = []
    for linea in salida.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        if linea.upper().startswith("VEREDICTO"):
            fue_corregida = "MALA" in linea.upper()
            continue
        # Caso encontrado en producción: a veces el modelo responde SOLO
        # "BUENA"/"MALA" pelado, sin el prefijo "VEREDICTO:" pedido. Sin este
        # chequeo, esa palabra suelta se trataba como si fuera el CONTENIDO
        # de la respuesta (texto_final terminaba siendo literalmente "BUENA",
        # mostrado tal cual al usuario en vez de la respuesta real).
        if linea.upper() in ("BUENA", "MALA"):
            fue_corregida = linea.upper() == "MALA"
            continue
        # La etiqueta ("TEXTO:", "CORRECCIÓN:", etc.) solo puede venir pegada a
        # la primera línea de la corrección — a las siguientes no hay que
        # pelarles nada, o se recortaría contenido real que arranca en
        # mayúsculas seguido de ":".
        if not lineas_texto:
            linea = re.sub(r"^[A-ZÁÉÍÓÚÑ ]+:\s*", "", linea, flags=re.IGNORECASE)
        lineas_texto.append(linea)

    # Si el modelo no devolvió ninguna línea de texto (solo el VEREDICTO,
    # o salida vacía), se cae de vuelta al borrador original en vez de perderlo.
    texto_final = "\n".join(lineas_texto) if lineas_texto else respuesta_generada
    return texto_final, fue_corregida
