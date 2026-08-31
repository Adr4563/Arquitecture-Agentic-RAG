# -*- coding: utf-8 -*-
"""
Busqueda en internet para la ruta BUSQUEDA_WEB.

Esta ruta existio antes y se elimino en d8b092c por alcance ("el proyecto se
queda con Trivia y Chat libre"), no porque fallara. Vuelve a pedido del
usuario, con el buscador ENCHUFABLE:

    export BUSCADOR=brave     BRAVE_API_KEY=...   # (default) ~1000/mes gratis
    export BUSCADOR=serpapi   SERPAPI_KEY=...     # resultados de Google, 250/mes

LOS DOS NECESITAN API KEY. Sin key, buscar() devuelve None y el robot dice
que no encontro nada -- no se cae, pero tampoco busca. Sacar la key es el
primer paso para que esta ruta sirva.

Por que no Google directo: la Custom Search JSON API de Google esta CERRADA a
clientes nuevos y se discontinua el 1 de enero de 2027 (verificado en su
documentacion). SerpAPI es la via viable para resultados de Google.

Por que no DuckDuckGo: se probo (su Instant Answer API no pide key) y se
descarto a pedido del usuario. Ademas tenia un limite serio medido aca: solo
responde a nombres propios y en INGLES -- "Albert Einstein" y "Peru" traian
texto, pero "fotosintesis", "dinosaurio" y "Japon" devolvian vacio.

Todas las funciones devuelven None ante cualquier problema -- sin key, sin
internet, sin resultados, timeout. El caller no distingue el motivo: le
alcanza con saber que no hay nada que decir. Un fallo de red nunca corta el
turno, mismo criterio que Carrito_Client y Musica_Client.
"""
import os

import requests

BUSCADOR = os.environ.get("BUSCADOR", "brave").strip().lower()

# 6s: por encima de esto el chico ya se aburrio. Preferimos "no encontre
# nada" rapido a una respuesta correcta que llega tarde.
TIMEOUT = float(os.environ.get("BUSQUEDA_TIMEOUT", "6"))

# Cuantos fragmentos se juntan. Mas de 2 no aporta: el robot lee la respuesta
# en voz alta y bloquea hasta terminar, asi que un texto largo son segundos
# de silencio.
MAX_RESULTADOS = int(os.environ.get("BUSQUEDA_MAX_RESULTADOS", "2"))


def _brave(consulta):
    clave = os.environ.get("BRAVE_API_KEY", "").strip()
    if not clave:
        print("[busqueda] falta BRAVE_API_KEY")
        return None
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": consulta, "count": MAX_RESULTADOS, "country": "AR",
                "search_lang": "es"},
        headers={"Accept": "application/json", "X-Subscription-Token": clave},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    resultados = (r.json().get("web") or {}).get("results") or []
    trozos = [x.get("description", "") for x in resultados[:MAX_RESULTADOS]]
    return " ".join(t for t in trozos if t) or None


def _serpapi(consulta):
    clave = os.environ.get("SERPAPI_KEY", "").strip()
    if not clave:
        print("[busqueda] falta SERPAPI_KEY")
        return None
    r = requests.get(
        "https://serpapi.com/search",
        params={"q": consulta, "api_key": clave, "hl": "es", "gl": "ar",
                "num": MAX_RESULTADOS},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    d = r.json()
    # answer_box es la respuesta directa que Google muestra arriba de todo:
    # cuando existe es mucho mejor que un snippet suelto.
    caja = d.get("answer_box") or {}
    directa = caja.get("answer") or caja.get("snippet")
    if directa:
        return directa
    trozos = [x.get("snippet", "") for x in (d.get("organic_results") or [])[:MAX_RESULTADOS]]
    return " ".join(t for t in trozos if t) or None


_BUSCADORES = {"brave": _brave, "serpapi": _serpapi}

# Palabras de envoltorio conversacional que no aportan a la busqueda. Se
# sacan del PRINCIPIO del mensaje, no del medio: "que es la fotosintesis" ->
# "la fotosintesis", pero "quien invento el telefono" conserva "invento".
_PREFIJOS = [
    "busca en internet", "busca en google", "buscame", "busca", "buscá",
    "googlea", "google", "averigua", "averiguá", "investiga",
    "decime", "dime", "contame", "cuentame", "sabes", "sabés",
    "me podes decir", "me podés decir", "podes decirme", "podés decirme",
    "quiero saber", "necesito saber",
    # Pedidos indirectos: "puedes traer informacion de X" dejaba el
    # envoltorio entero como termino de busqueda.
    "puedes traer informacion de", "podes traer informacion de",
    "podés traer información de", "puedes traer informacion",
    "traeme informacion de", "traeme informacion", "traeme datos de",
    "me traes informacion de", "me consigues informacion de",
    "puedes buscar informacion de", "puedes buscar informacion sobre",
    "podes buscar informacion de", "buscame informacion de",
    "que sabes de", "que sabes sobre", "qué sabés de", "qué sabés sobre",
    "contame de", "hablame de", "háblame de",
]


# Conectores que quedan colgando DESPUES de sacar el verbo: "busca sobre
# Peru" -> "sobre peru", que no matchea nada. Van aparte de _PREFIJOS porque
# solo se sacan cuando encabezan la frase ya recortada: "el clima de lima"
# tiene que conservar su "de".
_CONECTORES = ["sobre", "acerca de", "acerca", "respecto a", "info de",
               "informacion de", "información de", "datos de"]


def extraer_termino(mensaje_usuario):
    """Saca el envoltorio conversacional del mensaje. Sin LLM a proposito:
    la version vieja usaba VERIFICADOR_MODEL para esto, un modelo que ya no
    existe en el proyecto, y le sumaba segundos al turno. Un recorte de
    prefijos alcanza para lo que se le pide a este robot."""
    t = (mensaje_usuario or "").strip().lower().strip("¿?¡!.,")
    # 3 pasadas: "busca en google sobre X" necesita sacar prefijo, luego
    # conector, y todavia puede quedar otro conector ("sobre el tema de X").
    for _ in range(3):
        for p in _PREFIJOS + _CONECTORES:
            if t.startswith(p + " "):
                t = t[len(p) + 1:].strip()
                break
        else:
            break
    return t.strip("¿?¡!.,").strip() or (mensaje_usuario or "").strip()


def buscar(mensaje_usuario):
    """(texto_encontrado, termino_buscado). texto None si no hubo resultado.

    El termino se devuelve para poder decirle al usuario QUE se busco cuando
    no se encontro nada: "no encontre nada sobre X" es mucho mas util que un
    "no encontre nada" a secas."""
    termino = extraer_termino(mensaje_usuario)
    motor = _BUSCADORES.get(BUSCADOR)
    if motor is None:
        print(f"[busqueda] BUSCADOR={BUSCADOR!r} desconocido "
              f"(usa: {', '.join(sorted(_BUSCADORES))})")
        return None, termino
    try:
        return motor(termino), termino
    except requests.RequestException as e:
        print(f"[busqueda] fallo la consulta ({type(e).__name__})")
        return None, termino
    except (ValueError, KeyError, TypeError) as e:
        # respuesta con formato inesperado: el servicio cambio su JSON
        print(f"[busqueda] respuesta inesperada del buscador ({type(e).__name__})")
        return None, termino
