"""
Agente corrector: decide si la respuesta del usuario a una pregunta de Trivia
es correcta o no -- SIN LLM. Reemplaza el corrector anterior (llamada a
Ollama con temperature=0) por comparación determinista contra
respuesta_esperada, normalización + número/substring/fuzzy.

Por qué: el 74% de las respuesta_esperada del dataset son UNA sola palabra
(y otro ~27% son puramente numéricas, de los juegos de multiplicar) -- no
hace falta un modelo de lenguaje para comparar "parís" contra "es paris, la
capital". Benchmark propio contra las 187 preguntas reales del dataset (ver
../router_training/, mismo patrón: variantes correctas parafraseadas/con
typo + variantes incorrectas con la respuesta de otra pregunta al azar):
98.4% accuracy, 100% recall en incorrectas (nunca acredita una respuesta
mal), 96.8% recall en correctas (solo falla con typos agresivos en palabras
de 4-5 letras). De hecho el corrector con LLM tenía un bug real que este
enfoque no tiene: en una prueba llegó a leer "7 por 8" como "7/8" y decir
que la respuesta correcta era 0.875 -- comparar el número extraído de la
respuesta del usuario contra el número exacto de respuesta_esperada no
tiene ese riesgo, es una comparación matemática directa.

Qué cara le corresponde al veredicto ya NO se decide acá tampoco: eso lo
resuelve Agent_Behavior.py (a partir de las columnas cara_respuesta_buena/
cara_respuesta_mala de la pregunta).
"""

import difflib
import re
import unicodedata

import perf_monitor

_ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al"}
_UMBRAL_SOLAPAMIENTO = 0.6  # fracción de palabras clave de la respuesta esperada que alcanza con mencionar
_UMBRAL_FUZZY = 0.8  # similitud mínima (difflib) para tolerar typos


def _normalizar(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[¿?¡!.,;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _palabras_significativas(s):
    return [w for w in _normalizar(s).split() if w not in _ARTICULOS and len(w) > 1]


def _es_numerico(s):
    return bool(re.fullmatch(r"-?\d+([.,]\d+)?", s.strip()))


def _extraer_numeros(s):
    return [float(n.replace(",", ".")) for n in re.findall(r"-?\d+(?:[.,]\d+)?", s)]


@perf_monitor.medir("agent_corrector")
def evaluar_respuesta(esperada, respuesta_usuario):
    """Compara la respuesta del usuario con la esperada. Devuelve True/False.

    No recibe la pregunta -- el corrector anterior (LLM) sí la necesitaba
    como contexto para juzgar; esta comparación determinista no usa nada
    de eso, solo `esperada` y `respuesta_usuario` (ver el motivo abajo), así
    que se sacó el parámetro en vez de dejarlo sin usar.

    El usuario escribe en libre: "paris", "es parís", "la capital es paris"
    son todas correctas para una esperada de "París" -- por eso no es un ==
    de strings, sino normalización + substring + solapamiento de palabras
    clave (para respuestas con varios elementos) + fuzzy (para typos). Si la
    respuesta esperada es puramente numérica (juegos de multiplicar), en vez
    de todo eso se extrae el/los número(s) de la respuesta del usuario y se
    compara matemáticamente -- ahí no hay ambigüedad de texto que resolver.
    """
    esperada = (esperada or "").strip()
    respuesta_usuario = (respuesta_usuario or "").strip()
    if not esperada or not respuesta_usuario:
        return False

    if _es_numerico(esperada):
        numero_esperado = float(esperada.replace(",", "."))
        return numero_esperado in _extraer_numeros(respuesta_usuario)

    esp_norm = _normalizar(esperada)
    resp_norm = _normalizar(respuesta_usuario)

    # Substring directo: cubre "paris", "es paris", "la capital es paris".
    if esp_norm in resp_norm:
        return True

    # Multi-elemento (ej. una respuesta con varias palabras/nombres): alcanza
    # con mencionar la mayoría de las palabras clave, no todas exactas --
    # cada palabra se compara también con fuzzy, por si vino con un typo.
    palabras_esp = _palabras_significativas(esperada)
    palabras_resp = set(_palabras_significativas(respuesta_usuario))
    if palabras_esp:
        coincididas = sum(
            1 for p in palabras_esp
            if p in palabras_resp or any(
                difflib.SequenceMatcher(None, p, w).ratio() >= 0.84 for w in palabras_resp
            )
        )
        if coincididas / len(palabras_esp) >= _UMBRAL_SOLAPAMIENTO:
            return True

    # Fuzzy global como último recurso (typos en la frase completa).
    return difflib.SequenceMatcher(None, esp_norm, resp_norm).ratio() >= _UMBRAL_FUZZY


if __name__ == "__main__":
    # python Agents/Agent_Corrector.py -- reproduce el benchmark contra las
    # ~187 preguntas reales del dataset (con respuesta_esperada): por cada
    # una arma una variante CORRECTA (parafraseada/con typo) y una
    # INCORRECTA (la respuesta de OTRA pregunta al azar), y mide accuracy.
    # Referencia (commit original): 98.4% accuracy, 100% recall en
    # incorrectas, 96.8% recall en correctas.
    import json
    import os
    import random

    random.seed(0)
    RUTA_DATASET = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "base_datos", "preguntas.jsonl"
    )

    def _typo(s):
        if len(s) < 4:
            return s
        i = random.randint(1, len(s) - 2)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]  # swap 2 letras adyacentes

    datos = []
    with open(RUTA_DATASET, encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            r = (d.get("respuesta_esperada") or "").strip()
            if r:
                datos.append(r)
    todas_respuestas = datos

    envoltorios = ["{r}", "es {r}", "la respuesta es {r}", "creo que es {r}", "{r}, obvio"]
    tp = fn = tn = fp = 0
    for esperada in datos:
        variante = random.choice(envoltorios).format(r=esperada.lower())
        if not _es_numerico(esperada) and random.random() < 0.3:
            variante = _typo(variante)
        if evaluar_respuesta(esperada, variante):
            tp += 1
        else:
            fn += 1

        otra = esperada
        while otra.strip().lower() == esperada.strip().lower():
            otra = random.choice(todas_respuestas)
        if not evaluar_respuesta(esperada, otra):
            tn += 1
        else:
            fp += 1

    print(f"Recall en correctas: {tp}/{tp + fn} ({tp / (tp + fn) * 100:.1f}%)")
    print(f"Recall en incorrectas: {tn}/{tn + fp} ({tn / (tn + fp) * 100:.1f}%)")
    print(f"Accuracy total: {(tp + tn) / (tp + fn + tn + fp) * 100:.1f}%")
