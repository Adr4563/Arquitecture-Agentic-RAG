#!/usr/bin/env bash
# Levanta el stack de Arquitecture-RAG en esta misma Raspberry Pi, con todo
# el código (Orchestrator_Management.py, preguntas.py, Agents/, Clients/,
# etc.) viviendo junto en esta misma carpeta — ver README.md para el detalle
# de cada archivo.
#
# Uso:
#   chmod +x start-all.sh
#   ./start-all.sh
#
# Orchestrator_Management.py es el único proceso propio que hace falta
# levantar: preguntas.py (búsqueda BM25 sobre preguntas.jsonl) ya no es un
# servidor aparte, corre en el mismo proceso — solo queda Ollama como
# servicio externo.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1/2: verificando Ollama =="
if ! curl -sf -m 5 http://localhost:11434/api/tags >/dev/null; then
  echo "[!] Ollama no responde en localhost:11434."
  echo "    Revisa: systemctl status ollama"
  echo "    Si no está instalado, ver README.md de esta carpeta."
  exit 1
fi

# Los 3 modelos que usa el stack, con los mismos defaults que
# Clients/Llama_Client.py (si cambian allá, cambiarlos acá). Los tres son
# fine-tunes LoRA propios: NO se bajan con `ollama pull`, se importan con
# `ollama create` desde su carpeta de entrenamiento -- por eso el mensaje de
# error apunta al README de cada una y no a un pull.
# Una sola llamada a `ollama list`, cacheada: los grep de abajo van contra
# esta variable con here-string (<<<) y NO contra un pipe. Con `set -o
# pipefail`, `ollama list | grep -q` reporta fallo aunque el grep encuentre
# el modelo: grep -q corta al primer match, ollama list recibe SIGPIPE y sale
# 141, y pipefail propaga ese 141 como fallo del pipeline entero. El
# here-string no tiene proceso upstream, así que no puede pasar.
MODELOS_INSTALADOS="$(ollama list)"

MODELOS_FALTANTES=0
MODELOS_REQUERIDOS=(
  "${CHAT_MODEL:-lora-chat}|chat_training"
  "${TRIVIA_MODEL:-lora-trivia}|trivia_training"
)
for entrada in "${MODELOS_REQUERIDOS[@]}"; do
  modelo="${entrada%%|*}"
  carpeta="${entrada##*|}"
  if ! grep -q "^${modelo}" <<<"$MODELOS_INSTALADOS"; then
    echo "[!] Falta el modelo '$modelo' — ver ${carpeta}/README.md (ollama create)"
    MODELOS_FALTANTES=1
  fi
done
if [ "$MODELOS_FALTANTES" -eq 1 ]; then
  exit 1
fi

# lora-salida-trivia va aparte y NO es fatal: si no está,
# Orchestrator_Management.py cae solo a las listas de palabras clave
# (_SALIR_TRIVIA/_TEMA_PERSONAL) para decidir si el usuario se fue de la
# trivia. Se pierde precisión, no funcionalidad -- ver _quiere_salir_trivia().
SALIDA_MODELO="${SALIDA_TRIVIA_MODEL:-lora-salida-trivia-v2}"
if ! grep -q "^${SALIDA_MODELO}" <<<"$MODELOS_INSTALADOS"; then
  echo "[i] Sin '$SALIDA_MODELO' — la salida de trivia se decide por palabras"
  echo "    clave (menos preciso). Para importarlo: salida_trivia_training/README.md"
fi

echo "OK — Ollama arriba y los modelos necesarios están importados."

echo "== 2/2: arrancando Orchestrator_Management.py =="
cd "$HERE"
python3 Orchestrator_Management.py
