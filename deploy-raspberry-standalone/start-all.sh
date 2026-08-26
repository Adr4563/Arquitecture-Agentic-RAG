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

MODELOS_FALTANTES=0
for modelo in "${CHAT_MODEL:-llama3.2:3b-q4s}" qwen2.5:0.5b; do
  if ! ollama list | grep -q "^${modelo}"; then
    echo "[!] Falta el modelo '$modelo' — corre: ollama pull $modelo"
    MODELOS_FALTANTES=1
  fi
done
if [ "$MODELOS_FALTANTES" -eq 1 ]; then
  exit 1
fi
echo "OK — Ollama arriba y los modelos necesarios están descargados."

echo "== 2/2: arrancando Orchestrator_Management.py =="
cd "$HERE"
python3 Orchestrator_Management.py
