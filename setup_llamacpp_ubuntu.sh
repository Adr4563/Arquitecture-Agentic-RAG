#!/usr/bin/env bash
# Instala llama.cpp (sin Ollama, sin systemd) y levanta dos servidores:
# uno para embeddings (local, este equipo) y uno para chat, cada uno
# sirviendo un .gguf publicado en los Releases del repo.
#
# Uso:
#   chmod +x setup_llamacpp_ubuntu.sh
#   ./setup_llamacpp_ubuntu.sh

set -euo pipefail

REPO_URL="https://github.com/Adr4563/Arquitecture-RAG"

if [ ! -d llama.cpp ]; then
    git clone https://github.com/ggml-org/llama.cpp
fi
cmake -B llama.cpp/build llama.cpp
cmake --build llama.cpp/build --config Release -j"$(nproc)"

mkdir -p models
curl -L -o models/llama3.2-3b-q4_K_S.gguf \
  "$REPO_URL/releases/download/llama3.2-3b-q4_K_S/llama3.2-3b-q4_K_S.gguf"
curl -L -o models/qwen2.5-0.5b.gguf \
  "$REPO_URL/releases/download/qwen2.5-0.5b/qwen2.5-0.5b.gguf"
curl -L -o models/nomic-embed-text.gguf \
  "$REPO_URL/releases/download/nomic-embed-text/nomic-embed-text.gguf"

echo "Modelos listos en ./models/"
echo ""
echo "Levanta el servidor de embeddings (puerto 8081, lo usa chat.py siempre):"
echo "  ./llama.cpp/build/bin/llama-server -m models/nomic-embed-text.gguf --embedding --port 8081 &"
echo ""
echo "Levanta el servidor de chat (puerto 8080; elige uno de los dos segun tu hardware):"
echo "  ./llama.cpp/build/bin/llama-server -m models/llama3.2-3b-q4_K_S.gguf --port 8080 -c 2048"
echo "  ./llama.cpp/build/bin/llama-server -m models/qwen2.5-0.5b.gguf --port 8080 -c 2048"
echo ""
echo "Luego corre 'python chat.py' apuntando GPU_SERVER_HOST a donde quede el de chat."
