#!/usr/bin/env bash
# Instala todo lo necesario para arrancar Arquitecture-RAG: primero los
# paquetes de sistema (Ubuntu, via apt) que hacen falta para compilar
# llama-cpp-python, y despues los paquetes de Python (via pip).
#
# Uso:
#   chmod +x install.sh
#   ./install.sh

set -euo pipefail

sudo apt update
sudo apt install -y build-essential cmake python3-dev python3-pip git curl btop chrony

pip3 install -r requirements.txt

# Node.js + npm (via NodeSource)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# Claude Code
sudo npm install -g @anthropic-ai/claude-code

# Deja que 'user' pueda hacer auto-update sin sudo (evita el error de
# "no write permission to npm prefix" al haber instalado como root)
sudo chown -R user /usr/local/lib/node_modules
sudo chown user /usr/local/bin
