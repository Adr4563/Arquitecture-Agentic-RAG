#!/usr/bin/env bash
# Instala las dependencias necesarias para correr los scripts de ai-camera
# (hdmi_live.py, live_stream.py, snapshot_deteccion.py): deteccion de objetos
# en vivo con la camara IMX500 de Raspberry Pi.
#
# Uso:
#   chmod +x install.sh
#   ./install.sh

set -euo pipefail

# --- Paquetes de sistema ---
# picamera2 depende de las bindings de libcamera (no se pueden instalar
# solo con pip); imx500-all trae el firmware y los modelos .rpk que usan
# estos scripts (/usr/share/imx500-models/*.rpk).
sudo apt update
sudo apt install -y python3-libcamera python3-kms++ libcap-dev imx500-all

# --- Paquetes de Python ---
# --break-system-packages porque Ubuntu/Debian recientes bloquean pip
# instalando fuera de un venv (PEP 668); necesitamos instalar a nivel de
# sistema para que picamera2 vea las bindings de libcamera del paso anterior.
sudo pip3 install --break-system-packages -r requirements.txt
