@echo off
rem Compila y sube mecanum_car_esp32s3.ino al ESP32-S3.
rem Uso: subir.bat [PUERTO]   (si no pones puerto, usa COM3 por defecto)
setlocal
set "SKETCH=mecanum_car_esp32s3"
set "HERE=%~dp0"
set "BUILD=%HERE%.build\%SKETCH%"
set "CLI=%HERE%..\tools\arduino-cli.exe"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=COM3"

if not exist "%CLI%" (
  echo [!] No se encontro ..\tools\arduino-cli.exe
  echo     Descargalo de https://github.com/arduino/arduino-cli/releases
  echo     y colocalo en carrito-mecanum-esp32\tools\arduino-cli.exe
  exit /b 1
)

if not exist "%HERE%credentials.h" (
  echo [!] Falta credentials.h en esta carpeta.
  echo     Copia credentials.example.h como credentials.h y pon tu WiFi real.
  exit /b 1
)

if not exist "%BUILD%" mkdir "%BUILD%"
copy /Y "%HERE%%SKETCH%.ino" "%BUILD%\" >nul
copy /Y "%HERE%credentials.h" "%BUILD%\" >nul

echo Subiendo a %PORT% ...
"%CLI%" upload -p %PORT% --fqbn esp32:esp32:esp32s3 "%BUILD%"
endlocal
