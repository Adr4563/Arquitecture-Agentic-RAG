@echo off
rem Compila carrito_tanque_1l298n_esp32s3.ino (arduino-cli exige que la carpeta
rem se llame igual que el .ino, asi que este script copia el sketch a .build\ antes).
setlocal
set "SKETCH=carrito_tanque_1l298n_esp32s3"
set "HERE=%~dp0"
set "BUILD=%HERE%.build\%SKETCH%"
set "CLI=%HERE%..\tools\arduino-cli.exe"

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

"%CLI%" compile --fqbn esp32:esp32:esp32s3 "%BUILD%"
endlocal
