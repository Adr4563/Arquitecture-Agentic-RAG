# Carrito Mecanum — ESP32-S3

Carrito con 4 ruedas mecanum sobre placa **ESP32-S3-DevKitC-1**. Basado en
el proyecto de
[robotlk.com](https://robotlk.com/web-controlled-mecanum-wheel-robot-car-using-esp32/),
adaptado para el S3 (el original es para ESP32 clásico).

Hay 2 versiones de firmware, según cuántos drivers **L298N** tengas — cada
una vive en su propia carpeta, con su propio `.ino`, `README.md`, diagrama
de cableado y scripts de compilar/subir:

## Versiones

| Carpeta | Drivers L298N | Movimientos disponibles | Control |
|---|---|---|---|
| [`2-l298n-mecanum/`](2-l298n-mecanum/) | 2 (4 canales) | Las 9: adelante, atrás, strafe izq/der, rotar izq/der, 4 diagonales | **Serial (USB) directo** contra la Raspberry Pi — ver el README de esa carpeta |
| [`1-l298n-tanque/`](1-l298n-tanque/) | 1 (2 canales) | Solo 4: adelante, atrás, rotar izq/der (modo tanque — sin strafe ni diagonales) | WiFi/HTTP desde un navegador |

`1-l298n-tanque/` existe porque con 1 solo L298N (2 canales) los motores
quedan agrupados en paralelo por lado del chasis (FL+BL en un canal, FR+BR
en el otro) — eso impide moverlos de forma independiente, así que se pierde
el desplazamiento lateral y las diagonales. Ver el `README.md` de esa
carpeta para el detalle completo.

## Herramientas compartidas

`tools/arduino-cli.exe` — usado por los scripts `compilar.bat`/`subir.bat`
de **ambas** versiones. No se sube a git (pesa ~38MB, ver `.gitignore`); si
falta, descárgalo de
[github.com/arduino/arduino-cli/releases](https://github.com/arduino/arduino-cli/releases)
y colócalo ahí como `arduino-cli.exe`.

## Cómo compilar y flashear (cualquiera de las 2 versiones)

**Desde Windows**, con `compilar.bat`/`subir.bat` de la carpeta de la
versión que quieras usar:

```
compilar.bat
subir.bat COM3
```

(Ajusta `COM3` al puerto real — si no pasas puerto, usan `COM3` por
defecto. Ambos scripts copian el sketch a una carpeta `.build\` temporal
antes de compilar, porque `arduino-cli` exige que el nombre de la carpeta
coincida exactamente con el del `.ino`; `.build\` está en `.gitignore`.)

**Desde Linux** (ej. la propia Raspberry Pi):

Primero, instalar `arduino-cli` y el core `esp32:esp32` — **no vienen con
el repo, hay que instalarlos una vez por máquina**:

```bash
# 1. arduino-cli (deja el binario en ~/.local/bin/arduino-cli)
mkdir -p ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh \
  | BINDIR=~/.local/bin sh
export PATH="$HOME/.local/bin:$PATH"   # agregalo a ~/.bashrc para que persista

# 2. Core del ESP32 (~7GB en ~/.arduino15/, toolchains incluidas — tarda
#    varios minutos la primera vez)
arduino-cli config init
arduino-cli config set board_manager.additional_urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index --additional-urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core install esp32:esp32
```

También hace falta estar en el grupo `dialout` para poder escribir al
puerto serial: `sudo usermod -aG dialout $USER` (reiniciar sesión para que
aplique).

Con eso instalado, compilar y flashear:

```
arduino-cli compile --fqbn esp32:esp32:esp32s3 .
arduino-cli upload -p /dev/ttyACM0 --fqbn esp32:esp32:esp32s3 .
```

(desde la carpeta de la versión que quieras usar; ajustá el puerto si no es
`/dev/ttyACM0` — `ls /dev/ttyACM* /dev/ttyUSB*` para verlo).

`credentials.h` (copiado desde `credentials.example.h`, con tu SSID/
contraseña real) solo hace falta para **`1-l298n-tanque/`**, que sigue por
WiFi — `2-l298n-mecanum/` ya no lo usa (ver su propio README, sección
"Control: Serial (USB), no WiFi").

## Placa

**ESP32-S3-DevKitC-1.** Pines de motor usados: `4, 5, 6, 7` (+ `15, 16, 17, 18`
en la versión de 2 L298N) — se evitan strapping (`0,3,45,46`), USB nativo
(`19,20`), UART consola (`43,44`) y flash/PSRAM (`26`–`32`), que en el S3
están reservados o no expuestos.
