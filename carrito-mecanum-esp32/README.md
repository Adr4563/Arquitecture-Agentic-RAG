# Carrito Mecanum — ESP32-S3

Carrito con 4 ruedas mecanum controlado por WiFi desde un navegador, sobre
placa **ESP32-S3-DevKitC-1**. Basado en el proyecto de
[robotlk.com](https://robotlk.com/web-controlled-mecanum-wheel-robot-car-using-esp32/),
adaptado para el S3 (el original es para ESP32 clásico).

Hay 2 versiones de firmware, según cuántos drivers **L298N** tengas — cada
una vive en su propia carpeta, con su propio `.ino`, `README.md`, diagrama
de cableado y scripts de compilar/subir:

## Versiones

| Carpeta | Drivers L298N | Movimientos disponibles |
|---|---|---|
| [`2-l298n-mecanum/`](2-l298n-mecanum/) | 2 (4 canales) | Las 9: adelante, atrás, strafe izq/der, rotar izq/der, 4 diagonales |
| [`1-l298n-tanque/`](1-l298n-tanque/) | 1 (2 canales) | Solo 4: adelante, atrás, rotar izq/der (modo tanque — sin strafe ni diagonales) |

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

Desde la carpeta de la versión que quieras usar:

```
compilar.bat
subir.bat COM3
```

(Ajusta `COM3` al puerto real — si no pasas puerto, usan `COM3` por
defecto. Ambos scripts copian el sketch a una carpeta `.build\` temporal
antes de compilar, porque `arduino-cli` exige que el nombre de la carpeta
coincida exactamente con el del `.ino`; `.build\` está en `.gitignore`.)

Antes de compilar, asegúrate de tener `credentials.h` en esa misma carpeta
(copia `credentials.example.h` y pon tu SSID/contraseña real — cada versión
tiene su propio `credentials.h` independiente, ninguno se sube a git).

## Placa

**ESP32-S3-DevKitC-1.** Pines de motor usados: `4, 5, 6, 7` (+ `15, 16, 17, 18`
en la versión de 2 L298N) — se evitan strapping (`0,3,45,46`), USB nativo
(`19,20`), UART consola (`43,44`) y flash/PSRAM (`26`–`32`), que en el S3
están reservados o no expuestos.
