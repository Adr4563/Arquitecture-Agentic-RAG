# Carrito Mecanum — ESP32-S3

Carrito con 4 ruedas mecanum controlado por WiFi desde un navegador, sobre
placa **ESP32-S3-DevKitC-1** y dos drivers **L298N**. Basado en el proyecto
de [robotlk.com](https://robotlk.com/web-controlled-mecanum-wheel-robot-car-using-esp32/),
adaptado para el S3 (el original es para ESP32 clásico).

## Archivos

| Archivo | Qué es |
|---|---|
| `mecanum_car_esp32s3.ino` | Sketch principal (Arduino). |
| `credentials.example.h` | Plantilla de credenciales WiFi. |
| `credentials.h` | Tus credenciales reales — **no se sube a git** (ver `.gitignore`). Cópialo desde el `.example` la primera vez. |
| `docs/cableado-mecanum-s3.html` | Diagrama de cableado (ESP32-S3 ↔ L298N ↔ motores ↔ batería). Ábrelo en el navegador. |

## Hecho hasta ahora

- [x] Pines remapeados de ESP32 clásico → ESP32-S3 (el original usaba GPIO 26-33,
      reservados a flash/PSRAM en el S3; ahora usa 4, 5, 6, 7, 15, 16, 17, 18).
- [x] Corregidas las 4 funciones de movimiento diagonal (`forwardLeft`,
      `forwardRight`, `backLeft`, `backRight`) — en el código original estaban
      cruzadas izquierda↔derecha respecto a la tabla de movimientos del propio
      tutorial.
- [x] Watchdog de seguridad: si no llega ningún comando en 500 ms (WiFi caído,
      pestaña cerrada, etc.) el carrito se detiene solo.
- [x] Cambiado de modo Access Point (`MecanumCar` propia) a modo estación:
      el ESP32 se conecta a la red de casa, así no hay que cambiar de WiFi en
      el celular/PC para controlarlo.
- [x] Credenciales WiFi movidas a `credentials.h` (fuera de git) en vez de
      estar hardcodeadas en el `.ino`.
- [x] Compilado y flasheado con éxito por USB (`arduino-cli`, FQBN
      `esp32:esp32:esp32s3`).
- [x] Diagnosticado y resuelto un bucle de brownout al arrancar — no era el
      cableado ni el código, era el cable/puerto USB (probar con un puerto
      trasero de la torre lo resolvió).
- [x] Probado: el carrito se conecta a la red de casa y sirve la página de
      control (botones F, B, S-L, S-R, R-L, R-R, diagonales, STOP).

## Pendiente / por revisar

- [ ] **Probar cada botón con el carrito levantado del piso** para confirmar
      que ninguna dirección quedó invertida tras el fix de las diagonales.
- [ ] Confirmar que el diagrama de cableado (`docs/cableado-mecanum-s3.html`)
      coincide con el cableado físico real.
- [ ] Decidir si se fija una IP estática para el ESP32 en el router (para que
      no cambie la IP cada vez que se reconecta).
- [ ] Evaluar reemplazar el 5V del ESP32 (actualmente desde el L298N Front)
      por un regulador dedicado, para no depender del regulador on-board del
      L298N.

## Cómo compilar y flashear

Con `arduino-cli` instalado y el core `esp32:esp32` (paquete Espressif):

```
arduino-cli compile --fqbn esp32:esp32:esp32s3 .
arduino-cli upload -p COM3 --fqbn esp32:esp32:esp32s3 .
```

(Ajusta `COM3` al puerto real. Si el ESP32-S3 se reinicia en bucle por
brownout apenas conectas USB, prueba otro cable/puerto — ver nota en
"Hecho hasta ahora").

Antes de compilar, asegúrate de tener `credentials.h` (copia
`credentials.example.h` y pon tu SSID/contraseña real).
