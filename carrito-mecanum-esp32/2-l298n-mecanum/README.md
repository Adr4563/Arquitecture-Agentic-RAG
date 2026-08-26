# Versión completa — 2 L298N

Firmware con las **9 funciones de movimiento** (adelante, atrás, strafe
izquierda/derecha, rotar izquierda/derecha, y 4 diagonales), usando **2
drivers L298N** — uno para las ruedas delanteras (FL/FR) y otro para las
traseras (BL/BR), cada motor con su propio canal independiente.

Si solo tienes 1 L298N, usa la variante en
[`../1-l298n-tanque/`](../1-l298n-tanque/) en su lugar (esa sigue por
WiFi/HTTP — el cambio a Serial de más abajo es solo de esta versión).

## Control: Serial (USB), no WiFi

El ESP32-S3 va conectado por cable USB directo a la Raspberry Pi que corre
`Orchestrator_Management.py` (antes era WiFi/HTTP, con el ESP32 sirviendo una
página de control) — al estar cableado directo ya no hace falta radio de por
medio, ni depender de la LAN, ni de que el ESP32 tenga IP.

**Protocolo**: un comando de texto por línea (`\n`), por Serial a 115200
baudios — `F`, `B`, `SL`, `SR`, `RL`, `RR`, `FL`, `FR`, `BL`, `BR`, o `S`
(parar; también cualquier texto no reconocido para). Mismos códigos que
antes iban como `?move=<código>` por HTTP, solo cambió el transporte. El
watchdog de 500ms se mantiene igual: si no llega otro comando antes de que
pasen esos 500ms, el firmware frena los motores solo.

El cliente del lado Python es `Clients/Carrito_Client.py` en
`deploy-raspberry-standalone/` — abre `/dev/ttyACM0` (configurable con
`CARRITO_PORT`) con `pyserial` y mantiene la conexión abierta entre
llamadas.

**Ya no hace falta `credentials.h`/WiFi para esta versión** — el `.ino` no
incluye `WiFi.h` ni `WebServer.h`. `credentials.example.h` queda en el
repo como referencia por si en algún momento se vuelve a un control
inalámbrico, pero no se usa.

## Archivos

| Archivo | Qué es |
|---|---|
| `mecanum_car_esp32s3.ino` | Sketch principal (Arduino) — lee comandos por Serial. |
| `credentials.example.h` | Plantilla de credenciales WiFi, sin uso actual (ver arriba). |
| `docs/cableado-mecanum-s3.html` | Diagrama de cableado (ESP32-S3 ↔ 2×L298N ↔ motores ↔ batería). Ábrelo en el navegador. |
| `compilar.bat` / `subir.bat` | Scripts para compilar / compilar+flashear desde Windows (ver README principal, un nivel arriba, y la sección de abajo para flashear desde Linux). |

## Hecho hasta ahora

- [x] Pines remapeados de ESP32 clásico → ESP32-S3 (el original usaba GPIO 26-33,
      reservados a flash/PSRAM en el S3; ahora usa 4, 5, 6, 7, 15, 16, 17, 18).
- [x] Corregidas las 4 funciones de movimiento diagonal (`forwardLeft`,
      `forwardRight`, `backLeft`, `backRight`) — en el código original estaban
      cruzadas izquierda↔derecha respecto a la tabla de movimientos del propio
      tutorial.
- [x] Watchdog de seguridad: si no llega ningún comando en 500 ms (cable
      desconectado, proceso Python colgado, etc.) el carrito se detiene solo.
- [x] Compilado y flasheado con éxito por USB (`arduino-cli`, FQBN
      `esp32:esp32:esp32s3`).
- [x] Diagnosticado y resuelto un bucle de brownout al arrancar — no era el
      cableado ni el código, era el cable/puerto USB (probar con un puerto
      trasero de la torre lo resolvió).
- [x] Cambiado de WiFi/HTTP a Serial (USB) directo contra la Raspberry Pi —
      se sacó `WiFi.h`/`WebServer.h`/`credentials.h`, el `.ino` ahora lee
      comandos de texto por línea desde `Serial`. Ver `Carrito_Client.py`.

## Pendiente / por revisar

- [ ] **Probar cada botón con el carrito levantado del piso** para confirmar
      que ninguna dirección quedó invertida tras el fix de las diagonales.
- [ ] Confirmar que el diagrama de cableado (`docs/cableado-mecanum-s3.html`)
      coincide con el cableado físico real.
- [ ] Evaluar reemplazar el 5V del ESP32 (actualmente desde el L298N Front)
      por un regulador dedicado, para no depender del regulador on-board del
      L298N.

## Cómo compilar y flashear

**Desde la Raspberry Pi (Linux)**, con `arduino-cli` instalado y el core
`esp32:esp32` agregado (ver "Cómo compilar y flashear" en el README
principal, un nivel arriba, para los comandos de instalación si todavía no
los tenés — no vienen con el repo):

```
arduino-cli compile --fqbn esp32:esp32:esp32s3 .
arduino-cli upload -p /dev/ttyACM0 --fqbn esp32:esp32:esp32s3 .
```

(Ajusta `/dev/ttyACM0` si el puerto es otro — `ls /dev/ttyACM* /dev/ttyUSB*`
para verlo. El usuario que corre esto necesita estar en el grupo `dialout`:
`sudo usermod -aG dialout $USER`, reiniciar sesión para que aplique.)

**Desde Windows**, con `arduino-cli.exe` (ver README principal, un nivel
arriba):

```
compilar.bat
subir.bat COM3
```

(Ajusta `COM3` al puerto real. Si el ESP32-S3 se reinicia en bucle por
brownout apenas conectas USB, prueba otro cable/puerto — ver nota en
"Hecho hasta ahora".)

Ya no hace falta `credentials.h` para compilar esta versión (ver la
sección de arriba).
