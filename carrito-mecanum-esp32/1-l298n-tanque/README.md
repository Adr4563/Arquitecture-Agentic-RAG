# Variante: 1 solo L298N (modo tanque)

Versión del carrito mecanum para quien solo tiene **1 driver L298N** (4 pines
de control IN1-IN4, 2 canales) en vez de los 2 L298N (8 pines, 4 canales) del
proyecto principal en [`../2-l298n-mecanum/`](../2-l298n-mecanum/).

## Qué cambia respecto a la versión de 2 L298N

Un L298N solo trae 2 canales independientes. Con 4 motores y solo 2 canales,
los 2 motores de cada **lado** del chasis quedan cableados en paralelo al
mismo canal:

- Canal A (IN1/IN2 → OUT1/OUT2): motores **FL + BL** (lado izquierdo), en paralelo.
- Canal B (IN3/IN4 → OUT3/OUT4): motores **FR + BR** (lado derecho), en paralelo.

Al no poder mover el motor delantero distinto del trasero en un mismo lado,
se pierden los movimientos que dependen de eso:

| Movimiento | ¿Disponible en esta variante? |
|---|---|
| Adelante / Atrás | ✅ Sí |
| Giro en el sitio (tipo tanque) | ✅ Sí |
| Desplazamiento lateral (strafe) | ❌ No |
| Diagonales | ❌ No |

Ver la explicación completa (por qué) en `docs/cableado-1l298n-tanque.html`.

## ⚠️ Corriente: motores en paralelo

Al conectar 2 motores a la misma salida del L298N, **ese canal entrega la
corriente de los 2 motores a la vez**. Un L298N típico soporta ~2 A por canal
(y bastante menos en continuo por el calentamiento del chip). Antes de armar
esto:

- Revisa el consumo (stall current incluido) de tus motores y multiplícalo x2.
- Si te acercas o pasas el límite del canal, considera un disipador en el
  L298N, bajar el voltaje de alimentación, o volver a la versión de 2 L298N
  ([`../2-l298n-mecanum/`](../2-l298n-mecanum/)) que no tiene este problema
  porque cada motor tiene su propio canal.

## Archivos

| Archivo | Qué es |
|---|---|
| `carrito_tanque_1l298n_esp32s3.ino` | Sketch principal (Arduino), modo tanque. |
| `credentials.example.h` | Plantilla de credenciales WiFi (mismo formato que el proyecto principal). |
| `credentials.h` | Tus credenciales reales — **no se sube a git**. Cópialo desde el `.example` la primera vez. |
| `docs/cableado-1l298n-tanque.html` | Diagrama de cableado (ESP32-S3 ↔ 1 L298N ↔ 4 motores en paralelo por lado). Ábrelo en el navegador. |
| `docs/preview-control-tanque.html` | Vista previa standalone de la página de control (los mismos botones que sirve el `.ino`), para verla sin tener el ESP32 conectado. |
| `compilar.bat` / `subir.bat` | Scripts para compilar / compilar+flashear (ver README principal, un nivel arriba, para el detalle). |

## Cómo compilar y flashear

Desde esta carpeta:

```
compilar.bat
subir.bat COM3
```

(Ajusta `COM3` al puerto real; si no pasas puerto, usa `COM3` por defecto.)
Antes de compilar, copia `credentials.example.h` a `credentials.h` y pon tu
SSID/contraseña real — es un archivo independiente del `credentials.h` de
`2-l298n-mecanum/`.
