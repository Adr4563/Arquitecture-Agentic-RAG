# musica/

Archivos de audio para las preguntas de trivia que traen algo en la columna
`musical` de `preguntas.jsonl` (ej. las de tema "Reconocimiento Musical",
que hoy tienen el valor `"musica mario"`).

Mismo criterio que [`../faces/`](../faces/) con las caras: un archivo por
valor posible de la columna, nombrado en snake_case (espacios -> `_`), para
poder mapear el valor tal cual viene del dataset al nombre del archivo.

| Valor en `musical` | Archivo esperado acá |
|---|---|
| `musica mario` | `musica_mario.mp3` |

Por ahora `reactor.py` (`expresar_musica()`) solo loguea el valor por
consola — todavía no reproduce nada. Cuando se conecte la reproducción real
(bocina en la Raspberry Pi), este es el lugar de donde leerlo.
