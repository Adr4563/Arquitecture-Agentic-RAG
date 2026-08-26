# musica/

Archivos de audio para las preguntas de trivia que traen algo en la columna
`musical` de `preguntas.jsonl` (tema "Reconocimiento Musical").

A diferencia de [`../faces/`](../faces/), acá no hay mapeo: el valor de
`musical` en el dataset YA es el nombre de archivo tal cual, así que solo
tiene que existir un archivo con ese nombre en esta carpeta.

| Valor en `musical` | Archivo esperado acá |
|---|---|
| `danza-kuduro.mp3` | `danza-kuduro.mp3` |
| `more-than-words-heaven.mp3` | `more-than-words-heaven.mp3` |

`expresar_musica()` en `Agents/Agent_Behavior.py` reproduce el audio de
verdad vía `Clients/Musica_Client.py` (mpv, recortado a
`Musica_Client.REPRODUCCION_MAX_SEG`). Si el hardware no responde (mpv no
instalado, archivo inexistente), la llamada falla en silencio — ver el
comentario en `Agent_Behavior.py`.
