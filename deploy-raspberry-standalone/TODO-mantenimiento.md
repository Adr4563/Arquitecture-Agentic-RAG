# TODO de mantenimiento

Pendientes detectados en una revisión (2026-08-26) que no se resolvieron
solos por ser cambios de comportamiento, no de documentación:

## `router_modelo.joblib` serializado con una versión vieja de scikit-learn

`Agents/router_modelo.joblib` fue guardado con scikit-learn 1.8.0. El
entorno de esta Pi ya tiene 1.9.0 (fijado en `requirements.txt`), así que al
cargarlo aparece:

```
InconsistentVersionWarning: Trying to unpickle estimator TfidfTransformer
from version 1.8.0 when using version 1.9.0.
```

Funciona bien hoy, pero es frágil ante un futuro upgrade de scikit-learn.
Para limpiarlo del todo (no solo fijar la versión en `requirements.txt`,
que ya está hecho):

```bash
cd router_training
python entrenar_router.py
```

Esto reentrena y sobreescribe `Agents/router_modelo.joblib` con la versión
de scikit-learn actualmente instalada. Es determinístico a partir de
`dataset_router.jsonl` (que sí está en git), así que es seguro repetirlo —
pero se dejó pendiente en vez de aplicarlo directo por tratarse del modelo
que corre en producción en el robot.

## `OLLAMA_KEEP_ALIVE`: se probó bajarlo a 2 min, se revirtió al default (5 min)

Con 3 modelos separados por rol (`CHAT_MODEL`=ereberus-chat,
`TRIVIA_MODEL`=ereberus-trivia, `VERIFICADOR_MODEL`=qwen2.5:0.5b), Ollama
puede mantener los 3 cargados en RAM a la vez (~1.6GB juntos) con su
`keep_alive` default de 5 minutos. Se probó bajarlo a 2 minutos a nivel de
servicio systemd (no hay forma de pasar `keep_alive` por request en el
endpoint `/v1/chat/completions` que usa `Clients/Llama_Client.py`, es
compatible OpenAI, no la API nativa de Ollama) con un drop-in en
`/etc/systemd/system/ollama.service.d/override.conf`.

Validado en vivo: SÍ funcionaba (el modelo se descargaba a los ~2 min), pero
el turno siguiente a una pausa >2 min tardaba ~28s en vez de los ~10-12s
normales (recarga desde disco antes de generar la reacción) -- notorio en
uso real. Como esta Pi no tiene problema de RAM hoy (5.9GB libres de sobra
incluso con los 3 modelos cargados), no valía la pena pagar esa latencia
por liberar memoria que no hacía falta liberar -- **se revirtió** al
default de Ollama (sin override, sin `OLLAMA_KEEP_ALIVE` seteada).

Si en algún momento SÍ hace falta (memoria más ajustada, u otro proceso
pesado corriendo a la vez), el override queda documentado acá para
reaplicarlo:

```
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'EOF'
[Service]
Environment="OLLAMA_KEEP_ALIVE=2m"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Ojo: esto es configuración de sistema, no de código -- **no se sube a git
ni se replica sola** si se reinstala esta Pi desde cero.

## Ya corregido en esta misma revisión

- `llama3.2:3b-q4s` (tag inexistente en Ollama) → `llama3.2:3b` en
  `README.md` y `Clients/Llama_Client.py`. El tag real y ya descargado en
  esta Pi es `llama3.2:3b` (`ollama list`).
- `musica/README.md` actualizado: `Agent_Behavior.py` (`expresar_musica()`)
  sí reproduce audio real (ya no es solo un log), y la tabla de valores
  ahora lista los archivos reales del dataset (`danza-kuduro.mp3`,
  `more-than-words-heaven.mp3`) en vez del ejemplo obsoleto `musica mario`.
- Comentario en `preguntas.py::pregunta_aleatoria()` corregido: el dataset
  hoy tiene 248 preguntas, 187 con `respuesta_esperada` (~75%, no "casi la
  mitad / 111 de 246").
