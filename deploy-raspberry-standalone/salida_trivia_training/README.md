# Entrenar `lora-salida-trivia`

Fine-tuning LoRA de `Qwen2.5-0.5B-Instruct` para una sola decisión binaria:
dado el turno de Trivia pendiente y el mensaje del usuario, ¿es un intento
de **RESPUESTA**, o el usuario se fue a hablar de otra cosa (**SALIR** —
tema personal, problema, charla sin relación, o pide explícitamente
parar/cambiar de tema)?

Por qué existe: se probó primero con `qwen2.5:0.5b` sin reentrenar (solo
few-shot) y falló -- se equivocó en 4 de 5 casos de prueba real. El modelo
grande (`llama3.2:3b-q4s`) sí acertó 5/5, pero se descartó por RAM/latencia
en la Raspberry Pi. Este fine-tune busca esa misma precisión en el tamaño
chico (~397MB, igual que `CHAT_MODEL`/`TRIVIA_MODEL`).

Esto es **ENTRENAMIENTO** -- corre en una PC de desarrollo (con CPU alcanza,
no hace falta GPU), NO en la Raspberry Pi. El resultado se copia a la Pi
como un modelo más de Ollama.

## 0. Requisitos

```bash
pip install torch transformers peft datasets
```

Necesitás también `llama.cpp` clonado (para el paso de conversión a GGUF)
y el binario `llama-quantize` (release prebuilt, no hace falta compilar) --
mismo requisito que `personalidad_training/`, `chat_training/` y
`trivia_training/`, que ya deberían tenerlo instalado si veniías de ahí.

Por default los archivos intermedios y el caché de Hugging Face van a
`D:/...` (para no llenar `C:`) -- si tu máquina no tiene una unidad `D:`,
sobreescribí las variables de entorno de cada paso con una ruta que sí
exista.

## 1. Generar el dataset (ya está hecho, opcional repetirlo)

`dataset_salida.jsonl` ya viene armado en esta carpeta (542 ejemplos: 222
RESPUESTA / 320 SALIR), generado a partir de `../base_datos/preguntas.jsonl`
y `../router_training/dataset_router.jsonl`. Solo hace falta correr esto de
nuevo si querés regenerarlo (ej. después de cambiar el dataset de trivia):

```bash
python generar_dataset.py
```

## 2. Entrenar

```bash
export PERSONALIDAD_HF_CACHE=D:/hf-cache          # cache de Hugging Face, opcional
python entrenar_salida.py
```

CPU, puede tardar **2-3 horas** (183 pasos, ~183 * 60-70s medido en una
laptop sin GPU). Corré esto solo, no al mismo tiempo que otro entrenamiento
LoRA en la misma máquina -- compiten por CPU y los dos tardan mucho más.

Al terminar deja el modelo fusionado en `D:/lora-salida-trivia-merged`
(configurable con `SALIDA_TRIVIA_SALIDA_MERGED`).

## 3. Convertir a GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py D:/lora-salida-trivia-merged \
    --outfile lora-salida-trivia-f16.gguf --outtype f16
```

## 4. Cuantizar

```bash
llama-quantize lora-salida-trivia-f16.gguf \
    lora-salida-trivia-q4_k_m.gguf Q4_K_M
```

## 5. Importar a Ollama

Desde esta carpeta (ya tiene el `Modelfile` armado, apuntando al `.gguf`
cuantizado del paso anterior):

```bash
ollama create lora-salida-trivia -f Modelfile
```

## 6. Probarlo

```bash
python probar_salida.py
```

Corre 14 casos reales (respuestas cortas, opiniones largas legítimas de
dilemas, frases explícitas de salir, temas personales, charla sin relación)
contra el modelo recién importado y reporta cuántos acertó. Sirve también
para comparar contra otro modelo sin reentrenar nada (`--modelo
qwen2.5:0.5b`), o contra un Ollama remoto (`--host http://<ip>:11434`).

## 7. Activarlo en Ereberus/Lora

En `deploy-raspberry-standalone` (o en la Raspberry Pi, una vez que copiaste
el modelo ahí):

```bash
export SALIDA_TRIVIA_MODEL=lora-salida-trivia
./start-all.sh
```

`SALIDA_TRIVIA_MODEL` ya apunta a `lora-salida-trivia` por default (ver
`Clients/Llama_Client.py`) -- si el modelo no está importado en Ollama,
`Orchestrator_Management.py` cae de vuelta a las listas de palabras clave
(`_SALIR_TRIVIA`/`_TEMA_PERSONAL`) sin cortar el turno, así que probarlo no
tiene riesgo de romper nada si algo sale mal en el camino.

## Archivos de esta carpeta

| Archivo | Qué es |
|---|---|
| `generar_dataset.py` | Arma `dataset_salida.jsonl` a partir del dataset real de trivia + ejemplos reusados de `router_training/`. |
| `dataset_salida.jsonl` | 542 ejemplos etiquetados RESPUESTA/SALIR, ya generado. |
| `entrenar_salida.py` | El fine-tuning LoRA en sí (paso 2 de arriba). |
| `Modelfile` | Para el `ollama create` del paso 5. |
| `probar_salida.py` | 14 casos reales para validar el modelo entrenado (paso 6). |
| `entrenamiento.log` | Log de la última corrida (si existe) -- útil para ver el progreso o retomar referencia si algo falló a mitad. |
