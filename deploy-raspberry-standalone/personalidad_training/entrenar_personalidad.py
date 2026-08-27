# -*- coding: utf-8 -*-
"""Fine-tuning LoRA de Qwen2.5-0.5B-Instruct para hornear la personalidad y
las reglas de estilo de Lora en los pesos, en vez de mandar el system
prompt completo (~200 tokens) en cada turno. Ver "Personalidad horneada en
el modelo" en el README para el benchmark de resultado y cómo usarlo.

Esto es ENTRENAMIENTO -- corre en una PC de desarrollo (¡NO en la Raspberry
Pi, no tiene ni de cerca la potencia!), y el resultado (unos cientos de MB)
se copia a la Pi como un modelo más de Ollama. Nada de esta carpeta ni sus
dependencias (torch/transformers/peft/datasets/accelerate, NO están en el
requirements.txt principal) hace falta en la Pi para lo que ya está
entrenado.

Corre en CPU (no todas las PCs de desarrollo tienen GPU) -- por eso LoRA
(pocos parámetros entrenables) y un modelo chico (0.5B). Con ~250 ejemplos
tardó bajo condiciones normales (sin otros programas pesados compitiendo
por CPU) unos 35-40 minutos en una PC sin GPU dedicada.

Pipeline completo (este script es solo el paso 1):
    1. python entrenar_personalidad.py
       -> deja el modelo fusionado (LoRA + base) en D:/lora-merged
          (o donde apunte SALIDA_MERGED/HF_HOME, configurables por env var)
    2. Convertir a GGUF (requiere clonar llama.cpp y sus deps de conversión
       -- pip install gguf sentencepiece protobuf):
       python llama.cpp/convert_hf_to_gguf.py D:/lora-merged \
           --outfile lora-personalidad-f16.gguf --outtype f16
    3. Cuantizar (requiere el binario llama-quantize -- bajar el release
       prebuilt de Windows/Linux/Mac de https://github.com/ggml-org/llama.cpp/releases,
       NO hace falta compilar nada):
       llama-quantize lora-personalidad-f16.gguf \
           lora-personalidad-q4_k_m.gguf Q4_K_M
    4. Importar a Ollama (Modelfile con FROM apuntando al .gguf + el mismo
       SYSTEM_PROMPT_CORTO de acá abajo):
       ollama create lora-personalidad -f Modelfile
    5. En deploy-raspberry-standalone: export CHAT_MODEL=lora-personalidad
       (Clients/Llama_Client.py + personalidad.py detectan el nombre solos
       y dejan de mandar el system prompt largo -- ver ese archivo).
"""
import json
import os

os.environ.setdefault("HF_HOME", os.environ.get("PERSONALIDAD_HF_CACHE", "D:/hf-cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ.get("PERSONALIDAD_HF_CACHE", "D:/hf-cache"))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset_personalidad.jsonl")
SALIDA_ADAPTER = os.environ.get("PERSONALIDAD_SALIDA_ADAPTER", "D:/lora-lora-adapter")
SALIDA_MERGED = os.environ.get("PERSONALIDAD_SALIDA_MERGED", "D:/lora-merged")

# OJO: si se cambia esto, hay que reentrenar -- el modelo aprendió a
# comportarse bien específicamente con ESTE texto de system prompt exacto
# (ver personalidad.py, que lo manda igual cuando CHAT_MODEL=lora-personalidad).
SYSTEM_PROMPT_CORTO = "Eres Lora, un robot con personalidad seca y directa. Respondés en español, corto y natural."

MAX_LEN = 384


def cargar_dataset(tokenizer):
    ejemplos = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8")]
    print(f"Cargados {len(ejemplos)} ejemplos de entrenamiento")

    def _tokenizar(ej):
        mensajes_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT_CORTO},
            {"role": "user", "content": ej["instruccion"]},
        ]
        # tokenize=False + tokenizer() aparte (en vez de tokenize=True directo):
        # en transformers 5.15.1 apply_chat_template(tokenize=True) devuelve un
        # BatchEncoding, no una lista plana -- este camino evita la ambigüedad.
        prompt_texto = tokenizer.apply_chat_template(
            mensajes_prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_texto, add_special_tokens=False)["input_ids"]
        objetivo_ids = tokenizer(ej["objetivo"].strip(), add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id]
        input_ids = prompt_ids + objetivo_ids + eos
        # Loss solo sobre la respuesta del asistente (-100 = ignorado por la loss).
        labels = [-100] * len(prompt_ids) + objetivo_ids + eos
        input_ids = input_ids[:MAX_LEN]
        labels = labels[:MAX_LEN]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    ds = Dataset.from_list(ejemplos).map(_tokenizar, remove_columns=["instruccion", "objetivo"])
    return ds


def main():
    print("Cargando tokenizer y modelo base...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    modelo = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    modelo = get_peft_model(modelo, lora_config)
    modelo.print_trainable_parameters()

    ds = cargar_dataset(tokenizer)
    ds_split = ds.train_test_split(test_size=0.1, seed=42)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=os.environ.get("PERSONALIDAD_CHECKPOINTS", "D:/lora-lora-checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        warmup_steps=5,
    )

    trainer = Trainer(
        model=modelo, args=args, train_dataset=ds_split["train"],
        eval_dataset=ds_split["test"], data_collator=collator,
    )
    print("Entrenando (CPU, puede tardar) ...")
    trainer.train()

    os.makedirs(SALIDA_ADAPTER, exist_ok=True)
    modelo.save_pretrained(SALIDA_ADAPTER)
    tokenizer.save_pretrained(SALIDA_ADAPTER)
    print(f"Adapter LoRA guardado en {SALIDA_ADAPTER}")

    print("Fusionando LoRA con el modelo base...")
    modelo_fusionado = modelo.merge_and_unload()
    os.makedirs(SALIDA_MERGED, exist_ok=True)
    modelo_fusionado.save_pretrained(SALIDA_MERGED)
    tokenizer.save_pretrained(SALIDA_MERGED)
    print(f"Modelo fusionado guardado en {SALIDA_MERGED} -- listo para convertir a GGUF.")


if __name__ == "__main__":
    main()
