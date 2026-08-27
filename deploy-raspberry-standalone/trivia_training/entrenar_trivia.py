# -*- coding: utf-8 -*-
"""Fine-tuning LoRA de Qwen2.5-0.5B-Instruct especializado SOLO en
reacciones de Trivia (comentar_resultado()/reaccionar_libre() en
Orchestrator_Management.py) -- a diferencia de personalidad_training/
(que entrena sobre las 4 categorías: trivia + chat libre + búsqueda web),
este solo ve los 121 ejemplos de trivia, para un modelo más especializado
en ese rol puntual (TRIVIA_MODEL, ver Clients/Llama_Client.py).

Mismo enfoque que personalidad_training/entrenar_personalidad.py (léelo
primero si esto es lo primero que abrís): LoRA, CPU, destilación (las
respuestas objetivo del dataset se generaron con llama3.2:3b-q4s + el
prompt completo, para que el modelo chico aprenda a reproducirlas con un
system prompt corto).

Esto es ENTRENAMIENTO -- corre en una PC de desarrollo, NO en la
Raspberry Pi. El resultado se copia a la Pi como un modelo más de Ollama.

Pipeline completo (este script es solo el paso 1, igual que en
personalidad_training/):
    1. python entrenar_trivia.py
       -> deja el modelo fusionado en D:/lora-trivia-merged
          (configurable por env var, ver más abajo)
    2. Convertir a GGUF (llama.cpp/convert_hf_to_gguf.py, ver
       personalidad_training/entrenar_personalidad.py para el detalle
       completo de este paso y el siguiente -- es idéntico):
       python llama.cpp/convert_hf_to_gguf.py D:/lora-trivia-merged \
           --outfile lora-trivia-f16.gguf --outtype f16
    3. Cuantizar con llama-quantize (release prebuilt, sin compilar):
       llama-quantize lora-trivia-f16.gguf \
           lora-trivia-q4_k_m.gguf Q4_K_M
    4. Importar a Ollama (Modelfile en esta carpeta):
       ollama create lora-trivia -f Modelfile
    5. En deploy-raspberry-standalone: export TRIVIA_MODEL=lora-trivia
       (por default TRIVIA_MODEL ya apunta a "lora-personalidad", el
       modelo general -- este es un reemplazo más especializado, opcional).
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
DATASET_PATH = os.path.join(HERE, "dataset_trivia.jsonl")
SALIDA_ADAPTER = os.environ.get("TRIVIA_SALIDA_ADAPTER", "D:/lora-trivia-lora-adapter")
SALIDA_MERGED = os.environ.get("TRIVIA_SALIDA_MERGED", "D:/lora-trivia-merged")

# Mismo system prompt corto que personalidad_training/ (ver ese archivo):
# se mantiene igual a propósito, para que ambos modelos sean intercambiables
# desde el punto de vista de Orchestrator_Management.py -- lo único que
# cambia es CON QUÉ EJEMPLOS se entrenó cada uno, no la instrucción base.
SYSTEM_PROMPT_CORTO = "Eres Lora, un robot con personalidad seca y directa. Respondés en español, corto y natural."

MAX_LEN = 384


def cargar_dataset(tokenizer):
    ejemplos = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8")]
    print(f"Cargados {len(ejemplos)} ejemplos de entrenamiento (solo Trivia)")

    def _tokenizar(ej):
        mensajes_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT_CORTO},
            {"role": "user", "content": ej["instruccion"]},
        ]
        # tokenize=False + tokenizer() aparte: en transformers 5.15.1
        # apply_chat_template(tokenize=True) devuelve un BatchEncoding, no
        # una lista plana -- este camino evita la ambigüedad (mismo fix que
        # en personalidad_training/entrenar_personalidad.py).
        prompt_texto = tokenizer.apply_chat_template(
            mensajes_prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_texto, add_special_tokens=False)["input_ids"]
        objetivo_ids = tokenizer(ej["objetivo"].strip(), add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id]
        input_ids = prompt_ids + objetivo_ids + eos
        labels = [-100] * len(prompt_ids) + objetivo_ids + eos  # loss solo sobre la respuesta
        input_ids = input_ids[:MAX_LEN]
        labels = labels[:MAX_LEN]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    ds = Dataset.from_list(ejemplos).map(_tokenizar, remove_columns=["instruccion", "objetivo"])
    return ds


def main():
    print("Cargando tokenizer y modelo base...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    modelo = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

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
        output_dir=os.environ.get("TRIVIA_CHECKPOINTS", "D:/lora-trivia-checkpoints"),
        num_train_epochs=4,  # dataset mas chico (121 vs 251) -- una epoca mas para compensar
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
