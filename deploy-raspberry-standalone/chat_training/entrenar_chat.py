# -*- coding: utf-8 -*-
"""Fine-tuning LoRA de Qwen2.5-0.5B-Instruct especializado en Chat libre
(responder() en Orchestrator_Management.py) -- a diferencia de
personalidad_training/ (que entrena sobre las 4 categorías) y
trivia_training/ (solo trivia), este solo ve los ejemplos de chat, para un
modelo especializado en el rol de CHAT_MODEL (ver Clients/Llama_Client.py).

⚠️ Búsqueda web se sacó del proyecto después de entrenar el
`lora-chat` que ya está en producción -- ese dataset original (130
ejemplos) todavía mezclaba chat + búsqueda web, así que el modelo ya
desplegado vio algunos de esos ejemplos. No rompe nada (sigue sirviendo
bien para Chat libre), pero si se reentrena desde cero conviene primero
sacar del dataset los ejemplos de búsqueda web que ya no aplican.

Mismo enfoque que trivia_training/entrenar_trivia.py y personalidad_
training/entrenar_personalidad.py (léelos primero si esto es lo primero
que abrís): LoRA, CPU, destilación.

Esto es ENTRENAMIENTO -- corre en una PC de desarrollo, NO en la
Raspberry Pi. El resultado se copia a la Pi como un modelo más de Ollama.

⚠️ Si se corre esto AL MISMO TIEMPO que otro entrenamiento (ej.
trivia_training/entrenar_trivia.py) en la misma máquina sin GPU, compiten
por CPU y los dos tardan mucho más -- mejor uno por vez.

Pipeline completo (este script es solo el paso 1, igual que los otros 2):
    1. python entrenar_chat.py
       -> deja el modelo fusionado en D:/lora-chat-merged
          (configurable por env var, ver más abajo)
    2. Convertir a GGUF (ver personalidad_training/entrenar_personalidad.py
       para el detalle completo de este paso y el siguiente -- es idéntico):
       python llama.cpp/convert_hf_to_gguf.py D:/lora-chat-merged \
           --outfile lora-chat-f16.gguf --outtype f16
    3. Cuantizar con llama-quantize (release prebuilt, sin compilar):
       llama-quantize lora-chat-f16.gguf \
           lora-chat-q4_k_m.gguf Q4_K_M
    4. Importar a Ollama (Modelfile en esta carpeta):
       ollama create lora-chat -f Modelfile
    5. En deploy-raspberry-standalone: export CHAT_MODEL=lora-chat
       (por default CHAT_MODEL apunta a "qwen2.5:0.5b" -- este es un
       reemplazo más especializado, opcional).
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
DATASET_PATH = os.path.join(HERE, "dataset_chat.jsonl")
SALIDA_ADAPTER = os.environ.get("CHAT_SALIDA_ADAPTER", "D:/lora-chat-lora-adapter")
SALIDA_MERGED = os.environ.get("CHAT_SALIDA_MERGED", "D:/lora-chat-merged")

# Mismo system prompt corto que los otros dos (ver personalidad_training/ y
# trivia_training/): se mantiene igual a propósito -- lo único que cambia
# entre los 3 modelos es CON QUÉ EJEMPLOS se entrenó cada uno.
SYSTEM_PROMPT_CORTO = "Eres Lora, un robot con personalidad seca y directa. Respondés en español, corto y natural."

MAX_LEN = 384


def cargar_dataset(tokenizer):
    ejemplos = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8")]
    print(f"Cargados {len(ejemplos)} ejemplos de entrenamiento (solo Chat libre/Búsqueda web)")

    def _tokenizar(ej):
        mensajes_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT_CORTO},
            {"role": "user", "content": ej["instruccion"]},
        ]
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
        output_dir=os.environ.get("CHAT_CHECKPOINTS", "D:/lora-chat-checkpoints"),
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
