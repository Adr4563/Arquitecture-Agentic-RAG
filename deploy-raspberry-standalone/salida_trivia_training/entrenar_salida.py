# -*- coding: utf-8 -*-
"""Fine-tuning LoRA de Qwen2.5-0.5B-Instruct especializado en UNA sola
decisión binaria: dado el turno de Trivia pendiente y el mensaje del
usuario, ¿es un intento de RESPUESTA, o el usuario se fue a hablar de otra
cosa (SALIR) -- un tema personal, un problema, charla sin relación, o una
frase explícita de "quiero parar"?

Por qué un LLM reentrenado y no el prompt genérico contra qwen2.5:0.5b: se
probó primero (few-shot, sin reentrenar) y falló -- se equivocó en 4 de 5
casos de prueba real (ver Orchestrator_Management.py, sección
_TEMA_PERSONAL). 0.5B es demasiado chico para este juicio semántico abierto
CON SOLO few-shot; fine-tuneado específicamente en esta tarea (como ya se
hizo con CHAT_MODEL/TRIVIA_MODEL para sus roles) es la apuesta de este
pipeline. El modelo grande (llama3.2:3b-q4s) sí acertó 5/5 con few-shot,
pero se descartó por RAM/latencia -- este fine-tune busca esa misma
precisión en el tamaño chico (~397MB, igual que CHAT_MODEL/TRIVIA_MODEL).

Mismo enfoque que trivia_training/entrenar_trivia.py (léelo primero si esto
es lo primero que abrís): LoRA, CPU. A diferencia de los otros 3 pipelines
(personalidad/chat/trivia, que entrenan REACCIONES de Lora con
destilación), acá el objetivo es una sola palabra (RESPUESTA o SALIR) --
clasificación, no generación en personaje. Por eso el system prompt es
distinto: no habla de la personalidad de Lora, describe la tarea de
clasificar.

Esto es ENTRENAMIENTO -- corre en una PC de desarrollo, NO en la Raspberry
Pi. El resultado se copia a la Pi como un modelo más de Ollama.

Pipeline completo (este script es solo el paso 1):
    1. python entrenar_salida.py
       -> deja el modelo fusionado en D:/lora-salida-trivia-merged
          (configurable por env var, ver más abajo)
    2. Convertir a GGUF (ver personalidad_training/entrenar_personalidad.py
       para el detalle completo de este paso y el siguiente -- es idéntico):
       python llama.cpp/convert_hf_to_gguf.py D:/lora-salida-trivia-merged \
           --outfile lora-salida-trivia-f16.gguf --outtype f16
    3. Cuantizar con llama-quantize (release prebuilt, sin compilar):
       llama-quantize lora-salida-trivia-f16.gguf \
           lora-salida-trivia-q4_k_m.gguf Q4_K_M
    4. Importar a Ollama (Modelfile en esta carpeta):
       ollama create lora-salida-trivia -f Modelfile
    5. En deploy-raspberry-standalone: export SALIDA_TRIVIA_MODEL=lora-salida-trivia
       (por default ya apunta a ese nombre, ver Clients/Llama_Client.py --
       si el modelo no está importado en Ollama, Orchestrator_Management.py
       cae de vuelta a las listas de palabras clave, nunca corta el turno).
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
DATASET_PATH = os.path.join(HERE, "dataset_salida.jsonl")
SALIDA_ADAPTER = os.environ.get("SALIDA_TRIVIA_SALIDA_ADAPTER", "D:/lora-salida-trivia-lora-adapter")
SALIDA_MERGED = os.environ.get("SALIDA_TRIVIA_SALIDA_MERGED", "D:/lora-salida-trivia-merged")

# System prompt de TAREA, no de personalidad -- este modelo no le habla al
# usuario nunca, solo clasifica; distinto a propósito de
# SYSTEM_PROMPT_CORTO en los otros 3 pipelines. Se guarda igual en el
# Modelfile (ver ese archivo) para que Ollama lo aplique solo, sin que
# Orchestrator_Management.py tenga que mandarlo en cada llamada.
SYSTEM_PROMPT = (
    "Sos un clasificador. Te dan la pregunta de una trivia y el mensaje del "
    "usuario. Respondé EXACTAMENTE una palabra: RESPUESTA si el usuario "
    "intenta responder esa pregunta (aunque esté mal, o sea una opinión "
    "larga si la pregunta la pide), o SALIR si se puso a hablar de otra "
    "cosa (un tema personal, un problema, charla sin relación, o pide "
    "explícitamente parar/cambiar de tema)."
)

MAX_LEN = 320


def cargar_dataset(tokenizer):
    ejemplos = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8")]
    print(f"Cargados {len(ejemplos)} ejemplos de entrenamiento (RESPUESTA vs SALIR)")

    def _tokenizar(ej):
        mensajes_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ej["instruccion"]},
        ]
        # tokenize=False + tokenizer() aparte: mismo fix que en los otros 3
        # pipelines (transformers 5.15.1, apply_chat_template(tokenize=True)
        # devuelve un BatchEncoding, no una lista plana).
        prompt_texto = tokenizer.apply_chat_template(
            mensajes_prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_texto, add_special_tokens=False)["input_ids"]
        objetivo_ids = tokenizer(ej["objetivo"].strip(), add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id]
        input_ids = prompt_ids + objetivo_ids + eos
        labels = [-100] * len(prompt_ids) + objetivo_ids + eos  # loss solo sobre la palabra objetivo
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
        output_dir=os.environ.get("SALIDA_TRIVIA_CHECKPOINTS", "D:/lora-salida-trivia-checkpoints"),
        num_train_epochs=3,  # dataset mas grande (542) que trivia_training (121) -- alcanza con menos epocas
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        warmup_steps=10,
        # En la corrida v1 la epoca 3 quedo PEOR que la 2 (eval_loss 0.042 vs
        # 0.0016) y el modelo que se exporto fue el de la 3, porque sin esto
        # merge_and_unload() fusiona el estado final en memoria, no el mejor
        # checkpoint. Con load_best_model_at_end el Trainer recarga el mejor
        # antes de devolver el control, asi que lo que se fusiona y se
        # convierte a GGUF es ese.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
