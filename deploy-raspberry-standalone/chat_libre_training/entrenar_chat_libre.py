# -*- coding: utf-8 -*-
"""Fine-tuning LoRA de qwen2.5:0.5b para conversar -- paso 4 del pipeline.

Cada corrida crea una VERSION nueva (v1, v2, v3...) sin pisar la anterior:
queda anotada en versiones.json con su fecha, cuantos ejemplos uso y su
eval_loss. Asi se puede volver atras si una version sale peor -- que pasa, y
no es raro.

QUE PUEDE Y QUE NO (medido en este proyecto, ver README.md):

  SI  -- estilo y comportamiento: acompanar en vez de desentenderse,
         repreguntar para mantener el hilo, admitir que no sabe.
  NO  -- conocimiento del mundo. El base de 0.5B ya cree que el cielo es
         blanco y que la Mona Lisa la pinto Frida Kahlo. Ningun fine-tuning
         arregla eso: el conocimiento viene del preentrenamiento.

Corre en la PC de desarrollo (CPU alcanza), NO en la Raspberry Pi.

Uso:
    pip install torch transformers peft datasets
    python entrenar_chat_libre.py
"""
import io
import json
import os
import time

os.environ.setdefault("HF_HOME", os.environ.get("PERSONALIDAD_HF_CACHE", "D:/hf-cache"))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset_chat_libre.jsonl")
VERSIONES_PATH = os.path.join(HERE, "versiones.json")
BASE_SALIDA = os.environ.get("CHAT_LIBRE_SALIDA", "D:/lora-chat-libre")

# El mismo prompt que usa el robot en runtime (personalidad.py). Tiene que
# coincidir: si se entrena con un system prompt y se sirve con otro, el
# modelo queda fuera de distribucion -- que es exactamente lo que le paso a
# lora-chat cuando se le saco el RAG.
SYSTEM_PROMPT = (
    "Hablas español natural, como una persona real. Responde en máximo 25 "
    "palabras, una idea puntual, sin relleno ni repetir la pregunta. Nunca "
    "digas que eres una IA o un asistente."
)
MAX_LEN = 320


def proxima_version():
    """v1, v2, v3... segun lo que ya haya en versiones.json."""
    if not os.path.isfile(VERSIONES_PATH):
        return 1
    with io.open(VERSIONES_PATH, encoding="utf-8") as f:
        return max((v["version"] for v in json.load(f)), default=0) + 1


def anotar_version(version, n_ejemplos, eval_loss, ruta):
    historial = []
    if os.path.isfile(VERSIONES_PATH):
        with io.open(VERSIONES_PATH, encoding="utf-8") as f:
            historial = json.load(f)
    historial.append({
        "version": version,
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "ejemplos": n_ejemplos,
        "eval_loss": eval_loss,
        "modelo_ollama": "lora-chat-libre-v" + str(version),
        "ruta_merged": ruta,
        # Lo completa probar_conversacion.py. Sin esto la version esta
        # entrenada pero NO validada, y no se despliega.
        "eval_adversario": None,
    })
    with io.open(VERSIONES_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)
    return historial


def cargar_dataset(tokenizer):
    with io.open(DATASET_PATH, encoding="utf-8") as f:
        ejemplos = [json.loads(l) for l in f if l.strip()]
    print("Cargados " + str(len(ejemplos)) + " ejemplos curados")

    def _tok(ej):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": ej["instruccion"]}],
            tokenize=False, add_generation_prompt=True,
        )
        pid = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        oid = tokenizer(ej["objetivo"].strip(), add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id]
        ids = (pid + oid + eos)[:MAX_LEN]
        # loss solo sobre la respuesta: el prompt no se aprende
        labels = ([-100] * len(pid) + oid + eos)[:MAX_LEN]
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    ds = Dataset.from_list(ejemplos).map(_tok, remove_columns=["instruccion", "objetivo"])
    return ds, len(ejemplos)


def main():
    version = proxima_version()
    salida_merged = BASE_SALIDA + "-v" + str(version) + "-merged"
    print("=== Entrenando lora-chat-libre v" + str(version) + " ===")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    modelo = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    modelo = get_peft_model(modelo, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    modelo.print_trainable_parameters()

    ds, n_ejemplos = cargar_dataset(tokenizer)
    split = ds.train_test_split(test_size=0.1, seed=42)

    args = TrainingArguments(
        output_dir=BASE_SALIDA + "-v" + str(version) + "-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        warmup_steps=10,
        # Sin esto se exporta la ULTIMA epoca, que no es la mejor: en las dos
        # corridas de salida_trivia_training la ultima quedo peor que la
        # anterior, y merge_and_unload() fusiona el estado en memoria.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Trainer(model=modelo, args=args, train_dataset=split["train"],
                      eval_dataset=split["test"],
                      data_collator=DataCollatorForSeq2Seq(
                          tokenizer, padding=True, label_pad_token_id=-100))
    print("Entrenando (CPU, puede tardar)...")
    trainer.train()
    eval_loss = trainer.evaluate().get("eval_loss")

    print("Fusionando LoRA con el base...")
    fusionado = modelo.merge_and_unload()
    os.makedirs(salida_merged, exist_ok=True)
    fusionado.save_pretrained(salida_merged)
    tokenizer.save_pretrained(salida_merged)

    historial = anotar_version(version, n_ejemplos, eval_loss, salida_merged)
    print("\nv" + str(version) + " lista: " + salida_merged)
    print("\nHistorial:")
    for v in historial:
        marca = "  <- nueva" if v["version"] == version else ""
        ev = v.get("eval_adversario") or "sin evaluar"
        print("  v%d  %s  %4d ej  eval_loss=%.5f  adversario=%s%s"
              % (v["version"], v["fecha"], v["ejemplos"], v["eval_loss"], ev, marca))
    print("\nSiguiente: GGUF -> Ollama -> EVALUAR con probar_conversacion.py")
    print("ANTES de desplegar (ver README.md, pasos 5 y 6).")


if __name__ == "__main__":
    main()
