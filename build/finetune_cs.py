# finetune_cs.py (optimized for speed on multi-GPU + resume)

from pathlib import Path
import os

from datasets import Dataset, DatasetDict
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

import numpy as np
import torch
from evaluate import load as load_metric

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "facebook/m2m100_418M"  


def load_parallel_dataset(en_path: Path, ko_path: Path, val_ratio: float = 0.1) -> DatasetDict:
    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    ko_lines = ko_path.read_text(encoding="utf-8").splitlines()

    assert len(en_lines) == len(ko_lines), \
        f"EN lines ({len(en_lines)}) != KO lines ({len(ko_lines)})"

    n = len(en_lines)
    n_val = max(1, int(n * val_ratio))

    train_en = en_lines[:-n_val]
    train_ko = ko_lines[:-n_val]
    val_en   = en_lines[-n_val:]
    val_ko   = ko_lines[-n_val:]

    train_ds = Dataset.from_dict({"en": train_en, "ko": train_ko})
    val_ds   = Dataset.from_dict({"en": val_en, "ko": val_ko})

    return DatasetDict({"train": train_ds, "validation": val_ds})


def main():
    en_path = DATA_DIR / "cs_train_en.aligned.txt"
    ko_path = DATA_DIR / "cs_train_ko.aligned.txt"

    if not en_path.exists() or not ko_path.exists():
        raise FileNotFoundError(
            f"Parallel data not found. Expected:\n  {en_path}\n  {ko_path}\n"
            "먼저 clean_cs_en.py 와 build_cs_parallel_by_translate.py 를 실행해줘."
        )

    dataset = load_parallel_dataset(en_path, ko_path, val_ratio=0.1)
    print(dataset)

    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "ko"

    max_source_length = 256
    max_target_length = 256

    def preprocess(batch):
        inputs = batch["en"]
        targets = batch["ko"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["en", "ko"],
        num_proc=4,       
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
    )

    training_args = TrainingArguments(
        output_dir="./m2m100_cs_finetuned",
        overwrite_output_dir=True,

        num_train_epochs=1,                 
        per_device_train_batch_size=16,      
        per_device_eval_batch_size=16,

        learning_rate=5e-5,
        warmup_ratio=0.03,

        fp16=False,
        bf16=torch.cuda.is_available(),    
        optim="adamw_torch",

        logging_steps=100,
        evaluation_strategy="epoch",     
        save_strategy="epoch",             
        save_total_limit=2,

        dataloader_num_workers=8,
        report_to="none",

        ddp_find_unused_parameters=True,   
    )

    bleu_metric = load_metric("sacrebleu")
    rouge_metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        bleu_refs = [[l.strip()] for l in decoded_labels]  
        rouge_refs = [l.strip() for l in decoded_labels]

        bleu_result = bleu_metric.compute(
            predictions=decoded_preds,
            references=bleu_refs,
        )
        bleu = bleu_result["score"]

        rouge_result = rouge_metric.compute(
            predictions=decoded_preds,
            references=rouge_refs,
            use_stemmer=True,
        )
        rouge_scores = {
            "rouge1": rouge_result["rouge1"].mid.fmeasure * 100,
            "rouge2": rouge_result["rouge2"].mid.fmeasure * 100,
            "rougeL": rouge_result["rougeL"].mid.fmeasure * 100,
            "rougeLsum": rouge_result["rougeLsum"].mid.fmeasure * 100,
        }

        result = {"bleu": bleu}
        result.update(rouge_scores)
        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is not None:
            print(f"[RESUME] Found checkpoint at {last_ckpt}, resuming training...")

    if last_ckpt is not None:
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()

    if trainer.is_world_process_zero():
        save_dir = BASE_DIR / "m2m100_cs_finetuned"
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"[DONE] Saved fine-tuned model at {save_dir}")

if __name__ == "__main__":
    main()
