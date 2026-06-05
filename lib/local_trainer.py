import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainerCallback, EarlyStoppingCallback, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import os, shutil

def main():
    # 1. Load Dataset
    print("Loading dataset from local file node_3.json...")
    dataset = load_dataset("json", data_files="node_3.json", split="train")

    split_1 = dataset.train_test_split(test_size=0.2, seed=42)
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
    dataset = DatasetDict({
        "train": split_1["train"],
        "validation": split_2["train"],
        "test": split_2["test"],
    })
    print("Split sizes:", {k: len(dataset[k]) for k in dataset.keys()})

    # Format into Hugging Face Dataset format
    def process_messages(item):
        prompt = item["Question"]

        # Reasoning style answer combining CoT and final response
        cot = item.get('Complex_CoT', '') or ""
        response = item.get('Response', '') or ""
        answer = f"<think>\n{cot}\n</think>\n\n{response}"

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        return {"messages": messages}

    dataset = dataset.map(process_messages)

    # 2. Load Model and Tokenizer from HuggingFace
    model_name = "openai-community/gpt2"
    print(f"Loading {model_name} from HuggingFace Hub...")

    # Optional: If the model is gated or private, uncomment the following line and log in:
    # from huggingface_hub import login
    # login(token="HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        texts = []
        for msgs in examples["messages"]:
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n<|im_start|>assistant\n{msgs[1]['content']}<|im_end|>"
            texts.append(text)

        encodings = tokenizer(texts, truncation=True, max_length=1024)
        return encodings   

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 3. Setup DoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        use_dora=False  # DoRA (Weight-Decomposed Low-Rank Adaptation)
    )

    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    _orig_load_adapter = model.load_adapter
    def _load_adapter_no_autocast(*args, **kwargs):
        kwargs.setdefault("autocast_adapter_dtype", False)
        return _orig_load_adapter(*args, **kwargs)
    model.load_adapter = _load_adapter_no_autocast
    model.print_trainable_parameters()

    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    class KeepBestNCheckpointsCallback(TrainerCallback):
        def __init__(self, output_dir: str, n_best: int = 5):
            self.output_dir = output_dir
            self.n_best = n_best
            self.checkpoint_scores = {}  # checkpoint_dir_name -> eval_loss

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return control
            eval_loss = metrics.get("eval_loss")
            if eval_loss is None:
                return control
            ckpt_name = f"checkpoint-{state.global_step}"
            self.checkpoint_scores[ckpt_name] = float(eval_loss)
            return control

        def on_save(self, args, state, control, **kwargs):
            if not self.output_dir or not os.path.isdir(self.output_dir):
                return control

            ckpt_dirs = []
            for name in os.listdir(self.output_dir):
                full_path = os.path.join(self.output_dir, name)
                if name.startswith("checkpoint-") and os.path.isdir(full_path):
                    ckpt_dirs.append(name)

            if len(ckpt_dirs) <= self.n_best:
                return control

            def score(name: str) -> float:
                # Unknown scores get treated as worst so they are deleted first
                return self.checkpoint_scores.get(name, float("inf"))

            ckpt_dirs_sorted = sorted(ckpt_dirs, key=score)
            keep = set(ckpt_dirs_sorted[: self.n_best])

            for name in ckpt_dirs:
                if name in keep:
                    continue
                full_path = os.path.join(self.output_dir, name)
                try:
                    shutil.rmtree(full_path)
                except Exception:
                    pass

            return control

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./full_gpt2_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        num_train_epochs=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=None,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        args=training_args,
        data_collator=data_collator,
        callbacks=[
            KeepBestNCheckpointsCallback(output_dir=training_args.output_dir, n_best=5),
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-5),
        ],
    )

    # 7. Start Training
    print("Starting baseline finetuning for full dataset...")
    trainer.train()

    import math
    try:
        from transformers.utils.notebook import NotebookProgressCallback
        trainer.remove_callback(NotebookProgressCallback)
    except Exception:
        pass
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("Test metrics:", test_metrics)
    if "eval_loss" in test_metrics and test_metrics["eval_loss"] is not None:
        print("Test perplexity:", math.exp(test_metrics["eval_loss"]))

    # 8. Save final model
    print("Saving model...")
    trainer.model.save_pretrained("./node3_gpt2_lora_final")
    tokenizer.save_pretrained("./node3_gpt2_lora_final")
    print("Done!")

if __name__ == "__main__":
    main()
