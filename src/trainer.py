"""
src/trainer.py — Local LoRA fine-tuning for DFL rounds.

Pure blocking code — runs in a thread executor (called from phase4.py).
Never import asyncio here.

Round N flow:
  - Input:  round{N}_{node_id}_adapter.safetensors (written by Phase 3 FedAvg)
  - Output: round{N+1}_{node_id}.safetensors  (stamped for Phase 2 next round)

Round 0 (or when merged adapter is missing):
  - Fresh LoRA init (no pre-existing adapter to load).
  - Output still stamped as round1_{node_id}.safetensors.
"""

import json
import logging
import os
import shutil
import traceback
from pathlib import Path
from safetensors.torch import load_file, save_file

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

log = logging.getLogger(__name__)


class LocalTrainer:
    """Thin wrapper around Hugging Face Trainer for DFL round training."""

    def __init__(
        self,
        node_id:      str,
        round_num:    int,
        model_dir:    Path,
        dataset_path: Path,
    ):
        self.node_id      = node_id
        self.round_num    = round_num
        self.model_dir    = model_dir
        self.dataset_path = dataset_path
        self._safe_id     = node_id.replace(":", "_")

    # ── public entry point ──────────────────────────────────────────────────
    def train(self):
        safe = self.node_id.replace(":", "_")
        out = self.model_dir / f"round{self.round_num + 1}_{safe}.safetensors"
        
        # Match YOUR model's actual key naming (GPT-2 style)
        fake = {}
        for layer in range(6):  # match your model's layer count
            fake[f"base_model.model.transformer.h.{layer}.attn.c_attn.lora_A.weight"] = torch.zeros(8, 768, dtype=torch.float16)
            fake[f"base_model.model.transformer.h.{layer}.attn.c_attn.lora_B.weight"] = torch.zeros(2304, 8, dtype=torch.float16)
        
        save_file(fake, str(out))
        return out

    """ Below is real code, above is fake one for testing"""
    # def train(self) -> Path | None:
    #     """
    #     Run training for this round.

    #     Returns the stamped Path (round{N+1}_{node_id}.safetensors)
    #     on success, or None on failure.
    #     """
    #     if not self.dataset_path.exists():
    #         log.warning(f"Dataset not found at {self.dataset_path} — skipping training")
    #         return None

    #     try:
    #         self._run_hf_trainer()

    #         src  = self.model_dir / "adapter_model.safetensors"
    #         dest = self.model_dir / f"round{self.round_num + 1}_{self._safe_id}.safetensors"

    #         if not src.exists():
    #             log.error(f"Training finished but adapter not found at {src}")
    #             return None

    #         shutil.copy2(src, dest)
    #         log.info(f"Adapter stamped for next round → {dest}")
    #         return dest

    #     except Exception:
    #         log.error(f"Training raised:\n{traceback.format_exc()}")
    #         return None

    def get_dataset_size(self) -> int:
        """Return number of samples in this node's dataset."""
        if not self.dataset_path.exists():
            return 0
        try:
            if self.dataset_path.suffix == ".json":
                with open(self.dataset_path, encoding="utf-8") as f:
                    return len(json.load(f))
            else:
                with open(self.dataset_path, encoding="utf-8") as f:
                    return sum(1 for _ in f)
        except Exception as e:
            log.warning(f"Could not read dataset size: {e}")
            return 0

    # ── internal training pipeline ──────────────────────────────────────────

    def _run_hf_trainer(self) -> None:
        """Full Hugging Face training pipeline. Writes adapter_model.safetensors then stamps a copy."""

        # 1. Load & split dataset
        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        split_1 = dataset.train_test_split(test_size=0.2, seed=42)
        split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
        dataset = DatasetDict({
            "train":      split_1["train"],
            "validation": split_2["train"],
            "test":       split_2["test"],
        })
        log.info(f"Dataset splits: train={len(dataset['train'])} "
                 f"val={len(dataset['validation'])} test={len(dataset['test'])}")

        def _format_messages(item):
            prompt   = item["Question"]
            response = item.get("Response", "") or ""
            return {
                "messages": [
                    {"role": "user",      "content": prompt},
                    {"role": "assistant",  "content": response},
                ]
            }

        dataset = dataset.map(_format_messages)

        # 2. Load model & tokenizer
        model_name = "openai-community/gpt2"
        log.info(f"Loading {model_name} from HuggingFace Hub...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _tokenize(examples):
            texts = []
            for msgs in examples["messages"]:
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    text = (
                        f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{msgs[1]['content']}<|im_end|>"
                    )
                texts.append(text)
            enc = tokenizer(texts, truncation=True, max_length=1024)
            return enc

        tokenized = dataset.map(
            _tokenize,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        _has_gpu = torch.cuda.is_available()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map  = "auto" if _has_gpu else None,
            torch_dtype = torch.bfloat16 if _has_gpu else torch.float32,
            trust_remote_code=True,
        )

        # 3. LoRA setup
        output_str = str(self.model_dir)
        merged_path = self.model_dir / f"round{self.round_num}_{self._safe_id}_adapter.safetensors"
        adapter_path = self.model_dir / "adapter_model.safetensors"

        if self.round_num >= 1 and merged_path.exists():
            log.info(f"Round {self.round_num}: loading merged adapter from {merged_path}")
            # Copy node-specific merged file to adapter_model.safetensors so
            # PeftModel.from_pretrained can find it alongside adapter_config.json
            if not adapter_path.exists() or adapter_path.stat().st_mtime < merged_path.stat().st_mtime:
                shutil.copy2(merged_path, adapter_path)
            model = PeftModel.from_pretrained(model, output_str, is_trainable=True)
            self._enable_lora_grads(model)
            model.train()
        elif self.round_num >= 1 and adapter_path.exists():
            log.info(f"Round {self.round_num}: loading adapter_model.safetensors directly")
            model = PeftModel.from_pretrained(model, output_str, is_trainable=True)
            self._enable_lora_grads(model)
            model.train()
        else:
            log.info(f"Round {self.round_num}: fresh LoRA adapter")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["c_attn", "c_proj"],
                use_dora=False,
            )
            model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
            orig = model.load_adapter
            model.load_adapter = lambda *a, **kw: orig(
                *a, **{**kw, "autocast_adapter_dtype": False}
            )

        model.print_trainable_parameters()

        # 4. Data collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )

        # 5. Training arguments — fast-demo settings (~30 seconds)
        training_args = TrainingArguments(
            output_dir                  = output_str,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 2,
            learning_rate               = 3e-4,
            logging_steps               = 1,
            max_steps                   = 5,
            save_strategy               = "no",
            eval_strategy               = "no",
            save_total_limit            = None,
            load_best_model_at_end      = False,
            bf16                        = _has_gpu,
            optim                       = "adamw_torch",
            report_to                   = "none",
            gradient_checkpointing      = False,
            dataloader_num_workers      = 0 if not _has_gpu else 2,
            dataloader_pin_memory       = _has_gpu,
        )

        # 6. Trainer
        trainer = Trainer(
            model=model,
            train_dataset=tokenized["train"],
            eval_dataset=None,
            args=training_args,
            data_collator=collator,
        )

        # 7. Train
        log.info("Starting training (fast demo — 5 steps)...")
        trainer.train()

        # 8. Save
        log.info("Saving adapter...")
        trainer.model.save_pretrained(output_str)
        tokenizer.save_pretrained(output_str)
        log.info(f"Training complete — adapter saved to {output_str}")

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _enable_lora_grads(model) -> None:
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable params: {trainable:,}")
