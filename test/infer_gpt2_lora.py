import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with the fine-tuned GPT-2 LoRA adapter."
    )
    parser.add_argument(
        "--adapter-path",
        default="full_gpt2_lora_final",
        help="Path to a saved PEFT LoRA adapter or Trainer checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name/path. Defaults to the value in adapter_config.json.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Question to answer. If omitted, examples are read from --dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="medical_o1_sft_mix.json",
        help="JSON dataset used when --prompt is not provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of dataset examples to test when --prompt is omitted.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "full"],
        default="test",
        help=(
            "Dataset split to sample from. Default recreates the notebook split "
            "and samples from test."
        ),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Use a specific row inside the selected split instead of random samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty above 1.0 discourages repeated text.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Block repeated n-grams of this size. 0 disables the constraint.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling. Pass --no-do-sample for deterministic greedy decoding.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device selection. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype. 'auto' uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--show-reference",
        action="store_true",
        help="Print reference dataset responses when testing from --dataset.",
    )
    return parser.parse_args()


def select_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def select_dtype(dtype_arg, device):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


def load_model_and_tokenizer(adapter_path, base_model_arg, device_arg, dtype_arg):
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    base_model_name = base_model_arg or peft_config.base_model_name_or_path
    device = select_device(device_arg)
    dtype = select_dtype(dtype_arg, device)

    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model_name}")
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    if device == "cpu":
        base_model.to(device)

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        autocast_adapter_dtype=False,
    )
    model.eval()
    return model, tokenizer, device


def format_prompt(question):
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def clean_generation(full_text, prompt_text):
    if full_text.startswith(prompt_text):
        full_text = full_text[len(prompt_text) :]
    return full_text.split("<|im_end|>", 1)[0].strip()


def generate_answer(model, tokenizer, question, args):
    prompt_text = format_prompt(question)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )

    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    if args.do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return clean_generation(decoded, prompt_text)


def split_dataset(data, seed):
    data = [{**example, "_dataset_index": index} for index, example in enumerate(data)]
    dataset = Dataset.from_list(data)

    split_1 = dataset.train_test_split(test_size=0.2, seed=seed)
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=seed)

    return {
        "train": split_1["train"],
        "validation": split_2["train"],
        "test": split_2["test"],
        "full": dataset,
    }


def load_examples(dataset_path, split_name, num_samples, sample_index, seed):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    splits = split_dataset(data, seed)
    selected_split = splits[split_name]
    split_size = len(selected_split)

    if sample_index is not None:
        if sample_index < 0 or sample_index >= split_size:
            raise IndexError(
                f"--sample-index must be between 0 and {split_size - 1} "
                f"for split '{split_name}', got {sample_index}"
            )
        example = dict(selected_split[sample_index])
        original_index = example.pop("_dataset_index")
        return [(sample_index, original_index, example)]

    rng = random.Random(seed)
    indices = list(range(split_size))
    rng.shuffle(indices)
    selected = indices[: min(num_samples, len(indices))]

    examples = []
    for split_index in selected:
        example = dict(selected_split[split_index])
        original_index = example.pop("_dataset_index")
        examples.append((split_index, original_index, example))
    return examples


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer, _ = load_model_and_tokenizer(
        args.adapter_path, args.base_model, args.device, args.dtype
    )

    if args.prompt:
        answer = generate_answer(model, tokenizer, args.prompt, args)
        print("\nQuestion:")
        print(args.prompt)
        print("\nModel answer:")
        print(answer)
        return

    examples = load_examples(
        args.dataset, args.split, args.num_samples, args.sample_index, args.seed
    )
    for position, (split_index, original_index, example) in enumerate(examples, start=1):
        question = example["Question"]
        answer = generate_answer(model, tokenizer, question, args)

        print("\n" + "=" * 80)
        print(
            f"Example {position} | split {args.split} index {split_index} "
            f"| original dataset index {original_index}"
        )
        print("\nQuestion:")
        print(question)
        print("\nModel answer:")
        print(answer)
        if args.show_reference:
            print("\nReference response:")
            print(example.get("Response", ""))


if __name__ == "__main__":
    main()
