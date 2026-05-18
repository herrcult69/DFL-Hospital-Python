import threading
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

_model      = None
_tokenizer  = None
_model_lock = threading.Lock()
_output_dir = None   # set on first call; updated via invalidate_cache()


def run_inference(symptoms: str, output_dir: str) -> str:
    global _output_dir
    _output_dir = output_dir
    _ensure_loaded(output_dir)

    with _model_lock:
        prompt = f"Question: {symptoms}\nReasoning:"
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        return _tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def invalidate_cache():
    """Call after aggregation to force model reload on next /predict."""
    global _model, _tokenizer
    with _model_lock:
        _model     = None
        _tokenizer = None
    print("[Inference] Model cache invalidated — will reload on next /predict.")


def _ensure_loaded(output_dir: str):
    global _model, _tokenizer
    with _model_lock:
        if _model is not None:
            return
        tok = GPT2Tokenizer.from_pretrained("distilbert/distilgpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = GPT2LMHeadModel.from_pretrained("distilbert/distilgpt2")
        adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(base, output_dir)
        else:
            model = base
        model.eval()
        _model     = model
        _tokenizer = tok
        print(f"[Inference] Model loaded from {output_dir!r}.")