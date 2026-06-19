"""
Phase 3 — FedAvg LoRA aggregation.
Reads .safetensors files already on disk from Phase 2 ring transfer.
No HTTP — all files delivered by gRPC ring in Phase 2.
"""

import os
import logging
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)


def aggregate(
    node_id:        str,
    round_num:      int,
    chunk_dir:      Path,
    output_dir:     Path,
    participants:   list[str],   # global_table minus dead_this_round minus no_model_set
    has_own_model:  bool = True,
    dataset_size:   int  = 1,
) -> Path | None:
    """
    Load all .safetensors files from chunk_dir for this round.
    Merge using FedAvg with SVD re-decomposition.
    Write merged adapter to output_dir/round{round_num}_{node_id_safe}_adapter.safetensors.
    Returns path to merged file, or None if aggregation failed.
    """
    node_safe = node_id.replace(":", "_")
    state_dicts:   dict = {}
    dataset_sizes: dict = {}

    # Load own adapter if we have one
    if has_own_model:
        own_path = output_dir / f"round{round_num}_{node_id.replace(':', '_')}.safetensors"
        if own_path.exists():
            state_dicts[node_id]   = load_file(str(own_path))
            dataset_sizes[node_id] = dataset_size
            log.info(f"Loaded own adapter: {own_path}")
        else:
            log.warning(f"Own adapter not found at {own_path} — skipping self")

    # Load peer adapters from chunk_dir (delivered by Phase 2)
    for peer_id in participants:
        if peer_id == node_id:
            continue
        safe      = peer_id.replace(":", "_")
        peer_path = chunk_dir / f"round{round_num}_{safe}.safetensors"
        if peer_path.exists():
            state_dicts[peer_id]   = load_file(str(peer_path))
            dataset_sizes[peer_id] = 1   # equal weighting — no dataset size exchange yet
            log.info(f"Loaded peer adapter: {peer_path}")
        else:
            log.warning(f"Peer adapter missing: {peer_path} — skipping")

    if not state_dicts:
        log.error("No adapters loaded — aggregation aborted")
        return None

    if len(state_dicts) == 1:
        log.warning("Only one adapter available — using as-is without merging")
        merged = next(iter(state_dicts.values()))
    else:
        log.info(f"Merging {len(state_dicts)} adapters with FedAvg+SVD")
        merged = _fedit_merge(state_dicts, rank=16, dataset_sizes=dataset_sizes)

    merged = {k: v.contiguous() for k, v in merged.items()}

    # Write merged adapter atomically — node-unique to prevent races
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"round{round_num}_{node_safe}_adapter.safetensors"
    tmp_path = output_dir / f"round{round_num}_{node_safe}_adapter.safetensors.tmp"
    save_file(merged, str(tmp_path))
    os.replace(tmp_path, out_path)
    log.info(f"Merged adapter saved → {out_path}")

    return out_path


def _fedit_merge(state_dicts: dict, rank: int, dataset_sizes: dict) -> dict:
    """
    FedAvg for LoRA with dataset-size weighting.
    Computes weighted_avg(B_i @ A_i) per layer, then SVD re-decomposes into A, B.
    Non-LoRA keys are averaged directly.
    """
    all_keys     = list(next(iter(state_dicts.values())).keys())
    merged       = {}
    processed    : set = set()

    total_samples = sum(dataset_sizes.get(n, 1) for n in state_dicts)
    weights       = {n: dataset_sizes.get(n, 1) / max(total_samples, 1) for n in state_dicts}
    log.info(f"Merge weights: { {n: f'{w:.3f}' for n, w in weights.items()} }")

    lora_prefixes = {
        k.replace(".lora_A.weight", "")
        for k in all_keys
        if "lora_A" in k
    }

    for prefix in lora_prefixes:
        key_A = f"{prefix}.lora_A.weight"
        key_B = f"{prefix}.lora_B.weight"

        if key_A in all_keys and key_B in all_keys:
            delta_sum = sum(
                (state_dicts[n][key_B].float() @ state_dicts[n][key_A].float()) * weights[n]
                for n in state_dicts
            )
            U, S, Vh = torch.linalg.svd(delta_sum, full_matrices=False)
            sqrt_S   = torch.sqrt(S[:rank].clamp(min=1e-8))

            merged[key_B] = (U[:, :rank] * sqrt_S).contiguous().to(torch.float16)
            merged[key_A] = (torch.diag(sqrt_S) @ Vh[:rank, :]).contiguous().to(torch.float16)
            processed.update([key_A, key_B])

    for k in all_keys:
        if k not in processed:
            merged[k] = (
                sum(state_dicts[n][k].float() * weights[n] for n in state_dicts)
            ).contiguous().to(torch.float16)

    return merged