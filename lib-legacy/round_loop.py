# lib/round_loop.py
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import requests

from lib.state import get_state, set_state
from lib import local_trainer, aggregator, inference

log = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    node_id:        int
    output_dir:     str
    peer_addresses: dict[int, str]   # {peer_id: "ip:port"}
    total_rounds:   int
    poll_interval:  float = 3.0
    poll_timeout:   float = 600.0


# ── Peer polling ───────────────────────────────────────────────────────────────

def _wait_all_peers_done(cfg: NodeConfig, round_num: int) -> None:
    remaining = set(cfg.peer_addresses.keys())
    start     = time.time()

    while remaining:
        if time.time() - start > cfg.poll_timeout:
            log.warning(f"Timeout — skipping peers {remaining} for round {round_num}.")
            break

        for peer_id in list(remaining):
            addr = cfg.peer_addresses[peer_id]
            try:
                resp = requests.get(
                    f"http://{addr}/status",
                    timeout=5,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    peer_phase = body.get("phase")
                    peer_round = body.get("round")
                    
                    # Peer is done with training for `round_num` if it's in a later state 
                    # of the same round ("done", "aggregating", "idle") or in any state of a FUTURE round.
                    is_ready = (peer_round > round_num) or (
                        peer_round == round_num and peer_phase in ["done", "aggregating", "idle"]
                    )
                    
                    if is_ready:
                        log.info(f"Peer {peer_id} done for round {round_num}.")
                        remaining.discard(peer_id)
                    else:
                        pass
            except requests.exceptions.ConnectionError:
                log.warning(f"Peer {peer_id} @ {addr} — connection refused, retrying...")
            except Exception as e:
                log.warning(f"Peer {peer_id} @ {addr} — {e}, retrying...")

        if remaining:
            time.sleep(cfg.poll_interval)


# ── Main round loop ────────────────────────────────────────────────────────────

def run(cfg: NodeConfig) -> None:
    for round_num in range(1, cfg.total_rounds + 1):
        log.info(f"=== Round {round_num}/{cfg.total_rounds} ===")

        # Training — runs directly on main thread
        set_state("training", round_num)
        try:
            local_trainer.train(round_num)
            set_state("done", round_num)
        except Exception as e:
            log.error(f"Training failed: {e}")
            continue

        # Polling barrier — also main thread
        log.info("Local training done. Polling peers...")
        _wait_all_peers_done(cfg, round_num)

        # Aggregation — also main thread
        set_state("aggregating", round_num)
        try:
            aggregator.collect_and_aggregate(
                peer_addresses=cfg.peer_addresses,
                round_num=round_num,
                output_dir=cfg.output_dir,
                self_node_id=cfg.node_id,
            )
            inference.invalidate_cache()
            set_state("idle", round_num)
        except Exception as e:
            log.error(f"Aggregation failed: {e}")
            set_state("idle", round_num)

        log.info(f"Round {round_num} complete.")

    log.info("All rounds finished. Flask stays up for /predict.")