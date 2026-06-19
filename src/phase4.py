"""
phase4.py — Round completion, training, and multi-round reset.

Responsibilities:
  1. Archive the merged adapter produced by Phase 3.
  2. Run local training on that merged adapter (in thread executor).
  3. Stamp trained adapter as round{N+1}_{node_id}.safetensors
     so Phase 2 of the next round can find it.
  4. Dead-node K-graph eviction (bootstrap only, calls /rewire-evict).
  5. Chunk directory cleanup.
  6. Increment round counter + call state.reset_phase1() → back to Phase 1.

Called from _phase2_loop() in bootstrap.py / worker.py after
AggregationPhase.run() sets state.phase = PHASE_4.

Training note:
  - Phase 3 writes:  model_dir/round{N}_{node_id}_adapter.safetensors  (merged FedAvg result)
  - Phase 4 trains on that file and stamps output as round{N+1}_{node_id}.safetensors
  - Phase 2 (next round) reads:  model_dir/round{N+1}_{node_id}.safetensors
  - Round 0: no merged adapter exists yet → lib/local_trainer does fresh LoRA init.
    Output stamped as round1_{node_id}.safetensors, picked up by Phase 2 round 1.
"""

import asyncio
import logging
import time
import traceback
from pathlib import Path

import httpx

from .config  import NetworkConfig
from .state   import NodeState, Phase
from .gossip  import GossipEngine

log = logging.getLogger(__name__)


class RoundCompletionPhase:

    def __init__(self, state: NodeState, config: NetworkConfig, gossip: GossipEngine):
        self.state  = state
        self.config = config
        self.gossip = gossip

    async def run(self, chunk_dir: Path) -> None:
        safe_id = self.state.node_id.replace(":", "_")
        log.info(f"=== Phase 4 started — training + round reset | round: {self.state.round} ===")

        # ── 1. Archive the merged adapter from Phase 3 ────────────────────────
        merged_basename = f"round{self.state.round}_{safe_id}_adapter.safetensors"
        merged_path = self.config.model_path / merged_basename
        if merged_path.exists():
            archive = self.config.model_path / merged_basename  # already per-node
            log.info(f"Phase 3 merged adapter ready: {merged_path}")
        else:
            log.warning(f"Merged adapter not found at {merged_path} — "
                        "Phase 3 may have failed; training will start from a fresh LoRA")

        # ── 2. Local training on merged adapter ───────────────────────────────
        adapter_path = await self._run_training()

        if adapter_path is None:
            log.warning(
                "Training produced no adapter — this node will gossip NO_MODEL "
                "in Phase 2 of the next round (file missing → ring_transfer handles it)"
            )
        else:
            log.info(f"Training complete → {adapter_path}")

        # ── 3. Dead-node K-graph eviction (bootstrap only) ────────────────────
        await self._evict_dead_nodes()

        # ── 4. Chunk directory cleanup ────────────────────────────────────────
        self._cleanup_chunks(chunk_dir)

        # ── 5. READY barrier — wait for all nodes to finish training ─────────
        advanced = await self._ready_barrier_phase1()
        if not advanced:
            log.warning("Phase 4 barrier failed — holding")
            return

        # ── 6. Advance round + reset ──────────────────────────────────────────
        completed = self.state.round
        self.state.round += 1
        self.state.reset_phase1()
        log.info(
            f"=== Round {completed} complete → Phase 1 | "
            f"new round: {self.state.round} ==="
        )

    # ── training ──────────────────────────────────────────────────────────────

    async def _run_training(self) -> Path | None:
        """Runs blocking training in a thread executor so the event loop is free."""
        from .trainer import LocalTrainer

        loop = asyncio.get_event_loop()

        def _sync() -> Path | None:
            try:
                trainer = LocalTrainer(
                    node_id      = self.state.node_id,
                    round_num    = self.state.round,
                    model_dir    = self.config.model_path,
                    dataset_path = Path(self.config.dataset_path),
                )
                return trainer.train()
            except Exception:
                log.error(f"Training raised:\n{traceback.format_exc()}")
                return None

        return await loop.run_in_executor(None, _sync)

    # ── dead-node eviction ────────────────────────────────────────────────────

    async def _evict_dead_nodes(self) -> None:
        """Bootstrap calls /rewire-evict to remove dead nodes from the K-graph."""
        if not self.state.is_bootstrap:
            return
        if not self.state.dead_this_round:
            return

        dead = list(self.state.dead_this_round)
        log.info(f"Bootstrap evicting dead nodes before round reset: {dead}")
        try:
            async with httpx.AsyncClient(timeout=self.config.http_timeout) as client:
                await client.post(
                    f"http://{self.config.host}:{self.config.port}/rewire-evict",
                    json={"dead_nodes": dead},
                )
        except Exception as e:
            log.warning(f"Rewire-evict call failed (non-fatal): {e}")

    # ── chunk cleanup ─────────────────────────────────────────────────────────

    def _cleanup_chunks(self, chunk_dir: Path) -> None:
        """Remove round{N}_*.safetensors files left over from Phase 2."""
        pattern = f"round{self.state.round}_*.safetensors"
        removed = 0
        for f in chunk_dir.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                log.warning(f"Could not remove chunk {f}: {e}")
        log.info(f"Chunk cleanup: {removed} file(s) removed from {chunk_dir}")

    async def _ready_barrier_phase1(self) -> bool:
        from .models import Rumor, RumorType
        self.state.ready_set_p1.add(self.state.node_id)  # ← need new set in state
        deadline = time.time() + self.config.phase4_timeout

        rumor = Rumor.build(
            type          = RumorType.READY,
            originator_id = self.state.node_id,
            round         = self.state.round,
            ttl           = self.config.gossip_ttl,
            payload       = {"target_phase": "PHASE_1"},
        )
        self.state.mark_seen(rumor.rumor_id)
        await self.gossip.spread(rumor)

        while True:
            await asyncio.sleep(0.5)
            if self.state.ready_set_p1 >= set(self.state.global_table):
                log.info("Phase 4→1 barrier cleared — all nodes done training")
                return True
            if time.time() > deadline:
                missing = set(self.state.global_table) - self.state.ready_set_p1
                log.warning(f"Phase 4→1 timeout — dropping: {missing}")
                for node in missing:
                    self.state.global_table.remove(node)
                return len(self.state.global_table) >= 2