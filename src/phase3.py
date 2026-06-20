"""
Phase 3 — Aggregation orchestrator.
Called after Phase 2 ring transfer completes on all nodes.
"""

import asyncio
import logging
import time
from pathlib import Path

from .config  import NetworkConfig
from .state   import NodeState, Phase
from .gossip  import GossipEngine
from .models  import Rumor, RumorType
from .aggregator import aggregate

log = logging.getLogger(__name__)


class AggregationPhase:

    def __init__(self, state: NodeState, config: NetworkConfig, gossip: GossipEngine):
        self.state  = state
        self.config = config
        self.gossip = gossip

    async def run(self, chunk_dir: Path) -> None:
        log.info("=== Phase 3 started — FedAvg aggregation ===")

        # Determine who contributed files this round
        participants = [
            n for n in self.state.global_table
            if n not in self.state.no_model_set
        ]
        has_own = self.state.node_id not in self.state.no_model_set
        log.info(f"participant: {participants}, has own {has_own}")
        
        if not participants and not has_own:
            log.info("No models available this round — skipping aggregation")
            advanced = await self._ready_barrier_phase4()
            if advanced:
                self.state.phase = Phase.PHASE_4
                log.info("=== Phase flipped to PHASE_4 (no-model round) ===")
            return
        
        log.info("Submitting aggregation to thread executor...")
        # Run aggregation in a thread — CPU/IO bound, don't block event loop
        loop        = asyncio.get_event_loop()
        output_dir  = self.config.model_path
        chunk_dir_p = chunk_dir
        import traceback
        def _run_aggregate():
            try:
                return aggregate(
                    node_id       = self.state.node_id,
                    round_num     = self.state.round,
                    chunk_dir     = chunk_dir_p,
                    output_dir    = output_dir,
                    participants  = participants,
                    has_own_model = has_own,
                    dataset_size  = 1,
                    dataset_sizes = self.state.dataset_sizes,
                )
            except Exception:
                log.error(f"Aggregation thread raised:\n{traceback.format_exc()}")
                return None

        result = await loop.run_in_executor(None, _run_aggregate)
        log.info(f"Aggregation executor returned: {result}")

        if result is None:
            log.error("Aggregation failed — holding in Phase 3")
            return

        log.info(f"Aggregation complete → {result}")

        # READY barrier Phase 3→4
        advanced = await self._ready_barrier_phase4()
        if advanced:
            self.state.phase = Phase.PHASE_4
            log.info(f"=== Phase flipped to PHASE_4 | round: {self.state.round} ===")
        else:
            log.warning("Phase 3→4 barrier failed — holding")

    async def _ready_barrier_phase4(self) -> bool:
        self.state.ready_set_p4.add(self.state.node_id)
        ref             = self.state.phase2_start_ts if self.state.phase2_start_ts > 0 else time.time()
        shared_deadline = ref + self.config.phase3_total_budget

        rumor = Rumor.build(
            type          = RumorType.READY,
            originator_id = self.state.node_id,
            round         = self.state.round,
            ttl           = self.config.gossip_ttl,
            payload       = {"target_phase": "PHASE_4"},
        )
        self.state.mark_seen(rumor.rumor_id, rumor.model_dump())
        await self.gossip.spread(rumor)

        while True:
            await asyncio.sleep(0.5)

            if self.state.ready_set_p4 >= set(self.state.global_table):
                log.info("Phase 3→4 READY barrier cleared")
                return True

            if time.time() > shared_deadline:
                missing = set(self.state.global_table) - self.state.ready_set_p4
                log.warning(f"Phase 3→4 deadline reached — dropping: {missing}")
                for node in missing:
                    self.state.global_table.remove(node)
                    self.state.global_table.remove(node)
                if len(self.state.global_table) < 2:
                    self.state.table_locked           = False
                    self.state.ready_set_p4           = set()
                    self.state.last_table_change_time = time.time()
                    return False
                return True