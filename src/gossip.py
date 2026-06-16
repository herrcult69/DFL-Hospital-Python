"""
GossipEngine — runs on every node (bootstrap and worker).
Handles SIR rumor spreading.
"""
import logging
import httpx, time

from .models import Rumor, RumorType
from .state import NodeState, Phase
from .config import NetworkConfig

log = logging.getLogger(__name__)


class GossipEngine:

    def __init__(self, state: NodeState, config: NetworkConfig):
        self.state  = state
        self.config = config

    def _gossip_addr(self, node_id: str) -> str:
        host, gossip_port, _ = node_id.split(":")
        return f"http://{host}:{gossip_port}"

    async def spread(self, rumor: Rumor, exclude: set[str] | None = None) -> None:
        """Send rumor to all K neighbors except excluded node_ids."""
        targets = self.state.neighbor_map - (exclude or set()) 
        async with httpx.AsyncClient(timeout=self.config.http_timeout) as client:
            for neighbor in targets:
                try:
                    await client.post(
                        f"{self._gossip_addr(neighbor)}/gossip",
                        json=rumor.model_dump(),
                        headers={"X-Sender-Id": self.state.node_id},
                    )
                except Exception as e:
                    log.warning(f"Could not reach {neighbor}: {e}")

    async def receive(self, rumor: Rumor, sender_id: str) -> None:
        """
        SIR receive logic:
        1. Discard if already seen (dedup via rumor_id)
        2. Discard if round mismatch
        3. Discard if TTL is 0
        4. Process the rumor (update local state)
        5. Decrement TTL and forward to neighbors except sender
        """
        # 1. dedup
        if self.state.is_seen(rumor.rumor_id):
            return
        self.state.mark_seen(rumor.rumor_id)

        # 2. round check
        if rumor.round != self.state.round:
            log.debug(f"Stale rumor discarded: {rumor.rumor_id} (round {rumor.round} != {self.state.round})")
            return

        # 3. TTL check
        if rumor.ttl <= 0:
            log.debug(f"Rumor TTL exhausted: {rumor.rumor_id}")
            return

        # 4. process
        self._process(rumor)

        # 5. forward with decremented TTL
        forwarded = rumor.model_copy(update={"ttl": rumor.ttl - 1})
        await self.spread(forwarded, exclude={sender_id})

    def _process(self, rumor: Rumor) -> None:
        """Apply rumor effects to local state."""
        if rumor.type == RumorType.JOIN:
            new_node = rumor.payload.get("node_id")
            if new_node and new_node not in self.state.global_table:
                self.state.add_node(new_node)
                log.info(f"Learned about new node via gossip: {new_node}")

        elif rumor.type == RumorType.HEARTBEAT:
            self.state.heartbeat_seen.add(rumor.originator_id)
            log.debug(f"Heartbeat received from: {rumor.originator_id}")
            
        elif rumor.type == RumorType.READY:
            target = rumor.payload.get("target_phase")
            if (
                target == Phase.PHASE_2.value                
                and rumor.originator_id in self.state.global_table #  sanity check
            ):
                self.state.ready_set.add(rumor.originator_id)
                log.debug(f"READY received from {rumor.originator_id} | ready_set: {len(self.state.ready_set)}")
    
    async def originate_heartbeat(self) -> None:
        """Build and spread a fresh heartbeat. beat is a monotonic counter."""
        ts = time.time()
        rumor = Rumor(
            type=RumorType.HEARTBEAT,
            originator_id=self.state.node_id,
            round=self.state.round,
            rumor_id=f"HEARTBEAT:{self.state.node_id}:{self.state.round}:{ts}",
            ttl=self.config.gossip_ttl,
        )
        # Mark it seen locally so we don't re-process our own heartbeat
        self.state.mark_seen(rumor.rumor_id)
        self.state.heartbeat_seen.add(self.state.node_id)  # count ourselves as alive
        await self.spread(rumor)