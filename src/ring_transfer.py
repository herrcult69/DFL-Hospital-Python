"""
Phase 2: Ring All-Gather via gRPC Bidirectional Streaming.

This module implements the data plane of the DFL protocol.

Key design:
  1. Pre-allocation: Split Safetensors into N equal chunks.
  2. Zero-Copy Disk Writes: Incoming chunk_bytes are written directly to
     disk via `safetensors` mmap. We never load the full model into RAM.
  3. Push Pipeline: Each received chunk is immediately forwarded to the
     right neighbor's gRPC stream.
  4. Dynamic TTL: Stream terminates when
       hops_taken == (total_nodes - 1) - len(dead_nodes_detected)
  5. Fault Tolerance: On gRPC timeout, skip to next valid node in the
     sorted ring, open new stream, append dead IP.

Supports Phase 3: All chunks land in `data/received_models/{originator_ip}.safetensors`
for mmap-based FedAvg.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Dict, List, Optional, Set

import grpc
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

# These imports assume the generated proto stubs exist at:
# protos/ring_transfer_pb2.py and protos/ring_transfer_pb2_grpc.py
import protos.ring_transfer_pb2 as pb2
import protos.ring_transfer_pb2_grpc as pb2_grpc

from src.config import NodeConfig
from src.state import NodeState, Phase, grpc_addr_from_node_id

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_table_hash(table: List[str]) -> str:
    raw = ",".join(sorted(table)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _host_from_node_id(node_id: str) -> str:
    """Convert a 'IP:Port' node ID to 'IP:GossipPort' for gRPC use."""
    # node_id is "ip:gossip_port" — we use the same host with the grpc_port
    return node_id


# ── Chunk splitting (Safetensors → N equal parts) ────────────────────────────

def split_safetensors(
    model_path: str,
    n_chunks: int,
    chunk_size: int,
    output_dir: str,
) -> List[str]:
    """
    Split a .safetensors file into N equal-sized chunk files on disk.

    Uses memory-mapped reading to avoid loading the entire model into RAM.

    Returns a list of chunk file paths.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Get total file size
    total_size = os.path.getsize(model_path)
    chunk_size_bytes = max(chunk_size, total_size // n_chunks + 1)

    chunk_paths: List[str] = []
    offset = 0
    chunk_idx = 0

    os.makedirs(output_dir, exist_ok=True)

    # Read the safetensors header to get tensor metadata
    with open(model_path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header_bytes = f.read(header_len)
        # We'll use safetensors' safe_open for mmap-based reading later

    while offset < total_size:
        end = min(offset + chunk_size_bytes, total_size)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx}.bin")

        # Memory-map the source file for this chunk range
        with open(model_path, "rb") as src:
            src.seek(offset)
            data = src.read(end - offset)

        with open(chunk_path, "wb") as dst:
            dst.write(data)

        chunk_paths.append(chunk_path)
        offset = end
        chunk_idx += 1

    log.info(
        "Split %s into %d chunks (%d bytes each)",
        model_path,
        len(chunk_paths),
        chunk_size_bytes,
    )
    return chunk_paths


def merge_safetensors(
    chunk_dir: str,
    originator_id: str,
    n_chunks: int,
    output_path: str,
) -> str:
    """
    Reassemble chunk files for a given originator into a single .safetensors file.

    Used during Phase 3 aggregation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as out:
        for i in range(n_chunks):
            chunk_path = os.path.join(chunk_dir, f"chunk_{i}.bin")
            if not os.path.exists(chunk_path):
                log.warning("Missing chunk %d for %s, skipping", i, originator_id)
                continue
            with open(chunk_path, "rb") as f:
                while True:
                    buf = f.read(8 * 1024 * 1024)  # 8 MB buffer
                    if not buf:
                        break
                    out.write(buf)

    log.info("Merged %d chunks for %s into %s", n_chunks, originator_id, output_path)
    return output_path


# ── gRPC Servicer ─────────────────────────────────────────────────────────────

class RingTransferServicer(pb2_grpc.RingTransferServicer):
    """
    gRPC servicer for bidirectional streaming.

    Each node runs one of these. The servicer:
      - Accepts incoming chunks from the left neighbor.
      - Writes each chunk directly to disk via a file handle.
      - Forwards each chunk to the right neighbor using a persistent gRPC
        channel (no per-chunk TCP handshake).
      - Tracks TTL and dead nodes.
    """

    def __init__(
        self, node_state: NodeState, config: NodeConfig
    ) -> None:
        self.state = node_state
        self.config = config
        self._data_dir = config.received_models_dir

        # Persistent gRPC channel for forwarding
        self._fwd_channel: Optional[grpc.aio.Channel] = None
        self._fwd_stub: Optional[pb2_grpc.RingTransferStub] = None
        self._fwd_target: str = ""

        # Track open file handles for zero-copy disk writes
        self._file_handles: Dict[str, object] = {}
        # Track total chunks received per originator (for Phase 3)
        self._received_counts: Dict[str, int] = {}

    async def start(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)

    async def stop(self) -> None:
        await self._close_fwd_channel()
        for fh in self._file_handles.values():
            if hasattr(fh, "close"):
                fh.close()
        self._file_handles.clear()

    async def _close_fwd_channel(self) -> None:
        if self._fwd_channel:
            await self._fwd_channel.close()
            self._fwd_channel = None
            self._fwd_stub = None
            self._fwd_target = ""

    async def _ensure_fwd_channel(self, target: str) -> bool:
        """Ensure the persistent forward channel targets `target`.
        Close + reopen if the target changed. Returns True on success."""
        if target == self._fwd_target and self._fwd_stub is not None:
            return True
        await self._close_fwd_channel()
        grpc_addr = grpc_addr_from_node_id(target)
        try:
            self._fwd_channel = grpc.aio.insecure_channel(grpc_addr)
            self._fwd_stub = pb2_grpc.RingTransferStub(self._fwd_channel)
            self._fwd_target = target
            return True
        except Exception as e:
            log.warning("Failed to create channel to %s: %s", target, e)
            return False

    # ── gRPC StreamChunks handler ─────────────────────────────────────────

    async def StreamChunks(  # noqa: N802
        self,
        request_iterator: grpc.aio.AsyncIterable[pb2.Chunk],
        context: grpc.aio.ServicerContext,
    ) -> grpc.aio.AsyncIterable[pb2.Ack]:
        """
        Handle incoming bidirectional stream from the left neighbor.

        For each Chunk:
          1. Write chunk_bytes to disk (zero-copy via append).
          2. Forward the chunk to the right neighbor.
          3. Check TTL: if hops_taken >= TTL, terminate.
          4. Send Ack back to sender.
        """
        total_nodes = len(self.state.global_table)

        async for chunk in request_iterator:
            originator = chunk.originator_ip
            chunk_id = chunk.chunk_id
            chunk_data = chunk.chunk_bytes
            hops = chunk.hops_taken
            dead_nodes = list(chunk.dead_nodes_detected)

            # ── Zero-copy disk write ──────────────────────────────────────
            await self._write_chunk_to_disk(originator, chunk_id, chunk_data)

            # ── TTL check ─────────────────────────────────────────────────
            ttl = (total_nodes - 1) - len(dead_nodes)
            if hops >= ttl:
                log.info(
                    "TTL reached for %s chunk %d (hops=%d, ttl=%d) — terminating stream",
                    originator,
                    chunk_id,
                    hops,
                    ttl,
                )
                yield pb2.Ack(
                    chunk_id=chunk_id,
                    status="ok",
                )
                continue

            # ── Forward to right neighbor ─────────────────────────────────
            forwarded = await self._forward_chunk(chunk, dead_nodes)
            if not forwarded:
                log.warning("Failed to forward chunk %d from %s", chunk_id, originator)
                # Try ring patching: skip dead right neighbor
                dead_nodes.append(self.state.right_neighbor)
                new_right = self._find_next_alive(dead_nodes)
                if new_right:
                    log.info("Patching ring: forwarding via %s instead", new_right)
                    forwarded = await self._forward_chunk(chunk, dead_nodes, override_target=new_right)

            yield pb2.Ack(
                chunk_id=chunk_id,
                status="ok" if forwarded else "error",
            )

    # ── Zero-copy disk write ──────────────────────────────────────────────

    async def _write_chunk_to_disk(
        self, originator: str, chunk_id: int, data: bytes
    ) -> None:
        """
        Append chunk bytes directly to disk.

        Each originator gets one file:
          data/received_models/{originator_hash}.safetensors

        We write chunks sequentially by chunk_id so Phase 3 can mmap-read
        the complete file.
        """
        # Sanitize filename
        safe_name = originator.replace(":", "_").replace(".", "_")
        file_path = os.path.join(self._data_dir, f"{safe_name}.safetensors")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Append write: open in binary append mode
        # (In production this would use a proper mmap region; for now
        #  write sequentially to disk which Safetensors can mmap later.)
        loop = asyncio.get_event_loop()

        def _write() -> None:
            with open(file_path, "ab") as f:
                f.write(data)

        await loop.run_in_executor(None, _write)

        # Track count for merge validation
        self._received_counts[originator] = self._received_counts.get(originator, 0) + 1
        log.debug("Wrote chunk %d from %s to disk (%d bytes)", chunk_id, originator, len(data))

    # ── Forward chunk to right neighbor ───────────────────────────────────

    async def _forward_chunk(
        self,
        chunk: pb2.Chunk,
        dead_nodes: List[str],
        override_target: Optional[str] = None,
    ) -> bool:
        """Forward a chunk to the right neighbor (or override target).
        Uses a persistent gRPC channel — no per-chunk TCP handshake."""
        target = override_target or self.state.right_neighbor
        if not target:
            log.error("No right neighbor to forward to")
            return False

        # Ensure the persistent channel is pointed at the right target
        if not await self._ensure_fwd_channel(target):
            return False

        assert self._fwd_stub is not None

        # Build the forwarded chunk with incremented hop
        forwarded = pb2.Chunk(
            chunk_bytes=chunk.chunk_bytes,
            originator_ip=chunk.originator_ip,
            chunk_id=chunk.chunk_id,
            hops_taken=chunk.hops_taken + 1,
            total_chunks=chunk.total_chunks,
            dead_nodes_detected=dead_nodes,
        )

        try:
            async for ack in self._fwd_stub.StreamChunks(iter([forwarded])):
                if ack.status == "ok":
                    return True
                else:
                    log.warning("Nack for chunk %d: %s", chunk.chunk_id, ack.status)
                    return False
        except grpc.aio.AioRpcError as e:
            log.warning(
                "gRPC forward to %s failed: %s — marking as dead",
                target,
                e,
            )
            return False
        except Exception as e:
            log.warning("Unexpected error forwarding to %s: %s", target, e)
            return False

    # ── Ring patching helper ──────────────────────────────────────────────

    def _find_next_alive(self, dead_nodes: List[str]) -> Optional[str]:
        """Find the next valid node in the sorted ring after skipping dead ones."""
        table = sorted(self.state.global_table)
        dead_set = set(dead_nodes)
        my_id = self.state.snapshot()["node_id"]

        try:
            idx = table.index(my_id)
        except ValueError:
            return None

        # Scan forward in the ring for a live node
        for offset in range(1, len(table) + 1):
            candidate = table[(idx + offset) % len(table)]
            if candidate not in dead_set and candidate != my_id:
                return candidate

        return None


# ── Ring Transfer Client (initiated by Phase 1 → Phase 2 transition) ─────────

class RingTransferClient:
    """
    Client that initiates the Phase 2 ring all-gather.

    Steps:
      1. Split the local model into N chunks.
      2. Open a persistent gRPC stream to the right neighbor.
      3. Send all chunks, each with originator=self, hops=1.
      4. Wait for acks.
      5. Also receive chunks from the left neighbor (handled by the server).
    """

    def __init__(
        self,
        node_state: NodeState,
        config: NodeConfig,
        servicer: RingTransferServicer,
    ) -> None:
        self.state = node_state
        self.config = config
        self.servicer = servicer
        # Persistent gRPC channel for the client side
        self._client_channel: Optional[grpc.aio.Channel] = None
        self._client_stub: Optional[pb2_grpc.RingTransferStub] = None
        self._client_target: str = ""

    async def _close_client_channel(self) -> None:
        if self._client_channel:
            await self._client_channel.close()
            self._client_channel = None
            self._client_stub = None
            self._client_target = ""

    async def _ensure_client_channel(self, target: str) -> bool:
        """Ensure the persistent client channel targets `target`."""
        if target == self._client_target and self._client_stub is not None:
            return True
        await self._close_client_channel()
        grpc_addr = grpc_addr_from_node_id(target)
        try:
            self._client_channel = grpc.aio.insecure_channel(grpc_addr)
            self._client_stub = pb2_grpc.RingTransferStub(self._client_channel)
            self._client_target = target
            return True
        except Exception as e:
            log.warning("Failed to create client channel to %s: %s", target, e)
            return False

    async def run_ring_all_gather(self) -> None:
        """
        Execute the full Phase 2 ring transfer.

        Preconditions:
          - Phase 1 quiescence reached, ring neighbors computed.
          - Local model saved as safetensors in output_adapter_dir.
        """
        my_id = self.state.snapshot()["node_id"]
        total_nodes = len(self.state.global_table)
        right = self.state.right_neighbor

        if total_nodes < 2:
            log.info("Only node in network — skipping ring transfer")
            await self._transition_to_phase_3()
            return

        # 1. Split local model into chunks
        model_path = os.path.join(self.config.output_adapter_dir, "model.safetensors")
        if not os.path.exists(model_path):
            log.warning("No local model at %s — using empty chunk", model_path)
            # Create an empty safetensors file
            os.makedirs(self.config.output_adapter_dir, exist_ok=True)
            save_file({}, model_path)

        chunk_dir = os.path.join(self.config.data_dir, "outgoing_chunks")
        chunk_paths = split_safetensors(
            model_path=model_path,
            n_chunks=total_nodes,
            chunk_size=self.config.chunk_size_bytes,
            output_dir=chunk_dir,
        )
        n_chunks = len(chunk_paths)

        log.info(
            "Starting ring all-gather: %d chunks, %d nodes, right=%s",
            n_chunks,
            total_nodes,
            right,
        )

        # 2. Open stream and send chunks
        dead_nodes: List[str] = []

        for i, chunk_path in enumerate(chunk_paths):
            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()

            chunk_msg = pb2.Chunk(
                chunk_bytes=chunk_bytes,
                originator_ip=my_id,
                chunk_id=i,
                hops_taken=1,
                total_chunks=n_chunks,
                dead_nodes_detected=dead_nodes,
            )

            success = await self._send_chunk_with_retry(chunk_msg, dead_nodes)
            if success:
                log.debug("Sent chunk %d/%d", i + 1, n_chunks)
            else:
                log.error("Failed to send chunk %d/%d after retries", i + 1, n_chunks)

        log.info("Ring all-gather complete for round %d", self.state.round)

        # 3. Wait briefly for left neighbor's chunks to finish streaming
        await asyncio.sleep(2.0)

    async def _send_chunk_with_retry(
        self,
        chunk: pb2.Chunk,
        dead_nodes: List[str],
        max_retries: int = 3,
    ) -> bool:
        """Send a chunk to the right neighbor, with ring-patching retry.
        Uses a persistent gRPC channel."""
        target = self.state.right_neighbor

        for attempt in range(max_retries):
            # Point the persistent channel at this target
            if not await self._ensure_client_channel(target):
                # Channel creation failed — treat as dead
                if target not in dead_nodes:
                    dead_nodes.append(target)
                new_target = self.servicer._find_next_alive(dead_nodes)
                if new_target and new_target != self.state.snapshot()["node_id"]:
                    target = new_target
                    chunk.dead_nodes_detected.extend(dead_nodes)
                    log.info("Retrying chunk via %s (attempt %d)", target, attempt + 1)
                    continue
                else:
                    log.error("No alive target found for ring patching")
                    return False

            assert self._client_stub is not None

            try:
                async for ack in self._client_stub.StreamChunks(iter([chunk])):
                    if ack.status == "ok":
                        return True
                    return False
            except grpc.aio.AioRpcError as e:
                log.warning(
                    "Attempt %d: gRPC error sending to %s: %s",
                    attempt + 1,
                    target,
                    e,
                )
                # Close the broken channel so the next attempt reconnects
                await self._close_client_channel()
                # Ring patching: try next alive node
                if target not in dead_nodes:
                    dead_nodes.append(target)
                new_target = self.servicer._find_next_alive(dead_nodes)
                if new_target and new_target != self.state.snapshot()["node_id"]:
                    target = new_target
                    chunk.dead_nodes_detected.extend(dead_nodes)
                    log.info("Retrying chunk via %s (attempt %d)", target, attempt + 1)
                    continue
                else:
                    log.error("No alive target found for ring patching")
                    return False

        return False

    async def _transition_to_phase_3(self) -> None:
        """Move the state machine to Phase 3 (aggregation)."""
        self.state.phase = Phase.PHASE_3_AGG
        log.info("Transitioned to Phase 3 (Aggregation)")