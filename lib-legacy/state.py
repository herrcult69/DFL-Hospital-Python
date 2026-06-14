# lib/state.py
from __future__ import annotations
import threading

_lock: threading.Lock = threading.Lock()
_state: dict = {"phase": "idle", "round": 0}


def set_state(phase: str, round_num: int) -> None:
    with _lock:
        _state["phase"] = phase
        _state["round"] = round_num


def get_state() -> dict:
    with _lock:
        return dict(_state)