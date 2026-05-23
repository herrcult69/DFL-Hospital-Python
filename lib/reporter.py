import threading
import time
import requests
import logging
from lib.state import get_state

log = logging.getLogger(__name__)

class Reporter:
    """
    Handles state reporting to a central dashboard.
    Reports both on explicit transitions and periodically (heartbeat).
    """
    def __init__(self):
        self.node_id = None
        self.dashboard_url = None
        self.interval = 10
        self._thread = None
        self._stop_event = threading.Event()

    def start(self, node_id: int, dashboard_url: str, interval: int = 10):
        """Initializes settings and starts the periodic heartbeat thread"""
        self.node_id = node_id
        self.dashboard_url = dashboard_url
        self.interval = interval

        if self.dashboard_url:
            self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._thread.start()
            log.info(f"Reporter started, reporting to {dashboard_url} every {interval}s")

    def stop(self):
        """Stops the periodic heartbeat loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _heartbeat_loop(self):
        """Periodic loop to report state."""
        while not self._stop_event.is_set():
            # Wait first, so we don't immediately report after start/transition
            self._stop_event.wait(self.interval)
            
            if not self._stop_event.is_set():
                state = get_state()
                self.report_status(state["phase"], state["round"])

    def report_status(self, phase: str, round_num: int):
        """Sends data immediately to the dashboard if configured."""
        if not self.dashboard_url:
            return
        
        payload = {
            "node_id": self.node_id,
            "status": phase,
            "round": round_num,
            "timestamp": time.time()
        }
        try:
            requests.post(self.dashboard_url, json=payload, timeout=5)
        except Exception as e:
            log.debug(f"Failed to report status to dashboard: {e}")

# Global singleton
reporter = Reporter()
