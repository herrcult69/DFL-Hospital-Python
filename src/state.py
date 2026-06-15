from dataclasses import dataclass, field


@dataclass
class NodeState:
    node_id:      str
    is_bootstrap: bool = False
    global_table: list[str] = field(default_factory=list)
    neighbor_map: set[str]  = field(default_factory=set)

    def add_node(self, node_id: str) -> bool:
        """Add a node to the global table. Returns True if it was new."""
        if node_id not in self.global_table:
            self.global_table.append(node_id)
            self.global_table.sort()
            return True
        return False

    def snapshot(self) -> dict:
        return {
            "node_id":      self.node_id,
            "is_bootstrap": self.is_bootstrap,
            "global_table": list(self.global_table),
            "neighbor_map": list(self.neighbor_map),
        }