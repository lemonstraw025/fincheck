"""Transaction-graph construction and structural pattern detection."""

from fincheck.graph.builder import build_graph, edge_frame
from fincheck.graph.patterns import (
    PatternHit,
    detect_cycles,
    detect_fan_outs,
    detect_peeling_chains,
    detect_all_patterns,
)

__all__ = [
    "build_graph",
    "edge_frame",
    "PatternHit",
    "detect_cycles",
    "detect_fan_outs",
    "detect_peeling_chains",
    "detect_all_patterns",
]
