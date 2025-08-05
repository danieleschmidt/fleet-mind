"""Coordination components for Fleet-Mind swarm management."""

from .swarm_coordinator import SwarmCoordinator

# Optional imports - these modules may not exist yet
try:
    from .hierarchical_planner import HierarchicalPlanner
except ImportError:
    HierarchicalPlanner = None

try:
    from .consensus import ConsensusManager
except ImportError:
    ConsensusManager = None

try:
    from .fault_tolerance import FaultTolerance
except ImportError:
    FaultTolerance = None

__all__ = ["SwarmCoordinator"]

if HierarchicalPlanner:
    __all__.append("HierarchicalPlanner")
if ConsensusManager:
    __all__.append("ConsensusManager")
if FaultTolerance:
    __all__.append("FaultTolerance")