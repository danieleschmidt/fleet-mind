"""Quantum-Inspired Coordination for Next-Generation Drone Swarms.

This module implements cutting-edge quantum-inspired algorithms for:
- Quantum superposition-based path planning
- Entanglement-inspired coordination protocols
- Quantum error correction for ultra-reliable communication
- Quantum annealing for optimization problems
"""

from .quantum_coordinator import QuantumSwarmCoordinator, QuantumState, EntanglementPair
from .quantum_optimizer import QuantumOptimizer, QuantumAnnealingSchedule
from .quantum_communication import QuantumCommunication, QuantumChannel
from .quantum_error_correction import QuantumErrorCorrection, SyndromeMeasurement

__all__ = [
    "QuantumSwarmCoordinator",
    "QuantumState", 
    "EntanglementPair",
    "QuantumOptimizer",
    "QuantumAnnealingSchedule",
    "QuantumCommunication",
    "QuantumChannel",
    "QuantumErrorCorrection",
    "SyndromeMeasurement",
]