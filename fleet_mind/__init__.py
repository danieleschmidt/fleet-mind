"""Fleet-Mind: Realtime Swarm LLM Coordination Platform.

A ROS 2 & WebRTC stack that streams latent-action plans from a central
GPT-4o-style coordinator to 100+ drones with <100ms end-to-end latency.

GENERATION 4: QUANTUM LEAP ENHANCEMENTS
- Quantum-inspired coordination algorithms
- Neuromorphic processing systems  
- 5G/6G edge computing integration
- Advanced research framework
"""

__version__ = "4.0.0"  # Generation 4: Quantum Leap
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

# Core Coordination (Generations 1-3)
from .coordination.swarm_coordinator import SwarmCoordinator, MissionStatus, MissionConstraints
from .communication.webrtc_streamer import WebRTCStreamer, MessagePriority, ReliabilityMode
from .communication.latent_encoder import LatentEncoder, CompressionType
from .planning.llm_planner import LLMPlanner, PlanningLevel
from .fleet.drone_fleet import DroneFleet, DroneStatus, DroneCapability
from .security import SecurityManager, SecurityLevel
from .monitoring import HealthMonitor, HealthStatus, AlertSeverity
from .utils.performance import performance_monitor, get_performance_summary
from .utils.concurrency import execute_concurrent, get_concurrency_stats
from .utils.auto_scaling import update_scaling_metric, get_autoscaling_stats

# Generation 4: Quantum-Inspired Systems
from .quantum import QuantumSwarmCoordinator, QuantumState, EntanglementPair, QuantumOptimizer

# Generation 4: Neuromorphic Processing
from .neuromorphic import SpikingCoordinator, SpikingNeuron, BioSwarmIntelligence

# Generation 4: Edge Computing
from .edge_computing import EdgeCoordinator, EdgeNode, FiveGOptimizer

# Generation 4: Research Framework
from .research_framework import AlgorithmResearcher, NovelAlgorithm, ExperimentalFramework

__all__ = [
    # Core Coordination (Generations 1-3)
    "SwarmCoordinator",
    "MissionStatus",
    "MissionConstraints",
    "WebRTCStreamer", 
    "MessagePriority",
    "ReliabilityMode",
    "LatentEncoder",
    "CompressionType",
    "LLMPlanner",
    "PlanningLevel",
    "DroneFleet",
    "DroneStatus",
    "DroneCapability",
    "SecurityManager",
    "SecurityLevel",
    "HealthMonitor", 
    "HealthStatus",
    "AlertSeverity",
    "performance_monitor",
    "get_performance_summary",
    "execute_concurrent",
    "get_concurrency_stats",
    "update_scaling_metric",
    "get_autoscaling_stats",
    
    # Generation 4: Quantum-Inspired Systems
    "QuantumSwarmCoordinator",
    "QuantumState", 
    "EntanglementPair",
    "QuantumOptimizer",
    
    # Generation 4: Neuromorphic Processing
    "SpikingCoordinator",
    "SpikingNeuron",
    "BioSwarmIntelligence",
    
    # Generation 4: Edge Computing
    "EdgeCoordinator",
    "EdgeNode",
    "FiveGOptimizer",
    
    # Generation 4: Research Framework
    "AlgorithmResearcher",
    "NovelAlgorithm",
    "ExperimentalFramework",
]