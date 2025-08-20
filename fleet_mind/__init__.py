"""Fleet-Mind: Realtime Swarm LLM Coordination Platform.

A ROS 2 & WebRTC stack that streams latent-action plans from a central
GPT-4o-style coordinator to 100+ drones with <100ms end-to-end latency.

GENERATION 8: PROGRESSIVE QUALITY GATES
- Intelligent quality monitoring with ML-powered prediction
- Progressive testing framework with adaptive test generation
- Continuous performance optimization with real-time improvement
- Advanced compliance automation with dynamic regulatory adherence
- Proactive reliability engineering with predictive prevention and self-healing
- Comprehensive quality gate orchestration and certification
"""

__version__ = "8.0.0"  # Generation 8: Progressive Quality Gates
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

# Generation 5: Next Frontier Systems
from .consciousness import SwarmConsciousness, CollectiveIntelligence, EmergentBehavior
from .bio_hybrid import BioHybridDrone, SynapticInterface, BiologicalSensor
from .dimensional import MultidimensionalCoordinator, QuantumTunnel, SpacetimeNavigator
from .evolution import SelfEvolvingSwarm, GeneticOptimizer, AutonomousDesigner

# Generation 6: Ultimate Convergence
from .convergence import UltimateCoordinator

# Generation 8: Progressive Quality Gates
from .quality_gates import (
    IntelligentQualityMonitor,
    QualityMetric,
    QualityThreshold,
    ProgressiveTestingFramework,
    AdaptiveTestGenerator,
    ContinuousPerformanceOptimizer,
    OptimizationStrategy,
    ComplianceAutomation,
    ComplianceFramework,
    ProactiveReliabilityEngine,
    ReliabilityPrediction,
    QualityGateOrchestrator
)

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
    
    # Generation 5: Next Frontier Systems
    "SwarmConsciousness",
    "CollectiveIntelligence", 
    "EmergentBehavior",
    "BioHybridDrone",
    "SynapticInterface",
    "BiologicalSensor",
    "MultidimensionalCoordinator",
    "QuantumTunnel",
    "SpacetimeNavigator",
    "SelfEvolvingSwarm",
    "GeneticOptimizer",
    "AutonomousDesigner",
    
    # Generation 6: Ultimate Convergence
    "UltimateCoordinator",
    
    # Generation 8: Progressive Quality Gates
    "IntelligentQualityMonitor",
    "QualityMetric",
    "QualityThreshold",
    "ProgressiveTestingFramework",
    "AdaptiveTestGenerator",
    "ContinuousPerformanceOptimizer",
    "OptimizationStrategy",
    "ComplianceAutomation",
    "ComplianceFramework",
    "ProactiveReliabilityEngine", 
    "ReliabilityPrediction",
    "QualityGateOrchestrator",
]