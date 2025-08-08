"""Fleet-Mind: Realtime Swarm LLM Coordination Platform.

A ROS 2 & WebRTC stack that streams latent-action plans from a central
GPT-4o-style coordinator to 100+ drones with <100ms end-to-end latency.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

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

__all__ = [
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
]