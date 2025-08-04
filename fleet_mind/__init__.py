"""Fleet-Mind: Realtime Swarm LLM Coordination Platform.

A ROS 2 & WebRTC stack that streams latent-action plans from a central
GPT-4o-style coordinator to 100+ drones with <100ms end-to-end latency.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

from .coordination.swarm_coordinator import SwarmCoordinator
from .communication.webrtc_streamer import WebRTCStreamer
from .communication.latent_encoder import LatentEncoder
from .planning.llm_planner import LLMPlanner
from .fleet.drone_fleet import DroneFleet

__all__ = [
    "SwarmCoordinator",
    "WebRTCStreamer", 
    "LatentEncoder",
    "LLMPlanner",
    "DroneFleet",
]