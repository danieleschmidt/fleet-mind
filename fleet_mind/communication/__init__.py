"""Communication components for Fleet-Mind WebRTC networking."""

from .webrtc_streamer import WebRTCStreamer
from .latent_encoder import LatentEncoder

# Optional imports - these modules may not exist yet
try:
    from .mesh_network import MeshNetwork
except ImportError:
    MeshNetwork = None

try:
    from .qos_manager import QoSManager
except ImportError:
    QoSManager = None

__all__ = [
    "WebRTCStreamer",
    "LatentEncoder",
]

if MeshNetwork:
    __all__.append("MeshNetwork")
if QoSManager:
    __all__.append("QoSManager")