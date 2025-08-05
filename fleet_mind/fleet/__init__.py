"""Fleet management components for drone coordination."""

from .drone_fleet import DroneFleet

# Optional imports - these modules may not exist yet
try:
    from .drone import Drone
except ImportError:
    Drone = None

try:
    from .fleet_manager import FleetManager
except ImportError:
    FleetManager = None

__all__ = ["DroneFleet"]

if Drone:
    __all__.append("Drone")
if FleetManager:
    __all__.append("FleetManager")