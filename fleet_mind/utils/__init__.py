"""Utility components for Fleet-Mind."""

from .logging import get_logger, setup_logging
from .validation import validate_mission_constraints, validate_drone_state
from .security import SecurityManager, encrypt_message, decrypt_message

# Optional imports - these modules may not exist yet
try:
    from .metrics import MetricsCollector, PerformanceMonitor
except ImportError:
    MetricsCollector = None
    PerformanceMonitor = None

try:
    from .config import ConfigManager, load_config
except ImportError:
    ConfigManager = None
    load_config = None

try:
    from .performance import cached, async_cached, performance_monitor, get_performance_summary
except ImportError:
    cached = lambda *args, **kwargs: lambda f: f
    async_cached = lambda *args, **kwargs: lambda f: f
    performance_monitor = lambda f: f
    get_performance_summary = lambda: {}

try:
    from .concurrency import execute_concurrent, get_concurrency_stats
except ImportError:
    execute_concurrent = None
    get_concurrency_stats = lambda: {}

try:
    from .auto_scaling import update_scaling_metric, get_autoscaling_stats
except ImportError:
    update_scaling_metric = lambda name, value: None
    get_autoscaling_stats = lambda: {}

__all__ = [
    "get_logger",
    "setup_logging",
    "validate_mission_constraints",
    "validate_drone_state", 
    "SecurityManager",
    "encrypt_message",
    "decrypt_message",
]

# Add performance utilities if available
if 'performance' in locals():
    __all__.extend(["cached", "async_cached", "performance_monitor", "get_performance_summary"])
    
if MetricsCollector:
    __all__.extend(["MetricsCollector", "PerformanceMonitor"])
    
if ConfigManager:
    __all__.extend(["ConfigManager", "load_config"])