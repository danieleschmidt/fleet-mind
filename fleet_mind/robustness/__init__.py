"""
Generation 2 Robustness Module

Enhanced error handling, validation, monitoring, health checks, and security
measures for reliable swarm coordination under adverse conditions.
"""

from .fault_tolerance import FaultToleranceManager, FaultType, RecoveryStrategy
from .resilience_engine import ResilienceEngine, SystemResilience, FailureMode
from .adaptive_defense import AdaptiveDefenseSystem, ThreatLevel, DefenseResponse
from .health_diagnostics import AdvancedHealthDiagnostics, DiagnosticLevel
from .robust_communication import RobustCommunication, CommunicationReliability

__all__ = [
    "FaultToleranceManager",
    "FaultType",
    "RecoveryStrategy", 
    "ResilienceEngine",
    "SystemResilience",
    "FailureMode",
    "AdaptiveDefenseSystem",
    "ThreatLevel",
    "DefenseResponse",
    "AdvancedHealthDiagnostics",
    "DiagnosticLevel",
    "RobustCommunication",
    "CommunicationReliability"
]