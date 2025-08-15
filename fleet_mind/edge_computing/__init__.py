"""Edge Computing Integration for Ultra-Low Latency Coordination.

This module implements next-generation edge computing capabilities:
- 5G/6G network optimization for drone communication
- Edge AI deployment with distributed inference
- Dynamic edge-cloud workload balancing
- Federated learning across edge nodes
"""

from .edge_coordinator import EdgeCoordinator, EdgeNode, ComputeWorkload
from .network_5g import FiveGOptimizer, NetworkSlice, QualityOfService
from .federated_learning import FederatedLearning, ModelAggregation, PrivacyPreservation
from .edge_ai_deployment import EdgeAIDeployment, ModelPartitioning, DistributedInference

__all__ = [
    "EdgeCoordinator",
    "EdgeNode", 
    "ComputeWorkload",
    "FiveGOptimizer",
    "NetworkSlice",
    "QualityOfService",
    "FederatedLearning",
    "ModelAggregation",
    "PrivacyPreservation",
    "EdgeAIDeployment",
    "ModelPartitioning",
    "DistributedInference",
]