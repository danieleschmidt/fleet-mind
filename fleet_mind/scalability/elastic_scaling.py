"""
Elastic Scaling Manager

Advanced auto-scaling system for massive swarm coordination with
predictive scaling, multi-tier resource management, and cost optimization.
"""

import asyncio
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from ..utils.logging import get_logger
from ..utils.performance import performance_monitor

logger = get_logger(__name__)

class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ResourceType(Enum):
    """Types of resources that can be scaled."""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    COORDINATION_NODES = "coordination_nodes"
    COMMUNICATION_CHANNELS = "communication_channels"

@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_utilization: float = 0.0
    gpu_utilization: float = 0.0
    coordination_load: float = 0.0
    communication_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    resource_type: ResourceType
    min_instances: int
    max_instances: int
    target_utilization: float  # 0.0 to 1.0
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period: float  # seconds
    predictive_scaling: bool = True
    cost_optimization: bool = True
    priority: int = 1  # Higher number = higher priority

@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    id: str
    resource_type: ResourceType
    direction: ScalingDirection
    current_instances: int
    target_instances: int
    trigger_metric: str
    metric_value: float
    timestamp: float
    cost_impact: float = 0.0
    predicted: bool = False

class ElasticScalingManager:
    """
    Advanced elastic scaling manager for massive swarm coordination.
    
    Features:
    - Predictive scaling based on historical patterns
    - Multi-tier resource optimization
    - Cost-aware scaling decisions
    - Automatic load balancing integration
    - Real-time performance monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the elastic scaling manager."""
        self.config = config or self._default_config()
        
        # Scaling policies for different resource types
        self.scaling_policies: Dict[ResourceType, ScalingPolicy] = {}
        
        # Current resource state
        self.current_instances: Dict[ResourceType, int] = {}
        self.target_instances: Dict[ResourceType, int] = {}
        
        # Metrics and monitoring
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_history: List[ScalingEvent] = []
        
        # Predictive models
        self.demand_predictors: Dict[ResourceType, Any] = {}
        self.prediction_accuracy: Dict[ResourceType, float] = {}
        
        # Cooldown tracking
        self.last_scaling_time: Dict[ResourceType, float] = {}
        
        # Cost optimization
        self.cost_models: Dict[ResourceType, Callable[[int], float]] = {}
        self.budget_limits: Dict[str, float] = {}
        
        # Performance tracking
        self.scaling_effectiveness: Dict[str, float] = {}
        
        logger.info("Elastic Scaling Manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for elastic scaling."""
        return {
            'metrics_retention_hours': 24,
            'prediction_window_minutes': 30,
            'scaling_check_interval': 10.0,  # seconds
            'max_concurrent_scaling_ops': 3,
            'cost_optimization_enabled': True,
            'predictive_scaling_enabled': True,
            'aggressive_scaling_threshold': 0.9,
            'conservative_scaling_threshold': 0.3,
            'emergency_scaling_enabled': True,
            'min_prediction_confidence': 0.7,
            'max_scaling_velocity': 0.5,  # Max fraction of instances to scale at once
        }
    
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add a scaling policy for a resource type."""
        self.scaling_policies[policy.resource_type] = policy
        self.current_instances[policy.resource_type] = policy.min_instances
        self.target_instances[policy.resource_type] = policy.min_instances
        
        logger.info(f"Added scaling policy for {policy.resource_type.value}")
    
    async def start_scaling_monitor(self):
        """Start the scaling monitoring and decision loop."""
        logger.info("Starting elastic scaling monitor...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._scaling_decision_loop()),
            asyncio.create_task(self._predictive_scaling_loop()),
            asyncio.create_task(self._cost_optimization_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _metrics_collection_loop(self):
        """Collect resource metrics continuously."""
        while True:
            try:
                with performance_monitor("metrics_collection"):
                    # Collect current metrics
                    metrics = await self._collect_resource_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Trim history to configured retention
                    cutoff_time = time.time() - (self.config['metrics_retention_hours'] * 3600)
                    self.metrics_history = [
                        m for m in self.metrics_history if m.timestamp > cutoff_time
                    ]
                
                await asyncio.sleep(self.config['scaling_check_interval'])
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30.0)
    
    async def _scaling_decision_loop(self):
        """Main scaling decision loop."""
        while True:
            try:
                if not self.metrics_history:
                    await asyncio.sleep(self.config['scaling_check_interval'])
                    continue
                
                current_metrics = self.metrics_history[-1]
                
                # Check each resource type for scaling needs
                scaling_decisions = []
                for resource_type, policy in self.scaling_policies.items():
                    decision = await self._make_scaling_decision(resource_type, policy, current_metrics)
                    if decision:
                        scaling_decisions.append(decision)
                
                # Sort by priority and execute
                scaling_decisions.sort(key=lambda x: x[1], reverse=True)  # Sort by priority
                
                # Respect max concurrent scaling operations
                max_ops = self.config['max_concurrent_scaling_ops']
                for i, (decision, priority) in enumerate(scaling_decisions[:max_ops]):
                    asyncio.create_task(self._execute_scaling_decision(decision))
                
                await asyncio.sleep(self.config['scaling_check_interval'])
                
            except Exception as e:
                logger.error(f"Scaling decision loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _make_scaling_decision(
        self, 
        resource_type: ResourceType, 
        policy: ScalingPolicy, 
        metrics: ResourceMetrics
    ) -> Optional[Tuple[ScalingEvent, int]]:
        """Make a scaling decision for a specific resource type."""
        
        # Check cooldown period
        last_scaling = self.last_scaling_time.get(resource_type, 0)
        if time.time() - last_scaling < policy.cooldown_period:
            return None
        
        current_count = self.current_instances[resource_type]
        utilization = self._get_resource_utilization(resource_type, metrics)
        
        # Determine scaling direction
        direction = ScalingDirection.STABLE
        target_count = current_count
        
        if utilization > policy.scale_up_threshold:
            direction = ScalingDirection.UP
            # Calculate target instances based on utilization
            target_count = math.ceil(current_count * (utilization / policy.target_utilization))
            target_count = min(target_count, policy.max_instances)
            
        elif utilization < policy.scale_down_threshold:
            direction = ScalingDirection.DOWN
            # Calculate target instances based on utilization
            target_count = math.floor(current_count * (utilization / policy.target_utilization))
            target_count = max(target_count, policy.min_instances)
        
        # Apply max scaling velocity limit
        max_change = max(1, int(current_count * self.config['max_scaling_velocity']))
        if direction == ScalingDirection.UP:
            target_count = min(target_count, current_count + max_change)
        elif direction == ScalingDirection.DOWN:
            target_count = max(target_count, current_count - max_change)
        
        if target_count == current_count:
            return None
        
        # Create scaling event
        scaling_event = ScalingEvent(
            id=f"scaling_{resource_type.value}_{int(time.time())}",
            resource_type=resource_type,
            direction=direction,
            current_instances=current_count,
            target_instances=target_count,
            trigger_metric=f"{resource_type.value}_utilization",
            metric_value=utilization,
            timestamp=time.time(),
            predicted=False
        )
        
        # Calculate cost impact if cost optimization is enabled
        if self.config['cost_optimization_enabled']:
            scaling_event.cost_impact = await self._calculate_cost_impact(scaling_event)
        
        return (scaling_event, policy.priority)
    
    async def _execute_scaling_decision(self, scaling_event: ScalingEvent):
        """Execute a scaling decision."""
        try:
            logger.info(f"Executing scaling: {scaling_event.resource_type.value} "
                       f"{scaling_event.current_instances} -> {scaling_event.target_instances}")
            
            with performance_monitor(f"scaling_execution_{scaling_event.resource_type.value}"):
                # Execute the actual scaling
                success = await self._scale_resource(
                    scaling_event.resource_type,
                    scaling_event.current_instances,
                    scaling_event.target_instances
                )
                
                if success:
                    # Update state
                    self.current_instances[scaling_event.resource_type] = scaling_event.target_instances
                    self.target_instances[scaling_event.resource_type] = scaling_event.target_instances
                    self.last_scaling_time[scaling_event.resource_type] = time.time()
                    
                    # Record the scaling event
                    self.scaling_history.append(scaling_event)
                    
                    logger.info(f"Scaling successful: {scaling_event.id}")
                    
                    # Update effectiveness tracking
                    await self._update_scaling_effectiveness(scaling_event)
                    
                else:
                    logger.error(f"Scaling failed: {scaling_event.id}")
                    
        except Exception as e:
            logger.error(f"Scaling execution error: {e}")
    
    async def _predictive_scaling_loop(self):
        """Predictive scaling based on historical patterns."""
        if not self.config['predictive_scaling_enabled']:
            return
            
        while True:
            try:
                await asyncio.sleep(60.0)  # Check predictions every minute
                
                if len(self.metrics_history) < 100:  # Need sufficient history
                    continue
                
                # Generate predictions for each resource type
                for resource_type in self.scaling_policies.keys():
                    prediction = await self._predict_resource_demand(resource_type)
                    
                    if prediction and prediction['confidence'] > self.config['min_prediction_confidence']:
                        await self._create_predictive_scaling_event(resource_type, prediction)
                
            except Exception as e:
                logger.error(f"Predictive scaling error: {e}")
                await asyncio.sleep(300.0)
    
    async def _predict_resource_demand(self, resource_type: ResourceType) -> Optional[Dict[str, Any]]:
        """Predict future resource demand using historical patterns."""
        if len(self.metrics_history) < 50:
            return None
        
        # Extract recent utilization data
        recent_metrics = self.metrics_history[-50:]
        utilizations = [self._get_resource_utilization(resource_type, m) for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        # Simple trend analysis (would use more sophisticated ML in practice)
        if len(utilizations) < 10:
            return None
        
        # Calculate trend
        x = np.array(range(len(utilizations)))
        y = np.array(utilizations)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        trend_slope = coeffs[0]
        
        # Predict future utilization
        prediction_window = self.config['prediction_window_minutes'] * 60
        future_x = len(utilizations) + (prediction_window / (timestamps[-1] - timestamps[-2]))
        predicted_utilization = np.polyval(coeffs, future_x)
        
        # Calculate confidence based on recent variance
        recent_variance = np.var(utilizations[-10:])
        confidence = max(0.0, 1.0 - (recent_variance * 2))  # Simple confidence metric
        
        return {
            'predicted_utilization': float(predicted_utilization),
            'current_utilization': utilizations[-1],
            'trend_slope': float(trend_slope),
            'confidence': float(confidence),
            'prediction_horizon': prediction_window
        }
    
    async def _create_predictive_scaling_event(self, resource_type: ResourceType, prediction: Dict[str, Any]):
        """Create a predictive scaling event."""
        policy = self.scaling_policies[resource_type]
        current_count = self.current_instances[resource_type]
        predicted_utilization = prediction['predicted_utilization']
        
        # Determine if predictive scaling is needed
        if predicted_utilization > policy.scale_up_threshold:
            target_count = math.ceil(current_count * (predicted_utilization / policy.target_utilization))
            target_count = min(target_count, policy.max_instances)
            direction = ScalingDirection.UP
            
        elif predicted_utilization < policy.scale_down_threshold:
            target_count = math.floor(current_count * (predicted_utilization / policy.target_utilization))
            target_count = max(target_count, policy.min_instances)
            direction = ScalingDirection.DOWN
            
        else:
            return  # No predictive scaling needed
        
        if target_count == current_count:
            return
        
        # Create predictive scaling event
        scaling_event = ScalingEvent(
            id=f"predictive_{resource_type.value}_{int(time.time())}",
            resource_type=resource_type,
            direction=direction,
            current_instances=current_count,
            target_instances=target_count,
            trigger_metric="predicted_utilization",
            metric_value=predicted_utilization,
            timestamp=time.time(),
            predicted=True
        )
        
        logger.info(f"Predictive scaling triggered: {resource_type.value} "
                   f"(confidence: {prediction['confidence']:.2f})")
        
        # Execute predictive scaling with lower priority
        asyncio.create_task(self._execute_scaling_decision(scaling_event))
    
    async def _cost_optimization_loop(self):
        """Optimize costs by analyzing scaling patterns and efficiency."""
        if not self.config['cost_optimization_enabled']:
            return
            
        while True:
            try:
                await asyncio.sleep(300.0)  # Check every 5 minutes
                
                # Analyze cost efficiency of recent scaling decisions
                await self._analyze_cost_efficiency()
                
                # Optimize instance mix for cost
                await self._optimize_instance_mix()
                
                # Check budget utilization
                await self._check_budget_utilization()
                
            except Exception as e:
                logger.error(f"Cost optimization error: {e}")
                await asyncio.sleep(600.0)
    
    async def _cleanup_loop(self):
        """Clean up old data and optimize memory usage."""
        while True:
            try:
                await asyncio.sleep(3600.0)  # Run every hour
                
                # Clean up old scaling history
                cutoff_time = time.time() - (24 * 3600)  # Keep 24 hours
                self.scaling_history = [
                    event for event in self.scaling_history 
                    if event.timestamp > cutoff_time
                ]
                
                # Update prediction accuracy
                await self._update_prediction_accuracy()
                
                logger.info("Cleanup completed")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600.0)
    
    async def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # Simulate metric collection - would integrate with actual monitoring
        metrics = ResourceMetrics(
            cpu_utilization=np.random.uniform(0.3, 0.9),
            memory_utilization=np.random.uniform(0.4, 0.8),
            network_utilization=np.random.uniform(0.2, 0.7),
            gpu_utilization=np.random.uniform(0.1, 0.95),
            coordination_load=np.random.uniform(0.3, 0.85),
            communication_latency=np.random.uniform(10.0, 200.0),
            throughput=np.random.uniform(1000.0, 10000.0),
            error_rate=np.random.uniform(0.0, 0.05)
        )
        return metrics
    
    def _get_resource_utilization(self, resource_type: ResourceType, metrics: ResourceMetrics) -> float:
        """Get utilization for a specific resource type."""
        utilization_map = {
            ResourceType.COMPUTE: metrics.cpu_utilization,
            ResourceType.MEMORY: metrics.memory_utilization,
            ResourceType.STORAGE: 0.6,  # Would come from storage metrics
            ResourceType.NETWORK: metrics.network_utilization,
            ResourceType.GPU: metrics.gpu_utilization,
            ResourceType.COORDINATION_NODES: metrics.coordination_load,
            ResourceType.COMMUNICATION_CHANNELS: min(1.0, metrics.communication_latency / 100.0)
        }
        return utilization_map.get(resource_type, 0.5)
    
    async def _scale_resource(self, resource_type: ResourceType, current: int, target: int) -> bool:
        """Execute actual resource scaling."""
        try:
            # Simulate scaling operation
            scaling_time = abs(target - current) * 2.0  # 2 seconds per instance
            await asyncio.sleep(min(scaling_time, 30.0))  # Max 30 seconds
            
            logger.info(f"Scaled {resource_type.value}: {current} -> {target}")
            return True
            
        except Exception as e:
            logger.error(f"Resource scaling failed: {e}")
            return False
    
    async def _calculate_cost_impact(self, scaling_event: ScalingEvent) -> float:
        """Calculate the cost impact of a scaling decision."""
        resource_type = scaling_event.resource_type
        instance_change = scaling_event.target_instances - scaling_event.current_instances
        
        # Simple cost model - would be more sophisticated in practice
        cost_per_instance = {
            ResourceType.COMPUTE: 0.10,  # $/hour
            ResourceType.MEMORY: 0.05,
            ResourceType.STORAGE: 0.02,
            ResourceType.NETWORK: 0.08,
            ResourceType.GPU: 0.50,
            ResourceType.COORDINATION_NODES: 0.15,
            ResourceType.COMMUNICATION_CHANNELS: 0.03
        }
        
        hourly_cost_change = instance_change * cost_per_instance.get(resource_type, 0.10)
        return hourly_cost_change
    
    async def _update_scaling_effectiveness(self, scaling_event: ScalingEvent):
        """Update scaling effectiveness metrics."""
        # Wait for metrics to reflect scaling impact
        await asyncio.sleep(60.0)
        
        if len(self.metrics_history) < 2:
            return
        
        # Compare metrics before and after scaling
        pre_scaling_metrics = self.metrics_history[-2]
        post_scaling_metrics = self.metrics_history[-1]
        
        pre_utilization = self._get_resource_utilization(scaling_event.resource_type, pre_scaling_metrics)
        post_utilization = self._get_resource_utilization(scaling_event.resource_type, post_scaling_metrics)
        
        # Calculate effectiveness
        if scaling_event.direction == ScalingDirection.UP:
            effectiveness = max(0.0, 1.0 - (post_utilization - 0.7))  # Target 70% utilization
        else:
            effectiveness = max(0.0, post_utilization / 0.7)  # Should maintain reasonable utilization
        
        self.scaling_effectiveness[scaling_event.id] = effectiveness
        
        logger.info(f"Scaling effectiveness for {scaling_event.id}: {effectiveness:.2f}")
    
    async def _analyze_cost_efficiency(self):
        """Analyze cost efficiency of scaling decisions."""
        recent_events = [
            event for event in self.scaling_history
            if time.time() - event.timestamp < 3600  # Last hour
        ]
        
        if not recent_events:
            return
        
        total_cost_impact = sum(event.cost_impact for event in recent_events)
        avg_effectiveness = np.mean([
            self.scaling_effectiveness.get(event.id, 0.5)
            for event in recent_events
        ])
        
        cost_efficiency = avg_effectiveness / max(abs(total_cost_impact), 0.01)
        
        logger.info(f"Cost efficiency (last hour): {cost_efficiency:.2f}")
    
    async def _optimize_instance_mix(self):
        """Optimize the mix of instance types for cost efficiency."""
        # Placeholder for instance mix optimization
        # Would analyze workload patterns and optimize instance types
        pass
    
    async def _check_budget_utilization(self):
        """Check budget utilization and alert if approaching limits."""
        # Placeholder for budget checking
        # Would integrate with cost management systems
        pass
    
    async def _update_prediction_accuracy(self):
        """Update prediction accuracy metrics."""
        # Analyze how accurate recent predictions were
        for resource_type in self.scaling_policies.keys():
            recent_predictions = [
                event for event in self.scaling_history
                if (event.resource_type == resource_type and 
                    event.predicted and 
                    time.time() - event.timestamp < 3600)
            ]
            
            if recent_predictions:
                # Calculate accuracy based on actual vs predicted needs
                accuracy = np.random.uniform(0.7, 0.95)  # Placeholder
                self.prediction_accuracy[resource_type] = accuracy
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        return {
            'current_instances': dict(self.current_instances),
            'target_instances': dict(self.target_instances),
            'recent_metrics': self.metrics_history[-1].__dict__ if self.metrics_history else {},
            'recent_scaling_events': len([
                e for e in self.scaling_history 
                if time.time() - e.timestamp < 3600
            ]),
            'prediction_accuracy': dict(self.prediction_accuracy),
            'scaling_effectiveness': {
                event_id: eff for event_id, eff in self.scaling_effectiveness.items()
                if any(e.id == event_id for e in self.scaling_history[-10:])
            },
            'active_policies': len(self.scaling_policies),
            'cost_optimization_enabled': self.config['cost_optimization_enabled'],
            'predictive_scaling_enabled': self.config['predictive_scaling_enabled']
        }
    
    async def force_scale(self, resource_type: ResourceType, target_instances: int) -> bool:
        """Force scaling to a specific number of instances."""
        if resource_type not in self.scaling_policies:
            logger.error(f"No scaling policy for {resource_type.value}")
            return False
        
        policy = self.scaling_policies[resource_type]
        target_instances = max(policy.min_instances, min(target_instances, policy.max_instances))
        
        current_instances = self.current_instances[resource_type]
        
        if target_instances == current_instances:
            return True
        
        # Create forced scaling event
        scaling_event = ScalingEvent(
            id=f"forced_{resource_type.value}_{int(time.time())}",
            resource_type=resource_type,
            direction=ScalingDirection.UP if target_instances > current_instances else ScalingDirection.DOWN,
            current_instances=current_instances,
            target_instances=target_instances,
            trigger_metric="manual_override",
            metric_value=0.0,
            timestamp=time.time()
        )
        
        await self._execute_scaling_decision(scaling_event)
        return True