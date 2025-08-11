"""ML-Based Cost Optimization and Predictive Auto-Scaling for Fleet-Mind Generation 3.

This module implements advanced cost optimization including:
- ML-based predictive auto-scaling
- Spot instance management and cost optimization
- Dynamic resource allocation with cost awareness
- Multi-cloud cost optimization strategies
- Real-time cost monitoring and optimization
"""

import asyncio
import time
import math
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import threading

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.auto_scaling import AutoScaler, ScalingPolicy, ScalingAction, MetricType, ScalingMetric
from .ai_performance_optimizer import SimpleMLModel


class InstanceType(Enum):
    """Cloud instance types."""
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    PREEMPTIBLE = "preemptible"
    BURSTABLE = "burstable"


class CostOptimizationStrategy(Enum):
    """Cost optimization strategies."""
    AGGRESSIVE = "aggressive"      # Maximum cost savings, higher risk
    BALANCED = "balanced"         # Balance cost and reliability
    CONSERVATIVE = "conservative"  # Minimize risk, moderate savings
    PERFORMANCE_FIRST = "performance_first"  # Performance over cost


@dataclass
class InstanceConfig:
    """Instance configuration with cost information."""
    instance_type: InstanceType
    cpu_cores: int
    memory_gb: float
    cost_per_hour: float
    availability_zone: str = "default"
    preemption_probability: float = 0.0  # 0.0 to 1.0
    startup_time_seconds: float = 30.0
    termination_notice_seconds: float = 120.0
    performance_multiplier: float = 1.0  # Relative performance
    

@dataclass
class CostPrediction:
    """Cost prediction for scaling decisions."""
    predicted_cost: float
    confidence: float
    time_horizon_minutes: int
    cost_breakdown: Dict[str, float]
    savings_potential: float = 0.0
    risk_level: str = "low"


@dataclass
class ScalingRecommendation:
    """ML-based scaling recommendation with cost optimization."""
    action: str  # scale_up, scale_down, rebalance, no_action
    target_capacity: int
    instance_mix: Dict[InstanceType, int]
    predicted_cost_savings: float
    confidence: float
    reasoning: List[str]
    estimated_time_to_execute: float
    risk_assessment: Dict[str, float]


class MLCostOptimizer:
    """ML-based cost optimization system for auto-scaling."""
    
    def __init__(
        self,
        cost_optimization_strategy: CostOptimizationStrategy = CostOptimizationStrategy.BALANCED,
        spot_instance_max_ratio: float = 0.8,
        cost_prediction_horizon: int = 60,  # minutes
        enable_predictive_scaling: bool = True,
        cost_threshold_increase: float = 1.2,  # 20% cost increase threshold
    ):
        """Initialize ML cost optimizer.
        
        Args:
            cost_optimization_strategy: Cost optimization approach
            spot_instance_max_ratio: Maximum ratio of spot instances
            cost_prediction_horizon: Cost prediction horizon in minutes
            enable_predictive_scaling: Enable ML-based predictive scaling
            cost_threshold_increase: Maximum allowed cost increase ratio
        """
        self.cost_optimization_strategy = cost_optimization_strategy
        self.spot_instance_max_ratio = spot_instance_max_ratio
        self.cost_prediction_horizon = cost_prediction_horizon
        self.enable_predictive_scaling = enable_predictive_scaling
        self.cost_threshold_increase = cost_threshold_increase
        
        # ML models
        self.cost_predictor = SimpleMLModel(feature_size=8)
        self.demand_predictor = SimpleMLModel(feature_size=6)
        self.spot_price_predictor = SimpleMLModel(feature_size=4)
        
        # Cost tracking
        self.cost_history: deque = deque(maxlen=10000)
        self.instance_cost_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Instance configurations
        self.available_instances: Dict[InstanceType, InstanceConfig] = {
            InstanceType.ON_DEMAND: InstanceConfig(
                instance_type=InstanceType.ON_DEMAND,
                cpu_cores=4,
                memory_gb=16.0,
                cost_per_hour=0.20,
                preemption_probability=0.0,
                startup_time_seconds=30.0,
                performance_multiplier=1.0,
            ),
            InstanceType.SPOT: InstanceConfig(
                instance_type=InstanceType.SPOT,
                cpu_cores=4,
                memory_gb=16.0,
                cost_per_hour=0.06,  # 70% savings
                preemption_probability=0.05,
                startup_time_seconds=45.0,
                termination_notice_seconds=30.0,
                performance_multiplier=1.0,
            ),
            InstanceType.RESERVED: InstanceConfig(
                instance_type=InstanceType.RESERVED,
                cpu_cores=4,
                memory_gb=16.0,
                cost_per_hour=0.12,  # 40% savings with commitment
                preemption_probability=0.0,
                startup_time_seconds=25.0,
                performance_multiplier=1.05,
            ),
            InstanceType.BURSTABLE: InstanceConfig(
                instance_type=InstanceType.BURSTABLE,
                cpu_cores=2,
                memory_gb=8.0,
                cost_per_hour=0.05,
                preemption_probability=0.0,
                startup_time_seconds=20.0,
                performance_multiplier=0.7,
            ),
        }
        
        # Current deployment state
        self.current_instances: Dict[InstanceType, int] = defaultdict(int)
        self.spot_price_history: deque = deque(maxlen=1000)
        
        # Optimization metrics
        self.total_cost_savings = 0.0
        self.optimization_attempts = 0
        self.successful_optimizations = 0
        
        # Threading
        self.optimization_lock = threading.RLock()
        
        # Logging
        self.logger = get_logger("ml_cost_optimizer")
        
    def record_cost_metrics(
        self,
        current_cost: float,
        instance_counts: Dict[InstanceType, int],
        performance_metrics: Dict[str, float],
    ):
        """Record cost and performance metrics for ML training."""
        with self.optimization_lock:
            timestamp = time.time()
            
            cost_record = {
                "timestamp": timestamp,
                "total_cost": current_cost,
                "instance_counts": dict(instance_counts),
                "performance": dict(performance_metrics),
            }
            
            self.cost_history.append(cost_record)
            
            # Update current instance counts
            self.current_instances = defaultdict(int, instance_counts)
            
            # Train models with new data
            if len(self.cost_history) >= 10:
                asyncio.create_task(self._train_models())
    
    async def _train_models(self):
        """Train ML models with historical data."""
        try:
            if len(self.cost_history) < 20:
                return
            
            recent_data = list(self.cost_history)[-100:]  # Last 100 records
            
            # Prepare training data for cost prediction
            cost_features = []
            cost_targets = []
            
            for i in range(len(recent_data) - 1):
                current = recent_data[i]
                next_record = recent_data[i + 1]
                
                # Features: instance counts, performance metrics, time of day
                features = [
                    current["instance_counts"].get(InstanceType.ON_DEMAND.value, 0),
                    current["instance_counts"].get(InstanceType.SPOT.value, 0),
                    current["instance_counts"].get(InstanceType.RESERVED.value, 0),
                    current["instance_counts"].get(InstanceType.BURSTABLE.value, 0),
                    current["performance"].get("cpu_usage", 0),
                    current["performance"].get("memory_usage", 0),
                    current["performance"].get("response_time", 0),
                    (current["timestamp"] % 86400) / 86400,  # Time of day
                ]
                
                target_cost = next_record["total_cost"]
                
                cost_features.append(features)
                cost_targets.append(target_cost)
            
            # Train cost predictor
            if cost_features and cost_targets:
                self.cost_predictor.train(cost_features, cost_targets)
                
            # Train demand predictor
            demand_features = []
            demand_targets = []
            
            for record in recent_data:
                total_instances = sum(record["instance_counts"].values())
                cpu_usage = record["performance"].get("cpu_usage", 0)
                
                features = [
                    total_instances,
                    cpu_usage,
                    record["performance"].get("memory_usage", 0),
                    record["performance"].get("response_time", 0),
                    record["performance"].get("throughput", 0),
                    (record["timestamp"] % 86400) / 86400,  # Time of day
                ]
                
                # Target is future demand (simplified as current CPU usage trend)
                demand_features.append(features)
                demand_targets.append(cpu_usage)
            
            if demand_features and demand_targets:
                self.demand_predictor.train(demand_features, demand_targets)
                
            self.logger.debug("ML models trained with recent cost data")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
    
    async def predict_optimal_scaling(
        self,
        current_load: Dict[str, float],
        predicted_load: Dict[str, float],
        time_horizon_minutes: int = 60,
    ) -> ScalingRecommendation:
        """Predict optimal scaling action with cost optimization."""
        try:
            self.optimization_attempts += 1
            
            # Get current costs
            current_cost = self._calculate_current_cost()
            
            # Predict demand
            predicted_demand = await self._predict_demand(current_load, time_horizon_minutes)
            
            # Calculate required capacity
            required_capacity = self._calculate_required_capacity(predicted_demand)
            current_capacity = sum(self.current_instances.values())
            
            if abs(required_capacity - current_capacity) < 2:
                # No significant scaling needed
                return ScalingRecommendation(
                    action="no_action",
                    target_capacity=current_capacity,
                    instance_mix=dict(self.current_instances),
                    predicted_cost_savings=0.0,
                    confidence=0.8,
                    reasoning=["Predicted demand within current capacity"],
                    estimated_time_to_execute=0.0,
                    risk_assessment={"preemption_risk": 0.0, "performance_risk": 0.0},
                )
            
            # Generate cost-optimized instance mix
            optimal_mix = await self._optimize_instance_mix(required_capacity, predicted_demand)
            
            # Calculate cost predictions
            cost_prediction = await self._predict_costs(optimal_mix, time_horizon_minutes)
            
            # Determine action type
            if required_capacity > current_capacity:
                action = "scale_up"
            elif required_capacity < current_capacity:
                action = "scale_down"
            else:
                action = "rebalance"  # Same capacity but different mix
            
            # Calculate confidence based on model performance and data availability
            confidence = self._calculate_confidence(current_load, predicted_demand)
            
            # Risk assessment
            risk_assessment = self._assess_risks(optimal_mix)
            
            recommendation = ScalingRecommendation(
                action=action,
                target_capacity=required_capacity,
                instance_mix=optimal_mix,
                predicted_cost_savings=current_cost - cost_prediction.predicted_cost,
                confidence=confidence,
                reasoning=self._generate_reasoning(action, optimal_mix, cost_prediction),
                estimated_time_to_execute=self._estimate_execution_time(optimal_mix),
                risk_assessment=risk_assessment,
            )
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error predicting optimal scaling: {e}")
            return ScalingRecommendation(
                action="no_action",
                target_capacity=sum(self.current_instances.values()),
                instance_mix=dict(self.current_instances),
                predicted_cost_savings=0.0,
                confidence=0.0,
                reasoning=[f"Error in prediction: {str(e)}"],
                estimated_time_to_execute=0.0,
                risk_assessment={"error": 1.0},
            )
    
    async def _predict_demand(
        self,
        current_load: Dict[str, float],
        time_horizon_minutes: int,
    ) -> Dict[str, float]:
        """Predict future demand using ML models."""
        try:
            if not self.demand_predictor.trained:
                # Fallback to simple trend analysis
                return self._simple_demand_prediction(current_load, time_horizon_minutes)
            
            # Prepare features for demand prediction
            features = [
                sum(self.current_instances.values()),
                current_load.get("cpu_usage", 0),
                current_load.get("memory_usage", 0),
                current_load.get("response_time", 0),
                current_load.get("throughput", 0),
                (time.time() % 86400) / 86400,  # Time of day
            ]
            
            # Predict future CPU usage as demand proxy
            predicted_cpu = self.demand_predictor.predict(features)
            
            # Scale other metrics proportionally
            cpu_ratio = predicted_cpu / max(current_load.get("cpu_usage", 1), 1)
            
            predicted_demand = {
                "cpu_usage": predicted_cpu,
                "memory_usage": current_load.get("memory_usage", 0) * cpu_ratio,
                "response_time": current_load.get("response_time", 0) * (1 + (cpu_ratio - 1) * 0.5),
                "throughput": current_load.get("throughput", 0) * cpu_ratio,
            }
            
            return predicted_demand
            
        except Exception as e:
            self.logger.error(f"Error predicting demand: {e}")
            return current_load
    
    def _simple_demand_prediction(
        self,
        current_load: Dict[str, float],
        time_horizon_minutes: int,
    ) -> Dict[str, float]:
        """Simple trend-based demand prediction."""
        if len(self.cost_history) < 5:
            return current_load
        
        # Calculate trend from recent history
        recent_records = list(self.cost_history)[-5:]
        cpu_values = [r["performance"].get("cpu_usage", 0) for r in recent_records]
        
        if len(cpu_values) >= 2:
            # Simple linear trend
            trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
            
            # Project trend forward
            future_cpu = current_load.get("cpu_usage", 0) + (trend * time_horizon_minutes / 10)
            future_cpu = max(0, min(100, future_cpu))  # Clamp to 0-100%
            
            cpu_ratio = future_cpu / max(current_load.get("cpu_usage", 1), 1)
            
            return {
                "cpu_usage": future_cpu,
                "memory_usage": current_load.get("memory_usage", 0) * cpu_ratio,
                "response_time": current_load.get("response_time", 0) * (1 + (cpu_ratio - 1) * 0.3),
                "throughput": current_load.get("throughput", 0) * cpu_ratio,
            }
        
        return current_load
    
    def _calculate_required_capacity(self, predicted_demand: Dict[str, float]) -> int:
        """Calculate required capacity based on predicted demand."""
        cpu_usage = predicted_demand.get("cpu_usage", 0)
        memory_usage = predicted_demand.get("memory_usage", 0)
        
        # Calculate capacity needed to handle load with safety margin
        cpu_target = 70.0  # Target 70% CPU utilization
        memory_target = 75.0  # Target 75% memory utilization
        
        cpu_capacity_needed = math.ceil(cpu_usage / cpu_target)
        memory_capacity_needed = math.ceil(memory_usage / memory_target)
        
        # Take the higher requirement
        required_capacity = max(cpu_capacity_needed, memory_capacity_needed)
        
        # Apply safety margin based on strategy
        if self.cost_optimization_strategy == CostOptimizationStrategy.CONSERVATIVE:
            safety_margin = 1.3
        elif self.cost_optimization_strategy == CostOptimizationStrategy.BALANCED:
            safety_margin = 1.15
        else:  # AGGRESSIVE or PERFORMANCE_FIRST
            safety_margin = 1.05
        
        return max(1, int(required_capacity * safety_margin))
    
    async def _optimize_instance_mix(
        self,
        target_capacity: int,
        predicted_demand: Dict[str, float],
    ) -> Dict[InstanceType, int]:
        """Optimize instance mix for cost and performance."""
        if target_capacity <= 0:
            return defaultdict(int)
        
        # Initialize mix
        optimal_mix = defaultdict(int)
        
        # Strategy-based optimization
        if self.cost_optimization_strategy == CostOptimizationStrategy.AGGRESSIVE:
            optimal_mix = self._aggressive_cost_optimization(target_capacity)
        elif self.cost_optimization_strategy == CostOptimizationStrategy.CONSERVATIVE:
            optimal_mix = self._conservative_optimization(target_capacity)
        elif self.cost_optimization_strategy == CostOptimizationStrategy.PERFORMANCE_FIRST:
            optimal_mix = self._performance_first_optimization(target_capacity)
        else:  # BALANCED
            optimal_mix = self._balanced_optimization(target_capacity)
        
        # Ensure we meet capacity requirements
        total_capacity = sum(optimal_mix.values())
        if total_capacity < target_capacity:
            # Add on-demand instances to meet capacity
            optimal_mix[InstanceType.ON_DEMAND] += target_capacity - total_capacity
        
        return dict(optimal_mix)
    
    def _aggressive_cost_optimization(self, target_capacity: int) -> Dict[InstanceType, int]:
        """Aggressive cost optimization - maximize spot instances."""
        mix = defaultdict(int)
        
        # Use maximum spot instances allowed
        spot_instances = min(target_capacity, int(target_capacity * self.spot_instance_max_ratio))
        mix[InstanceType.SPOT] = spot_instances
        
        # Fill remaining with cheapest alternatives
        remaining = target_capacity - spot_instances
        if remaining > 0:
            # Mix of burstable and reserved
            burstable_instances = min(remaining, int(remaining * 0.3))
            mix[InstanceType.BURSTABLE] = burstable_instances
            
            remaining -= burstable_instances
            if remaining > 0:
                mix[InstanceType.RESERVED] = remaining
        
        return mix
    
    def _conservative_optimization(self, target_capacity: int) -> Dict[InstanceType, int]:
        """Conservative optimization - prioritize reliability."""
        mix = defaultdict(int)
        
        # Mostly on-demand and reserved instances
        reserved_instances = min(target_capacity, int(target_capacity * 0.6))
        mix[InstanceType.RESERVED] = reserved_instances
        
        remaining = target_capacity - reserved_instances
        if remaining > 0:
            # Small portion of spot instances for cost savings
            spot_instances = min(remaining, int(remaining * 0.3))
            mix[InstanceType.SPOT] = spot_instances
            
            remaining -= spot_instances
            if remaining > 0:
                mix[InstanceType.ON_DEMAND] = remaining
        
        return mix
    
    def _performance_first_optimization(self, target_capacity: int) -> Dict[InstanceType, int]:
        """Performance-first optimization."""
        mix = defaultdict(int)
        
        # Prioritize high-performance instances
        reserved_instances = min(target_capacity, int(target_capacity * 0.8))
        mix[InstanceType.RESERVED] = reserved_instances
        
        remaining = target_capacity - reserved_instances
        if remaining > 0:
            mix[InstanceType.ON_DEMAND] = remaining
        
        return mix
    
    def _balanced_optimization(self, target_capacity: int) -> Dict[InstanceType, int]:
        """Balanced optimization - mix of cost savings and reliability."""
        mix = defaultdict(int)
        
        # Balanced mix
        spot_instances = min(target_capacity, int(target_capacity * 0.5))
        mix[InstanceType.SPOT] = spot_instances
        
        remaining = target_capacity - spot_instances
        if remaining > 0:
            reserved_instances = min(remaining, int(remaining * 0.6))
            mix[InstanceType.RESERVED] = reserved_instances
            
            remaining -= reserved_instances
            if remaining > 0:
                # Split between on-demand and burstable
                on_demand = int(remaining * 0.7)
                mix[InstanceType.ON_DEMAND] = on_demand
                
                if remaining - on_demand > 0:
                    mix[InstanceType.BURSTABLE] = remaining - on_demand
        
        return mix
    
    async def _predict_costs(
        self,
        instance_mix: Dict[InstanceType, int],
        time_horizon_minutes: int,
    ) -> CostPrediction:
        """Predict costs for given instance mix."""
        try:
            total_cost = 0.0
            cost_breakdown = {}
            
            for instance_type, count in instance_mix.items():
                if count == 0:
                    continue
                    
                config = self.available_instances[instance_type]
                hourly_cost = config.cost_per_hour * count
                horizon_cost = hourly_cost * (time_horizon_minutes / 60.0)
                
                # Apply spot price fluctuation for spot instances
                if instance_type == InstanceType.SPOT:
                    # Simulate spot price prediction (in real implementation, use actual spot price API)
                    spot_multiplier = self._predict_spot_price_multiplier()
                    horizon_cost *= spot_multiplier
                
                total_cost += horizon_cost
                cost_breakdown[instance_type.value] = horizon_cost
            
            # Calculate confidence based on model training and data availability
            confidence = 0.8 if self.cost_predictor.trained else 0.5
            
            return CostPrediction(
                predicted_cost=total_cost,
                confidence=confidence,
                time_horizon_minutes=time_horizon_minutes,
                cost_breakdown=cost_breakdown,
                risk_level="low" if confidence > 0.7 else "medium",
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting costs: {e}")
            return CostPrediction(
                predicted_cost=0.0,
                confidence=0.0,
                time_horizon_minutes=time_horizon_minutes,
                cost_breakdown={},
                risk_level="high",
            )
    
    def _predict_spot_price_multiplier(self) -> float:
        """Predict spot price fluctuation multiplier."""
        try:
            if self.spot_price_predictor.trained and len(self.spot_price_history) > 5:
                # Use historical data for prediction
                recent_prices = list(self.spot_price_history)[-10:]
                features = [
                    statistics.mean(recent_prices),
                    statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0.0,
                    recent_prices[-1],
                    (time.time() % 86400) / 86400,  # Time of day
                ]
                
                multiplier = self.spot_price_predictor.predict(features)
                return max(0.5, min(2.0, multiplier))  # Clamp to reasonable range
            else:
                # Simple heuristic based on time of day
                hour = int((time.time() % 86400) / 3600)
                if 8 <= hour <= 18:  # Business hours
                    return 1.2  # 20% higher prices
                else:
                    return 0.9  # 10% lower prices
                    
        except Exception:
            return 1.0  # Default multiplier
    
    def _calculate_confidence(
        self,
        current_load: Dict[str, float],
        predicted_demand: Dict[str, float],
    ) -> float:
        """Calculate confidence in the scaling recommendation."""
        base_confidence = 0.7
        
        # Increase confidence if models are trained
        if self.cost_predictor.trained:
            base_confidence += 0.1
        if self.demand_predictor.trained:
            base_confidence += 0.1
        
        # Decrease confidence for large demand changes
        cpu_change = abs(predicted_demand.get("cpu_usage", 0) - current_load.get("cpu_usage", 0))
        if cpu_change > 30:  # >30% CPU change
            base_confidence -= 0.2
        elif cpu_change > 15:  # >15% CPU change
            base_confidence -= 0.1
        
        # Increase confidence if we have good historical data
        if len(self.cost_history) > 100:
            base_confidence += 0.1
        
        return max(0.3, min(1.0, base_confidence))
    
    def _assess_risks(self, instance_mix: Dict[InstanceType, int]) -> Dict[str, float]:
        """Assess risks associated with instance mix."""
        risks = {}
        
        total_instances = sum(instance_mix.values())
        if total_instances == 0:
            return {"no_instances": 1.0}
        
        # Preemption risk
        spot_ratio = instance_mix.get(InstanceType.SPOT, 0) / total_instances
        preemptible_ratio = instance_mix.get(InstanceType.PREEMPTIBLE, 0) / total_instances
        risks["preemption_risk"] = (spot_ratio * 0.05) + (preemptible_ratio * 0.10)
        
        # Performance risk
        burstable_ratio = instance_mix.get(InstanceType.BURSTABLE, 0) / total_instances
        risks["performance_risk"] = burstable_ratio * 0.3  # 30% performance penalty
        
        # Cost volatility risk
        spot_cost_risk = spot_ratio * 0.2  # Spot prices can fluctuate
        risks["cost_volatility_risk"] = spot_cost_risk
        
        # Availability risk
        single_type_ratio = max(instance_mix.values()) / total_instances
        if single_type_ratio > 0.8:  # >80% single instance type
            risks["availability_risk"] = 0.3
        else:
            risks["availability_risk"] = 0.1
        
        return risks
    
    def _generate_reasoning(
        self,
        action: str,
        instance_mix: Dict[InstanceType, int],
        cost_prediction: CostPrediction,
    ) -> List[str]:
        """Generate human-readable reasoning for the recommendation."""
        reasoning = []
        
        if action == "scale_up":
            reasoning.append("Predicted demand increase requires scaling up")
        elif action == "scale_down":
            reasoning.append("Predicted demand decrease allows scaling down")
        elif action == "rebalance":
            reasoning.append("Rebalancing instance mix for cost optimization")
        
        # Cost savings reasoning
        if cost_prediction.savings_potential > 0:
            reasoning.append(f"Estimated cost savings: ${cost_prediction.savings_potential:.2f}")
        
        # Instance mix reasoning
        spot_count = instance_mix.get(InstanceType.SPOT, 0)
        total_count = sum(instance_mix.values())
        
        if spot_count > 0 and total_count > 0:
            spot_ratio = spot_count / total_count
            if spot_ratio > 0.6:
                reasoning.append("High spot instance ratio for maximum cost savings")
            elif spot_ratio > 0.3:
                reasoning.append("Balanced spot instance usage for cost efficiency")
        
        # Strategy-specific reasoning
        if self.cost_optimization_strategy == CostOptimizationStrategy.AGGRESSIVE:
            reasoning.append("Aggressive cost optimization strategy applied")
        elif self.cost_optimization_strategy == CostOptimizationStrategy.CONSERVATIVE:
            reasoning.append("Conservative strategy prioritizing reliability")
        
        return reasoning
    
    def _estimate_execution_time(self, instance_mix: Dict[InstanceType, int]) -> float:
        """Estimate time to execute scaling action."""
        max_startup_time = 0.0
        
        for instance_type, count in instance_mix.items():
            if count > 0:
                config = self.available_instances[instance_type]
                max_startup_time = max(max_startup_time, config.startup_time_seconds)
        
        # Add overhead for coordination and configuration
        return max_startup_time + 15.0
    
    def _calculate_current_cost(self) -> float:
        """Calculate current hourly cost."""
        total_cost = 0.0
        
        for instance_type, count in self.current_instances.items():
            if count > 0:
                config = self.available_instances[instance_type]
                total_cost += config.cost_per_hour * count
        
        return total_cost
    
    async def execute_cost_optimization(
        self,
        recommendation: ScalingRecommendation,
    ) -> Dict[str, Any]:
        """Execute cost optimization recommendation."""
        try:
            if recommendation.action == "no_action":
                return {"status": "skipped", "reason": "No optimization needed"}
            
            # Calculate current cost
            old_cost = self._calculate_current_cost()
            
            # Simulate execution (replace with actual infrastructure calls)
            await asyncio.sleep(0.1)
            
            # Update instance counts
            self.current_instances = defaultdict(int, recommendation.instance_mix)
            
            # Calculate new cost
            new_cost = self._calculate_current_cost()
            actual_savings = old_cost - new_cost
            
            # Update optimization metrics
            if actual_savings > 0:
                self.successful_optimizations += 1
                self.total_cost_savings += actual_savings
            
            result = {
                "status": "success",
                "action": recommendation.action,
                "old_cost_per_hour": old_cost,
                "new_cost_per_hour": new_cost,
                "actual_savings": actual_savings,
                "predicted_savings": recommendation.predicted_cost_savings,
                "instance_mix": dict(recommendation.instance_mix),
                "execution_time_seconds": recommendation.estimated_time_to_execute,
            }
            
            self.logger.info(f"Cost optimization executed: {recommendation.action}, "
                           f"savings: ${actual_savings:.4f}/hour")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cost optimization execution failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive cost optimization statistics."""
        current_cost = self._calculate_current_cost()
        
        # Calculate efficiency metrics
        success_rate = self.successful_optimizations / max(1, self.optimization_attempts)
        
        return {
            "strategy": self.cost_optimization_strategy.value,
            "current_hourly_cost": current_cost,
            "total_cost_savings": self.total_cost_savings,
            "optimization_attempts": self.optimization_attempts,
            "successful_optimizations": self.successful_optimizations,
            "success_rate": success_rate,
            "current_instance_mix": dict(self.current_instances),
            "spot_instance_ratio": self.current_instances[InstanceType.SPOT] / max(1, sum(self.current_instances.values())),
            "model_training_status": {
                "cost_predictor": self.cost_predictor.trained,
                "demand_predictor": self.demand_predictor.trained,
                "spot_price_predictor": self.spot_price_predictor.trained,
            },
            "cost_history_size": len(self.cost_history),
            "prediction_confidence": self._calculate_confidence({}, {}),
        }
    
    def save_optimization_state(self, filepath: str):
        """Save optimization state and models."""
        try:
            state_data = {
                "current_instances": dict(self.current_instances),
                "cost_history": list(self.cost_history),
                "total_cost_savings": self.total_cost_savings,
                "optimization_attempts": self.optimization_attempts,
                "successful_optimizations": self.successful_optimizations,
                "models": {
                    "cost_predictor": {
                        "weights": self.cost_predictor.weights,
                        "bias": self.cost_predictor.bias,
                        "trained": self.cost_predictor.trained,
                    },
                    "demand_predictor": {
                        "weights": self.demand_predictor.weights,
                        "bias": self.demand_predictor.bias,
                        "trained": self.demand_predictor.trained,
                    },
                    "spot_price_predictor": {
                        "weights": self.spot_price_predictor.weights,
                        "bias": self.spot_price_predictor.bias,
                        "trained": self.spot_price_predictor.trained,
                    },
                },
                "timestamp": time.time(),
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"Saved cost optimization state to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization state: {e}")


# Global cost optimizer instance
_cost_optimizer: Optional[MLCostOptimizer] = None

def get_cost_optimizer() -> MLCostOptimizer:
    """Get or create global ML cost optimizer."""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = MLCostOptimizer(
            cost_optimization_strategy=CostOptimizationStrategy.BALANCED,
            spot_instance_max_ratio=0.7,
            enable_predictive_scaling=True,
        )
    return _cost_optimizer

def record_cost_metrics(
    current_cost: float,
    instance_counts: Dict[str, int],
    performance_metrics: Dict[str, float],
):
    """Record cost metrics for ML optimization."""
    try:
        optimizer = get_cost_optimizer()
        
        # Convert string keys to InstanceType enums
        instance_type_counts = {}
        for key, count in instance_counts.items():
            try:
                instance_type = InstanceType(key)
                instance_type_counts[instance_type] = count
            except ValueError:
                pass  # Skip unknown instance types
        
        optimizer.record_cost_metrics(current_cost, instance_type_counts, performance_metrics)
        
    except Exception as e:
        # Silently handle errors to avoid disrupting main application
        pass

def get_cost_optimization_stats() -> Dict[str, Any]:
    """Get cost optimization statistics."""
    try:
        optimizer = get_cost_optimizer()
        return optimizer.get_optimization_stats()
    except Exception:
        return {"error": "Cost optimizer not available"}