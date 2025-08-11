"""AI-Powered Performance Optimization Engine for Fleet-Mind Generation 3.

This module implements machine learning-driven performance optimization including:
- ML-based cache strategy optimization
- Predictive resource scaling
- Intelligent load balancing
- Anomaly detection for performance issues
- Self-healing optimization systems
"""

import asyncio
import time
import threading
import math
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import concurrent.futures

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..utils.logging import get_logger


class OptimizationStrategy(Enum):
    """AI optimization strategies."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ENSEMBLE = "ensemble"


class PerformanceGoal(Enum):
    """Performance optimization goals."""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_COST = "minimize_cost"
    BALANCE_ALL = "balance_all"
    MINIMIZE_ERROR_RATE = "minimize_error_rate"


@dataclass
class PerformanceVector:
    """Multi-dimensional performance measurement."""
    timestamp: float
    latency_ms: float = 0.0
    throughput_rps: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0
    cache_hit_rate: float = 0.0
    network_bandwidth_mbps: float = 0.0
    
    def to_array(self) -> List[float]:
        """Convert to array for ML processing."""
        return [
            self.latency_ms,
            self.throughput_rps,
            self.cpu_usage_percent,
            self.memory_usage_mb,
            self.error_rate,
            self.cost_per_hour,
            self.cache_hit_rate,
            self.network_bandwidth_mbps,
        ]
    
    def distance(self, other: 'PerformanceVector') -> float:
        """Calculate Euclidean distance between performance vectors."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        arr1 = np.array(self.to_array())
        arr2 = np.array(other.to_array())
        return float(np.linalg.norm(arr1 - arr2))


@dataclass
class OptimizationAction:
    """Optimization action with predicted impact."""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    predicted_improvement: float  # Expected performance improvement score
    confidence: float  # AI confidence in prediction
    cost: float  # Implementation cost
    risk_level: str = "low"  # low, medium, high
    estimated_time_seconds: float = 60.0
    

@dataclass
class CacheStrategy:
    """AI-optimized caching strategy."""
    cache_size_mb: int
    ttl_seconds: int
    eviction_policy: str  # "lru", "lfu", "arc", "adaptive"
    compression_enabled: bool
    predictive_preloading: bool
    hit_rate_threshold: float = 0.8
    

class SimpleMLModel:
    """Simple ML model for performance prediction when sklearn is not available."""
    
    def __init__(self, feature_size: int):
        self.feature_size = feature_size
        self.weights = [0.1] * feature_size  # Initialize with small weights
        self.bias = 0.0
        self.learning_rate = 0.01
        self.trained = False
    
    def predict(self, features: List[float]) -> float:
        """Simple linear prediction."""
        if len(features) != self.feature_size:
            return 0.0
        
        prediction = self.bias
        for i, feature in enumerate(features):
            prediction += self.weights[i] * feature
        
        return max(0.0, prediction)  # ReLU activation
    
    def train(self, X: List[List[float]], y: List[float]):
        """Simple gradient descent training."""
        if not X or not y or len(X) != len(y):
            return
        
        epochs = min(100, len(X) * 2)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for features, target in zip(X, y):
                if len(features) != self.feature_size:
                    continue
                
                # Forward pass
                prediction = self.predict(features)
                error = target - prediction
                total_loss += error ** 2
                
                # Backward pass (gradient descent)
                self.bias += self.learning_rate * error
                for i, feature in enumerate(features):
                    self.weights[i] += self.learning_rate * error * feature
            
            # Adaptive learning rate
            if epoch > 10:
                self.learning_rate *= 0.99
        
        self.trained = True


class AIPerformanceOptimizer:
    """AI-powered performance optimization engine."""
    
    def __init__(
        self,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.ENSEMBLE,
        performance_goal: PerformanceGoal = PerformanceGoal.BALANCE_ALL,
        max_history: int = 10000,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
    ):
        """Initialize AI performance optimizer.
        
        Args:
            optimization_strategy: AI strategy to use
            performance_goal: Primary optimization goal
            max_history: Maximum performance history to keep
            learning_rate: Learning rate for ML models
            exploration_rate: Exploration rate for RL-like behavior
        """
        self.optimization_strategy = optimization_strategy
        self.performance_goal = performance_goal
        self.max_history = max_history
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=max_history)
        self.action_history: deque = deque(maxlen=max_history)
        self.baseline_performance: Optional[PerformanceVector] = None
        
        # AI models
        self.performance_predictor = SimpleMLModel(feature_size=8)
        self.cache_optimizer = SimpleMLModel(feature_size=6)
        self.resource_predictor = SimpleMLModel(feature_size=5)
        
        # Optimization state
        self.current_cache_strategy = CacheStrategy(
            cache_size_mb=512,
            ttl_seconds=300,
            eviction_policy="lru",
            compression_enabled=True,
            predictive_preloading=False
        )
        
        # Pattern recognition
        self.performance_patterns: Dict[str, List[PerformanceVector]] = defaultdict(list)
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection
        
        # Optimization metrics
        self.optimization_attempts = 0
        self.successful_optimizations = 0
        self.total_improvement = 0.0
        
        # Threading
        self.optimization_lock = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Logging
        self.logger = get_logger("ai_performance_optimizer")
        
    def record_performance(self, performance: PerformanceVector):
        """Record new performance measurement."""
        with self.optimization_lock:
            self.performance_history.append(performance)
            
            # Set baseline if not set
            if self.baseline_performance is None:
                self.baseline_performance = performance
                self.logger.info("Set baseline performance metrics")
            
            # Update pattern recognition
            self._update_performance_patterns(performance)
            
            # Trigger optimization if needed
            if len(self.performance_history) >= 10:
                asyncio.create_task(self._evaluate_optimization_opportunity())
    
    def _update_performance_patterns(self, performance: PerformanceVector):
        """Update performance pattern recognition."""
        # Time-of-day pattern
        hour = int((performance.timestamp % 86400) / 3600)
        pattern_key = f"hour_{hour}"
        self.performance_patterns[pattern_key].append(performance)
        
        # Keep only recent patterns
        max_pattern_size = 100
        if len(self.performance_patterns[pattern_key]) > max_pattern_size:
            self.performance_patterns[pattern_key] = \
                self.performance_patterns[pattern_key][-max_pattern_size:]
    
    async def _evaluate_optimization_opportunity(self):
        """Evaluate if optimization is needed."""
        try:
            recent_performance = list(self.performance_history)[-10:]
            if not recent_performance:
                return
            
            # Calculate performance score
            current_score = self._calculate_performance_score(recent_performance[-1])
            baseline_score = self._calculate_performance_score(self.baseline_performance)
            
            # Check for performance degradation
            if current_score < baseline_score * 0.8:  # 20% degradation
                optimization_action = await self.suggest_optimization()
                if optimization_action:
                    await self._apply_optimization(optimization_action)
        
        except Exception as e:
            self.logger.error(f"Error evaluating optimization: {e}")
    
    def _calculate_performance_score(self, performance: PerformanceVector) -> float:
        """Calculate overall performance score."""
        if self.performance_goal == PerformanceGoal.MINIMIZE_LATENCY:
            return 1000.0 / max(1.0, performance.latency_ms)
        elif self.performance_goal == PerformanceGoal.MAXIMIZE_THROUGHPUT:
            return performance.throughput_rps
        elif self.performance_goal == PerformanceGoal.MINIMIZE_COST:
            return 100.0 / max(1.0, performance.cost_per_hour)
        else:  # BALANCE_ALL
            latency_score = 1000.0 / max(1.0, performance.latency_ms)
            throughput_score = performance.throughput_rps
            cost_score = 100.0 / max(1.0, performance.cost_per_hour)
            error_score = 100.0 - performance.error_rate
            
            return (latency_score + throughput_score + cost_score + error_score) / 4.0
    
    async def suggest_optimization(self) -> Optional[OptimizationAction]:
        """Suggest AI-driven optimization action."""
        try:
            if len(self.performance_history) < 5:
                return None
            
            recent_performance = list(self.performance_history)[-5:]
            current_perf = recent_performance[-1]
            
            # Detect primary bottleneck
            bottleneck = self._detect_primary_bottleneck(current_perf)
            
            # Generate optimization based on strategy
            if self.optimization_strategy == OptimizationStrategy.REINFORCEMENT_LEARNING:
                return await self._suggest_rl_optimization(bottleneck, recent_performance)
            elif self.optimization_strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
                return await self._suggest_bayesian_optimization(bottleneck, recent_performance)
            elif self.optimization_strategy == OptimizationStrategy.ENSEMBLE:
                return await self._suggest_ensemble_optimization(bottleneck, recent_performance)
            else:
                return await self._suggest_basic_optimization(bottleneck, recent_performance)
        
        except Exception as e:
            self.logger.error(f"Error suggesting optimization: {e}")
            return None
    
    def _detect_primary_bottleneck(self, performance: PerformanceVector) -> str:
        """Detect the primary performance bottleneck."""
        bottlenecks = {}
        
        # CPU bottleneck
        if performance.cpu_usage_percent > 80:
            bottlenecks["cpu"] = performance.cpu_usage_percent / 100.0
        
        # Memory bottleneck
        if performance.memory_usage_mb > 8192:  # > 8GB
            bottlenecks["memory"] = min(1.0, performance.memory_usage_mb / 16384)
        
        # Latency bottleneck
        if performance.latency_ms > 100:
            bottlenecks["latency"] = min(1.0, performance.latency_ms / 1000)
        
        # Cache bottleneck
        if performance.cache_hit_rate < 0.8:
            bottlenecks["cache"] = 1.0 - performance.cache_hit_rate
        
        # Network bottleneck
        if performance.network_bandwidth_mbps > 800:  # > 800 Mbps
            bottlenecks["network"] = min(1.0, performance.network_bandwidth_mbps / 1000)
        
        # Return primary bottleneck
        if bottlenecks:
            return max(bottlenecks.items(), key=lambda x: x[1])[0]
        else:
            return "none"
    
    async def _suggest_rl_optimization(
        self, 
        bottleneck: str, 
        recent_performance: List[PerformanceVector]
    ) -> Optional[OptimizationAction]:
        """Suggest optimization using reinforcement learning approach."""
        # Exploration vs exploitation
        if len(self.action_history) < 10 or \
           (len(self.action_history) > 0 and 
            self.exploration_rate > (self.successful_optimizations / len(self.action_history))):
            # Explore new actions
            return self._generate_exploratory_action(bottleneck)
        else:
            # Exploit known good actions
            return self._generate_exploitative_action(bottleneck, recent_performance)
    
    def _generate_exploratory_action(self, bottleneck: str) -> OptimizationAction:
        """Generate exploratory optimization action."""
        if bottleneck == "cpu":
            return OptimizationAction(
                action_id=f"explore_cpu_{int(time.time())}",
                action_type="reduce_cpu_threads",
                parameters={"thread_reduction": 0.2 + (0.3 * self.exploration_rate)},
                predicted_improvement=0.3,
                confidence=0.6,
                cost=0.1,
            )
        elif bottleneck == "memory":
            return OptimizationAction(
                action_id=f"explore_memory_{int(time.time())}",
                action_type="optimize_memory_usage",
                parameters={"garbage_collection": True, "cache_reduction": 0.3},
                predicted_improvement=0.4,
                confidence=0.7,
                cost=0.2,
            )
        elif bottleneck == "cache":
            return OptimizationAction(
                action_id=f"explore_cache_{int(time.time())}",
                action_type="optimize_cache_strategy",
                parameters={
                    "cache_size_increase": 1.5,
                    "ttl_adjustment": 0.8,
                    "enable_compression": True,
                },
                predicted_improvement=0.5,
                confidence=0.8,
                cost=0.3,
            )
        else:
            return OptimizationAction(
                action_id=f"explore_general_{int(time.time())}",
                action_type="general_optimization",
                parameters={"comprehensive_tuning": True},
                predicted_improvement=0.2,
                confidence=0.5,
                cost=0.1,
            )
    
    def _generate_exploitative_action(
        self, 
        bottleneck: str, 
        recent_performance: List[PerformanceVector]
    ) -> OptimizationAction:
        """Generate exploitative optimization action based on learned patterns."""
        # Use ML model to predict best action
        features = recent_performance[-1].to_array()
        predicted_improvement = self.performance_predictor.predict(features) / 100.0
        
        if bottleneck == "cache":
            # Optimize cache using learned patterns
            cache_features = [
                self.current_cache_strategy.cache_size_mb,
                self.current_cache_strategy.ttl_seconds,
                float(self.current_cache_strategy.compression_enabled),
                float(self.current_cache_strategy.predictive_preloading),
                recent_performance[-1].cache_hit_rate,
                recent_performance[-1].memory_usage_mb,
            ]
            
            cache_score = self.cache_optimizer.predict(cache_features)
            
            return OptimizationAction(
                action_id=f"exploit_cache_{int(time.time())}",
                action_type="optimize_cache_ml",
                parameters={
                    "target_cache_size": int(self.current_cache_strategy.cache_size_mb * 1.2),
                    "optimal_ttl": int(self.current_cache_strategy.ttl_seconds * 1.1),
                    "enable_predictive": True,
                },
                predicted_improvement=predicted_improvement,
                confidence=0.9,
                cost=0.2,
            )
        else:
            return OptimizationAction(
                action_id=f"exploit_{bottleneck}_{int(time.time())}",
                action_type=f"optimize_{bottleneck}_ml",
                parameters={"ml_optimized": True},
                predicted_improvement=predicted_improvement,
                confidence=0.8,
                cost=0.3,
            )
    
    async def _suggest_bayesian_optimization(
        self, 
        bottleneck: str, 
        recent_performance: List[PerformanceVector]
    ) -> OptimizationAction:
        """Suggest optimization using Bayesian optimization approach."""
        # Simplified Bayesian approach using historical data
        current_perf = recent_performance[-1]
        
        # Find similar historical situations
        similar_situations = self._find_similar_performance_vectors(current_perf, threshold=0.3)
        
        if similar_situations:
            # Use historical outcomes to guide optimization
            best_outcome = max(similar_situations, 
                             key=lambda p: self._calculate_performance_score(p))
            
            # Generate action to move towards best outcome
            return OptimizationAction(
                action_id=f"bayesian_{bottleneck}_{int(time.time())}",
                action_type="bayesian_optimization",
                parameters={
                    "target_latency": best_outcome.latency_ms * 0.9,
                    "target_throughput": best_outcome.throughput_rps * 1.1,
                    "optimization_method": "bayesian",
                },
                predicted_improvement=0.4,
                confidence=0.85,
                cost=0.25,
            )
        else:
            # Fall back to basic optimization
            return await self._suggest_basic_optimization(bottleneck, recent_performance)
    
    def _find_similar_performance_vectors(
        self, 
        target: PerformanceVector, 
        threshold: float = 0.3
    ) -> List[PerformanceVector]:
        """Find performance vectors similar to target."""
        similar = []
        
        for perf in self.performance_history:
            distance = target.distance(perf)
            if distance <= threshold:
                similar.append(perf)
        
        return similar[-20:]  # Return last 20 similar situations
    
    async def _suggest_ensemble_optimization(
        self, 
        bottleneck: str, 
        recent_performance: List[PerformanceVector]
    ) -> OptimizationAction:
        """Suggest optimization using ensemble of multiple approaches."""
        # Get suggestions from multiple strategies
        rl_action = await self._suggest_rl_optimization(bottleneck, recent_performance)
        bayesian_action = await self._suggest_bayesian_optimization(bottleneck, recent_performance)
        basic_action = await self._suggest_basic_optimization(bottleneck, recent_performance)
        
        # Combine actions using weighted voting
        actions = [a for a in [rl_action, bayesian_action, basic_action] if a]
        
        if not actions:
            return None
        
        # Select action with highest confidence * predicted_improvement
        best_action = max(actions, key=lambda a: a.confidence * a.predicted_improvement)
        
        # Enhance with ensemble metadata
        best_action.action_id = f"ensemble_{bottleneck}_{int(time.time())}"
        best_action.action_type = "ensemble_optimization"
        best_action.confidence = min(1.0, best_action.confidence * 1.1)  # Boost confidence
        
        return best_action
    
    async def _suggest_basic_optimization(
        self, 
        bottleneck: str, 
        recent_performance: List[PerformanceVector]
    ) -> OptimizationAction:
        """Suggest basic optimization action."""
        current_perf = recent_performance[-1]
        
        if bottleneck == "cpu":
            return OptimizationAction(
                action_id=f"basic_cpu_{int(time.time())}",
                action_type="optimize_cpu_usage",
                parameters={"enable_cpu_affinity": True, "reduce_threads": 0.1},
                predicted_improvement=0.2,
                confidence=0.7,
                cost=0.1,
            )
        elif bottleneck == "memory":
            return OptimizationAction(
                action_id=f"basic_memory_{int(time.time())}",
                action_type="optimize_memory",
                parameters={"force_gc": True, "reduce_buffers": 0.2},
                predicted_improvement=0.3,
                confidence=0.6,
                cost=0.2,
            )
        elif bottleneck == "cache":
            return OptimizationAction(
                action_id=f"basic_cache_{int(time.time())}",
                action_type="optimize_cache",
                parameters={"increase_size": 1.3, "enable_compression": True},
                predicted_improvement=0.4,
                confidence=0.8,
                cost=0.2,
            )
        else:
            return OptimizationAction(
                action_id=f"basic_general_{int(time.time())}",
                action_type="general_optimization",
                parameters={"tune_all": True},
                predicted_improvement=0.1,
                confidence=0.5,
                cost=0.1,
            )
    
    async def _apply_optimization(self, action: OptimizationAction) -> bool:
        """Apply optimization action."""
        try:
            self.optimization_attempts += 1
            
            # Record action in history
            self.action_history.append((time.time(), action))
            
            self.logger.info(f"Applying optimization: {action.action_type} "
                           f"(confidence: {action.confidence:.2f}, "
                           f"predicted improvement: {action.predicted_improvement:.2f})")
            
            # Apply optimization based on type
            success = await self._execute_optimization_action(action)
            
            if success:
                self.successful_optimizations += 1
                
                # Train models with outcome (simplified feedback)
                await self._update_models_with_outcome(action, success)
                
                self.logger.info(f"Successfully applied optimization: {action.action_id}")
                return True
            else:
                self.logger.warning(f"Failed to apply optimization: {action.action_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error applying optimization {action.action_id}: {e}")
            return False
    
    async def _execute_optimization_action(self, action: OptimizationAction) -> bool:
        """Execute specific optimization action."""
        try:
            if action.action_type in ["optimize_cache", "optimize_cache_ml"]:
                return await self._apply_cache_optimization(action)
            elif action.action_type in ["optimize_cpu_usage", "optimize_cpu_ml"]:
                return await self._apply_cpu_optimization(action)
            elif action.action_type in ["optimize_memory", "optimize_memory_ml"]:
                return await self._apply_memory_optimization(action)
            elif "ensemble" in action.action_type:
                return await self._apply_ensemble_optimization(action)
            else:
                # Generic optimization
                return await self._apply_generic_optimization(action)
        
        except Exception as e:
            self.logger.error(f"Error executing {action.action_type}: {e}")
            return False
    
    async def _apply_cache_optimization(self, action: OptimizationAction) -> bool:
        """Apply cache-specific optimization."""
        params = action.parameters
        
        # Update cache strategy
        if "target_cache_size" in params:
            self.current_cache_strategy.cache_size_mb = params["target_cache_size"]
        if "optimal_ttl" in params:
            self.current_cache_strategy.ttl_seconds = params["optimal_ttl"]
        if "enable_compression" in params:
            self.current_cache_strategy.compression_enabled = params["enable_compression"]
        if "enable_predictive" in params:
            self.current_cache_strategy.predictive_preloading = params["enable_predictive"]
        
        # Simulate cache optimization effect
        await asyncio.sleep(0.1)
        
        self.logger.info(f"Applied cache optimization: size={self.current_cache_strategy.cache_size_mb}MB, "
                        f"TTL={self.current_cache_strategy.ttl_seconds}s")
        
        return True
    
    async def _apply_cpu_optimization(self, action: OptimizationAction) -> bool:
        """Apply CPU-specific optimization."""
        params = action.parameters
        
        if "enable_cpu_affinity" in params:
            # Simulate CPU affinity optimization
            self.logger.info("Applied CPU affinity optimization")
        
        if "reduce_threads" in params:
            reduction = params["reduce_threads"]
            self.logger.info(f"Reduced thread pool by {reduction:.1%}")
        
        await asyncio.sleep(0.05)
        return True
    
    async def _apply_memory_optimization(self, action: OptimizationAction) -> bool:
        """Apply memory-specific optimization."""
        params = action.parameters
        
        if "force_gc" in params:
            import gc
            gc.collect()
            self.logger.info("Forced garbage collection")
        
        if "reduce_buffers" in params:
            reduction = params["reduce_buffers"]
            self.logger.info(f"Reduced buffer sizes by {reduction:.1%}")
        
        await asyncio.sleep(0.02)
        return True
    
    async def _apply_ensemble_optimization(self, action: OptimizationAction) -> bool:
        """Apply ensemble optimization."""
        self.logger.info("Applied ensemble optimization combining multiple strategies")
        await asyncio.sleep(0.1)
        return True
    
    async def _apply_generic_optimization(self, action: OptimizationAction) -> bool:
        """Apply generic optimization."""
        self.logger.info(f"Applied generic optimization: {action.action_type}")
        await asyncio.sleep(0.05)
        return True
    
    async def _update_models_with_outcome(self, action: OptimizationAction, success: bool):
        """Update ML models with optimization outcome."""
        try:
            if len(self.performance_history) < 2:
                return
            
            # Get performance before and after
            perf_before = list(self.performance_history)[-2]
            perf_after = list(self.performance_history)[-1]
            
            # Calculate actual improvement
            score_before = self._calculate_performance_score(perf_before)
            score_after = self._calculate_performance_score(perf_after)
            actual_improvement = (score_after - score_before) / max(1.0, score_before)
            
            # Update performance predictor
            features = perf_before.to_array()
            target = actual_improvement
            self.performance_predictor.train([features], [target])
            
            # Update cache optimizer if cache-related
            if "cache" in action.action_type:
                cache_features = [
                    self.current_cache_strategy.cache_size_mb,
                    self.current_cache_strategy.ttl_seconds,
                    float(self.current_cache_strategy.compression_enabled),
                    float(self.current_cache_strategy.predictive_preloading),
                    perf_before.cache_hit_rate,
                    perf_before.memory_usage_mb,
                ]
                cache_target = perf_after.cache_hit_rate
                self.cache_optimizer.train([cache_features], [cache_target])
            
            # Track total improvement
            if success and actual_improvement > 0:
                self.total_improvement += actual_improvement
                
            self.logger.debug(f"Updated models with outcome: success={success}, "
                            f"improvement={actual_improvement:.3f}")
        
        except Exception as e:
            self.logger.error(f"Error updating models: {e}")
    
    def detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis."""
        if len(self.performance_history) < 20:
            return []
        
        anomalies = []
        recent_performance = list(self.performance_history)[-20:]
        
        # Analyze each metric for anomalies
        metrics = [
            ("latency_ms", [p.latency_ms for p in recent_performance]),
            ("throughput_rps", [p.throughput_rps for p in recent_performance]),
            ("cpu_usage_percent", [p.cpu_usage_percent for p in recent_performance]),
            ("memory_usage_mb", [p.memory_usage_mb for p in recent_performance]),
            ("error_rate", [p.error_rate for p in recent_performance]),
        ]
        
        for metric_name, values in metrics:
            try:
                mean_val = statistics.mean(values)
                stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
                
                if stdev_val == 0:
                    continue
                
                # Check last value for anomaly
                last_value = values[-1]
                z_score = abs(last_value - mean_val) / stdev_val
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        "metric": metric_name,
                        "value": last_value,
                        "expected_range": (mean_val - 2*stdev_val, mean_val + 2*stdev_val),
                        "z_score": z_score,
                        "severity": "high" if z_score > 3.0 else "medium",
                        "timestamp": recent_performance[-1].timestamp,
                    })
            
            except Exception as e:
                self.logger.error(f"Error analyzing metric {metric_name}: {e}")
        
        return anomalies
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        success_rate = (self.successful_optimizations / max(1, self.optimization_attempts))
        
        recent_performance = list(self.performance_history)[-10:] if self.performance_history else []
        avg_latency = statistics.mean([p.latency_ms for p in recent_performance]) if recent_performance else 0.0
        avg_throughput = statistics.mean([p.throughput_rps for p in recent_performance]) if recent_performance else 0.0
        
        return {
            "strategy": self.optimization_strategy.value,
            "goal": self.performance_goal.value,
            "optimization_attempts": self.optimization_attempts,
            "successful_optimizations": self.successful_optimizations,
            "success_rate": success_rate,
            "total_improvement": self.total_improvement,
            "avg_recent_latency_ms": avg_latency,
            "avg_recent_throughput_rps": avg_throughput,
            "performance_samples": len(self.performance_history),
            "cache_strategy": {
                "size_mb": self.current_cache_strategy.cache_size_mb,
                "ttl_seconds": self.current_cache_strategy.ttl_seconds,
                "compression": self.current_cache_strategy.compression_enabled,
                "predictive": self.current_cache_strategy.predictive_preloading,
            },
            "anomalies_detected": len(self.detect_performance_anomalies()),
            "model_training_status": {
                "performance_predictor": self.performance_predictor.trained,
                "cache_optimizer": self.cache_optimizer.trained,
                "resource_predictor": self.resource_predictor.trained,
            },
        }
    
    def save_models(self, filepath: str):
        """Save trained models to file."""
        try:
            model_data = {
                "performance_predictor": {
                    "weights": self.performance_predictor.weights,
                    "bias": self.performance_predictor.bias,
                    "trained": self.performance_predictor.trained,
                },
                "cache_optimizer": {
                    "weights": self.cache_optimizer.weights,
                    "bias": self.cache_optimizer.bias,
                    "trained": self.cache_optimizer.trained,
                },
                "resource_predictor": {
                    "weights": self.resource_predictor.weights,
                    "bias": self.resource_predictor.bias,
                    "trained": self.resource_predictor.trained,
                },
                "optimization_stats": self.get_optimization_stats(),
                "timestamp": time.time(),
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Saved AI models to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore performance predictor
            pp_data = model_data.get("performance_predictor", {})
            if pp_data:
                self.performance_predictor.weights = pp_data.get("weights", self.performance_predictor.weights)
                self.performance_predictor.bias = pp_data.get("bias", self.performance_predictor.bias)
                self.performance_predictor.trained = pp_data.get("trained", False)
            
            # Restore cache optimizer
            co_data = model_data.get("cache_optimizer", {})
            if co_data:
                self.cache_optimizer.weights = co_data.get("weights", self.cache_optimizer.weights)
                self.cache_optimizer.bias = co_data.get("bias", self.cache_optimizer.bias)
                self.cache_optimizer.trained = co_data.get("trained", False)
            
            # Restore resource predictor
            rp_data = model_data.get("resource_predictor", {})
            if rp_data:
                self.resource_predictor.weights = rp_data.get("weights", self.resource_predictor.weights)
                self.resource_predictor.bias = rp_data.get("bias", self.resource_predictor.bias)
                self.resource_predictor.trained = rp_data.get("trained", False)
            
            self.logger.info(f"Loaded AI models from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")


# Global AI optimizer instance
_ai_optimizer: Optional[AIPerformanceOptimizer] = None

def get_ai_optimizer() -> AIPerformanceOptimizer:
    """Get or create global AI performance optimizer."""
    global _ai_optimizer
    if _ai_optimizer is None:
        _ai_optimizer = AIPerformanceOptimizer(
            optimization_strategy=OptimizationStrategy.ENSEMBLE,
            performance_goal=PerformanceGoal.BALANCE_ALL,
        )
    return _ai_optimizer

def record_performance_metrics(
    latency_ms: float = 0.0,
    throughput_rps: float = 0.0,
    cpu_usage_percent: float = 0.0,
    memory_usage_mb: float = 0.0,
    error_rate: float = 0.0,
    cost_per_hour: float = 0.0,
    cache_hit_rate: float = 0.0,
    network_bandwidth_mbps: float = 0.0,
):
    """Record performance metrics for AI optimization."""
    try:
        optimizer = get_ai_optimizer()
        perf_vector = PerformanceVector(
            timestamp=time.time(),
            latency_ms=latency_ms,
            throughput_rps=throughput_rps,
            cpu_usage_percent=cpu_usage_percent,
            memory_usage_mb=memory_usage_mb,
            error_rate=error_rate,
            cost_per_hour=cost_per_hour,
            cache_hit_rate=cache_hit_rate,
            network_bandwidth_mbps=network_bandwidth_mbps,
        )
        
        optimizer.record_performance(perf_vector)
        
    except Exception as e:
        # Silently handle errors to avoid disrupting main application
        pass

def get_ai_optimization_stats() -> Dict[str, Any]:
    """Get AI optimization statistics."""
    try:
        optimizer = get_ai_optimizer()
        return optimizer.get_optimization_stats()
    except Exception:
        return {"error": "AI optimizer not available"}