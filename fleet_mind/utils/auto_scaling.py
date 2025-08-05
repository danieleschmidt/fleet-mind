"""Intelligent auto-scaling system for Fleet-Mind with predictive scaling and load balancing."""

import asyncio
import time
import threading
import math
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import statistics
import json


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"      # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predicted load
    ADAPTIVE = "adaptive"      # Learn from patterns and adapt
    HYBRID = "hybrid"         # Combination of all approaches


class MetricType(Enum):
    """Types of metrics for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


@dataclass
class ScalingMetric:
    """A metric used for scaling decisions."""
    metric_type: MetricType
    name: str
    current_value: float = 0.0
    threshold_up: float = 80.0
    threshold_down: float = 30.0
    weight: float = 1.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_value(self, value: float):
        """Add a new metric value."""
        self.current_value = value
        self.history.append((time.time(), value))
    
    @property
    def average(self) -> float:
        """Get average value from history."""
        if not self.history:
            return 0.0
        return statistics.mean(val for _, val in self.history)
    
    @property
    def trend(self) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)."""
        if len(self.history) < 2:
            return 0.0
        
        # Simple linear regression
        times = [t for t, _ in self.history]
        values = [v for _, v in self.history]
        
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(t * v for t, v in zip(times, values))
        sum_x2 = sum(t * t for t in times)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


@dataclass
class ScalingAction:
    """Represents a scaling action."""
    action_type: str  # "scale_up", "scale_down", "no_action"
    target_capacity: int
    reason: str
    confidence: float = 1.0
    estimated_time: float = 60.0  # seconds
    cost_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action_type': self.action_type,
            'target_capacity': self.target_capacity,
            'reason': self.reason,
            'confidence': self.confidence,
            'estimated_time': self.estimated_time,
            'cost_estimate': self.cost_estimate,
        }


class LoadPredictor:
    """Predicts future load based on historical patterns."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize load predictor.
        
        Args:
            history_size: Maximum size of history to keep
        """
        self.history_size = history_size
        self.load_history: deque = deque(maxlen=history_size)
        self.pattern_cache: Dict[str, Any] = {}
        
        # Pattern detection parameters
        self.seasonality_periods = [60, 300, 900, 3600, 86400]  # 1min, 5min, 15min, 1hour, 1day
        self.pattern_threshold = 0.8  # Correlation threshold for pattern detection
    
    def add_load_data(self, load: float, timestamp: Optional[float] = None):
        """Add load data point.
        
        Args:
            load: Current load value
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.load_history.append((timestamp, load))
        
        # Update patterns periodically
        if len(self.load_history) % 10 == 0:
            self._update_patterns()
    
    def predict_load(
        self,
        horizon_seconds: int = 300,
        confidence_interval: float = 0.95
    ) -> Tuple[float, float, float]:
        """Predict future load.
        
        Args:
            horizon_seconds: Prediction horizon in seconds
            confidence_interval: Confidence interval (0.0 to 1.0)
            
        Returns:
            Tuple of (predicted_load, lower_bound, upper_bound)
        """
        if len(self.load_history) < 10:
            # Not enough data, return current load
            current_load = self.load_history[-1][1] if self.load_history else 0.0
            return current_load, current_load * 0.8, current_load * 1.2
        
        # Try different prediction methods
        predictions = []
        
        # Method 1: Trend-based prediction
        trend_pred = self._predict_by_trend(horizon_seconds)
        if trend_pred is not None:
            predictions.append(trend_pred)
        
        # Method 2: Seasonal pattern prediction
        seasonal_pred = self._predict_by_seasonality(horizon_seconds)
        if seasonal_pred is not None:
            predictions.append(seasonal_pred)
        
        # Method 3: Moving average
        ma_pred = self._predict_by_moving_average(horizon_seconds)
        if ma_pred is not None:
            predictions.append(ma_pred)
        
        # Ensemble prediction
        if predictions:
            predicted_load = statistics.mean(predictions)
            std_dev = statistics.stdev(predictions) if len(predictions) > 1 else predicted_load * 0.1
            
            # Calculate confidence bounds
            z_score = 1.96 if confidence_interval >= 0.95 else 1.645  # Simplified
            margin = z_score * std_dev
            
            return predicted_load, predicted_load - margin, predicted_load + margin
        else:
            # Fallback to current load
            current_load = self.load_history[-1][1]
            return current_load, current_load * 0.8, current_load * 1.2
    
    def _predict_by_trend(self, horizon_seconds: int) -> Optional[float]:
        """Predict using linear trend."""
        if len(self.load_history) < 5:
            return None
        
        # Use last 50 points or all available
        recent_data = list(self.load_history)[-50:]
        times = [t for t, _ in recent_data]
        loads = [l for _, l in recent_data]
        
        # Linear regression
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(loads)
        sum_xy = sum(t * l for t, l in zip(times, loads))
        sum_x2 = sum(t * t for t in times)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return None
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict
        future_time = times[-1] + horizon_seconds
        predicted_load = slope * future_time + intercept
        
        return max(0, predicted_load)  # Load cannot be negative
    
    def _predict_by_seasonality(self, horizon_seconds: int) -> Optional[float]:
        """Predict using seasonal patterns."""
        current_time = time.time()
        
        # Check each seasonality period
        for period in self.seasonality_periods:
            if len(self.load_history) < period / 10:  # Need at least 10% of period
                continue
            
            # Find historical data at same time in cycle
            target_phase = (current_time + horizon_seconds) % period
            
            matching_loads = []
            for timestamp, load in self.load_history:
                phase = timestamp % period
                if abs(phase - target_phase) < period * 0.05:  # Within 5% of period
                    matching_loads.append(load)
            
            if len(matching_loads) >= 3:
                return statistics.mean(matching_loads)
        
        return None
    
    def _predict_by_moving_average(self, horizon_seconds: int) -> Optional[float]:
        """Predict using weighted moving average."""
        if len(self.load_history) < 5:
            return None
        
        # Use recent data with exponential weighting
        recent_data = list(self.load_history)[-20:]
        weights = [math.exp(-i * 0.1) for i in range(len(recent_data))]
        weights.reverse()  # Most recent gets highest weight
        
        weighted_sum = sum(w * l for w, (_, l) in zip(weights, recent_data))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else None
    
    def _update_patterns(self):
        """Update detected patterns in the data."""
        if len(self.load_history) < 100:
            return
        
        # Simple pattern detection - look for recurring cycles
        data = [load for _, load in self.load_history]
        
        for period in self.seasonality_periods:
            if len(data) < period:
                continue
            
            # Calculate autocorrelation at this period
            correlation = self._calculate_autocorrelation(data, period)
            
            pattern_key = f"period_{period}"
            self.pattern_cache[pattern_key] = {
                'correlation': correlation,
                'strength': 'strong' if correlation > 0.8 else 'weak' if correlation > 0.5 else 'none',
                'last_updated': time.time(),
            }
    
    def _calculate_autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if len(data) <= lag:
            return 0.0
        
        n = len(data) - lag
        if n <= 1:
            return 0.0
        
        mean_x = statistics.mean(data[:-lag])
        mean_y = statistics.mean(data[lag:])
        
        numerator = sum((data[i] - mean_x) * (data[i + lag] - mean_y) for i in range(n))
        
        sum_sq_x = sum((data[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((data[i + lag] - mean_y) ** 2 for i in range(n))
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        return numerator / denominator if denominator > 0 else 0.0


class AutoScaler:
    """Intelligent auto-scaling system with multiple strategies."""
    
    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 100,
        scaling_policy: ScalingPolicy = ScalingPolicy.ADAPTIVE,
        cooldown_period: float = 300.0,  # 5 minutes
        prediction_horizon: int = 300,   # 5 minutes
        enable_cost_optimization: bool = True
    ):
        """Initialize auto-scaler.
        
        Args:
            min_capacity: Minimum number of instances
            max_capacity: Maximum number of instances
            scaling_policy: Scaling policy to use
            cooldown_period: Cooldown between scaling actions
            prediction_horizon: Prediction horizon in seconds
            enable_cost_optimization: Enable cost-aware scaling
        """
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.scaling_policy = scaling_policy
        self.cooldown_period = cooldown_period
        self.prediction_horizon = prediction_horizon
        self.enable_cost_optimization = enable_cost_optimization
        
        # Current state
        self.current_capacity = min_capacity
        self.target_capacity = min_capacity
        self.last_scaling_time = 0.0
        self.last_scaling_action = "no_action"
        
        # Metrics tracking
        self.metrics: Dict[str, ScalingMetric] = {}
        self.load_predictor = LoadPredictor()
        
        # Scaling history
        self.scaling_history: deque = deque(maxlen=1000)
        
        # Cost tracking
        self.cost_per_instance_hour = 0.10  # Default cost
        self.performance_cost_ratio = 1.0
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        # Start monitoring
        self._monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def _initialize_default_metrics(self):
        """Initialize default scaling metrics."""
        self.metrics['cpu'] = ScalingMetric(
            MetricType.CPU_UTILIZATION,
            'cpu_utilization',
            threshold_up=75.0,
            threshold_down=25.0,
            weight=1.0
        )
        
        self.metrics['memory'] = ScalingMetric(
            MetricType.MEMORY_UTILIZATION,
            'memory_utilization',
            threshold_up=80.0,
            threshold_down=30.0,
            weight=0.8
        )
        
        self.metrics['queue'] = ScalingMetric(
            MetricType.QUEUE_LENGTH,
            'queue_length',
            threshold_up=50.0,
            threshold_down=5.0,
            weight=1.2
        )
        
        self.metrics['response_time'] = ScalingMetric(
            MetricType.RESPONSE_TIME,
            'response_time_ms',
            threshold_up=1000.0,
            threshold_down=200.0,
            weight=0.9
        )
    
    def add_metric(self, name: str, metric: ScalingMetric):
        """Add custom scaling metric.
        
        Args:
            name: Metric name
            metric: Scaling metric instance
        """
        self.metrics[name] = metric
    
    def update_metric(self, name: str, value: float):
        """Update metric value.
        
        Args:
            name: Metric name
            value: New metric value
        """
        if name in self.metrics:
            self.metrics[name].add_value(value)
            
            # Update load predictor with aggregated load
            if name == 'cpu':
                self.load_predictor.add_load_data(value)
    
    def evaluate_scaling(self) -> ScalingAction:
        """Evaluate whether scaling is needed.
        
        Returns:
            Scaling action recommendation
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.cooldown_period:
            return ScalingAction(
                "no_action",
                self.current_capacity,
                f"Cooldown period active (remaining: {self.cooldown_period - (current_time - self.last_scaling_time):.1f}s)"
            )
        
        # Evaluate based on policy
        if self.scaling_policy == ScalingPolicy.REACTIVE:
            return self._evaluate_reactive()
        elif self.scaling_policy == ScalingPolicy.PREDICTIVE:
            return self._evaluate_predictive()
        elif self.scaling_policy == ScalingPolicy.ADAPTIVE:
            return self._evaluate_adaptive()
        else:  # HYBRID
            return self._evaluate_hybrid()
    
    def _evaluate_reactive(self) -> ScalingAction:
        """Reactive scaling based on current metrics."""
        scale_up_score = 0.0
        scale_down_score = 0.0
        reasons = []
        
        for name, metric in self.metrics.items():
            if metric.current_value > metric.threshold_up:
                score = ((metric.current_value - metric.threshold_up) / metric.threshold_up) * metric.weight
                scale_up_score += score
                reasons.append(f"{name}: {metric.current_value:.1f} > {metric.threshold_up}")
            
            elif metric.current_value < metric.threshold_down:
                score = ((metric.threshold_down - metric.current_value) / metric.threshold_down) * metric.weight
                scale_down_score += score
                reasons.append(f"{name}: {metric.current_value:.1f} < {metric.threshold_down}")
        
        # Make scaling decision
        if scale_up_score > scale_down_score and scale_up_score > 1.0:
            # Scale up
            scale_factor = min(2.0, 1.0 + scale_up_score * 0.5)
            target = min(self.max_capacity, int(self.current_capacity * scale_factor))
            target = max(target, self.current_capacity + 1)  # At least +1
            
            return ScalingAction(
                "scale_up",
                target,
                f"High load detected: {'; '.join(reasons)}",
                confidence=min(1.0, scale_up_score / 2.0)
            )
        
        elif scale_down_score > 1.0 and self.current_capacity > self.min_capacity:
            # Scale down
            scale_factor = max(0.5, 1.0 - scale_down_score * 0.3)
            target = max(self.min_capacity, int(self.current_capacity * scale_factor))
            target = min(target, self.current_capacity - 1)  # At least -1
            
            return ScalingAction(
                "scale_down",
                target,
                f"Low load detected: {'; '.join(reasons)}",
                confidence=min(1.0, scale_down_score / 2.0)
            )
        
        return ScalingAction(
            "no_action",
            self.current_capacity,
            "Metrics within acceptable ranges"
        )
    
    def _evaluate_predictive(self) -> ScalingAction:
        """Predictive scaling based on forecasted load."""
        # Get load prediction
        predicted_load, lower_bound, upper_bound = self.load_predictor.predict_load(
            self.prediction_horizon
        )
        
        # Use CPU as primary load indicator
        current_cpu = self.metrics['cpu'].current_value
        cpu_threshold_up = self.metrics['cpu'].threshold_up
        cpu_threshold_down = self.metrics['cpu'].threshold_down
        
        # Calculate required capacity based on prediction
        if predicted_load > cpu_threshold_up:
            # Need to scale up proactively
            scale_factor = predicted_load / cpu_threshold_up
            target = min(self.max_capacity, int(self.current_capacity * scale_factor))
            target = max(target, self.current_capacity + 1)
            
            confidence = min(1.0, (predicted_load - cpu_threshold_up) / cpu_threshold_up)
            
            return ScalingAction(
                "scale_up",
                target,
                f"Predicted load increase: {predicted_load:.1f}% (current: {current_cpu:.1f}%)",
                confidence=confidence
            )
        
        elif predicted_load < cpu_threshold_down and self.current_capacity > self.min_capacity:
            # Can scale down proactively
            scale_factor = max(0.5, predicted_load / current_cpu if current_cpu > 0 else 0.8)
            target = max(self.min_capacity, int(self.current_capacity * scale_factor))
            target = min(target, self.current_capacity - 1)
            
            confidence = min(1.0, (cpu_threshold_down - predicted_load) / cpu_threshold_down)
            
            return ScalingAction(
                "scale_down",
                target,
                f"Predicted load decrease: {predicted_load:.1f}% (current: {current_cpu:.1f}%)",
                confidence=confidence
            )
        
        return ScalingAction(
            "no_action",
            self.current_capacity,
            f"Predicted load acceptable: {predicted_load:.1f}%"
        )
    
    def _evaluate_adaptive(self) -> ScalingAction:
        """Adaptive scaling that learns from patterns."""
        # Combine reactive and predictive approaches
        reactive_action = self._evaluate_reactive()
        predictive_action = self._evaluate_predictive()
        
        # Learn from historical performance
        recent_history = list(self.scaling_history)[-20:]  # Last 20 actions
        
        if recent_history:
            # Calculate success rate of recent scaling decisions
            successful_actions = sum(1 for entry in recent_history if entry.get('success', False))
            success_rate = successful_actions / len(recent_history)
            
            # Adjust confidence based on historical success
            confidence_multiplier = 0.5 + success_rate * 0.5
        else:
            confidence_multiplier = 1.0
        
        # Choose best action based on confidence and patterns
        actions = [reactive_action, predictive_action]
        best_action = max(actions, key=lambda a: a.confidence)
        
        # Adjust confidence
        best_action.confidence *= confidence_multiplier
        
        # Add adaptive reasoning
        if best_action.action_type != "no_action":
            best_action.reason += f" (adaptive confidence: {best_action.confidence:.2f})"
        
        return best_action
    
    def _evaluate_hybrid(self) -> ScalingAction:
        """Hybrid approach combining all strategies."""
        reactive_action = self._evaluate_reactive()
        predictive_action = self._evaluate_predictive()
        adaptive_action = self._evaluate_adaptive()
        
        actions = [reactive_action, predictive_action, adaptive_action]
        
        # Weighted voting
        weights = {'scale_up': 0, 'scale_down': 0, 'no_action': 0}
        action_details = {'scale_up': [], 'scale_down': [], 'no_action': []}
        
        for action in actions:
            weight = action.confidence
            weights[action.action_type] += weight
            action_details[action.action_type].append(action)
        
        # Choose winning action
        winning_action_type = max(weights.keys(), key=lambda k: weights[k])
        
        if winning_action_type == 'no_action':
            return ScalingAction(
                "no_action",
                self.current_capacity,
                "Hybrid analysis suggests no scaling needed"
            )
        
        # Get representative action
        representative_actions = action_details[winning_action_type]
        if representative_actions:
            best_action = max(representative_actions, key=lambda a: a.confidence)
            best_action.reason = f"Hybrid decision: {best_action.reason}"
            return best_action
        
        return ScalingAction(
            "no_action",
            self.current_capacity,
            "Hybrid analysis inconclusive"
        )
    
    def execute_scaling(self, action: ScalingAction) -> bool:
        """Execute scaling action.
        
        Args:
            action: Scaling action to execute
            
        Returns:
            True if scaling was executed successfully
        """
        if action.action_type == "no_action":
            return True
        
        # Validate capacity bounds
        target = max(self.min_capacity, min(self.max_capacity, action.target_capacity))
        
        if target == self.current_capacity:
            return True
        
        # Record scaling action
        scaling_record = {
            'timestamp': time.time(),
            'action': action.to_dict(),
            'old_capacity': self.current_capacity,
            'new_capacity': target,
            'success': False,  # Will be updated later
        }
        
        try:
            # Execute the scaling (this would integrate with actual infrastructure)
            success = self._perform_scaling(target)
            
            if success:
                self.current_capacity = target
                self.target_capacity = target
                self.last_scaling_time = time.time()
                self.last_scaling_action = action.action_type
                scaling_record['success'] = True
                
                print(f"Scaling {action.action_type}: {self.current_capacity} -> {target} ({action.reason})")
            
            self.scaling_history.append(scaling_record)
            return success
            
        except Exception as e:
            print(f"Scaling failed: {e}")
            scaling_record['error'] = str(e)
            self.scaling_history.append(scaling_record)
            return False
    
    def _perform_scaling(self, target_capacity: int) -> bool:
        """Perform actual scaling operation.
        
        This is a placeholder - in real implementation, this would
        interact with infrastructure APIs (AWS, Kubernetes, etc.)
        
        Args:
            target_capacity: Target number of instances
            
        Returns:
            True if scaling succeeded
        """
        # Simulate scaling delay
        time.sleep(0.1)
        
        # For demo purposes, always succeed
        return True
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Update system metrics if psutil is available
                if PSUTIL_AVAILABLE:
                    self.update_metric('cpu', psutil.cpu_percent())
                    self.update_metric('memory', psutil.virtual_memory().percent)
                
                # Evaluate and execute auto-scaling
                action = self.evaluate_scaling()
                if action.action_type != "no_action" and action.confidence > 0.7:
                    self.execute_scaling(action)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error in auto-scaling monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling statistics."""
        recent_actions = list(self.scaling_history)[-10:]
        
        return {
            'current_capacity': self.current_capacity,
            'target_capacity': self.target_capacity,
            'min_capacity': self.min_capacity,
            'max_capacity': self.max_capacity,
            'scaling_policy': self.scaling_policy.value,
            'last_scaling_time': self.last_scaling_time,
            'last_scaling_action': self.last_scaling_action,
            'cooldown_remaining': max(0, self.cooldown_period - (time.time() - self.last_scaling_time)),
            'metrics': {
                name: {
                    'current_value': metric.current_value,
                    'threshold_up': metric.threshold_up,
                    'threshold_down': metric.threshold_down,
                    'average': metric.average,
                    'trend': metric.trend,
                }
                for name, metric in self.metrics.items()
            },
            'recent_actions': recent_actions,
            'total_scaling_actions': len(self.scaling_history),
            'load_prediction': self.load_predictor.predict_load(self.prediction_horizon),
        }
    
    def shutdown(self):
        """Shutdown the auto-scaler."""
        self._monitoring_active = False


# Global auto-scaler instance (lazy-initialized)
_global_autoscaler = None

def _get_global_autoscaler():
    """Get or create global auto-scaler."""
    global _global_autoscaler
    if _global_autoscaler is None:
        _global_autoscaler = AutoScaler(
            min_capacity=2,
            max_capacity=50,
            scaling_policy=ScalingPolicy.ADAPTIVE,
            cooldown_period=300.0,
            prediction_horizon=300
        )
    return _global_autoscaler


def get_autoscaling_stats() -> Dict[str, Any]:
    """Get comprehensive auto-scaling statistics."""
    try:
        autoscaler = _get_global_autoscaler()
        return autoscaler.get_stats()
    except Exception:
        return {'error': 'Auto-scaler not initialized'}


def update_scaling_metric(name: str, value: float):
    """Update a scaling metric value.
    
    Args:
        name: Metric name
        value: New metric value
    """
    try:
        autoscaler = _get_global_autoscaler()
        autoscaler.update_metric(name, value)
    except Exception:
        pass  # Silently ignore if not initialized


def trigger_scaling_evaluation() -> ScalingAction:
    """Manually trigger scaling evaluation.
    
    Returns:
        Recommended scaling action
    """
    try:
        autoscaler = _get_global_autoscaler()
        return autoscaler.evaluate_scaling()
    except Exception:
        return ScalingAction('no_action', 1, 'Auto-scaler not initialized')