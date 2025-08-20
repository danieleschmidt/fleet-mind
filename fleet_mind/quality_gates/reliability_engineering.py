"""Proactive Reliability Engineering with Predictive Prevention and Self-Healing.

Advanced reliability system that predicts failures before they occur and
automatically implements preventive measures and self-healing capabilities.
"""

import asyncio
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import math

from ..utils.advanced_logging import get_logger
from ..utils.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of system failures that can be predicted and prevented."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_LEAK = "memory_leak"
    CONNECTION_FAILURE = "connection_failure"
    DISK_SPACE_EXHAUSTION = "disk_space_exhaustion"
    CPU_OVERLOAD = "cpu_overload"
    NETWORK_CONGESTION = "network_congestion"
    SERVICE_UNAVAILABILITY = "service_unavailability"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    CONFIGURATION_DRIFT = "configuration_drift"


class ReliabilityLevel(Enum):
    """System reliability levels."""
    CRITICAL = "critical"      # 99.99% uptime
    HIGH = "high"             # 99.9% uptime
    MEDIUM = "medium"         # 99.5% uptime
    BASIC = "basic"           # 99% uptime


class PredictionConfidence(Enum):
    """Confidence levels for failure predictions."""
    VERY_HIGH = "very_high"   # >95% confidence
    HIGH = "high"             # 85-95% confidence
    MEDIUM = "medium"         # 70-85% confidence
    LOW = "low"               # 50-70% confidence
    VERY_LOW = "very_low"     # <50% confidence


@dataclass
class ReliabilityMetric:
    """Reliability metric with predictive capabilities."""
    name: str
    current_value: float
    target_value: float
    unit: str
    failure_threshold: float
    warning_threshold: float
    trend_window: int = 100
    historical_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    prediction_model: Optional[str] = None
    prediction_accuracy: float = 0.0
    last_prediction: Optional[Tuple[datetime, float, PredictionConfidence]] = None


@dataclass
class FailurePrediction:
    """Failure prediction with preventive actions."""
    failure_type: FailureType
    predicted_time: datetime
    confidence: PredictionConfidence
    contributing_factors: List[str]
    impact_assessment: Dict[str, Any]
    preventive_actions: List[str]
    estimated_prevention_time: int  # minutes
    cost_of_failure: float
    cost_of_prevention: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealingAction:
    """Self-healing action definition."""
    id: str
    name: str
    description: str
    trigger_conditions: List[str]
    implementation: Callable
    success_criteria: Dict[str, float]
    rollback_function: Optional[Callable] = None
    max_retry_attempts: int = 3
    cooldown_period: int = 300  # seconds
    last_executed: Optional[datetime] = None
    success_rate: float = 0.0
    execution_count: int = 0


@dataclass
class ReliabilityIncident:
    """Reliability incident record."""
    id: str
    incident_type: FailureType
    severity: str
    description: str
    detected_at: datetime
    predicted: bool = False
    prevention_attempted: bool = False
    resolved_at: Optional[datetime] = None
    resolution_actions: List[str] = field(default_factory=list)
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)


class FailurePredictionEngine:
    """Advanced failure prediction engine using multiple ML techniques."""
    
    def __init__(self):
        """Initialize failure prediction engine."""
        self.prediction_models = {}
        self.feature_extractors = {}
        self.prediction_history = defaultdict(list)
        
        self._setup_prediction_models()
        self._setup_feature_extractors()
    
    def _setup_prediction_models(self):
        """Setup prediction models for different failure types."""
        self.prediction_models = {
            FailureType.PERFORMANCE_DEGRADATION: self._predict_performance_degradation,
            FailureType.MEMORY_LEAK: self._predict_memory_leak,
            FailureType.CONNECTION_FAILURE: self._predict_connection_failure,
            FailureType.DISK_SPACE_EXHAUSTION: self._predict_disk_exhaustion,
            FailureType.CPU_OVERLOAD: self._predict_cpu_overload,
            FailureType.NETWORK_CONGESTION: self._predict_network_congestion,
            FailureType.SERVICE_UNAVAILABILITY: self._predict_service_unavailability,
            FailureType.CONFIGURATION_DRIFT: self._predict_configuration_drift
        }
    
    def _setup_feature_extractors(self):
        """Setup feature extractors for prediction models."""
        self.feature_extractors = {
            "trend_analysis": self._extract_trend_features,
            "statistical_analysis": self._extract_statistical_features,
            "pattern_analysis": self._extract_pattern_features,
            "anomaly_detection": self._extract_anomaly_features,
            "seasonal_analysis": self._extract_seasonal_features
        }
    
    async def predict_failures(self, metrics: Dict[str, ReliabilityMetric], 
                             prediction_horizon: int = 3600) -> List[FailurePrediction]:
        """Predict potential failures within the given time horizon."""
        predictions = []
        
        for failure_type, prediction_model in self.prediction_models.items():
            try:
                # Extract features for this failure type
                features = await self._extract_features_for_failure_type(failure_type, metrics)
                
                # Generate prediction
                prediction_result = await prediction_model(features, prediction_horizon)
                
                if prediction_result and prediction_result["probability"] > 0.5:
                    # Create failure prediction
                    predicted_time = datetime.now() + timedelta(seconds=prediction_result["time_to_failure"])
                    confidence = self._calculate_confidence(prediction_result["probability"])
                    
                    prediction = FailurePrediction(
                        failure_type=failure_type,
                        predicted_time=predicted_time,
                        confidence=confidence,
                        contributing_factors=prediction_result.get("factors", []),
                        impact_assessment=prediction_result.get("impact", {}),
                        preventive_actions=self._get_preventive_actions(failure_type),
                        estimated_prevention_time=prediction_result.get("prevention_time", 30),
                        cost_of_failure=prediction_result.get("failure_cost", 1000.0),
                        cost_of_prevention=prediction_result.get("prevention_cost", 100.0)
                    )
                    
                    predictions.append(prediction)
                    logger.info(f"Failure predicted: {failure_type.value} at {predicted_time} "
                              f"with {confidence.value} confidence")
            
            except Exception as e:
                logger.error(f"Error predicting {failure_type.value}: {e}")
        
        # Sort predictions by urgency (time to failure and impact)
        predictions.sort(key=lambda p: (p.predicted_time, -p.cost_of_failure))
        
        return predictions
    
    async def _extract_features_for_failure_type(self, failure_type: FailureType, 
                                               metrics: Dict[str, ReliabilityMetric]) -> Dict[str, Any]:
        """Extract relevant features for specific failure type prediction."""
        features = {}
        
        # Map failure types to relevant metrics
        metric_mapping = {
            FailureType.PERFORMANCE_DEGRADATION: ["response_time", "throughput", "cpu_usage"],
            FailureType.MEMORY_LEAK: ["memory_usage", "gc_frequency", "heap_size"],
            FailureType.CONNECTION_FAILURE: ["connection_count", "connection_errors", "network_latency"],
            FailureType.DISK_SPACE_EXHAUSTION: ["disk_usage", "disk_io", "log_size"],
            FailureType.CPU_OVERLOAD: ["cpu_usage", "cpu_temperature", "process_count"],
            FailureType.NETWORK_CONGESTION: ["network_latency", "packet_loss", "bandwidth_usage"],
            FailureType.SERVICE_UNAVAILABILITY: ["error_rate", "timeout_rate", "health_checks"],
            FailureType.CONFIGURATION_DRIFT: ["config_changes", "restart_frequency", "error_patterns"]
        }
        
        relevant_metrics = metric_mapping.get(failure_type, [])
        
        for metric_name in relevant_metrics:
            if metric_name in metrics:
                metric = metrics[metric_name]
                
                # Extract features using all extractors
                for extractor_name, extractor_func in self.feature_extractors.items():
                    metric_features = extractor_func(metric)
                    features[f"{metric_name}_{extractor_name}"] = metric_features
        
        return features
    
    def _extract_trend_features(self, metric: ReliabilityMetric) -> Dict[str, float]:
        """Extract trend-based features from metric."""
        if len(metric.historical_values) < 10:
            return {"trend_slope": 0.0, "trend_acceleration": 0.0}
        
        values = [v[1] if isinstance(v, tuple) else v for v in list(metric.historical_values)[-50:]]
        x = np.arange(len(values))
        
        # Linear trend
        trend_coeffs = np.polyfit(x, values, 1)
        trend_slope = trend_coeffs[0]
        
        # Acceleration (second derivative)
        if len(values) >= 3:
            accel_coeffs = np.polyfit(x, values, 2)
            trend_acceleration = 2 * accel_coeffs[0]
        else:
            trend_acceleration = 0.0
        
        return {
            "trend_slope": trend_slope,
            "trend_acceleration": trend_acceleration,
            "trend_r_squared": self._calculate_r_squared(x, values, trend_coeffs)
        }
    
    def _extract_statistical_features(self, metric: ReliabilityMetric) -> Dict[str, float]:
        """Extract statistical features from metric."""
        if len(metric.historical_values) < 5:
            return {"mean": 0.0, "std": 0.0, "variance": 0.0}
        
        values = [v[1] if isinstance(v, tuple) else v for v in list(metric.historical_values)[-100:]]
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "variance": np.var(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "skewness": self._calculate_skewness(values),
            "kurtosis": self._calculate_kurtosis(values)
        }
    
    def _extract_pattern_features(self, metric: ReliabilityMetric) -> Dict[str, float]:
        """Extract pattern-based features from metric."""
        if len(metric.historical_values) < 20:
            return {"pattern_strength": 0.0}
        
        values = [v[1] if isinstance(v, tuple) else v for v in list(metric.historical_values)[-100:]]
        
        # Calculate autocorrelation at different lags
        autocorrelations = []
        for lag in [1, 5, 10, 20]:
            if len(values) > lag:
                autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                autocorrelations.append(autocorr if not np.isnan(autocorr) else 0.0)
        
        return {
            "autocorr_lag1": autocorrelations[0] if len(autocorrelations) > 0 else 0.0,
            "autocorr_lag5": autocorrelations[1] if len(autocorrelations) > 1 else 0.0,
            "autocorr_lag10": autocorrelations[2] if len(autocorrelations) > 2 else 0.0,
            "autocorr_lag20": autocorrelations[3] if len(autocorrelations) > 3 else 0.0,
            "pattern_strength": np.mean(np.abs(autocorrelations)) if autocorrelations else 0.0
        }
    
    def _extract_anomaly_features(self, metric: ReliabilityMetric) -> Dict[str, float]:
        """Extract anomaly detection features from metric."""
        if len(metric.historical_values) < 10:
            return {"anomaly_score": 0.0}
        
        values = [v[1] if isinstance(v, tuple) else v for v in list(metric.historical_values)[-50:]]
        
        # Simple anomaly detection using z-score
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return {"anomaly_score": 0.0}
        
        z_scores = [(v - mean_val) / std_val for v in values]
        anomaly_count = sum(1 for z in z_scores if abs(z) > 2)  # 2 standard deviations
        
        return {
            "anomaly_score": anomaly_count / len(values),
            "max_z_score": np.max(np.abs(z_scores)),
            "recent_anomaly_trend": np.mean(np.abs(z_scores[-10:])) if len(z_scores) >= 10 else 0.0
        }
    
    def _extract_seasonal_features(self, metric: ReliabilityMetric) -> Dict[str, float]:
        """Extract seasonal pattern features from metric."""
        if len(metric.historical_values) < 50:
            return {"seasonality_strength": 0.0}
        
        values = [v[1] if isinstance(v, tuple) else v for v in list(metric.historical_values)[-200:]]
        
        # Simple seasonality detection using FFT
        try:
            fft_values = np.fft.fft(values)
            power_spectrum = np.abs(fft_values) ** 2
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            
            seasonality_strength = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
            dominant_period = len(values) / dominant_freq_idx if dominant_freq_idx > 0 else 0
            
            return {
                "seasonality_strength": float(seasonality_strength),
                "dominant_period": float(dominant_period)
            }
        except Exception:
            return {"seasonality_strength": 0.0, "dominant_period": 0.0}
    
    def _calculate_r_squared(self, x: np.ndarray, y: List[float], coeffs: np.ndarray) -> float:
        """Calculate R-squared for trend line."""
        try:
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of distribution."""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean([(v - mean_val) ** 3 for v in values]) / (std_val ** 3)
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of distribution."""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean([(v - mean_val) ** 4 for v in values]) / (std_val ** 4) - 3
        except Exception:
            return 0.0
    
    def _calculate_confidence(self, probability: float) -> PredictionConfidence:
        """Calculate prediction confidence based on probability."""
        if probability >= 0.95:
            return PredictionConfidence.VERY_HIGH
        elif probability >= 0.85:
            return PredictionConfidence.HIGH
        elif probability >= 0.70:
            return PredictionConfidence.MEDIUM
        elif probability >= 0.50:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    # Specific prediction models
    async def _predict_performance_degradation(self, features: Dict[str, Any], 
                                             horizon: int) -> Optional[Dict[str, Any]]:
        """Predict performance degradation."""
        # Analyze response time and throughput trends
        response_time_features = features.get("response_time_trend_analysis", {})
        throughput_features = features.get("throughput_trend_analysis", {})
        
        # Check for degrading trends
        response_time_slope = response_time_features.get("trend_slope", 0)
        throughput_slope = throughput_features.get("trend_slope", 0)
        
        # Performance is degrading if response time is increasing or throughput is decreasing
        degradation_score = max(0, response_time_slope) + max(0, -throughput_slope)
        
        if degradation_score > 0.1:  # Threshold for concern
            probability = min(0.95, degradation_score * 5)  # Scale to probability
            time_to_failure = horizon * (1 - probability)  # Sooner failure for higher probability
            
            return {
                "probability": probability,
                "time_to_failure": time_to_failure,
                "factors": ["increasing_response_time", "decreasing_throughput"],
                "impact": {"service_degradation": "high", "user_experience": "poor"},
                "prevention_time": 15,
                "failure_cost": 5000.0,
                "prevention_cost": 200.0
            }
        
        return None
    
    async def _predict_memory_leak(self, features: Dict[str, Any], 
                                 horizon: int) -> Optional[Dict[str, Any]]:
        """Predict memory leak."""
        memory_features = features.get("memory_usage_trend_analysis", {})
        memory_slope = memory_features.get("trend_slope", 0)
        
        # Memory leak indicated by consistently increasing memory usage
        if memory_slope > 0.01:  # Positive trend in memory usage
            probability = min(0.90, memory_slope * 50)
            time_to_failure = horizon * (1 - probability)
            
            return {
                "probability": probability,
                "time_to_failure": time_to_failure,
                "factors": ["increasing_memory_usage", "poor_garbage_collection"],
                "impact": {"system_crash": "high", "performance": "degraded"},
                "prevention_time": 10,
                "failure_cost": 8000.0,
                "prevention_cost": 50.0
            }
        
        return None
    
    async def _predict_connection_failure(self, features: Dict[str, Any], 
                                        horizon: int) -> Optional[Dict[str, Any]]:
        """Predict connection failure."""
        connection_features = features.get("connection_errors_statistical_analysis", {})
        error_rate = connection_features.get("mean", 0)
        
        if error_rate > 0.05:  # 5% error rate threshold
            probability = min(0.85, error_rate * 10)
            time_to_failure = horizon * (1 - probability)
            
            return {
                "probability": probability,
                "time_to_failure": time_to_failure,
                "factors": ["high_connection_errors", "network_instability"],
                "impact": {"service_unavailability": "medium", "data_loss": "low"},
                "prevention_time": 20,
                "failure_cost": 3000.0,
                "prevention_cost": 150.0
            }
        
        return None
    
    async def _predict_disk_exhaustion(self, features: Dict[str, Any], 
                                     horizon: int) -> Optional[Dict[str, Any]]:
        """Predict disk space exhaustion."""
        disk_features = features.get("disk_usage_trend_analysis", {})
        disk_slope = disk_features.get("trend_slope", 0)
        
        if disk_slope > 0.001:  # Increasing disk usage
            # Estimate time to 100% based on current trend
            current_usage = features.get("disk_usage_statistical_analysis", {}).get("mean", 50)
            remaining_space = 100 - current_usage
            
            if disk_slope > 0:
                time_to_full = remaining_space / disk_slope
                if time_to_full < horizon:
                    probability = min(0.95, 1 - (time_to_full / horizon))
                    
                    return {
                        "probability": probability,
                        "time_to_failure": min(time_to_full, horizon),
                        "factors": ["increasing_disk_usage", "insufficient_cleanup"],
                        "impact": {"system_failure": "critical", "data_loss": "high"},
                        "prevention_time": 5,
                        "failure_cost": 10000.0,
                        "prevention_cost": 25.0
                    }
        
        return None
    
    async def _predict_cpu_overload(self, features: Dict[str, Any], 
                                  horizon: int) -> Optional[Dict[str, Any]]:
        """Predict CPU overload."""
        cpu_features = features.get("cpu_usage_statistical_analysis", {})
        cpu_mean = cpu_features.get("mean", 0)
        cpu_trend = features.get("cpu_usage_trend_analysis", {}).get("trend_slope", 0)
        
        # High CPU usage or increasing trend
        if cpu_mean > 80 or (cpu_mean > 60 and cpu_trend > 0.1):
            probability = min(0.90, (cpu_mean - 50) / 50 + cpu_trend * 5)
            time_to_failure = horizon * (1 - probability)
            
            return {
                "probability": probability,
                "time_to_failure": time_to_failure,
                "factors": ["high_cpu_usage", "increasing_load"],
                "impact": {"performance_degradation": "high", "response_time": "increased"},
                "prevention_time": 10,
                "failure_cost": 4000.0,
                "prevention_cost": 100.0
            }
        
        return None
    
    async def _predict_network_congestion(self, features: Dict[str, Any], 
                                        horizon: int) -> Optional[Dict[str, Any]]:
        """Predict network congestion."""
        latency_features = features.get("network_latency_trend_analysis", {})
        latency_slope = latency_features.get("trend_slope", 0)
        
        if latency_slope > 0.1:  # Increasing latency trend
            probability = min(0.80, latency_slope * 5)
            time_to_failure = horizon * (1 - probability)
            
            return {
                "probability": probability,
                "time_to_failure": time_to_failure,
                "factors": ["increasing_latency", "network_congestion"],
                "impact": {"communication_failure": "medium", "coordination_delays": "high"},
                "prevention_time": 25,
                "failure_cost": 2000.0,
                "prevention_cost": 300.0
            }
        
        return None
    
    async def _predict_service_unavailability(self, features: Dict[str, Any], 
                                            horizon: int) -> Optional[Dict[str, Any]]:
        """Predict service unavailability."""
        error_features = features.get("error_rate_statistical_analysis", {})
        error_rate = error_features.get("mean", 0)
        
        if error_rate > 0.1:  # 10% error rate
            probability = min(0.85, error_rate * 5)
            time_to_failure = horizon * (1 - probability)
            
            return {
                "probability": probability,
                "time_to_failure": time_to_failure,
                "factors": ["high_error_rate", "service_instability"],
                "impact": {"service_downtime": "critical", "business_impact": "high"},
                "prevention_time": 30,
                "failure_cost": 15000.0,
                "prevention_cost": 500.0
            }
        
        return None
    
    async def _predict_configuration_drift(self, features: Dict[str, Any], 
                                         horizon: int) -> Optional[Dict[str, Any]]:
        """Predict configuration drift."""
        # Simulate configuration drift detection
        config_changes = features.get("config_changes_statistical_analysis", {}).get("mean", 0)
        
        if config_changes > 5:  # Many recent config changes
            probability = min(0.70, config_changes / 20)
            time_to_failure = horizon * 0.8  # Configuration issues develop slowly
            
            return {
                "probability": probability,
                "time_to_failure": time_to_failure,
                "factors": ["frequent_config_changes", "lack_of_validation"],
                "impact": {"configuration_errors": "medium", "service_instability": "low"},
                "prevention_time": 45,
                "failure_cost": 1000.0,
                "prevention_cost": 75.0
            }
        
        return None
    
    def _get_preventive_actions(self, failure_type: FailureType) -> List[str]:
        """Get preventive actions for specific failure type."""
        action_mapping = {
            FailureType.PERFORMANCE_DEGRADATION: [
                "scale_up_resources", "optimize_queries", "enable_caching", "restart_services"
            ],
            FailureType.MEMORY_LEAK: [
                "restart_services", "trigger_garbage_collection", "optimize_memory_usage"
            ],
            FailureType.CONNECTION_FAILURE: [
                "increase_connection_pool", "restart_network_services", "check_network_health"
            ],
            FailureType.DISK_SPACE_EXHAUSTION: [
                "cleanup_old_logs", "compress_data", "expand_storage", "archive_data"
            ],
            FailureType.CPU_OVERLOAD: [
                "scale_horizontally", "optimize_algorithms", "reduce_workload", "upgrade_hardware"
            ],
            FailureType.NETWORK_CONGESTION: [
                "implement_qos", "optimize_protocols", "add_bandwidth", "cache_content"
            ],
            FailureType.SERVICE_UNAVAILABILITY: [
                "restart_services", "failover_to_backup", "check_dependencies", "scale_resources"
            ],
            FailureType.CONFIGURATION_DRIFT: [
                "validate_configuration", "restore_from_backup", "apply_standard_config"
            ]
        }
        
        return action_mapping.get(failure_type, ["investigate_issue", "alert_administrators"])


class ProactiveReliabilityEngine:
    """Proactive reliability engineering system with predictive prevention and self-healing."""
    
    def __init__(self, 
                 target_reliability: ReliabilityLevel = ReliabilityLevel.HIGH,
                 prediction_horizon: int = 3600,  # 1 hour
                 enable_auto_healing: bool = True):
        """Initialize proactive reliability engine.
        
        Args:
            target_reliability: Target reliability level
            prediction_horizon: Prediction horizon in seconds
            enable_auto_healing: Enable automatic self-healing
        """
        self.target_reliability = target_reliability
        self.prediction_horizon = prediction_horizon
        self.enable_auto_healing = enable_auto_healing
        
        self.reliability_metrics: Dict[str, ReliabilityMetric] = {}
        self.prediction_engine = FailurePredictionEngine()
        self.healing_actions: Dict[str, HealingAction] = {}
        self.incidents: List[ReliabilityIncident] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self._setup_reliability_metrics()
        self._setup_healing_actions()
        self._setup_circuit_breakers()
        
        logger.info(f"Proactive Reliability Engine initialized with {target_reliability.value} target")
    
    def _setup_reliability_metrics(self):
        """Setup reliability metrics to monitor."""
        metrics = [
            ReliabilityMetric(
                name="uptime_percentage",
                current_value=99.9,
                target_value=99.9 if self.target_reliability == ReliabilityLevel.HIGH else 99.99,
                unit="%",
                failure_threshold=99.0,
                warning_threshold=99.5
            ),
            ReliabilityMetric(
                name="response_time",
                current_value=50.0,
                target_value=100.0,
                unit="ms",
                failure_threshold=500.0,
                warning_threshold=200.0
            ),
            ReliabilityMetric(
                name="error_rate",
                current_value=0.1,
                target_value=1.0,
                unit="%",
                failure_threshold=5.0,
                warning_threshold=2.0
            ),
            ReliabilityMetric(
                name="throughput",
                current_value=1000.0,
                target_value=1000.0,
                unit="req/s",
                failure_threshold=100.0,
                warning_threshold=500.0
            ),
            ReliabilityMetric(
                name="cpu_usage",
                current_value=45.0,
                target_value=70.0,
                unit="%",
                failure_threshold=95.0,
                warning_threshold=80.0
            ),
            ReliabilityMetric(
                name="memory_usage",
                current_value=60.0,
                target_value=80.0,
                unit="%",
                failure_threshold=95.0,
                warning_threshold=85.0
            ),
            ReliabilityMetric(
                name="disk_usage",
                current_value=40.0,
                target_value=80.0,
                unit="%",
                failure_threshold=95.0,
                warning_threshold=90.0
            )
        ]
        
        for metric in metrics:
            self.reliability_metrics[metric.name] = metric
    
    def _setup_healing_actions(self):
        """Setup self-healing actions."""
        actions = [
            HealingAction(
                id="restart_service",
                name="Restart Service",
                description="Restart the affected service to recover from failure",
                trigger_conditions=["service_unavailable", "high_error_rate"],
                implementation=self._restart_service,
                success_criteria={"error_rate": 1.0, "response_time": 200.0},
                rollback_function=self._rollback_service_restart,
                max_retry_attempts=2,
                cooldown_period=600
            ),
            HealingAction(
                id="scale_resources",
                name="Scale Resources",
                description="Scale up system resources to handle increased load",
                trigger_conditions=["high_cpu_usage", "high_memory_usage", "performance_degradation"],
                implementation=self._scale_resources,
                success_criteria={"cpu_usage": 70.0, "memory_usage": 80.0, "response_time": 100.0},
                rollback_function=self._rollback_scaling,
                max_retry_attempts=1,
                cooldown_period=300
            ),
            HealingAction(
                id="clear_cache",
                name="Clear Cache",
                description="Clear system caches to free memory and resolve cache-related issues",
                trigger_conditions=["memory_leak", "high_memory_usage"],
                implementation=self._clear_cache,
                success_criteria={"memory_usage": 70.0},
                max_retry_attempts=3,
                cooldown_period=180
            ),
            HealingAction(
                id="optimize_database",
                name="Optimize Database",
                description="Optimize database queries and connections",
                trigger_conditions=["slow_database_queries", "connection_timeout"],
                implementation=self._optimize_database,
                success_criteria={"response_time": 150.0, "error_rate": 1.0},
                max_retry_attempts=1,
                cooldown_period=900
            ),
            HealingAction(
                id="cleanup_disk_space",
                name="Cleanup Disk Space",
                description="Clean up temporary files and logs to free disk space",
                trigger_conditions=["disk_space_low", "disk_space_exhaustion"],
                implementation=self._cleanup_disk_space,
                success_criteria={"disk_usage": 80.0},
                max_retry_attempts=2,
                cooldown_period=300
            ),
            HealingAction(
                id="network_optimization",
                name="Network Optimization",
                description="Optimize network settings and connections",
                trigger_conditions=["network_congestion", "high_latency"],
                implementation=self._optimize_network,
                success_criteria={"network_latency": 50.0, "throughput": 800.0},
                max_retry_attempts=1,
                cooldown_period=600
            )
        ]
        
        for action in actions:
            self.healing_actions[action.id] = action
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for critical services."""
        services = ["coordination_service", "communication_service", "planning_service", "monitoring_service"]
        
        for service in services:
            self.circuit_breakers[service] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception
            )
    
    async def start_monitoring(self):
        """Start proactive reliability monitoring."""
        if self.monitoring_active:
            logger.warning("Reliability monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Proactive reliability monitoring started")
    
    async def stop_monitoring(self):
        """Stop reliability monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Reliability monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main reliability monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect reliability metrics
                await self._collect_reliability_metrics()
                
                # Generate failure predictions
                predictions = await self.prediction_engine.predict_failures(
                    self.reliability_metrics, self.prediction_horizon
                )
                
                # Process predictions and take preventive actions
                await self._process_predictions(predictions)
                
                # Check for immediate healing opportunities
                await self._check_healing_triggers()
                
                # Update reliability status
                await self._update_reliability_status()
                
                # Generate reliability report
                await self._generate_reliability_report()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in reliability monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_reliability_metrics(self):
        """Collect current reliability metrics."""
        current_time = datetime.now()
        
        # Simulate metric collection with realistic variations
        for metric_name, metric in self.reliability_metrics.items():
            if metric_name == "uptime_percentage":
                # Uptime decreases slightly over time, with occasional incidents
                base_uptime = 99.95
                variation = random.gauss(0, 0.01)
                current_value = max(99.0, min(100.0, base_uptime + variation))
                
            elif metric_name == "response_time":
                # Response time varies with load and system health
                base_response = 75.0
                load_factor = 1 + (self.reliability_metrics["cpu_usage"].current_value - 50) / 100
                variation = random.gauss(0, 10)
                current_value = max(10.0, base_response * load_factor + variation)
                
            elif metric_name == "error_rate":
                # Error rate correlates with system stress
                base_error = 0.2
                stress_factor = (self.reliability_metrics["cpu_usage"].current_value - 40) / 60
                current_value = max(0.0, base_error + stress_factor * 2 + random.gauss(0, 0.1))
                
            elif metric_name == "cpu_usage":
                # CPU usage varies with workload
                base_cpu = 50.0
                trend = math.sin(time.time() / 3600) * 10  # Hourly cycle
                variation = random.gauss(0, 5)
                current_value = max(0.0, min(100.0, base_cpu + trend + variation))
                
            elif metric_name == "memory_usage":
                # Memory usage tends to increase over time (potential leak simulation)
                base_memory = 55.0
                time_factor = (time.time() % 86400) / 86400 * 5  # Slight increase over day
                variation = random.gauss(0, 3)
                current_value = max(0.0, min(100.0, base_memory + time_factor + variation))
                
            elif metric_name == "disk_usage":
                # Disk usage increases slowly over time
                base_disk = 35.0
                time_factor = (time.time() % 604800) / 604800 * 10  # Increase over week
                variation = random.gauss(0, 2)
                current_value = max(0.0, min(100.0, base_disk + time_factor + variation))
                
            else:  # throughput
                # Throughput inversely related to response time and CPU
                base_throughput = 1200.0
                performance_factor = 1 - (current_value / 200 if metric_name == "response_time" else 0)
                cpu_factor = 1 - (self.reliability_metrics["cpu_usage"].current_value - 50) / 100
                variation = random.gauss(0, 50)
                current_value = max(0.0, base_throughput * performance_factor * cpu_factor + variation)
            
            # Update metric
            metric.current_value = current_value
            metric.historical_values.append((current_time, current_value))
            
            # Update prediction accuracy (simplified)
            if metric.last_prediction:
                pred_time, pred_value, confidence = metric.last_prediction
                if current_time >= pred_time:
                    error = abs(pred_value - current_value) / metric.target_value
                    metric.prediction_accuracy = max(0.0, 1.0 - error)
    
    async def _process_predictions(self, predictions: List[FailurePrediction]):
        """Process failure predictions and take preventive actions."""
        for prediction in predictions:
            logger.info(f"Processing prediction: {prediction.failure_type.value} "
                       f"in {(prediction.predicted_time - datetime.now()).total_seconds():.0f}s "
                       f"with {prediction.confidence.value} confidence")
            
            # Take preventive actions for high-confidence predictions
            if (prediction.confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH] and
                prediction.cost_of_prevention < prediction.cost_of_failure * 0.5):
                
                await self._execute_preventive_actions(prediction)
    
    async def _execute_preventive_actions(self, prediction: FailurePrediction):
        """Execute preventive actions for a failure prediction."""
        logger.info(f"Executing preventive actions for {prediction.failure_type.value}")
        
        for action_name in prediction.preventive_actions:
            # Map action names to healing actions
            action_mapping = {
                "scale_up_resources": "scale_resources",
                "restart_services": "restart_service",
                "cleanup_old_logs": "cleanup_disk_space",
                "optimize_queries": "optimize_database",
                "enable_caching": "clear_cache",
                "implement_qos": "network_optimization"
            }
            
            healing_action_id = action_mapping.get(action_name)
            if healing_action_id and healing_action_id in self.healing_actions:
                healing_action = self.healing_actions[healing_action_id]
                
                # Check cooldown period
                if (healing_action.last_executed and 
                    (datetime.now() - healing_action.last_executed).total_seconds() < healing_action.cooldown_period):
                    logger.info(f"Skipping {healing_action.name} due to cooldown period")
                    continue
                
                # Execute healing action
                await self._execute_healing_action(healing_action, f"preventive_{prediction.failure_type.value}")
    
    async def _check_healing_triggers(self):
        """Check for immediate healing triggers based on current metrics."""
        current_time = datetime.now()
        
        for action_id, action in self.healing_actions.items():
            # Check if action should be triggered
            should_trigger = False
            triggered_by = []
            
            for condition in action.trigger_conditions:
                if await self._evaluate_trigger_condition(condition):
                    should_trigger = True
                    triggered_by.append(condition)
            
            if should_trigger:
                # Check cooldown period
                if (action.last_executed and 
                    (current_time - action.last_executed).total_seconds() < action.cooldown_period):
                    continue
                
                logger.info(f"Healing action triggered: {action.name} by {triggered_by}")
                await self._execute_healing_action(action, f"reactive_{','.join(triggered_by)}")
    
    async def _evaluate_trigger_condition(self, condition: str) -> bool:
        """Evaluate whether a healing trigger condition is met."""
        condition_mapping = {
            "service_unavailable": lambda: self.reliability_metrics["error_rate"].current_value > 5.0,
            "high_error_rate": lambda: self.reliability_metrics["error_rate"].current_value > 2.0,
            "high_cpu_usage": lambda: self.reliability_metrics["cpu_usage"].current_value > 80.0,
            "high_memory_usage": lambda: self.reliability_metrics["memory_usage"].current_value > 85.0,
            "performance_degradation": lambda: self.reliability_metrics["response_time"].current_value > 200.0,
            "memory_leak": lambda: self.reliability_metrics["memory_usage"].current_value > 90.0,
            "disk_space_low": lambda: self.reliability_metrics["disk_usage"].current_value > 90.0,
            "disk_space_exhaustion": lambda: self.reliability_metrics["disk_usage"].current_value > 95.0,
            "network_congestion": lambda: self.reliability_metrics["response_time"].current_value > 300.0,
            "high_latency": lambda: self.reliability_metrics["response_time"].current_value > 250.0
        }
        
        evaluator = condition_mapping.get(condition)
        if evaluator:
            try:
                return evaluator()
            except Exception as e:
                logger.error(f"Error evaluating condition {condition}: {e}")
                return False
        
        return False
    
    async def _execute_healing_action(self, action: HealingAction, trigger_reason: str):
        """Execute a healing action with retry logic."""
        start_time = datetime.now()
        
        for attempt in range(action.max_retry_attempts):
            try:
                logger.info(f"Executing healing action: {action.name} (attempt {attempt + 1}/{action.max_retry_attempts})")
                
                # Execute the healing action
                result = await action.implementation()
                
                action.last_executed = datetime.now()
                action.execution_count += 1
                
                if result.get("success", False):
                    # Check if success criteria are met
                    success_met = await self._check_success_criteria(action.success_criteria)
                    
                    if success_met:
                        action.success_rate = (action.success_rate * (action.execution_count - 1) + 1.0) / action.execution_count
                        logger.info(f"Healing action successful: {action.name}")
                        
                        # Create incident record
                        incident = ReliabilityIncident(
                            id=f"healing_{action.id}_{int(start_time.timestamp())}",
                            incident_type=FailureType.PERFORMANCE_DEGRADATION,  # Generic type for healing
                            severity="medium",
                            description=f"Healing action executed: {action.name}",
                            detected_at=start_time,
                            resolved_at=datetime.now(),
                            resolution_actions=[action.name],
                            impact_metrics={"trigger_reason": trigger_reason}
                        )
                        self.incidents.append(incident)
                        
                        return True
                    else:
                        logger.warning(f"Healing action executed but success criteria not met: {action.name}")
                else:
                    logger.warning(f"Healing action execution failed: {action.name}")
                
            except Exception as e:
                logger.error(f"Error executing healing action {action.name}: {e}")
            
            if attempt < action.max_retry_attempts - 1:
                await asyncio.sleep(30)  # Wait before retry
        
        # All attempts failed
        action.success_rate = (action.success_rate * (action.execution_count - 1)) / action.execution_count
        logger.error(f"Healing action failed after {action.max_retry_attempts} attempts: {action.name}")
        
        # Attempt rollback if available
        if action.rollback_function:
            try:
                await action.rollback_function()
                logger.info(f"Rollback executed for failed healing action: {action.name}")
            except Exception as e:
                logger.error(f"Rollback failed for {action.name}: {e}")
        
        return False
    
    async def _check_success_criteria(self, criteria: Dict[str, float]) -> bool:
        """Check if healing action success criteria are met."""
        for metric_name, target_value in criteria.items():
            if metric_name in self.reliability_metrics:
                current_value = self.reliability_metrics[metric_name].current_value
                
                # For most metrics, lower is better (error_rate, response_time, usage)
                if metric_name in ["error_rate", "response_time", "cpu_usage", "memory_usage", "disk_usage"]:
                    if current_value > target_value:
                        return False
                else:  # For throughput, uptime, higher is better
                    if current_value < target_value:
                        return False
        
        return True
    
    async def _update_reliability_status(self):
        """Update overall reliability status."""
        # Calculate reliability score based on metrics
        score_components = []
        
        for metric_name, metric in self.reliability_metrics.items():
            if metric_name == "uptime_percentage":
                score_components.append(metric.current_value)
            elif metric_name in ["error_rate"]:
                # Lower is better, convert to score
                score_components.append(max(0, 100 - metric.current_value * 20))
            elif metric_name in ["response_time"]:
                # Lower is better, convert to score
                score_components.append(max(0, 100 - metric.current_value / 5))
            elif metric_name in ["throughput"]:
                # Higher is better
                score_components.append(min(100, metric.current_value / 10))
        
        overall_score = np.mean(score_components) if score_components else 0.0
        
        # Determine reliability level
        if overall_score >= 99.99:
            current_level = ReliabilityLevel.CRITICAL
        elif overall_score >= 99.9:
            current_level = ReliabilityLevel.HIGH
        elif overall_score >= 99.5:
            current_level = ReliabilityLevel.MEDIUM
        else:
            current_level = ReliabilityLevel.BASIC
        
        logger.debug(f"Reliability Status: {current_level.value} (score: {overall_score:.2f}%)")
    
    async def _generate_reliability_report(self):
        """Generate comprehensive reliability report."""
        current_time = datetime.now()
        
        report = {
            "timestamp": current_time.isoformat(),
            "target_reliability": self.target_reliability.value,
            "monitoring_active": self.monitoring_active,
            "metrics": {},
            "recent_incidents": [],
            "healing_actions_summary": {},
            "predictions_summary": {}
        }
        
        # Current metrics
        for metric_name, metric in self.reliability_metrics.items():
            status = "healthy"
            if metric.current_value > metric.failure_threshold:
                status = "critical"
            elif metric.current_value > metric.warning_threshold:
                status = "warning"
            
            report["metrics"][metric_name] = {
                "current_value": metric.current_value,
                "target_value": metric.target_value,
                "status": status,
                "prediction_accuracy": metric.prediction_accuracy,
                "unit": metric.unit
            }
        
        # Recent incidents (last 24 hours)
        recent_cutoff = current_time - timedelta(hours=24)
        recent_incidents = [i for i in self.incidents if i.detected_at > recent_cutoff]
        
        for incident in recent_incidents[-10:]:  # Latest 10
            report["recent_incidents"].append({
                "id": incident.id,
                "type": incident.incident_type.value,
                "severity": incident.severity,
                "detected_at": incident.detected_at.isoformat(),
                "resolved": incident.resolved_at is not None,
                "resolution_time": (incident.resolved_at - incident.detected_at).total_seconds() if incident.resolved_at else None
            })
        
        # Healing actions summary
        for action_id, action in self.healing_actions.items():
            report["healing_actions_summary"][action_id] = {
                "name": action.name,
                "execution_count": action.execution_count,
                "success_rate": action.success_rate,
                "last_executed": action.last_executed.isoformat() if action.last_executed else None
            }
        
        logger.debug(f"Reliability Report: {json.dumps(report, indent=2)}")
    
    # Healing action implementations
    async def _restart_service(self) -> Dict[str, Any]:
        """Restart service healing action."""
        logger.info("Executing service restart")
        await asyncio.sleep(2)  # Simulate restart time
        
        # Simulate restart success
        success = random.random() > 0.1  # 90% success rate
        
        return {
            "success": success,
            "message": "Service restarted successfully" if success else "Service restart failed",
            "actions_taken": ["stopped_service", "started_service"]
        }
    
    async def _rollback_service_restart(self):
        """Rollback service restart."""
        logger.info("Rolling back service restart")
        await asyncio.sleep(1)
    
    async def _scale_resources(self) -> Dict[str, Any]:
        """Scale resources healing action."""
        logger.info("Executing resource scaling")
        await asyncio.sleep(3)  # Simulate scaling time
        
        return {
            "success": True,
            "message": "Resources scaled successfully",
            "actions_taken": ["increased_cpu_limit", "increased_memory_limit", "added_instances"]
        }
    
    async def _rollback_scaling(self):
        """Rollback resource scaling."""
        logger.info("Rolling back resource scaling")
        await asyncio.sleep(2)
    
    async def _clear_cache(self) -> Dict[str, Any]:
        """Clear cache healing action."""
        logger.info("Executing cache clear")
        await asyncio.sleep(1)  # Simulate cache clear time
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "actions_taken": ["cleared_application_cache", "cleared_system_cache"]
        }
    
    async def _optimize_database(self) -> Dict[str, Any]:
        """Optimize database healing action."""
        logger.info("Executing database optimization")
        await asyncio.sleep(5)  # Simulate optimization time
        
        return {
            "success": True,
            "message": "Database optimized successfully",
            "actions_taken": ["optimized_queries", "rebuilt_indexes", "updated_statistics"]
        }
    
    async def _cleanup_disk_space(self) -> Dict[str, Any]:
        """Cleanup disk space healing action."""
        logger.info("Executing disk cleanup")
        await asyncio.sleep(2)  # Simulate cleanup time
        
        return {
            "success": True,
            "message": "Disk space cleaned successfully",
            "actions_taken": ["removed_old_logs", "compressed_archives", "cleaned_temp_files"]
        }
    
    async def _optimize_network(self) -> Dict[str, Any]:
        """Optimize network healing action."""
        logger.info("Executing network optimization")
        await asyncio.sleep(3)  # Simulate optimization time
        
        return {
            "success": True,
            "message": "Network optimized successfully",
            "actions_taken": ["adjusted_tcp_settings", "optimized_routing", "enabled_compression"]
        }
    
    def add_custom_healing_action(self, action: HealingAction):
        """Add a custom healing action."""
        self.healing_actions[action.id] = action
        logger.info(f"Custom healing action added: {action.name}")
    
    def get_reliability_summary(self) -> Dict[str, Any]:
        """Get comprehensive reliability summary."""
        current_time = datetime.now()
        
        # Calculate overall reliability score
        metric_scores = []
        for metric_name, metric in self.reliability_metrics.items():
            if metric_name == "uptime_percentage":
                metric_scores.append(metric.current_value)
            elif metric_name in ["error_rate"]:
                metric_scores.append(max(0, 100 - metric.current_value * 20))
            elif metric_name in ["response_time"]:
                metric_scores.append(max(0, 100 - metric.current_value / 5))
            elif metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
                metric_scores.append(max(0, 100 - metric.current_value))
            else:  # throughput
                metric_scores.append(min(100, metric.current_value / 10))
        
        overall_reliability = np.mean(metric_scores) if metric_scores else 0.0
        
        # Count recent incidents
        recent_cutoff = current_time - timedelta(hours=24)
        recent_incidents = len([i for i in self.incidents if i.detected_at > recent_cutoff])
        
        return {
            "timestamp": current_time.isoformat(),
            "overall_reliability_score": overall_reliability,
            "target_reliability": self.target_reliability.value,
            "monitoring_active": self.monitoring_active,
            "auto_healing_enabled": self.enable_auto_healing,
            "total_metrics": len(self.reliability_metrics),
            "total_healing_actions": len(self.healing_actions),
            "recent_incidents_24h": recent_incidents,
            "total_incidents": len(self.incidents),
            "reliability_metrics": {
                name: {
                    "current_value": metric.current_value,
                    "target_value": metric.target_value,
                    "prediction_accuracy": metric.prediction_accuracy,
                    "status": "healthy" if metric.current_value <= metric.warning_threshold else 
                             "warning" if metric.current_value <= metric.failure_threshold else "critical"
                } for name, metric in self.reliability_metrics.items()
            }
        }