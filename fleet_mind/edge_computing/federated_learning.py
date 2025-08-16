"""Federated Learning for Privacy-Preserving Drone Swarm Intelligence.

Distributed machine learning across drone swarms without centralizing sensitive data,
enabling collaborative intelligence while preserving privacy and security.
"""

import asyncio
import math
import time
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

class AggregationStrategy(Enum):
    FEDERATED_AVERAGING = "federated_averaging"
    WEIGHTED_AGGREGATION = "weighted_aggregation"
    BYZANTINE_ROBUST = "byzantine_robust"
    DIFFERENTIAL_PRIVATE = "differential_private"

class PrivacyMechanism(Enum):
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_AGGREGATION = "secure_aggregation"
    LOCAL_DP = "local_differential_privacy"

@dataclass
class ModelUpdate:
    """Model update from individual drone."""
    drone_id: str
    update_vector: List[float]
    update_round: int
    sample_count: int
    loss_value: float
    timestamp: float
    privacy_budget: float = 1.0
    validation_accuracy: float = 0.0

@dataclass
class PrivacyPreservation:
    """Privacy preservation configuration."""
    mechanism: PrivacyMechanism
    epsilon: float = 1.0          # Privacy budget for differential privacy
    delta: float = 1e-5           # Privacy parameter
    noise_multiplier: float = 1.0  # Noise scaling factor
    clipping_threshold: float = 1.0 # Gradient clipping threshold
    
    def add_privacy_noise(self, values: List[float]) -> List[float]:
        """Add privacy-preserving noise to values."""
        
        if self.mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            return self._add_gaussian_noise(values)
        elif self.mechanism == PrivacyMechanism.LOCAL_DP:
            return self._add_laplace_noise(values)
        else:
            return values  # No noise for other mechanisms
    
    def _add_gaussian_noise(self, values: List[float]) -> List[float]:
        """Add Gaussian noise for differential privacy."""
        
        noisy_values = []
        sensitivity = self.clipping_threshold
        sigma = self.noise_multiplier * sensitivity / self.epsilon
        
        for value in values:
            noise = random.gauss(0, sigma)
            noisy_values.append(value + noise)
        
        return noisy_values
    
    def _add_laplace_noise(self, values: List[float]) -> List[float]:
        """Add Laplace noise for local differential privacy."""
        
        noisy_values = []
        sensitivity = self.clipping_threshold
        scale = sensitivity / self.epsilon
        
        for value in values:
            # Laplace noise using inverse transform sampling
            u = random.uniform(-0.5, 0.5)
            noise = -scale * math.copysign(math.log(1 - 2 * abs(u)), u)
            noisy_values.append(value + noise)
        
        return noisy_values

@dataclass
class ModelAggregation:
    """Model aggregation configuration and methods."""
    strategy: AggregationStrategy
    min_participants: int = 3
    max_participants: int = 100
    staleness_threshold: int = 5  # Maximum rounds a model can be stale
    byzantine_tolerance: float = 0.33  # Fraction of Byzantine participants to tolerate
    
    async def aggregate_updates(self, 
                              updates: List[ModelUpdate],
                              global_model: List[float]) -> List[float]:
        """Aggregate model updates using specified strategy."""
        
        if not updates:
            return global_model
        
        if self.strategy == AggregationStrategy.FEDERATED_AVERAGING:
            return await self._federated_averaging(updates)
        elif self.strategy == AggregationStrategy.WEIGHTED_AGGREGATION:
            return await self._weighted_aggregation(updates)
        elif self.strategy == AggregationStrategy.BYZANTINE_ROBUST:
            return await self._byzantine_robust_aggregation(updates, global_model)
        elif self.strategy == AggregationStrategy.DIFFERENTIAL_PRIVATE:
            return await self._differential_private_aggregation(updates)
        else:
            return global_model
    
    async def _federated_averaging(self, updates: List[ModelUpdate]) -> List[float]:
        """Standard federated averaging aggregation."""
        
        if not updates:
            return []
        
        total_samples = sum(update.sample_count for update in updates)
        model_size = len(updates[0].update_vector)
        
        aggregated_model = [0.0] * model_size
        
        for update in updates:
            weight = update.sample_count / total_samples
            
            for i, param in enumerate(update.update_vector):
                aggregated_model[i] += weight * param
        
        return aggregated_model
    
    async def _weighted_aggregation(self, updates: List[ModelUpdate]) -> List[float]:
        """Weighted aggregation based on model quality."""
        
        if not updates:
            return []
        
        # Calculate weights based on validation accuracy and sample count
        weights = []
        for update in updates:
            accuracy_weight = max(0.1, update.validation_accuracy)
            sample_weight = math.log(1 + update.sample_count)
            loss_weight = max(0.1, 1.0 / (1.0 + update.loss_value))
            
            combined_weight = accuracy_weight * sample_weight * loss_weight
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Aggregate with weights
        model_size = len(updates[0].update_vector)
        aggregated_model = [0.0] * model_size
        
        for update, weight in zip(updates, weights):
            for i, param in enumerate(update.update_vector):
                aggregated_model[i] += weight * param
        
        return aggregated_model
    
    async def _byzantine_robust_aggregation(self, 
                                          updates: List[ModelUpdate],
                                          global_model: List[float]) -> List[float]:
        """Byzantine-robust aggregation using coordinate-wise median."""
        
        if not updates:
            return global_model
        
        model_size = len(updates[0].update_vector)
        aggregated_model = []
        
        # Apply coordinate-wise median
        for i in range(model_size):
            coordinates = [update.update_vector[i] for update in updates]
            coordinates.sort()
            
            # Use trimmed mean instead of median for better performance
            trim_count = int(len(coordinates) * self.byzantine_tolerance)
            if trim_count > 0:
                trimmed_coords = coordinates[trim_count:-trim_count]
            else:
                trimmed_coords = coordinates
            
            if trimmed_coords:
                aggregated_value = sum(trimmed_coords) / len(trimmed_coords)
            else:
                aggregated_value = global_model[i] if i < len(global_model) else 0.0
            
            aggregated_model.append(aggregated_value)
        
        return aggregated_model
    
    async def _differential_private_aggregation(self, updates: List[ModelUpdate]) -> List[float]:
        """Differential private aggregation with noise addition."""
        
        # First perform standard aggregation
        aggregated = await self._federated_averaging(updates)
        
        # Add calibrated noise for differential privacy
        privacy_config = PrivacyPreservation(PrivacyMechanism.DIFFERENTIAL_PRIVACY)
        noisy_aggregated = privacy_config.add_privacy_noise(aggregated)
        
        return noisy_aggregated

class FederatedLearning:
    """Federated learning system for drone swarm intelligence."""
    
    def __init__(self,
                 model_size: int = 1000,
                 aggregation_config: ModelAggregation = None,
                 privacy_config: PrivacyPreservation = None):
        self.model_size = model_size
        self.aggregation_config = aggregation_config or ModelAggregation(AggregationStrategy.FEDERATED_AVERAGING)
        self.privacy_config = privacy_config or PrivacyPreservation(PrivacyMechanism.DIFFERENTIAL_PRIVACY)
        
        # Global model state
        self.global_model = [random.gauss(0, 0.1) for _ in range(model_size)]
        self.current_round = 0
        
        # Participant management
        self.registered_drones: Dict[str, Dict] = {}
        self.pending_updates: Dict[int, List[ModelUpdate]] = defaultdict(list)
        self.completed_rounds: List[Dict] = []
        
        # Privacy tracking
        self.privacy_budgets: Dict[str, float] = {}
        self.cumulative_privacy_loss = 0.0
        
        # Performance tracking
        self.convergence_history: List[float] = []
        self.participation_history: List[int] = []
        self.learning_stats = {
            'total_rounds': 0,
            'successful_aggregations': 0,
            'byzantine_detections': 0,
            'privacy_violations': 0,
            'convergence_rate': 0.0
        }
        
        # Start federated learning process
        asyncio.create_task(self._federated_learning_loop())
    
    async def register_drone(self, 
                           drone_id: str,
                           capabilities: Dict[str, Any]) -> bool:
        """Register drone for federated learning participation."""
        
        if drone_id in self.registered_drones:
            return False
        
        self.registered_drones[drone_id] = {
            'capabilities': capabilities,
            'registered_at': time.time(),
            'participation_count': 0,
            'last_update_round': -1,
            'reputation_score': 1.0,
            'privacy_budget_used': 0.0
        }
        
        # Initialize privacy budget
        self.privacy_budgets[drone_id] = self.privacy_config.epsilon * 10  # Total budget
        
        return True
    
    async def submit_model_update(self, 
                                drone_id: str,
                                local_update: List[float],
                                training_samples: int,
                                validation_accuracy: float = 0.0,
                                loss_value: float = 1.0) -> bool:
        """Submit model update from drone participant."""
        
        if drone_id not in self.registered_drones:
            return False
        
        # Check privacy budget
        if self.privacy_budgets[drone_id] < self.privacy_config.epsilon:
            self.learning_stats['privacy_violations'] += 1
            return False
        
        # Apply gradient clipping for privacy
        clipped_update = await self._clip_gradient(local_update)
        
        # Add privacy noise
        noisy_update = self.privacy_config.add_privacy_noise(clipped_update)
        
        # Create model update
        update = ModelUpdate(
            drone_id=drone_id,
            update_vector=noisy_update,
            update_round=self.current_round,
            sample_count=training_samples,
            loss_value=loss_value,
            timestamp=time.time(),
            privacy_budget=self.privacy_config.epsilon,
            validation_accuracy=validation_accuracy
        )
        
        # Store update for current round
        self.pending_updates[self.current_round].append(update)
        
        # Update participant stats
        participant = self.registered_drones[drone_id]
        participant['participation_count'] += 1
        participant['last_update_round'] = self.current_round
        
        # Deduct privacy budget
        self.privacy_budgets[drone_id] -= self.privacy_config.epsilon
        participant['privacy_budget_used'] += self.privacy_config.epsilon
        
        return True
    
    async def get_global_model(self, drone_id: str) -> Optional[List[float]]:
        """Get current global model for specific drone."""
        
        if drone_id not in self.registered_drones:
            return None
        
        return self.global_model.copy()
    
    async def _federated_learning_loop(self):
        """Main federated learning coordination loop."""
        
        while True:
            try:
                # Wait for sufficient participants
                await self._wait_for_participants()
                
                # Aggregate updates for current round
                await self._aggregate_round()
                
                # Evaluate convergence
                await self._evaluate_convergence()
                
                # Start next round
                self.current_round += 1
                
                # Sleep between rounds
                await asyncio.sleep(5.0)
                
            except Exception:
                # Continue learning even if individual rounds fail
                await asyncio.sleep(1.0)
    
    async def _wait_for_participants(self):
        """Wait for minimum number of participants."""
        
        max_wait_time = 30.0  # 30 seconds maximum wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            pending_count = len(self.pending_updates[self.current_round])
            
            if pending_count >= self.aggregation_config.min_participants:
                break
            
            await asyncio.sleep(1.0)
    
    async def _aggregate_round(self):
        """Aggregate model updates for current round."""
        
        current_updates = self.pending_updates[self.current_round]
        
        if not current_updates:
            return
        
        # Filter stale updates
        fresh_updates = [
            update for update in current_updates
            if self.current_round - update.update_round <= self.aggregation_config.staleness_threshold
        ]
        
        if len(fresh_updates) < self.aggregation_config.min_participants:
            return
        
        # Detect and filter Byzantine updates
        filtered_updates = await self._detect_byzantine_updates(fresh_updates)
        
        # Aggregate filtered updates
        new_global_model = await self.aggregation_config.aggregate_updates(
            filtered_updates, self.global_model
        )
        
        # Update global model
        self.global_model = new_global_model
        
        # Store round information
        round_info = {
            'round': self.current_round,
            'participants': len(filtered_updates),
            'byzantine_filtered': len(fresh_updates) - len(filtered_updates),
            'timestamp': time.time(),
            'convergence_metric': await self._calculate_convergence_metric()
        }
        
        self.completed_rounds.append(round_info)
        
        # Update statistics
        self.learning_stats['total_rounds'] += 1
        self.learning_stats['successful_aggregations'] += 1
        self.participation_history.append(len(filtered_updates))
        
        # Clean up old pending updates
        if len(self.pending_updates) > 10:
            old_rounds = sorted(self.pending_updates.keys())[:-10]
            for old_round in old_rounds:
                del self.pending_updates[old_round]
    
    async def _detect_byzantine_updates(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Detect and filter potentially Byzantine (malicious) updates."""
        
        if len(updates) < 3:
            return updates
        
        filtered_updates = []
        
        # Calculate update statistics
        update_norms = []
        for update in updates:
            norm = math.sqrt(sum(x**2 for x in update.update_vector))
            update_norms.append((update, norm))
        
        # Sort by norm
        update_norms.sort(key=lambda x: x[1])
        
        # Use interquartile range to detect outliers
        n = len(update_norms)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        if q3_idx < len(update_norms) and q1_idx >= 0:
            q1_norm = update_norms[q1_idx][1]
            q3_norm = update_norms[q3_idx][1]
            iqr = q3_norm - q1_norm
            
            # Outlier thresholds
            lower_threshold = q1_norm - 1.5 * iqr
            upper_threshold = q3_norm + 1.5 * iqr
            
            # Filter outliers
            for update, norm in update_norms:
                if lower_threshold <= norm <= upper_threshold:
                    filtered_updates.append(update)
                else:
                    # Mark as Byzantine
                    self.learning_stats['byzantine_detections'] += 1
                    
                    # Reduce reputation score
                    if update.drone_id in self.registered_drones:
                        participant = self.registered_drones[update.drone_id]
                        participant['reputation_score'] *= 0.9
        else:
            # Not enough data for outlier detection
            filtered_updates = [update for update, _ in update_norms]
        
        return filtered_updates
    
    async def _clip_gradient(self, gradient: List[float]) -> List[float]:
        """Apply gradient clipping for privacy protection."""
        
        # Calculate L2 norm
        norm = math.sqrt(sum(x**2 for x in gradient))
        
        # Clip if necessary
        if norm > self.privacy_config.clipping_threshold:
            scaling_factor = self.privacy_config.clipping_threshold / norm
            return [x * scaling_factor for x in gradient]
        
        return gradient
    
    async def _calculate_convergence_metric(self) -> float:
        """Calculate convergence metric for current round."""
        
        if len(self.completed_rounds) < 2:
            return 1.0
        
        # Compare current model with previous model
        if hasattr(self, '_previous_global_model'):
            model_diff = sum(
                (a - b)**2 for a, b in zip(self.global_model, self._previous_global_model)
            )
            convergence_metric = math.sqrt(model_diff / len(self.global_model))
        else:
            convergence_metric = 1.0
        
        self._previous_global_model = self.global_model.copy()
        self.convergence_history.append(convergence_metric)
        
        return convergence_metric
    
    async def _evaluate_convergence(self):
        """Evaluate if the federated learning has converged."""
        
        if len(self.convergence_history) < 5:
            return
        
        # Check if convergence metric has stabilized
        recent_metrics = self.convergence_history[-5:]
        avg_recent = sum(recent_metrics) / len(recent_metrics)
        
        # Calculate convergence rate
        if len(self.convergence_history) >= 10:
            older_metrics = self.convergence_history[-10:-5]
            avg_older = sum(older_metrics) / len(older_metrics)
            
            if avg_older > 0:
                convergence_rate = (avg_older - avg_recent) / avg_older
                self.learning_stats['convergence_rate'] = convergence_rate
    
    async def evaluate_model_quality(self, 
                                   test_data: List[Tuple[List[float], float]]) -> Dict[str, float]:
        """Evaluate global model quality on test data."""
        
        if not test_data:
            return {'error': 'no_test_data'}
        
        # Simplified model evaluation (would use actual ML metrics)
        total_samples = len(test_data)
        correct_predictions = 0
        total_loss = 0.0
        
        for features, label in test_data:
            # Simplified prediction (dot product with model)
            prediction = sum(f * m for f, m in zip(features, self.global_model[:len(features)]))
            predicted_class = 1 if prediction > 0 else 0
            
            # Calculate accuracy
            if predicted_class == int(label):
                correct_predictions += 1
            
            # Calculate loss (simplified squared error)
            loss = (prediction - label) ** 2
            total_loss += loss
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / total_samples
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total_samples,
            'model_size': len(self.global_model)
        }
    
    async def get_participation_statistics(self) -> Dict[str, Any]:
        """Get detailed participation statistics."""
        
        # Calculate per-drone statistics
        drone_stats = {}
        for drone_id, info in self.registered_drones.items():
            drone_stats[drone_id] = {
                'participation_count': info['participation_count'],
                'last_round': info['last_update_round'],
                'reputation_score': info['reputation_score'],
                'privacy_budget_remaining': self.privacy_budgets.get(drone_id, 0.0),
                'privacy_budget_used': info['privacy_budget_used']
            }
        
        # Calculate aggregate statistics
        total_participants = len(self.registered_drones)
        active_participants = sum(1 for info in self.registered_drones.values() 
                                if info['last_update_round'] >= self.current_round - 5)
        
        avg_participation = (sum(info['participation_count'] for info in self.registered_drones.values()) 
                           / max(total_participants, 1))
        
        return {
            'total_registered_drones': total_participants,
            'active_participants': active_participants,
            'average_participation_rate': avg_participation,
            'current_round': self.current_round,
            'drone_statistics': drone_stats,
            'participation_history': self.participation_history[-10:],  # Last 10 rounds
        }
    
    async def optimize_federated_learning(self, target: str = "balanced"):
        """Optimize federated learning parameters."""
        
        if target == "privacy":
            await self._optimize_for_privacy()
        elif target == "accuracy":
            await self._optimize_for_accuracy()
        elif target == "efficiency":
            await self._optimize_for_efficiency()
        else:
            await self._optimize_balanced()
    
    async def _optimize_for_privacy(self):
        """Optimize for maximum privacy protection."""
        
        # Increase noise multiplier
        self.privacy_config.noise_multiplier *= 1.2
        
        # Reduce epsilon (stronger privacy)
        self.privacy_config.epsilon *= 0.9
        
        # Use more robust aggregation
        self.aggregation_config.strategy = AggregationStrategy.DIFFERENTIAL_PRIVATE
    
    async def _optimize_for_accuracy(self):
        """Optimize for maximum model accuracy."""
        
        # Reduce noise for better accuracy
        self.privacy_config.noise_multiplier *= 0.9
        
        # Use weighted aggregation
        self.aggregation_config.strategy = AggregationStrategy.WEIGHTED_AGGREGATION
        
        # Increase minimum participants
        self.aggregation_config.min_participants = min(50, self.aggregation_config.min_participants + 5)
    
    async def _optimize_for_efficiency(self):
        """Optimize for computational efficiency."""
        
        # Reduce minimum participants for faster rounds
        self.aggregation_config.min_participants = max(3, self.aggregation_config.min_participants - 2)
        
        # Use simpler aggregation
        self.aggregation_config.strategy = AggregationStrategy.FEDERATED_AVERAGING
        
        # Increase staleness tolerance
        self.aggregation_config.staleness_threshold += 1
    
    async def _optimize_balanced(self):
        """Apply balanced optimization across all metrics."""
        
        # Moderate privacy-accuracy tradeoff
        self.privacy_config.noise_multiplier = 1.0
        self.privacy_config.epsilon = max(0.5, self.privacy_config.epsilon * 0.95)
        
        # Balanced aggregation strategy
        self.aggregation_config.strategy = AggregationStrategy.WEIGHTED_AGGREGATION
        
        # Moderate participation requirements
        target_participants = len(self.registered_drones) // 4
        self.aggregation_config.min_participants = max(3, min(20, target_participants))
    
    def get_federated_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive federated learning statistics."""
        
        # Calculate privacy statistics
        total_privacy_budget = sum(self.privacy_budgets.values())
        used_privacy_budget = sum(info['privacy_budget_used'] 
                                for info in self.registered_drones.values())
        
        # Calculate model statistics
        model_norm = math.sqrt(sum(x**2 for x in self.global_model))
        
        return {
            'learning_progress': {
                'current_round': self.current_round,
                'total_rounds_completed': len(self.completed_rounds),
                'convergence_rate': self.learning_stats.get('convergence_rate', 0.0),
                'model_norm': model_norm
            },
            'participation_metrics': {
                'total_registered_drones': len(self.registered_drones),
                'average_participation': (sum(self.participation_history) / 
                                        max(len(self.participation_history), 1)),
                'recent_participation': self.participation_history[-5:] if self.participation_history else []
            },
            'privacy_metrics': {
                'total_privacy_budget': total_privacy_budget,
                'used_privacy_budget': used_privacy_budget,
                'privacy_budget_utilization': used_privacy_budget / max(total_privacy_budget, 1),
                'privacy_violations': self.learning_stats['privacy_violations'],
                'privacy_mechanism': self.privacy_config.mechanism.value
            },
            'security_metrics': {
                'byzantine_detections': self.learning_stats['byzantine_detections'],
                'aggregation_strategy': self.aggregation_config.strategy.value,
                'byzantine_tolerance': self.aggregation_config.byzantine_tolerance
            },
            'configuration': {
                'model_size': self.model_size,
                'min_participants': self.aggregation_config.min_participants,
                'max_participants': self.aggregation_config.max_participants,
                'privacy_epsilon': self.privacy_config.epsilon,
                'noise_multiplier': self.privacy_config.noise_multiplier
            },
            'performance_stats': self.learning_stats.copy()
        }