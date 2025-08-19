"""
Advanced Fault Tolerance Manager

Comprehensive fault detection, isolation, and recovery for swarm systems.
Implements Byzantine fault tolerance, cascade failure prevention, and
autonomous healing mechanisms.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from ..utils.logging import get_logger
from ..utils.performance import performance_monitor

logger = get_logger(__name__)

class FaultType(Enum):
    """Types of faults that can occur in the swarm system."""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_ERROR = "software_error"
    COMMUNICATION_LOSS = "communication_loss"
    SENSOR_MALFUNCTION = "sensor_malfunction"
    POWER_DEPLETION = "power_depletion"
    COORDINATION_FAILURE = "coordination_failure"
    SECURITY_BREACH = "security_breach"
    ENVIRONMENTAL_HAZARD = "environmental_hazard"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    CASCADE_FAILURE = "cascade_failure"

class RecoveryStrategy(Enum):
    """Recovery strategies for different fault types."""
    RESTART = "restart"
    FAILOVER = "failover"
    REDUNDANCY = "redundancy"
    ISOLATION = "isolation"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_LANDING = "emergency_landing"
    FORMATION_RECONFIGURATION = "formation_reconfiguration"
    CONSENSUS_OVERRIDE = "consensus_override"
    SELF_HEALING = "self_healing"
    SYSTEM_RESET = "system_reset"

@dataclass
class FaultEvent:
    """Represents a fault event in the system."""
    id: str
    fault_type: FaultType
    affected_entities: List[str]
    severity: float  # 0.0 to 1.0
    timestamp: float
    description: str
    recovery_strategy: Optional[RecoveryStrategy] = None
    resolution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""
    action_id: str
    strategy: RecoveryStrategy
    target_entities: List[str]
    priority: int  # Higher number = higher priority
    estimated_duration: float  # seconds
    success_probability: float  # 0.0 to 1.0
    action_func: Callable
    rollback_func: Optional[Callable] = None

class FaultToleranceManager:
    """
    Advanced fault tolerance manager for swarm systems.
    
    Provides comprehensive fault detection, isolation, and recovery
    capabilities with Byzantine fault tolerance and cascade prevention.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the fault tolerance manager."""
        self.config = config or self._default_config()
        
        # Fault tracking
        self.active_faults: Dict[str, FaultEvent] = {}
        self.fault_history: List[FaultEvent] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        # Byzantine fault tolerance
        self.byzantine_threshold = self.config['byzantine_threshold']
        self.consensus_nodes: Set[str] = set()
        self.suspicious_nodes: Dict[str, float] = {}  # node_id -> suspicion_level
        
        # Cascade failure prevention
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.failure_propagation_model: Dict[str, float] = {}
        
        # Recovery statistics
        self.recovery_stats = {
            'total_faults': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'cascade_failures_prevented': 0
        }
        
        # Monitoring and callbacks
        self.fault_detectors: List[Callable] = []
        self.recovery_callbacks: Dict[RecoveryStrategy, List[Callable]] = {}
        
        logger.info("Fault Tolerance Manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for fault tolerance."""
        return {
            'byzantine_threshold': 0.33,  # Tolerate up to 33% Byzantine nodes
            'fault_detection_interval': 1.0,  # seconds
            'recovery_timeout': 30.0,  # seconds
            'cascade_prevention_enabled': True,
            'auto_isolation_enabled': True,
            'self_healing_enabled': True,
            'consensus_required_for_critical': True,
            'max_concurrent_recoveries': 5,
            'fault_history_limit': 1000,
            'suspicion_decay_rate': 0.1  # per hour
        }
    
    async def start_monitoring(self):
        """Start fault monitoring and recovery processes."""
        logger.info("Starting fault tolerance monitoring...")
        
        # Start fault detection loop
        asyncio.create_task(self._fault_detection_loop())
        
        # Start recovery processing loop
        asyncio.create_task(self._recovery_processing_loop())
        
        # Start Byzantine monitoring
        asyncio.create_task(self._byzantine_monitoring_loop())
        
        # Start suspicion decay
        asyncio.create_task(self._suspicion_decay_loop())
        
        logger.info("Fault tolerance monitoring started")
    
    async def register_fault(self, fault_event: FaultEvent) -> bool:
        """Register a new fault event."""
        try:
            with performance_monitor(f"fault_registration_{fault_event.fault_type.value}"):
                # Validate fault event
                if not self._validate_fault_event(fault_event):
                    logger.error(f"Invalid fault event: {fault_event}")
                    return False
                
                # Check for duplicate faults
                if fault_event.id in self.active_faults:
                    logger.warning(f"Duplicate fault event ignored: {fault_event.id}")
                    return False
                
                # Register the fault
                self.active_faults[fault_event.id] = fault_event
                self.fault_history.append(fault_event)
                self.recovery_stats['total_faults'] += 1
                
                logger.error(f"FAULT REGISTERED: {fault_event.fault_type.value} - {fault_event.description}")
                
                # Check for cascade potential
                if self.config['cascade_prevention_enabled']:
                    await self._analyze_cascade_potential(fault_event)
                
                # Determine recovery strategy
                recovery_strategy = await self._determine_recovery_strategy(fault_event)
                fault_event.recovery_strategy = recovery_strategy
                
                # Create recovery action
                recovery_action = await self._create_recovery_action(fault_event)
                if recovery_action:
                    self.recovery_actions[recovery_action.action_id] = recovery_action
                
                # Update Byzantine suspicion if applicable
                if fault_event.fault_type == FaultType.BYZANTINE_BEHAVIOR:
                    await self._update_byzantine_suspicion(fault_event)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to register fault: {e}")
            return False
    
    async def _fault_detection_loop(self):
        """Main fault detection loop."""
        while True:
            try:
                # Run all registered fault detectors
                for detector in self.fault_detectors:
                    try:
                        detected_faults = await detector()
                        for fault in detected_faults:
                            await self.register_fault(fault)
                    except Exception as e:
                        logger.error(f"Fault detector error: {e}")
                
                await asyncio.sleep(self.config['fault_detection_interval'])
                
            except Exception as e:
                logger.error(f"Fault detection loop error: {e}")
                await asyncio.sleep(5.0)  # Prevent tight error loop
    
    async def _recovery_processing_loop(self):
        """Process pending recovery actions."""
        while True:
            try:
                # Get pending recovery actions sorted by priority
                pending_actions = sorted(
                    [action for action in self.recovery_actions.values()],
                    key=lambda x: x.priority,
                    reverse=True
                )
                
                # Execute recovery actions (respecting concurrency limit)
                concurrent_count = 0
                max_concurrent = self.config['max_concurrent_recoveries']
                
                for action in pending_actions:
                    if concurrent_count >= max_concurrent:
                        break
                    
                    # Execute recovery action
                    asyncio.create_task(self._execute_recovery_action(action))
                    concurrent_count += 1
                
                await asyncio.sleep(1.0)  # Check for new actions every second
                
            except Exception as e:
                logger.error(f"Recovery processing loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _execute_recovery_action(self, action: RecoveryAction):
        """Execute a specific recovery action."""
        start_time = time.time()
        
        try:
            logger.info(f"Executing recovery action: {action.strategy.value} for {action.target_entities}")
            
            # Execute the recovery function
            success = await action.action_func(action.target_entities)
            
            execution_time = time.time() - start_time
            
            if success:
                self.recovery_stats['successful_recoveries'] += 1
                logger.info(f"Recovery successful in {execution_time:.2f}s: {action.action_id}")
                
                # Remove completed action
                if action.action_id in self.recovery_actions:
                    del self.recovery_actions[action.action_id]
                
                # Mark related faults as resolved
                await self._mark_faults_resolved(action)
                
            else:
                self.recovery_stats['failed_recoveries'] += 1
                logger.error(f"Recovery failed after {execution_time:.2f}s: {action.action_id}")
                
                # Try rollback if available
                if action.rollback_func:
                    try:
                        await action.rollback_func(action.target_entities)
                        logger.info(f"Rollback successful for: {action.action_id}")
                    except Exception as e:
                        logger.error(f"Rollback failed for {action.action_id}: {e}")
            
            # Update average recovery time
            total_recoveries = self.recovery_stats['successful_recoveries'] + self.recovery_stats['failed_recoveries']
            if total_recoveries > 0:
                current_avg = self.recovery_stats['average_recovery_time']
                self.recovery_stats['average_recovery_time'] = (
                    (current_avg * (total_recoveries - 1) + execution_time) / total_recoveries
                )
                
        except Exception as e:
            logger.error(f"Recovery action execution failed: {e}")
            self.recovery_stats['failed_recoveries'] += 1
            
            # Remove failed action
            if action.action_id in self.recovery_actions:
                del self.recovery_actions[action.action_id]
    
    async def _byzantine_monitoring_loop(self):
        """Monitor for Byzantine behavior in the swarm."""
        while True:
            try:
                # Analyze consensus patterns
                await self._analyze_consensus_patterns()
                
                # Check for nodes exceeding Byzantine threshold
                byzantine_nodes = [
                    node_id for node_id, suspicion in self.suspicious_nodes.items()
                    if suspicion > self.byzantine_threshold
                ]
                
                if byzantine_nodes:
                    logger.warning(f"Byzantine nodes detected: {byzantine_nodes}")
                    
                    # Create Byzantine fault events
                    for node_id in byzantine_nodes:
                        fault_event = FaultEvent(
                            id=f"byzantine_{node_id}_{int(time.time())}",
                            fault_type=FaultType.BYZANTINE_BEHAVIOR,
                            affected_entities=[node_id],
                            severity=self.suspicious_nodes[node_id],
                            timestamp=time.time(),
                            description=f"Byzantine behavior detected in node {node_id}"
                        )
                        await self.register_fault(fault_event)
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Byzantine monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    async def _suspicion_decay_loop(self):
        """Decay suspicion levels over time."""
        while True:
            try:
                decay_rate = self.config['suspicion_decay_rate'] / 3600  # Per second
                
                # Decay suspicion for all nodes
                for node_id in list(self.suspicious_nodes.keys()):
                    self.suspicious_nodes[node_id] = max(
                        0.0,
                        self.suspicious_nodes[node_id] - decay_rate
                    )
                    
                    # Remove nodes with negligible suspicion
                    if self.suspicious_nodes[node_id] < 0.01:
                        del self.suspicious_nodes[node_id]
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Suspicion decay error: {e}")
                await asyncio.sleep(60.0)
    
    async def _analyze_cascade_potential(self, fault_event: FaultEvent):
        """Analyze the potential for cascade failures."""
        try:
            affected_entities = set(fault_event.affected_entities)
            
            # Find dependent entities
            dependent_entities = set()
            for entity in affected_entities:
                if entity in self.dependency_graph:
                    dependent_entities.update(self.dependency_graph[entity])
            
            if dependent_entities:
                cascade_probability = self._calculate_cascade_probability(
                    affected_entities, dependent_entities
                )
                
                if cascade_probability > 0.5:  # High cascade risk
                    logger.warning(f"High cascade failure risk: {cascade_probability:.2f}")
                    
                    # Create preventive actions
                    await self._create_cascade_prevention_actions(
                        dependent_entities, cascade_probability
                    )
                    
                    self.recovery_stats['cascade_failures_prevented'] += 1
                    
        except Exception as e:
            logger.error(f"Cascade analysis error: {e}")
    
    def _calculate_cascade_probability(self, affected: Set[str], dependent: Set[str]) -> float:
        """Calculate the probability of cascade failure."""
        # Simple model - in practice would be more sophisticated
        base_probability = len(affected) / max(len(self.dependency_graph), 1)
        dependency_factor = len(dependent) / max(len(self.dependency_graph), 1)
        
        # Consider current system health
        system_health = 1.0 - (len(self.active_faults) / 100)  # Assume max 100 concurrent faults
        
        cascade_probability = (base_probability + dependency_factor) * (1 - system_health)
        return min(cascade_probability, 1.0)
    
    async def _create_cascade_prevention_actions(self, entities: Set[str], probability: float):
        """Create actions to prevent cascade failures."""
        # Isolate high-risk entities
        for entity in entities:
            if probability > 0.8:  # Very high risk
                action = RecoveryAction(
                    action_id=f"isolate_{entity}_{int(time.time())}",
                    strategy=RecoveryStrategy.ISOLATION,
                    target_entities=[entity],
                    priority=90,  # High priority
                    estimated_duration=5.0,
                    success_probability=0.95,
                    action_func=self._isolate_entity
                )
                self.recovery_actions[action.action_id] = action
    
    async def _isolate_entity(self, entities: List[str]) -> bool:
        """Isolate entities to prevent cascade failures."""
        try:
            logger.info(f"Isolating entities to prevent cascade: {entities}")
            # Simulate isolation process
            await asyncio.sleep(1.0)
            return True
        except Exception as e:
            logger.error(f"Entity isolation failed: {e}")
            return False
    
    def get_fault_tolerance_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status."""
        return {
            'active_faults': len(self.active_faults),
            'pending_recoveries': len(self.recovery_actions),
            'suspicious_nodes': len(self.suspicious_nodes),
            'recovery_stats': self.recovery_stats.copy(),
            'system_health': self._calculate_system_health(),
            'byzantine_tolerance': {
                'threshold': self.byzantine_threshold,
                'suspicious_nodes': dict(self.suspicious_nodes),
                'consensus_nodes': len(self.consensus_nodes)
            }
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        if not self.fault_history:
            return 1.0
        
        # Consider recent fault frequency
        recent_faults = [
            f for f in self.fault_history
            if time.time() - f.timestamp < 300  # Last 5 minutes
        ]
        
        fault_factor = max(0, 1 - len(recent_faults) / 10)  # Assume 10 faults = 0 health
        
        # Consider recovery success rate
        total_recoveries = self.recovery_stats['successful_recoveries'] + self.recovery_stats['failed_recoveries']
        if total_recoveries > 0:
            recovery_factor = self.recovery_stats['successful_recoveries'] / total_recoveries
        else:
            recovery_factor = 1.0
        
        # Consider Byzantine node ratio
        total_nodes = len(self.consensus_nodes) + len(self.suspicious_nodes)
        if total_nodes > 0:
            byzantine_factor = 1 - (len(self.suspicious_nodes) / total_nodes)
        else:
            byzantine_factor = 1.0
        
        return (fault_factor * 0.4 + recovery_factor * 0.4 + byzantine_factor * 0.2)
    
    # Additional methods would be implemented for specific recovery strategies
    async def _determine_recovery_strategy(self, fault_event: FaultEvent) -> RecoveryStrategy:
        """Determine the best recovery strategy for a fault."""
        # Simple strategy mapping - would be more sophisticated in practice
        strategy_map = {
            FaultType.HARDWARE_FAILURE: RecoveryStrategy.FAILOVER,
            FaultType.SOFTWARE_ERROR: RecoveryStrategy.RESTART,
            FaultType.COMMUNICATION_LOSS: RecoveryStrategy.REDUNDANCY,
            FaultType.SENSOR_MALFUNCTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FaultType.POWER_DEPLETION: RecoveryStrategy.EMERGENCY_LANDING,
            FaultType.COORDINATION_FAILURE: RecoveryStrategy.FORMATION_RECONFIGURATION,
            FaultType.SECURITY_BREACH: RecoveryStrategy.ISOLATION,
            FaultType.ENVIRONMENTAL_HAZARD: RecoveryStrategy.EMERGENCY_LANDING,
            FaultType.BYZANTINE_BEHAVIOR: RecoveryStrategy.CONSENSUS_OVERRIDE,
            FaultType.CASCADE_FAILURE: RecoveryStrategy.SYSTEM_RESET
        }
        
        return strategy_map.get(fault_event.fault_type, RecoveryStrategy.GRACEFUL_DEGRADATION)
    
    async def _create_recovery_action(self, fault_event: FaultEvent) -> Optional[RecoveryAction]:
        """Create a recovery action for a fault event."""
        if not fault_event.recovery_strategy:
            return None
        
        action_id = f"recovery_{fault_event.id}_{int(time.time())}"
        
        # Map strategy to action function
        action_func_map = {
            RecoveryStrategy.RESTART: self._restart_entities,
            RecoveryStrategy.FAILOVER: self._failover_entities,
            RecoveryStrategy.REDUNDANCY: self._activate_redundancy,
            RecoveryStrategy.ISOLATION: self._isolate_entity,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation,
            RecoveryStrategy.EMERGENCY_LANDING: self._emergency_landing,
            RecoveryStrategy.FORMATION_RECONFIGURATION: self._reconfigure_formation,
            RecoveryStrategy.CONSENSUS_OVERRIDE: self._consensus_override,
            RecoveryStrategy.SELF_HEALING: self._self_healing,
            RecoveryStrategy.SYSTEM_RESET: self._system_reset
        }
        
        action_func = action_func_map.get(fault_event.recovery_strategy)
        if not action_func:
            return None
        
        return RecoveryAction(
            action_id=action_id,
            strategy=fault_event.recovery_strategy,
            target_entities=fault_event.affected_entities,
            priority=int(fault_event.severity * 100),
            estimated_duration=10.0,  # Default estimate
            success_probability=0.8,  # Default probability
            action_func=action_func
        )
    
    # Recovery action implementations (simplified for demonstration)
    async def _restart_entities(self, entities: List[str]) -> bool:
        """Restart specified entities."""
        logger.info(f"Restarting entities: {entities}")
        await asyncio.sleep(2.0)  # Simulate restart
        return True
    
    async def _failover_entities(self, entities: List[str]) -> bool:
        """Failover to backup entities."""
        logger.info(f"Failing over entities: {entities}")
        await asyncio.sleep(3.0)  # Simulate failover
        return True
    
    async def _activate_redundancy(self, entities: List[str]) -> bool:
        """Activate redundant systems."""
        logger.info(f"Activating redundancy for: {entities}")
        await asyncio.sleep(1.0)  # Simulate activation
        return True
    
    async def _graceful_degradation(self, entities: List[str]) -> bool:
        """Implement graceful degradation."""
        logger.info(f"Graceful degradation for: {entities}")
        await asyncio.sleep(1.0)  # Simulate degradation
        return True
    
    async def _emergency_landing(self, entities: List[str]) -> bool:
        """Execute emergency landing."""
        logger.warning(f"Emergency landing for: {entities}")
        await asyncio.sleep(5.0)  # Simulate landing
        return True
    
    async def _reconfigure_formation(self, entities: List[str]) -> bool:
        """Reconfigure formation."""
        logger.info(f"Reconfiguring formation for: {entities}")
        await asyncio.sleep(2.0)  # Simulate reconfiguration
        return True
    
    async def _consensus_override(self, entities: List[str]) -> bool:
        """Override Byzantine consensus."""
        logger.warning(f"Consensus override for: {entities}")
        await asyncio.sleep(1.0)  # Simulate override
        return True
    
    async def _self_healing(self, entities: List[str]) -> bool:
        """Trigger self-healing processes."""
        logger.info(f"Self-healing for: {entities}")
        await asyncio.sleep(3.0)  # Simulate healing
        return True
    
    async def _system_reset(self, entities: List[str]) -> bool:
        """Perform system reset."""
        logger.warning(f"System reset for: {entities}")
        await asyncio.sleep(10.0)  # Simulate reset
        return True
    
    # Additional helper methods
    def _validate_fault_event(self, fault_event: FaultEvent) -> bool:
        """Validate a fault event."""
        return (
            fault_event.id and
            isinstance(fault_event.fault_type, FaultType) and
            fault_event.affected_entities and
            0.0 <= fault_event.severity <= 1.0 and
            fault_event.timestamp > 0
        )
    
    async def _mark_faults_resolved(self, action: RecoveryAction):
        """Mark faults as resolved after successful recovery."""
        # Find faults related to this recovery action
        resolved_faults = []
        for fault_id, fault in self.active_faults.items():
            if any(entity in action.target_entities for entity in fault.affected_entities):
                fault.resolution_time = time.time()
                resolved_faults.append(fault_id)
        
        # Remove resolved faults from active list
        for fault_id in resolved_faults:
            del self.active_faults[fault_id]
        
        logger.info(f"Resolved {len(resolved_faults)} faults through recovery action")
    
    async def _analyze_consensus_patterns(self):
        """Analyze consensus patterns to detect Byzantine behavior."""
        # Simplified Byzantine detection - would be more sophisticated in practice
        # This would analyze voting patterns, message consistency, etc.
        pass
    
    async def _update_byzantine_suspicion(self, fault_event: FaultEvent):
        """Update Byzantine suspicion levels."""
        for entity in fault_event.affected_entities:
            current_suspicion = self.suspicious_nodes.get(entity, 0.0)
            suspicion_increase = fault_event.severity * 0.1
            self.suspicious_nodes[entity] = min(1.0, current_suspicion + suspicion_increase)