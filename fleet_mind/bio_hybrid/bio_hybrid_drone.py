"""Bio-Hybrid Drone Integration System - Generation 5."""

import asyncio
# Fallback for numpy
try:
    import numpy as np
except ImportError:
    class MockNumPy:
        @staticmethod
        def random():
            import random
            class MockRandom:
                @staticmethod
                def uniform(low, high, size=None):
                    import random
                    if size is None:
                        return random.uniform(low, high)
                    return [random.uniform(low, high) for _ in range(size)]
                
                @staticmethod
                def normal(mean, std, size=None):
                    import random
                    if size is None:
                        return random.gauss(mean, std)
                    return [random.gauss(mean, std) for _ in range(size)]
            return MockRandom()
    np = MockNumPy()
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict


class BiologicalComponent(Enum):
    """Types of biological components integrated into drones."""
    NEURAL_TISSUE = "neural_tissue"
    MUSCLE_FIBER = "muscle_fiber"
    SENSORY_ORGAN = "sensory_organ"
    METABOLIC_CELL = "metabolic_cell"
    ADAPTIVE_MEMBRANE = "adaptive_membrane"


class SynapticState(Enum):
    """States of synaptic connections in bio-hybrid systems."""
    DORMANT = "dormant"
    ACTIVE = "active"
    POTENTIATED = "potentiated"  
    DEPRESSED = "depressed"
    PLASTIC = "plastic"


@dataclass
class BioComponent:
    """Represents a biological component in the drone."""
    component_type: BiologicalComponent
    health_level: float  # 0.0 to 1.0
    activity_level: float  # 0.0 to 1.0
    adaptation_rate: float  # How quickly it adapts
    metabolic_rate: float  # Energy consumption
    growth_factor: float  # Growth/healing rate
    integration_level: float  # How well integrated with drone systems
    age: float  # Age in seconds since integration
    
    
@dataclass
class BiologicalState:
    """Overall biological state of the bio-hybrid drone."""
    total_health: float
    neural_activity: float
    metabolic_efficiency: float
    adaptation_score: float
    bio_mechanical_sync: float  # How well bio and mechanical systems sync
    stress_level: float  # Biological stress indicators
    regeneration_rate: float
    timestamp: float = field(default_factory=time.time)


class BioHybridDrone:
    """Bio-hybrid drone with integrated biological components.
    
    Combines mechanical drone systems with living biological components
    for enhanced adaptability, efficiency, and emergent behaviors.
    """
    
    def __init__(
        self,
        drone_id: int,
        biological_components: List[BiologicalComponent],
        bio_integration_level: float = 0.7,
        enable_evolution: bool = True,
        metabolic_efficiency: float = 0.8,
        neural_plasticity: bool = True
    ):
        self.drone_id = drone_id
        self.bio_integration_level = bio_integration_level
        self.enable_evolution = enable_evolution
        self.metabolic_efficiency = metabolic_efficiency
        self.neural_plasticity = neural_plasticity
        
        # Initialize biological components
        self.bio_components: Dict[str, BioComponent] = {}
        for comp_type in biological_components:
            component_id = f"{comp_type.value}_{drone_id}"
            self.bio_components[component_id] = BioComponent(
                component_type=comp_type,
                health_level=np.random().uniform(0.7, 1.0),
                activity_level=np.random().uniform(0.5, 0.9),
                adaptation_rate=np.random().uniform(0.1, 0.5),
                metabolic_rate=np.random().uniform(0.3, 0.8),
                growth_factor=np.random().uniform(0.1, 0.3),
                integration_level=bio_integration_level,
                age=0.0
            )
        
        # Biological state tracking
        self.bio_state = BiologicalState(
            total_health=1.0,
            neural_activity=0.5,
            metabolic_efficiency=metabolic_efficiency,
            adaptation_score=0.0,
            bio_mechanical_sync=0.5,
            stress_level=0.2,
            regeneration_rate=0.1
        )
        
        # Synaptic network for neural components
        self.synaptic_network: Dict[str, Dict[str, float]] = {}
        self._initialize_synaptic_network()
        
        # Bio-hybrid metrics
        self.bio_metrics = {
            'adaptation_events': 0,
            'regeneration_cycles': 0,
            'evolution_steps': 0,
            'bio_mechanical_conflicts': 0,
            'metabolic_optimizations': 0,
            'neural_rewiring_events': 0
        }
        
        # Background biological processes
        self._bio_task = None
        self._is_bio_active = False
        
        # Learning and memory
        self.bio_memory: Dict[str, Any] = {}
        self.learned_behaviors: List[Dict[str, Any]] = []
    
    def _initialize_synaptic_network(self):
        """Initialize synaptic connections between neural components."""
        neural_components = [
            comp_id for comp_id, comp in self.bio_components.items()
            if comp.component_type == BiologicalComponent.NEURAL_TISSUE
        ]
        
        # Create all-to-all synaptic connections
        for comp1 in neural_components:
            self.synaptic_network[comp1] = {}
            for comp2 in neural_components:
                if comp1 != comp2:
                    # Initialize with random synaptic strength
                    self.synaptic_network[comp1][comp2] = np.random().uniform(0.1, 0.8)
    
    async def activate_biological_systems(self):
        """Activate the biological systems of the drone."""
        if self._is_bio_active:
            return
        
        self._is_bio_active = True
        print(f"🧬 Activating bio-hybrid systems for drone {self.drone_id}")
        print(f"   Bio components: {len(self.bio_components)}")
        print(f"   Integration level: {self.bio_integration_level:.1%}")
        
        # Start biological processes
        self._bio_task = asyncio.create_task(self._biological_loop())
        
        # Initial biological calibration
        await self._calibrate_bio_systems()
    
    async def _biological_loop(self):
        """Main biological processing loop."""
        while self._is_bio_active:
            try:
                # Update biological components
                await self._update_bio_components()
                
                # Process neural activity
                await self._process_neural_activity()
                
                # Handle metabolic processes
                await self._process_metabolism()
                
                # Check for adaptation opportunities
                await self._check_adaptation()
                
                # Handle regeneration and healing
                await self._process_regeneration()
                
                # Update overall biological state
                await self._update_bio_state()
                
                # Sleep for biological time step (1Hz for biological processes)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"❌ Biological processing error for drone {self.drone_id}: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_bio_components(self):
        """Update all biological components."""
        for comp_id, component in self.bio_components.items():
            # Age the component
            component.age += 1.0
            
            # Natural health decay and regeneration
            health_decay = 0.001 * (1.0 - component.growth_factor)
            health_regen = component.growth_factor * 0.002
            component.health_level = max(0.0, min(1.0, 
                component.health_level - health_decay + health_regen))
            
            # Activity fluctuations
            activity_noise = np.random().normal(0, 0.05)
            component.activity_level = max(0.0, min(1.0,
                component.activity_level + activity_noise))
            
            # Adaptation based on usage
            if component.activity_level > 0.8:
                component.adaptation_rate = min(1.0, component.adaptation_rate + 0.01)
                component.integration_level = min(1.0, component.integration_level + 0.005)
            
            # Metabolic efficiency improvements
            if component.component_type == BiologicalComponent.METABOLIC_CELL:
                if component.health_level > 0.9:
                    component.metabolic_rate = max(0.1, component.metabolic_rate - 0.002)
                    self.bio_metrics['metabolic_optimizations'] += 1
    
    async def _process_neural_activity(self):
        """Process neural tissue activity and synaptic changes."""
        neural_components = [
            (comp_id, comp) for comp_id, comp in self.bio_components.items()
            if comp.component_type == BiologicalComponent.NEURAL_TISSUE
        ]
        
        if not neural_components:
            return
        
        # Update neural activity based on stimulation
        total_neural_activity = 0.0
        for comp_id, component in neural_components:
            # Random neural firing patterns
            neural_noise = np.random().normal(0, 0.1)
            component.activity_level = max(0.0, min(1.0,
                component.activity_level + neural_noise))
            total_neural_activity += component.activity_level
        
        # Update synaptic strengths (Hebbian learning)
        if self.neural_plasticity and len(neural_components) > 1:
            for comp1_id, comp1 in neural_components:
                for comp2_id, comp2 in neural_components:
                    if comp1_id != comp2_id and comp1_id in self.synaptic_network:
                        # Hebbian rule: neurons that fire together, wire together
                        activity_product = comp1.activity_level * comp2.activity_level
                        if activity_product > 0.6:  # Both highly active
                            delta = 0.01 * activity_product
                            self.synaptic_network[comp1_id][comp2_id] = min(1.0,
                                self.synaptic_network[comp1_id][comp2_id] + delta)
                            self.bio_metrics['neural_rewiring_events'] += 1
                        elif activity_product < 0.1:  # Both inactive
                            delta = -0.005
                            self.synaptic_network[comp1_id][comp2_id] = max(0.0,
                                self.synaptic_network[comp1_id][comp2_id] + delta)
        
        self.bio_state.neural_activity = total_neural_activity / max(1, len(neural_components))
    
    async def _process_metabolism(self):
        """Process metabolic activities of biological components."""
        metabolic_components = [
            comp for comp in self.bio_components.values()
            if comp.component_type == BiologicalComponent.METABOLIC_CELL
        ]
        
        if not metabolic_components:
            return
        
        # Calculate total metabolic demand
        total_demand = sum(comp.metabolic_rate * comp.activity_level 
                          for comp in self.bio_components.values())
        
        # Calculate metabolic efficiency
        if metabolic_components:
            avg_efficiency = statistics.mean(
                (1.0 - comp.metabolic_rate) * comp.health_level 
                for comp in metabolic_components
            )
            self.bio_state.metabolic_efficiency = avg_efficiency
        
        # Energy redistribution to stressed components
        stressed_components = [
            comp for comp in self.bio_components.values()
            if comp.health_level < 0.5
        ]
        
        if stressed_components and metabolic_components:
            # Redirect metabolic resources to stressed components
            for stressed_comp in stressed_components:
                energy_boost = 0.02 * self.bio_state.metabolic_efficiency
                stressed_comp.health_level = min(1.0, 
                    stressed_comp.health_level + energy_boost)
    
    async def _check_adaptation(self):
        """Check for biological adaptation opportunities."""
        # Environmental stress adaptation
        if self.bio_state.stress_level > 0.7:
            for component in self.bio_components.values():
                if component.adaptation_rate > 0.3:
                    # Increase adaptation rate under stress
                    component.adaptation_rate = min(1.0, component.adaptation_rate + 0.05)
                    component.integration_level = min(1.0, component.integration_level + 0.02)
                    self.bio_metrics['adaptation_events'] += 1
        
        # Adaptive membrane responses
        membrane_components = [
            comp for comp in self.bio_components.values()
            if comp.component_type == BiologicalComponent.ADAPTIVE_MEMBRANE
        ]
        
        for membrane in membrane_components:
            if membrane.activity_level > 0.8:
                # Membrane adapts to high activity
                membrane.adaptation_rate = min(1.0, membrane.adaptation_rate + 0.03)
                
        # Calculate overall adaptation score
        if self.bio_components:
            self.bio_state.adaptation_score = statistics.mean(
                comp.adaptation_rate * comp.integration_level 
                for comp in self.bio_components.values()
            )
    
    async def _process_regeneration(self):
        """Process biological regeneration and healing."""
        damaged_components = [
            comp for comp in self.bio_components.values()
            if comp.health_level < 0.8
        ]
        
        for component in damaged_components:
            # Regeneration rate based on component health and growth factor
            regen_rate = component.growth_factor * (1.0 - component.health_level) * 0.01
            component.health_level = min(1.0, component.health_level + regen_rate)
            
            if regen_rate > 0.005:  # Significant regeneration
                self.bio_metrics['regeneration_cycles'] += 1
        
        # Update overall regeneration rate
        if self.bio_components:
            avg_growth = statistics.mean(comp.growth_factor for comp in self.bio_components.values())
            avg_health = statistics.mean(comp.health_level for comp in self.bio_components.values())
            self.bio_state.regeneration_rate = avg_growth * (1.0 - avg_health)
    
    async def _update_bio_state(self):
        """Update the overall biological state."""
        if not self.bio_components:
            return
        
        # Calculate total health
        self.bio_state.total_health = statistics.mean(
            comp.health_level for comp in self.bio_components.values()
        )
        
        # Calculate bio-mechanical synchronization
        avg_integration = statistics.mean(
            comp.integration_level for comp in self.bio_components.values()
        )
        avg_activity = statistics.mean(
            comp.activity_level for comp in self.bio_components.values()
        )
        self.bio_state.bio_mechanical_sync = (avg_integration + avg_activity) / 2.0
        
        # Calculate stress level
        health_stress = 1.0 - self.bio_state.total_health
        metabolic_stress = 1.0 - self.bio_state.metabolic_efficiency
        self.bio_state.stress_level = (health_stress + metabolic_stress) / 2.0
    
    async def _calibrate_bio_systems(self):
        """Initial calibration of biological systems."""
        print(f"🔬 Calibrating bio-hybrid systems for drone {self.drone_id}")
        
        # Run initial adaptation cycles
        for _ in range(5):
            await self._update_bio_components()
            await self._process_neural_activity()
            await self._process_metabolism()
            await asyncio.sleep(0.1)
        
        print(f"✅ Bio-hybrid calibration complete")
    
    async def stimulate_component(self, component_type: BiologicalComponent, 
                                intensity: float = 0.5):
        """Stimulate a specific type of biological component."""
        stimulated_components = [
            comp for comp in self.bio_components.values()
            if comp.component_type == component_type
        ]
        
        for component in stimulated_components:
            activity_boost = intensity * 0.3
            component.activity_level = min(1.0, component.activity_level + activity_boost)
        
        print(f"⚡ Stimulated {component_type.value} components with intensity {intensity}")
    
    async def learn_behavior(self, behavior_data: Dict[str, Any]):
        """Learn and adapt biological responses to new behaviors."""
        behavior_id = f"behavior_{len(self.learned_behaviors)}"
        
        # Store behavior in bio-memory
        self.bio_memory[behavior_id] = {
            'data': behavior_data,
            'timestamp': time.time(),
            'usage_count': 0,
            'effectiveness': 0.5
        }
        
        self.learned_behaviors.append(behavior_data)
        
        # Adapt neural components to new behavior
        neural_components = [
            comp for comp in self.bio_components.values()
            if comp.component_type == BiologicalComponent.NEURAL_TISSUE
        ]
        
        for component in neural_components:
            if self.neural_plasticity:
                component.adaptation_rate = min(1.0, component.adaptation_rate + 0.02)
                component.activity_level = min(1.0, component.activity_level + 0.1)
        
        print(f"🧠 Learned new behavior: {behavior_id}")
    
    async def evolve_bio_systems(self):
        """Trigger evolutionary changes in biological systems."""
        if not self.enable_evolution:
            return
        
        # Only evolve if bio-mechanical sync is high
        if self.bio_state.bio_mechanical_sync < 0.7:
            return
        
        evolution_occurred = False
        
        for component in self.bio_components.values():
            if component.adaptation_rate > 0.8 and np.random.random() < 0.1:
                # Evolutionary improvement
                component.health_level = min(1.0, component.health_level + 0.05)
                component.integration_level = min(1.0, component.integration_level + 0.03)
                component.growth_factor = min(1.0, component.growth_factor + 0.02)
                evolution_occurred = True
        
        if evolution_occurred:
            self.bio_metrics['evolution_steps'] += 1
            print(f"🧬 Evolutionary step completed for drone {self.drone_id}")
    
    async def shutdown_bio_systems(self):
        """Gracefully shutdown biological systems."""
        if not self._is_bio_active:
            return
        
        print(f"🌙 Shutting down bio-hybrid systems for drone {self.drone_id}")
        self._is_bio_active = False
        
        if self._bio_task:
            self._bio_task.cancel()
            try:
                await self._bio_task
            except asyncio.CancelledError:
                pass
        
        # Gradually reduce biological activity
        for component in self.bio_components.values():
            component.activity_level *= 0.1
        
        print(f"💤 Bio-hybrid systems dormant for drone {self.drone_id}")
    
    def get_bio_status(self) -> Dict[str, Any]:
        """Get comprehensive biological status report."""
        component_status = {}
        for comp_id, component in self.bio_components.items():
            component_status[comp_id] = {
                'type': component.component_type.value,
                'health': round(component.health_level, 3),
                'activity': round(component.activity_level, 3),
                'adaptation_rate': round(component.adaptation_rate, 3),
                'integration': round(component.integration_level, 3),
                'age_hours': round(component.age / 3600, 2)
            }
        
        return {
            'drone_id': self.drone_id,
            'bio_state': {
                'total_health': round(self.bio_state.total_health, 3),
                'neural_activity': round(self.bio_state.neural_activity, 3),
                'metabolic_efficiency': round(self.bio_state.metabolic_efficiency, 3),
                'adaptation_score': round(self.bio_state.adaptation_score, 3),
                'bio_mechanical_sync': round(self.bio_state.bio_mechanical_sync, 3),
                'stress_level': round(self.bio_state.stress_level, 3),
                'regeneration_rate': round(self.bio_state.regeneration_rate, 3)
            },
            'components': component_status,
            'metrics': self.bio_metrics.copy(),
            'learned_behaviors': len(self.learned_behaviors),
            'synaptic_connections': len(self.synaptic_network),
            'is_active': self._is_bio_active
        }