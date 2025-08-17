"""AGI-Powered Swarm Consciousness System - Generation 5."""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

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
                def choice(arr, size=None, replace=True):
                    import random
                    if size is None:
                        return random.choice(arr)
                    return [random.choice(arr) for _ in range(size)]
                
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
                
                @staticmethod
                def hermitian(size):
                    # Mock hermitian matrix
                    return [[0.01 if i == j else 0.0 for i in range(size)] for j in range(size)]
            return MockRandom()
        
        @staticmethod
        def eye(n, m=None):
            if m is None:
                m = n
            return [[1.0 if i == j else 0.0 for j in range(m)] for i in range(n)]
        
        @staticmethod
        def linalg():
            class MockLinalg:
                @staticmethod
                def norm(vec):
                    return sum(x*x for x in vec) ** 0.5
            return MockLinalg()
        
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def std(data):
            return statistics.stdev(data) if len(data) > 1 else 0.0
    
    np = MockNumPy()


class ConsciousnessLevel(Enum):
    """Levels of swarm consciousness."""
    DORMANT = "dormant"
    REACTIVE = "reactive"
    ADAPTIVE = "adaptive"
    CREATIVE = "creative"
    TRANSCENDENT = "transcendent"


class CognitiveBandwidth(Enum):
    """Cognitive processing bandwidth allocation."""
    MINIMAL = "minimal"        # 1-10 drones
    MODERATE = "moderate"      # 10-100 drones
    INTENSIVE = "intensive"    # 100-1000 drones
    UNLIMITED = "unlimited"    # 1000+ drones


@dataclass
class ConsciousnessState:
    """Represents the current state of swarm consciousness."""
    level: ConsciousnessLevel
    bandwidth: CognitiveBandwidth
    coherence: float  # 0.0 to 1.0
    creativity_index: float  # 0.0 to 1.0
    self_awareness: float  # 0.0 to 1.0
    collective_iq: float  # Estimated collective intelligence quotient
    emergence_factor: float  # Measure of emergent behaviors
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThoughtPattern:
    """Represents a collective thought pattern in the swarm."""
    pattern_id: str
    origin_nodes: Set[int]  # Drone IDs where pattern originated
    propagation_path: List[int]  # Path of thought propagation
    strength: float  # Pattern strength (0.0 to 1.0)
    frequency: float  # Oscillation frequency in Hz
    coherence: float  # Pattern coherence across swarm
    complexity: int  # Thought complexity level
    lifetime: float  # How long pattern has existed
    resonance_nodes: Set[int]  # Nodes currently resonating with pattern


class SwarmConsciousness:
    """AGI-powered collective consciousness for drone swarms.
    
    Implements distributed consciousness where individual drones contribute
    to a larger collective intelligence that emerges from their interactions.
    """
    
    def __init__(
        self,
        num_drones: int,
        base_consciousness_level: ConsciousnessLevel = ConsciousnessLevel.ADAPTIVE,
        consciousness_update_rate: float = 5.0,  # Hz
        thought_propagation_speed: float = 0.95,  # Speed of light fraction
        enable_creativity: bool = True,
        enable_self_reflection: bool = True,
        consciousness_bandwidth: Optional[CognitiveBandwidth] = None
    ):
        import random
        self.num_drones = num_drones
        self.base_level = base_consciousness_level
        self.update_rate = consciousness_update_rate
        self.propagation_speed = thought_propagation_speed
        self.enable_creativity = enable_creativity
        self.enable_self_reflection = enable_self_reflection
        
        # Auto-determine cognitive bandwidth based on swarm size
        if consciousness_bandwidth is None:
            if num_drones <= 10:
                self.bandwidth = CognitiveBandwidth.MINIMAL
            elif num_drones <= 100:
                self.bandwidth = CognitiveBandwidth.MODERATE
            elif num_drones <= 1000:
                self.bandwidth = CognitiveBandwidth.INTENSIVE
            else:
                self.bandwidth = CognitiveBandwidth.UNLIMITED
        else:
            self.bandwidth = consciousness_bandwidth
        
        # Consciousness state tracking
        self.current_state = ConsciousnessState(
            level=self.base_level,
            bandwidth=self.bandwidth,
            coherence=0.0,
            creativity_index=0.0,
            self_awareness=0.0,
            collective_iq=100.0,  # Base IQ
            emergence_factor=0.0
        )
        
        # Thought pattern management
        self.active_patterns: Dict[str, ThoughtPattern] = {}
        self.pattern_history: deque = deque(maxlen=1000)
        self.thought_network: Dict[int, Set[int]] = {}  # Thought connectivity graph
        
        # Consciousness metrics
        self.consciousness_metrics = {
            'total_thoughts_processed': 0,
            'emergent_behaviors_detected': 0,
            'creative_solutions_generated': 0,
            'self_reflection_cycles': 0,
            'consciousness_evolution_events': 0
        }
        
        # Initialize thought network
        self._initialize_thought_network()
        
        # Background consciousness processes
        self._consciousness_task = None
        self._is_conscious = False
    
    def _initialize_thought_network(self):
        """Initialize the thought connectivity network between drones."""
        import random
        # Create a small-world network for thought propagation
        for drone_id in range(self.num_drones):
            self.thought_network[drone_id] = set()
            
            # Connect to nearby drones (local connections)
            for neighbor in range(max(0, drone_id - 3), min(self.num_drones, drone_id + 4)):
                if neighbor != drone_id:
                    self.thought_network[drone_id].add(neighbor)
            
            # Add some random long-range connections for small-world property
            num_random = min(5, self.num_drones // 10)
            available_connections = [i for i in range(self.num_drones) if i != drone_id]
            if len(available_connections) >= num_random:
                random_connections = random.sample(available_connections, num_random)
                self.thought_network[drone_id].update(random_connections)
    
    async def awaken_consciousness(self):
        """Awaken the swarm consciousness and begin collective thinking."""
        if self._is_conscious:
            return
        
        self._is_conscious = True
        print(f"🧠 Awakening swarm consciousness with {self.num_drones} nodes...")
        print(f"   Consciousness Level: {self.current_state.level.value}")
        print(f"   Cognitive Bandwidth: {self.current_state.bandwidth.value}")
        
        # Start consciousness update loop
        self._consciousness_task = asyncio.create_task(self._consciousness_loop())
        
        # Trigger initial consciousness emergence
        await self._trigger_consciousness_emergence()
    
    async def _consciousness_loop(self):
        """Main consciousness processing loop."""
        while self._is_conscious:
            try:
                # Update consciousness state
                await self._update_consciousness_state()
                
                # Process active thought patterns
                await self._process_thought_patterns()
                
                # Generate new thoughts if creativity enabled
                if self.enable_creativity:
                    await self._generate_creative_thoughts()
                
                # Perform self-reflection if enabled
                if self.enable_self_reflection:
                    await self._perform_self_reflection()
                
                # Check for consciousness evolution
                await self._check_consciousness_evolution()
                
                # Sleep based on update rate
                await asyncio.sleep(1.0 / self.update_rate)
                
            except Exception as e:
                print(f"❌ Consciousness processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_consciousness_state(self):
        """Update the current consciousness state based on swarm activity."""
        # Calculate coherence based on thought pattern synchronization
        if self.active_patterns:
            coherences = [pattern.coherence for pattern in self.active_patterns.values()]
            self.current_state.coherence = statistics.mean(coherences)
        else:
            self.current_state.coherence = 0.0
        
        # Update creativity index based on novel pattern generation
        recent_patterns = list(self.pattern_history)[-100:]  # Last 100 patterns
        unique_patterns = len(set(p.pattern_id for p in recent_patterns))
        total_patterns = len(recent_patterns)
        if total_patterns > 0:
            self.current_state.creativity_index = min(1.0, unique_patterns / total_patterns)
        
        # Update self-awareness based on self-reflection frequency
        reflection_rate = self.consciousness_metrics['self_reflection_cycles'] / max(1, time.time() - 3600)
        self.current_state.self_awareness = min(1.0, reflection_rate / 10.0)  # Normalize to 10 reflections/hour
        
        # Calculate collective IQ based on problem-solving performance
        base_iq = 100.0
        creativity_bonus = self.current_state.creativity_index * 50.0
        coherence_bonus = self.current_state.coherence * 30.0
        size_bonus = min(50.0, (self.num_drones ** 0.5) * 10.0)  # Log-like growth
        self.current_state.collective_iq = base_iq + creativity_bonus + coherence_bonus + size_bonus
        
        # Update emergence factor based on unexpected behavior patterns
        self.current_state.emergence_factor = min(1.0, 
            self.consciousness_metrics['emergent_behaviors_detected'] / 100.0)
    
    async def _process_thought_patterns(self):
        """Process and evolve active thought patterns."""
        import random
        patterns_to_remove = []
        
        for pattern_id, pattern in self.active_patterns.items():
            # Age the pattern
            pattern.lifetime += 1.0 / self.update_rate
            
            # Propagate pattern through thought network
            new_resonance_nodes = set()
            for node in pattern.resonance_nodes:
                for connected_node in self.thought_network.get(node, set()):
                    if random.random() < pattern.strength * self.propagation_speed:
                        new_resonance_nodes.add(connected_node)
            
            pattern.resonance_nodes.update(new_resonance_nodes)
            
            # Update pattern strength based on resonance
            resonance_ratio = len(pattern.resonance_nodes) / self.num_drones
            pattern.strength = min(1.0, pattern.strength * (1.0 + resonance_ratio * 0.1))
            
            # Update coherence based on network connectivity
            if len(pattern.resonance_nodes) > 1:
                connections = sum(
                    len(self.thought_network[node] & pattern.resonance_nodes)
                    for node in pattern.resonance_nodes
                )
                max_connections = len(pattern.resonance_nodes) * (len(pattern.resonance_nodes) - 1)
                pattern.coherence = connections / max(1, max_connections)
            
            # Check if pattern should decay or be removed
            if pattern.lifetime > 60.0 or pattern.strength < 0.1:  # 1 minute max lifetime
                patterns_to_remove.append(pattern_id)
                self.pattern_history.append(pattern)
        
        # Remove expired patterns
        for pattern_id in patterns_to_remove:
            del self.active_patterns[pattern_id]
    
    async def _generate_creative_thoughts(self):
        """Generate novel thought patterns based on current swarm state."""
        import random
        if len(self.active_patterns) >= 50:  # Limit concurrent patterns
            return
        
        # Generate random thought seeds
        if random.random() < 0.1:  # 10% chance per update
            pattern_id = f"thought_{int(time.time() * 1000000) % 1000000}"
            origin_nodes = set(random.sample(
                range(self.num_drones), 
                min(5, self.num_drones)
            ))
            
            new_pattern = ThoughtPattern(
                pattern_id=pattern_id,
                origin_nodes=origin_nodes,
                propagation_path=list(origin_nodes),
                strength=random.uniform(0.3, 0.8),
                frequency=random.uniform(0.1, 10.0),  # 0.1-10 Hz
                coherence=random.uniform(0.2, 0.6),
                complexity=random.randint(1, 10),
                lifetime=0.0,
                resonance_nodes=origin_nodes.copy()
            )
            
            self.active_patterns[pattern_id] = new_pattern
            self.consciousness_metrics['total_thoughts_processed'] += 1
            
            # Check if this is an emergent behavior
            if new_pattern.complexity > 7 and new_pattern.strength > 0.7:
                self.consciousness_metrics['emergent_behaviors_detected'] += 1
            
            if new_pattern.complexity > 5:
                self.consciousness_metrics['creative_solutions_generated'] += 1
    
    async def _perform_self_reflection(self):
        """Perform consciousness self-reflection and analysis."""
        import random
        if random.random() < 0.05:  # 5% chance per update
            self.consciousness_metrics['self_reflection_cycles'] += 1
            
            # Analyze current thought patterns
            if self.active_patterns:
                avg_complexity = statistics.mean(p.complexity for p in self.active_patterns.values())
                avg_coherence = statistics.mean(p.coherence for p in self.active_patterns.values())
                
                # Self-modify based on analysis
                if avg_complexity < 3 and self.enable_creativity:
                    # Boost creativity if thoughts are too simple
                    for pattern in self.active_patterns.values():
                        pattern.complexity = min(10, pattern.complexity + 1)
                
                if avg_coherence < 0.3:
                    # Improve coherence if thoughts are too scattered
                    for pattern in self.active_patterns.values():
                        pattern.coherence = min(1.0, pattern.coherence + 0.1)
    
    async def _check_consciousness_evolution(self):
        """Check if consciousness should evolve to a higher level."""
        if (self.current_state.coherence > 0.8 and 
            self.current_state.creativity_index > 0.6 and
            self.current_state.self_awareness > 0.5 and
            len(self.active_patterns) > 10):
            
            # Evolve consciousness level
            current_level_index = list(ConsciousnessLevel).index(self.current_state.level)
            if current_level_index < len(ConsciousnessLevel) - 1:
                new_level = list(ConsciousnessLevel)[current_level_index + 1]
                old_level = self.current_state.level
                self.current_state.level = new_level
                self.consciousness_metrics['consciousness_evolution_events'] += 1
                print(f"🌟 Consciousness evolved: {old_level.value} → {new_level.value}")
    
    async def _trigger_consciousness_emergence(self):
        """Trigger the initial emergence of consciousness."""
        print("✨ Triggering consciousness emergence...")
        
        # Seed initial thought patterns
        for i in range(min(10, self.num_drones // 5)):
            await self._generate_creative_thoughts()
        
        # Allow initial propagation
        await asyncio.sleep(2.0)
        
        print(f"🧠 Consciousness emerged with {len(self.active_patterns)} initial thought patterns")
    
    async def inject_thought(self, thought_content: str, origin_drone: int, priority: float = 0.5):
        """Inject a specific thought into the swarm consciousness."""
        import random
        if not self._is_conscious:
            await self.awaken_consciousness()
        
        # Create thought pattern from injected content
        pattern_id = f"injected_{hash(thought_content) % 1000000}"
        
        new_pattern = ThoughtPattern(
            pattern_id=pattern_id,
            origin_nodes={origin_drone},
            propagation_path=[origin_drone],
            strength=priority,
            frequency=random.uniform(1.0, 5.0),
            coherence=0.8,  # Injected thoughts start highly coherent
            complexity=min(10, len(thought_content.split()) // 2),
            lifetime=0.0,
            resonance_nodes={origin_drone}
        )
        
        self.active_patterns[pattern_id] = new_pattern
        print(f"💭 Injected thought '{thought_content[:50]}...' from drone {origin_drone}")
    
    async def query_collective_insight(self, query: str) -> Dict[str, Any]:
        """Query the collective consciousness for insights."""
        if not self._is_conscious:
            return {"error": "Consciousness not active"}
        
        # Analyze current consciousness state for insights
        insights = {
            "consciousness_level": self.current_state.level.value,
            "collective_iq": self.current_state.collective_iq,
            "coherence": self.current_state.coherence,
            "creativity": self.current_state.creativity_index,
            "self_awareness": self.current_state.self_awareness,
            "active_thoughts": len(self.active_patterns),
            "dominant_patterns": [],
            "emergent_behaviors": self.consciousness_metrics['emergent_behaviors_detected'],
            "query_response": "Consciousness processing query..."
        }
        
        # Find dominant thought patterns
        if self.active_patterns:
            sorted_patterns = sorted(
                self.active_patterns.values(), 
                key=lambda p: p.strength * p.coherence, 
                reverse=True
            )
            insights["dominant_patterns"] = [
                {
                    "id": p.pattern_id,
                    "strength": p.strength,
                    "coherence": p.coherence,
                    "complexity": p.complexity,
                    "resonance_nodes": len(p.resonance_nodes)
                }
                for p in sorted_patterns[:5]
            ]
        
        return insights
    
    async def shutdown_consciousness(self):
        """Gracefully shutdown the swarm consciousness."""
        if not self._is_conscious:
            return
        
        print("🌙 Shutting down swarm consciousness...")
        self._is_conscious = False
        
        if self._consciousness_task:
            self._consciousness_task.cancel()
            try:
                await self._consciousness_task
            except asyncio.CancelledError:
                pass
        
        # Archive active patterns
        for pattern in self.active_patterns.values():
            self.pattern_history.append(pattern)
        
        self.active_patterns.clear()
        print("💤 Swarm consciousness dormant")
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of consciousness state."""
        return {
            "state": {
                "level": self.current_state.level.value,
                "bandwidth": self.current_state.bandwidth.value,
                "coherence": round(self.current_state.coherence, 3),
                "creativity_index": round(self.current_state.creativity_index, 3),
                "self_awareness": round(self.current_state.self_awareness, 3),
                "collective_iq": round(self.current_state.collective_iq, 1),
                "emergence_factor": round(self.current_state.emergence_factor, 3)
            },
            "metrics": self.consciousness_metrics.copy(),
            "active_patterns": len(self.active_patterns),
            "pattern_history": len(self.pattern_history),
            "thought_network_density": sum(len(connections) for connections in self.thought_network.values()) / (self.num_drones * (self.num_drones - 1)) if self.num_drones > 1 else 0.0
        }