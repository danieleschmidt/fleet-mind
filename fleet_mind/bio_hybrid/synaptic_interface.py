"""Synaptic Interface for Bio-Hybrid Systems - Generation 5."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class NeuronType(Enum):
    """Types of neurons in the synaptic interface."""
    SENSORY = "sensory"
    MOTOR = "motor"
    INTERNEURON = "interneuron"
    MEMORY = "memory"


class SynapticStrength(Enum):
    """Synaptic connection strengths."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    POTENTIATED = "potentiated"


@dataclass
class SynapticConnection:
    """Represents a synaptic connection."""
    pre_neuron: int
    post_neuron: int
    strength: float
    plasticity: float


class SynapticInterface:
    """Interface for synaptic communication in bio-hybrid systems."""
    
    def __init__(self, num_neurons: int = 100):
        self.num_neurons = num_neurons
        self.connections = []
        self.neuron_types = {}
        
    async def form_connection(self, pre_neuron: int, post_neuron: int, strength: float = 0.5):
        """Form a synaptic connection between neurons."""
        connection = SynapticConnection(
            pre_neuron=pre_neuron,
            post_neuron=post_neuron,
            strength=strength,
            plasticity=0.1
        )
        self.connections.append(connection)
        
    async def update_plasticity(self):
        """Update synaptic plasticity based on activity."""
        for connection in self.connections:
            # Simplified plasticity update
            connection.strength *= (1.0 + connection.plasticity * 0.01)