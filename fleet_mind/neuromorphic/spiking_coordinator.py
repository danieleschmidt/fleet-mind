"""Spiking Neural Network Coordinator for Ultra-Low Latency Drone Control.

Bio-inspired coordination using spiking neural networks:
- Event-driven processing for energy efficiency
- Temporal coding for precise timing control
- Plastic synapses for adaptive learning
- Distributed spike propagation
"""

import asyncio
import math
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import concurrent.futures

from ..utils.logging import get_logger

logger = get_logger(__name__)


class NeuronType(Enum):
    """Types of spiking neurons."""
    INTEGRATE_AND_FIRE = "integrate_and_fire"
    LEAKY_INTEGRATE_AND_FIRE = "leaky_integrate_and_fire"
    ADAPTIVE_EXPONENTIAL = "adaptive_exponential"
    IZHIKEVICH = "izhikevich"


class PlasticityType(Enum):
    """Synaptic plasticity types."""
    STDP = "spike_timing_dependent"  # Spike-timing dependent plasticity
    HOMEOSTATIC = "homeostatic"     # Homeostatic scaling
    METAPLASTIC = "metaplastic"     # Meta-plasticity
    STRUCTURAL = "structural"       # Structural plasticity


@dataclass
class SpikeEvent:
    """Spike event with precise timing."""
    neuron_id: int
    timestamp: float
    amplitude: float = 1.0
    dendrite_id: Optional[int] = None


@dataclass
class SpikingNeuron:
    """Spiking neuron model for real-time processing."""
    neuron_id: int
    neuron_type: NeuronType = NeuronType.LEAKY_INTEGRATE_AND_FIRE
    
    # Membrane dynamics
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0          # mV
    reset_potential: float = -80.0     # mV
    resting_potential: float = -70.0   # mV
    membrane_capacitance: float = 1.0  # nF
    leak_conductance: float = 0.1      # μS
    
    # Timing
    last_spike_time: Optional[float] = None
    refractory_period: float = 2.0     # ms
    
    # Adaptation
    adaptation_current: float = 0.0
    adaptation_time_constant: float = 100.0  # ms
    
    # Spike history for STDP
    spike_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        """Initialize neuron state."""
        if not hasattr(self, 'spike_history') or self.spike_history is None:
            self.spike_history = deque(maxlen=100)
    
    def integrate(self, input_current: float, dt: float) -> bool:
        """Integrate membrane equation and check for spike."""
        current_time = time.time() * 1000  # Convert to ms
        
        # Check refractory period
        if (self.last_spike_time is not None and 
            current_time - self.last_spike_time < self.refractory_period):
            return False
        
        # Leaky integrate-and-fire dynamics
        if self.neuron_type == NeuronType.LEAKY_INTEGRATE_AND_FIRE:
            # dV/dt = (-(V - V_rest) + R*I) / τ_m
            tau_m = self.membrane_capacitance / self.leak_conductance
            
            leak_current = self.leak_conductance * (self.membrane_potential - self.resting_potential)
            total_current = input_current - leak_current - self.adaptation_current
            
            dv_dt = total_current / self.membrane_capacitance
            self.membrane_potential += dv_dt * dt
            
            # Update adaptation current
            self.adaptation_current *= math.exp(-dt / self.adaptation_time_constant)
        
        elif self.neuron_type == NeuronType.IZHIKEVICH:
            # Izhikevich model: v' = 0.04v² + 5v + 140 - u + I
            v = self.membrane_potential + 70  # Shift to Izhikevich range
            u = self.adaptation_current
            
            dv_dt = 0.04 * v * v + 5 * v + 140 - u + input_current
            du_dt = 0.02 * (0.2 * v - u)  # a=0.02, b=0.2
            
            self.membrane_potential = v + dv_dt * dt - 70
            self.adaptation_current = u + du_dt * dt
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.spike()
            return True
        
        return False
    
    def spike(self) -> None:
        """Generate spike and reset neuron."""
        current_time = time.time() * 1000
        self.last_spike_time = current_time
        self.spike_history.append(current_time)
        
        # Reset membrane potential
        self.membrane_potential = self.reset_potential
        
        # Add adaptation current (spike-triggered adaptation)
        self.adaptation_current += 2.0  # nA
    
    def get_recent_spikes(self, time_window: float = 50.0) -> List[float]:
        """Get spikes within time window (ms)."""
        current_time = time.time() * 1000
        return [t for t in self.spike_history if current_time - t <= time_window]


@dataclass  
class SynapticConnection:
    """Synaptic connection with plasticity."""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float = 1.0
    delay: float = 1.0  # ms
    
    # Plasticity parameters
    plasticity_type: PlasticityType = PlasticityType.STDP
    learning_rate: float = 0.01
    stdp_window: float = 20.0  # ms
    
    # STDP traces
    pre_trace: float = 0.0
    post_trace: float = 0.0
    trace_decay: float = 20.0  # ms
    
    # Homeostatic parameters
    target_rate: float = 10.0  # Hz
    homeostatic_strength: float = 0.001
    
    def update_stdp(self, pre_spike_time: Optional[float], 
                   post_spike_time: Optional[float], dt: float) -> None:
        """Update synaptic weight using STDP."""
        if self.plasticity_type != PlasticityType.STDP:
            return
        
        current_time = time.time() * 1000
        
        # Decay traces
        decay_factor = math.exp(-dt / self.trace_decay)
        self.pre_trace *= decay_factor
        self.post_trace *= decay_factor
        
        # Update traces on spikes
        if pre_spike_time is not None and abs(current_time - pre_spike_time) < 1.0:
            self.pre_trace += 1.0
            # Depression: post-trace at time of pre-spike
            if self.post_trace > 0:
                delta_w = -self.learning_rate * self.post_trace
                self.weight = max(0.0, self.weight + delta_w)
        
        if post_spike_time is not None and abs(current_time - post_spike_time) < 1.0:
            self.post_trace += 1.0
            # Potentiation: pre-trace at time of post-spike
            if self.pre_trace > 0:
                delta_w = self.learning_rate * self.pre_trace
                self.weight = min(10.0, self.weight + delta_w)  # Upper bound
    
    def apply_homeostatic_scaling(self, post_neuron_rate: float) -> None:
        """Apply homeostatic scaling to maintain target firing rate."""
        if self.plasticity_type in [PlasticityType.HOMEOSTATIC, PlasticityType.METAPLASTIC]:
            rate_error = self.target_rate - post_neuron_rate
            scaling = 1.0 + self.homeostatic_strength * rate_error
            self.weight *= scaling
            self.weight = max(0.0, min(10.0, self.weight))


class SpikingCoordinator:
    """Spiking neural network coordinator for drone swarms."""
    
    def __init__(self, 
                 max_drones: int = 1000,
                 update_frequency: float = 1000.0,  # Hz
                 plasticity_enabled: bool = True):
        self.max_drones = max_drones
        self.dt = 1.0 / update_frequency  # ms
        self.plasticity_enabled = plasticity_enabled
        
        # Neural network components
        self.neurons: Dict[int, SpikingNeuron] = {}
        self.synapses: Dict[Tuple[int, int], SynapticConnection] = {}
        self.spike_queue: deque = deque()
        
        # Network topology
        self.input_neurons: Set[int] = set()    # Sensor inputs
        self.output_neurons: Set[int] = set()   # Motor outputs
        self.hidden_neurons: Set[int] = set()   # Processing layer
        
        # Performance metrics
        self.spike_rate: float = 0.0
        self.network_synchrony: float = 0.0
        self.processing_latency: float = 0.0
        
        # Background processing
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Spiking coordinator initialized: {max_drones} drones, {update_frequency}Hz")
    
    async def start(self) -> None:
        """Start neuromorphic processing."""
        self._running = True
        self._processing_task = asyncio.create_task(self._neural_processing_loop())
        logger.info("Neuromorphic processing started")
    
    async def stop(self) -> None:
        """Stop neuromorphic processing."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Neuromorphic processing stopped")
    
    async def _neural_processing_loop(self) -> None:
        """Main neural processing loop."""
        while self._running:
            try:
                start_time = time.time()
                
                await self._process_spike_events()
                await self._update_neural_dynamics()
                await self._update_synaptic_plasticity()
                
                # Maintain precise timing
                processing_time = (time.time() - start_time) * 1000  # ms
                self.processing_latency = processing_time
                
                sleep_time = max(0, self.dt - processing_time / 1000)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Neural processing error: {e}")
    
    async def _process_spike_events(self) -> None:
        """Process pending spike events."""
        current_time = time.time() * 1000
        processed_spikes = []
        
        while self.spike_queue:
            spike = self.spike_queue[0]
            if current_time - spike.timestamp > 100:  # 100ms max delay
                self.spike_queue.popleft()
                continue
            break
        
        # Propagate spikes through synapses
        for spike in processed_spikes:
            await self._propagate_spike(spike)
    
    async def _propagate_spike(self, spike: SpikeEvent) -> None:
        """Propagate spike through synaptic connections."""
        pre_neuron_id = spike.neuron_id
        
        # Find all outgoing synapses
        for (pre_id, post_id), synapse in self.synapses.items():
            if pre_id == pre_neuron_id:
                # Apply synaptic delay
                delayed_spike = SpikeEvent(
                    neuron_id=post_id,
                    timestamp=spike.timestamp + synapse.delay,
                    amplitude=spike.amplitude * synapse.weight
                )
                
                # Add to processing queue if within timing window
                current_time = time.time() * 1000
                if delayed_spike.timestamp <= current_time + 10:  # 10ms lookahead
                    self.spike_queue.append(delayed_spike)
    
    async def _update_neural_dynamics(self) -> None:
        """Update membrane dynamics for all neurons."""
        total_spikes = 0
        
        for neuron in self.neurons.values():
            # Calculate input current from synapses
            input_current = self._calculate_input_current(neuron.neuron_id)
            
            # Integrate membrane equation
            spiked = neuron.integrate(input_current, self.dt)
            if spiked:
                total_spikes += 1
                # Add spike event
                spike = SpikeEvent(
                    neuron_id=neuron.neuron_id,
                    timestamp=time.time() * 1000
                )
                self.spike_queue.append(spike)
        
        # Update spike rate
        self.spike_rate = total_spikes / (len(self.neurons) * self.dt / 1000)
    
    def _calculate_input_current(self, neuron_id: int) -> float:
        """Calculate total input current for neuron."""
        total_current = 0.0
        current_time = time.time() * 1000
        
        # Sum currents from all incoming synapses
        for (pre_id, post_id), synapse in self.synapses.items():
            if post_id == neuron_id:
                pre_neuron = self.neurons.get(pre_id)
                if pre_neuron:
                    # Check for recent spikes within synaptic delay
                    recent_spikes = pre_neuron.get_recent_spikes(synapse.delay + 1.0)
                    for spike_time in recent_spikes:
                        if abs(current_time - spike_time - synapse.delay) < 0.5:
                            total_current += synapse.weight
        
        return total_current
    
    async def _update_synaptic_plasticity(self) -> None:
        """Update synaptic weights based on spike timing."""
        if not self.plasticity_enabled:
            return
        
        for synapse in self.synapses.values():
            pre_neuron = self.neurons.get(synapse.pre_neuron_id)
            post_neuron = self.neurons.get(synapse.post_neuron_id)
            
            if pre_neuron and post_neuron:
                # Update STDP
                synapse.update_stdp(
                    pre_neuron.last_spike_time,
                    post_neuron.last_spike_time,
                    self.dt
                )
                
                # Homeostatic scaling
                post_rate = len(post_neuron.get_recent_spikes(1000)) / 1.0  # Hz
                synapse.apply_homeostatic_scaling(post_rate)
    
    async def create_neural_network(self, drone_id: int) -> None:
        """Create neural network for drone control."""
        base_id = drone_id * 100  # Neuron ID offset per drone
        
        # Create input neurons (sensors)
        for i in range(10):  # 10 sensor inputs
            neuron_id = base_id + i
            self.neurons[neuron_id] = SpikingNeuron(
                neuron_id=neuron_id,
                neuron_type=NeuronType.INTEGRATE_AND_FIRE
            )
            self.input_neurons.add(neuron_id)
        
        # Create hidden layer neurons
        for i in range(20):  # 20 hidden neurons
            neuron_id = base_id + 10 + i
            self.neurons[neuron_id] = SpikingNeuron(
                neuron_id=neuron_id,
                neuron_type=NeuronType.LEAKY_INTEGRATE_AND_FIRE
            )
            self.hidden_neurons.add(neuron_id)
        
        # Create output neurons (motors)
        for i in range(6):  # 6 motor outputs (x,y,z velocity + roll,pitch,yaw)
            neuron_id = base_id + 30 + i
            self.neurons[neuron_id] = SpikingNeuron(
                neuron_id=neuron_id,
                neuron_type=NeuronType.ADAPTIVE_EXPONENTIAL
            )
            self.output_neurons.add(neuron_id)
        
        # Create synaptic connections
        await self._create_synapses(base_id)
        
        logger.info(f"Neural network created for drone {drone_id}: 36 neurons, {len(self.synapses)} synapses")
    
    async def _create_synapses(self, base_id: int) -> None:
        """Create synaptic connections for neural network."""
        # Input to hidden connections
        for i in range(10):  # Input neurons
            for j in range(20):  # Hidden neurons
                if random.random() < 0.3:  # 30% connectivity
                    pre_id = base_id + i
                    post_id = base_id + 10 + j
                    
                    synapse = SynapticConnection(
                        pre_neuron_id=pre_id,
                        post_neuron_id=post_id,
                        weight=random.uniform(0.5, 2.0),
                        delay=random.uniform(0.5, 3.0)
                    )
                    self.synapses[(pre_id, post_id)] = synapse
        
        # Hidden to output connections
        for i in range(20):  # Hidden neurons
            for j in range(6):  # Output neurons
                if random.random() < 0.5:  # 50% connectivity
                    pre_id = base_id + 10 + i
                    post_id = base_id + 30 + j
                    
                    synapse = SynapticConnection(
                        pre_neuron_id=pre_id,
                        post_neuron_id=post_id,
                        weight=random.uniform(0.5, 2.0),
                        delay=random.uniform(0.5, 2.0)
                    )
                    self.synapses[(pre_id, post_id)] = synapse
    
    async def process_sensor_input(self, drone_id: int, 
                                 sensor_data: Dict[str, float]) -> None:
        """Convert sensor data to spike trains."""
        base_id = drone_id * 100
        
        # Convert sensor values to spike rates
        sensor_values = list(sensor_data.values())[:10]  # Max 10 sensors
        
        for i, value in enumerate(sensor_values):
            neuron_id = base_id + i
            neuron = self.neurons.get(neuron_id)
            
            if neuron:
                # Rate coding: higher values → higher spike probability
                spike_probability = min(0.8, max(0.0, value / 100.0))
                
                if random.random() < spike_probability:
                    # Inject current to trigger spike
                    neuron.membrane_potential += 20.0  # Strong current injection
    
    async def get_motor_output(self, drone_id: int) -> Dict[str, float]:
        """Extract motor commands from output neuron spikes."""
        base_id = drone_id * 100
        output_commands = {}
        
        motor_names = ['vel_x', 'vel_y', 'vel_z', 'roll', 'pitch', 'yaw']
        
        for i, motor_name in enumerate(motor_names):
            neuron_id = base_id + 30 + i
            neuron = self.neurons.get(neuron_id)
            
            if neuron:
                # Rate decoding: spike rate → motor command
                recent_spikes = neuron.get_recent_spikes(100.0)  # 100ms window
                spike_rate = len(recent_spikes) / 0.1  # Hz
                
                # Convert to motor command (-1 to 1)
                motor_value = min(1.0, max(-1.0, (spike_rate - 5.0) / 10.0))
                output_commands[motor_name] = motor_value
            else:
                output_commands[motor_name] = 0.0
        
        return output_commands
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get neuromorphic network status."""
        # Calculate network synchrony
        if len(self.neurons) > 0:
            recent_spike_times = []
            for neuron in self.neurons.values():
                if neuron.last_spike_time:
                    recent_spike_times.append(neuron.last_spike_time)
            
            # Synchrony measure based on spike time variance
            if len(recent_spike_times) > 1:
                variance = sum((t - sum(recent_spike_times)/len(recent_spike_times))**2 
                             for t in recent_spike_times) / len(recent_spike_times)
                self.network_synchrony = 1.0 / (1.0 + variance / 100.0)
            else:
                self.network_synchrony = 0.0
        
        return {
            "total_neurons": len(self.neurons),
            "total_synapses": len(self.synapses),
            "input_neurons": len(self.input_neurons),
            "hidden_neurons": len(self.hidden_neurons),
            "output_neurons": len(self.output_neurons),
            "spike_rate": self.spike_rate,
            "network_synchrony": self.network_synchrony,
            "processing_latency": self.processing_latency,
            "plasticity_enabled": self.plasticity_enabled,
            "running": self._running
        }