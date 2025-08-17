"""Biological Sensor Systems - Generation 5."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time


class BioSensorType(Enum):
    """Types of biological sensors."""
    CHEMICAL = "chemical"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    PH_LEVEL = "ph_level"
    GLUCOSE = "glucose"
    OXYGEN = "oxygen"


@dataclass
class BiometricData:
    """Biometric sensor data."""
    sensor_type: BioSensorType
    value: float
    timestamp: float
    reliability: float
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class BiologicalSensor:
    """Biological sensor for bio-hybrid drones."""
    
    def __init__(self, sensor_type: BioSensorType, sensitivity: float = 0.8):
        self.sensor_type = sensor_type
        self.sensitivity = sensitivity
        self.calibration_factor = 1.0
        self.readings_history = []
        
    async def read_sensor(self) -> BiometricData:
        """Read current sensor value."""
        import random
        
        # Simulate sensor reading based on type
        if self.sensor_type == BioSensorType.TEMPERATURE:
            base_value = 37.0  # Body temperature
            noise = random.gauss(0, 0.5)
        elif self.sensor_type == BioSensorType.PH_LEVEL:
            base_value = 7.4  # Blood pH
            noise = random.gauss(0, 0.1)
        elif self.sensor_type == BioSensorType.OXYGEN:
            base_value = 98.0  # Oxygen saturation %
            noise = random.gauss(0, 2.0)
        else:
            base_value = random.uniform(0.0, 100.0)
            noise = random.gauss(0, 5.0)
        
        value = (base_value + noise) * self.calibration_factor
        reliability = self.sensitivity * random.uniform(0.8, 1.0)
        
        reading = BiometricData(
            sensor_type=self.sensor_type,
            value=value,
            timestamp=time.time(),
            reliability=reliability
        )
        
        self.readings_history.append(reading)
        if len(self.readings_history) > 100:
            self.readings_history.pop(0)
            
        return reading
        
    async def calibrate_sensor(self, reference_value: float):
        """Calibrate sensor against reference value."""
        if self.readings_history:
            latest_reading = self.readings_history[-1]
            if latest_reading.value != 0:
                self.calibration_factor = reference_value / latest_reading.value
                
    def get_sensor_health(self) -> float:
        """Get sensor health status."""
        if len(self.readings_history) < 2:
            return 1.0
            
        # Calculate health based on reading consistency
        recent_readings = self.readings_history[-10:]
        values = [r.value for r in recent_readings]
        reliabilities = [r.reliability for r in recent_readings]
        
        if len(values) > 1:
            import statistics
            value_std = statistics.stdev(values)
            avg_reliability = statistics.mean(reliabilities)
            
            # Health decreases with high variance and low reliability
            health = avg_reliability * (1.0 - min(1.0, value_std / 10.0))
            return max(0.0, min(1.0, health))
        
        return 1.0