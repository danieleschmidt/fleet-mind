"""Comprehensive validation system for Fleet-Mind."""

import re
import json
import math
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# NumPy and Pydantic imports with fallback handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Fallback implementation for numpy functions
    class np:
        @staticmethod
        def array(data): return data
        @staticmethod
        def any(arr): return any(arr)
        @staticmethod
        def isnan(arr): return [False] * len(arr) if hasattr(arr, '__len__') else False
        @staticmethod
        def isinf(arr): return [False] * len(arr) if hasattr(arr, '__len__') else False
        @staticmethod
        def max(arr): return max(arr) if hasattr(arr, '__len__') else arr
        @staticmethod
        def abs(arr): return [abs(x) for x in arr] if hasattr(arr, '__len__') else abs(arr)
        class ndarray:
            def __init__(self, data): 
                self.data = data
                self.shape = (len(data),) if hasattr(data, '__len__') else (1,)
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available, using simplified array operations")

try:
    from pydantic import BaseModel, Field, validator, ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback implementations
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(*args, **kwargs): return None
    def validator(field): return lambda func: func
    
    class PydanticValidationError(Exception):
        def __init__(self, errors):
            self.errors = lambda: errors
            super().__init__(str(errors))
    
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic not available, using simplified validation")


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'severity': self.severity.value,
            'message': self.message,
            'field': self.field,
            'value': self.value,
            'expected': self.expected,
        }


class ValidationError(Exception):
    """Custom validation error with detailed information."""
    
    def __init__(self, results: List[ValidationResult]):
        self.results = results
        self.errors = [r for r in results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        self.warnings = [r for r in results if r.severity == ValidationSeverity.WARNING]
        
        error_messages = [r.message for r in self.errors]
        super().__init__(f"Validation failed: {'; '.join(error_messages)}")


class MissionConstraintsValidator(BaseModel):
    """Pydantic model for mission constraints validation."""
    
    max_altitude: float = Field(gt=0, le=500, description="Maximum altitude in meters")
    battery_time: float = Field(gt=0, le=300, description="Battery time in minutes")
    safety_distance: float = Field(gt=0, le=50, description="Safety distance in meters")
    geofence: Optional[List[Tuple[float, float]]] = Field(None, description="Geofence coordinates")
    no_fly_zones: Optional[List[Tuple[float, float]]] = Field(None, description="No-fly zone coordinates")
    
    @validator('max_altitude')
    def validate_altitude(cls, v):
        """Validate altitude constraints."""
        if v > 400:  # FAA limit
            raise ValueError("Altitude exceeds regulatory limit (400m)")
        return v
    
    @validator('geofence')
    def validate_geofence(cls, v):
        """Validate geofence coordinates."""
        if v is not None:
            if len(v) < 3:
                raise ValueError("Geofence must have at least 3 points")
            
            for lat, lon in v:
                if not (-90 <= lat <= 90):
                    raise ValueError(f"Invalid latitude: {lat}")
                if not (-180 <= lon <= 180):
                    raise ValueError(f"Invalid longitude: {lon}")
        
        return v


class DroneStateValidator(BaseModel):
    """Pydantic model for drone state validation."""
    
    drone_id: str = Field(regex=r'^[a-zA-Z0-9_-]+$', description="Drone identifier")
    position: Tuple[float, float, float] = Field(description="Position (x, y, z)")
    velocity: Tuple[float, float, float] = Field(description="Velocity (vx, vy, vz)")
    orientation: Tuple[float, float, float] = Field(description="Orientation (roll, pitch, yaw)")
    battery_percent: float = Field(ge=0, le=100, description="Battery percentage")
    health_score: float = Field(ge=0, le=1, description="Health score")
    
    @validator('position')
    def validate_position(cls, v):
        """Validate position coordinates."""
        x, y, z = v
        
        # Check for reasonable bounds
        if abs(x) > 10000 or abs(y) > 10000:  # 10km radius
            raise ValueError("Position coordinates exceed reasonable bounds")
        
        if z < -100 or z > 500:  # Below sea level or above regulatory limit
            raise ValueError("Altitude outside safe operating range")
        
        return v
    
    @validator('velocity')
    def validate_velocity(cls, v):
        """Validate velocity components."""
        vx, vy, vz = v
        
        # Check for reasonable velocity limits
        max_velocity = 30.0  # 30 m/s
        if abs(vx) > max_velocity or abs(vy) > max_velocity or abs(vz) > max_velocity:
            raise ValueError("Velocity exceeds safe operating limits")
        
        return v
    
    @validator('orientation')
    def validate_orientation(cls, v):
        """Validate orientation angles."""
        roll, pitch, yaw = v
        
        # Check angle ranges (radians)
        if not (-math.pi <= roll <= math.pi):
            raise ValueError("Roll angle outside valid range")
        if not (-math.pi/2 <= pitch <= math.pi/2):
            raise ValueError("Pitch angle outside valid range")
        if not (-math.pi <= yaw <= math.pi):
            raise ValueError("Yaw angle outside valid range")
        
        return v


class FleetValidator:
    """Comprehensive validation system for Fleet-Mind operations."""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules configuration."""
        return {
            'mission': {
                'max_description_length': 10000,
                'required_fields': ['mission_id', 'description'],
                'allowed_priorities': ['low', 'normal', 'high', 'critical'],
            },
            'drone': {
                'max_id_length': 64,
                'min_battery_warning': 20.0,
                'min_battery_critical': 10.0,
                'min_health_warning': 0.7,
                'min_health_critical': 0.5,
            },
            'fleet': {
                'max_fleet_size': 500,
                'min_separation_distance': 2.0,
                'max_communication_range': 10000.0,
            },
            'network': {
                'max_latency_ms': 1000,
                'min_bandwidth_mbps': 0.1,
                'max_packet_loss': 0.1,
            },
        }
    
    def validate_mission_constraints(self, constraints: Dict[str, Any]) -> List[ValidationResult]:
        """Validate mission constraints."""
        results = []
        
        try:
            # Use Pydantic model for basic validation
            MissionConstraintsValidator(**constraints)
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Mission constraints passed basic validation"
            ))
            
        except ValidationError as e:
            for error in e.errors():
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=error['msg'],
                    field='.'.join(str(loc) for loc in error['loc']),
                    value=error.get('input'),
                ))
        
        # Additional custom validations
        results.extend(self._validate_mission_safety(constraints))
        results.extend(self._validate_mission_feasibility(constraints))
        
        return results
    
    def validate_drone_state(self, state: Dict[str, Any]) -> List[ValidationResult]:
        """Validate drone state information."""
        results = []
        
        try:
            # Use Pydantic model for basic validation
            DroneStateValidator(**state)
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Drone state passed basic validation"
            ))
            
        except ValidationError as e:
            for error in e.errors():
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=error['msg'],
                    field='.'.join(str(loc) for loc in error['loc']),
                    value=error.get('input'),
                ))
        
        # Additional custom validations
        results.extend(self._validate_drone_health(state))
        results.extend(self._validate_drone_safety(state))
        
        return results
    
    def validate_fleet_configuration(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate fleet configuration."""
        results = []
        rules = self.validation_rules['fleet']
        
        # Validate fleet size
        fleet_size = config.get('num_drones', 0)
        if fleet_size <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Fleet size must be positive",
                field='num_drones',
                value=fleet_size,
            ))
        elif fleet_size > rules['max_fleet_size']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Fleet size exceeds recommended maximum ({rules['max_fleet_size']})",
                field='num_drones',
                value=fleet_size,
                expected=rules['max_fleet_size'],
            ))
        
        # Validate separation distance
        separation = config.get('min_separation', 0)
        if separation < rules['min_separation_distance']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Minimum separation distance too small (minimum: {rules['min_separation_distance']}m)",
                field='min_separation',
                value=separation,
                expected=rules['min_separation_distance'],
            ))
        
        # Validate communication range
        comm_range = config.get('communication_range', 0)
        if comm_range > rules['max_communication_range']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Communication range may be unrealistic ({comm_range}m)",
                field='communication_range',
                value=comm_range,
                expected=rules['max_communication_range'],
            ))
        
        return results
    
    def validate_latent_encoding(self, encoding: np.ndarray, expected_dim: int) -> List[ValidationResult]:
        """Validate latent encoding data."""
        results = []
        
        # Check dimensions
        if encoding.shape[0] != expected_dim:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Latent encoding dimension mismatch",
                field='latent_dim',
                value=encoding.shape[0],
                expected=expected_dim,
            ))
        
        # Check for NaN or inf values
        if np.any(np.isnan(encoding)):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Latent encoding contains NaN values",
                field='latent_encoding',
            ))
        
        if np.any(np.isinf(encoding)):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Latent encoding contains infinite values",
                field='latent_encoding',
            ))
        
        # Check value range
        if np.max(np.abs(encoding)) > 1000:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Latent encoding values are very large",
                field='latent_encoding',
                value=np.max(np.abs(encoding)),
            ))
        
        return results
    
    def validate_communication_message(self, message: Dict[str, Any]) -> List[ValidationResult]:
        """Validate communication message format."""
        results = []
        
        # Check required fields
        required_fields = ['type', 'timestamp', 'data']
        for field in required_fields:
            if field not in message:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required field: {field}",
                    field=field,
                ))
        
        # Validate message type
        valid_types = ['latent_action', 'telemetry', 'status', 'emergency', 'heartbeat']
        msg_type = message.get('type')
        if msg_type not in valid_types:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid message type: {msg_type}",
                field='type',
                value=msg_type,
                expected=valid_types,
            ))
        
        # Validate timestamp
        timestamp = message.get('timestamp', 0)
        import time
        current_time = time.time()
        if abs(timestamp - current_time) > 300:  # 5 minutes
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Message timestamp is significantly different from current time",
                field='timestamp',
                value=timestamp,
            ))
        
        # Validate message size
        try:
            message_size = len(json.dumps(message))
            max_size = 1024 * 1024  # 1MB
            if message_size > max_size:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Message size is very large ({message_size} bytes)",
                    field='message_size',
                    value=message_size,
                    expected=max_size,
                ))
        except (TypeError, ValueError):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Message is not JSON serializable",
                field='message',
            ))
        
        return results
    
    def _validate_mission_safety(self, constraints: Dict[str, Any]) -> List[ValidationResult]:
        """Validate mission safety constraints."""
        results = []
        
        # Check altitude safety
        max_alt = constraints.get('max_altitude', 0)
        if max_alt > 120:  # Standard drone altitude limit
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Maximum altitude exceeds standard drone operating limit (120m)",
                field='max_altitude',
                value=max_alt,
                expected=120,
            ))
        
        # Check battery time reasonableness
        battery_time = constraints.get('battery_time', 0)
        if battery_time > 60:  # 1 hour
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Battery time seems optimistic for most drones",
                field='battery_time',
                value=battery_time,
            ))
        
        return results
    
    def _validate_mission_feasibility(self, constraints: Dict[str, Any]) -> List[ValidationResult]:
        """Validate mission feasibility."""
        results = []
        
        # Check if constraints are self-consistent
        max_alt = constraints.get('max_altitude', 0)
        safety_dist = constraints.get('safety_distance', 0)
        
        if safety_dist > max_alt / 2:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Safety distance is very large relative to operating altitude",
                field='safety_distance',
                value=safety_dist,
            ))
        
        return results
    
    def _validate_drone_health(self, state: Dict[str, Any]) -> List[ValidationResult]:
        """Validate drone health indicators."""
        results = []
        rules = self.validation_rules['drone']
        
        # Check battery levels
        battery = state.get('battery_percent', 100)
        if battery < rules['min_battery_critical']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Battery critically low ({battery}%)",
                field='battery_percent',
                value=battery,
            ))
        elif battery < rules['min_battery_warning']:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Battery low ({battery}%)",
                field='battery_percent',
                value=battery,
            ))
        
        # Check health score
        health = state.get('health_score', 1.0)
        if health < rules['min_health_critical']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Drone health critically low ({health:.2f})",
                field='health_score',
                value=health,
            ))
        elif health < rules['min_health_warning']:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Drone health low ({health:.2f})",
                field='health_score',
                value=health,
            ))
        
        return results
    
    def _validate_drone_safety(self, state: Dict[str, Any]) -> List[ValidationResult]:
        """Validate drone safety parameters."""
        results = []
        
        # Check for extreme positions or velocities that might indicate issues
        position = state.get('position', (0, 0, 0))
        velocity = state.get('velocity', (0, 0, 0))
        
        # Check altitude safety
        altitude = position[2]
        if altitude < 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Drone below ground level (altitude: {altitude}m)",
                field='position.z',
                value=altitude,
            ))
        
        # Check velocity safety
        total_velocity = math.sqrt(sum(v**2 for v in velocity))
        if total_velocity > 25:  # 25 m/s = 90 km/h
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"High velocity detected ({total_velocity:.1f} m/s)",
                field='velocity',
                value=total_velocity,
            ))
        
        return results
    
    def validate_all(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Perform comprehensive validation on mixed data."""
        all_results = []
        
        # Detect data type and validate accordingly
        if 'mission_id' in data and 'description' in data:
            # Mission data
            all_results.extend(self.validate_mission_constraints(data))
        
        if 'drone_id' in data and 'position' in data:
            # Drone state data
            all_results.extend(self.validate_drone_state(data))
        
        if 'num_drones' in data:
            # Fleet configuration data
            all_results.extend(self.validate_fleet_configuration(data))
        
        return all_results
    
    def check_validation_results(self, results: List[ValidationResult]) -> None:
        """Check validation results and raise exception if critical errors found."""
        critical_errors = [r for r in results if r.severity == ValidationSeverity.CRITICAL and not r.is_valid]
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR and not r.is_valid]
        
        if critical_errors or errors:
            raise ValidationError(critical_errors + errors)


# Convenience functions
def validate_mission_constraints(constraints: Dict[str, Any]) -> List[ValidationResult]:
    """Validate mission constraints."""
    validator = FleetValidator()
    return validator.validate_mission_constraints(constraints)


def validate_drone_state(state: Dict[str, Any]) -> List[ValidationResult]:
    """Validate drone state."""
    validator = FleetValidator()
    return validator.validate_drone_state(state)


def validate_fleet_configuration(config: Dict[str, Any]) -> List[ValidationResult]:
    """Validate fleet configuration."""
    validator = FleetValidator()
    return validator.validate_fleet_configuration(config)


def validate_and_raise(data: Dict[str, Any]) -> None:
    """Validate data and raise exception if validation fails."""
    validator = FleetValidator()
    results = validator.validate_all(data)
    validator.check_validation_results(results)