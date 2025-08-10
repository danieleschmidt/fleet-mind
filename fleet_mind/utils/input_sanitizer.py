"""Input sanitization and validation for Fleet-Mind security."""

import re
import html
import json
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum

from .logging import get_logger


class SanitizationLevel(Enum):
    """Sanitization security levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class SanitizationConfig:
    """Configuration for input sanitization."""
    level: SanitizationLevel = SanitizationLevel.STRICT
    max_string_length: int = 10000
    max_list_items: int = 1000
    max_dict_keys: int = 100
    max_nesting_depth: int = 10
    allowed_html_tags: Set[str] = None
    blocked_patterns: List[str] = None
    
    def __post_init__(self):
        if self.allowed_html_tags is None:
            self.allowed_html_tags = set()  # No HTML allowed by default
        
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'<script[^>]*>.*?</script>',  # Script tags
                r'javascript:',                # JavaScript URLs
                r'vbscript:',                 # VBScript URLs
                r'on\w+\s*=',                 # Event handlers
                r'eval\s*\(',                 # eval() calls
                r'exec\s*\(',                 # exec() calls
                r'import\s+',                 # Import statements
                r'__\w+__',                   # Python dunder methods
                r'\.{2,}',                    # Path traversal attempts
                r'[;&|`$()]',                 # Shell metacharacters
            ]


class InputSanitizer:
    """Comprehensive input sanitization for security."""
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        self.config = config or SanitizationConfig()
        self.logger = get_logger("input_sanitizer")
        
        # Compile regex patterns for performance
        self.blocked_regex = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                             for pattern in self.config.blocked_patterns]
    
    def sanitize(self, data: Any, context: str = "unknown") -> Any:
        """Sanitize input data based on configuration."""
        try:
            return self._sanitize_recursive(data, 0, context)
        except Exception as e:
            self.logger.error(f"Sanitization failed for context '{context}': {e}")
            raise ValueError(f"Input sanitization failed: {e}")
    
    def _sanitize_recursive(self, data: Any, depth: int, context: str) -> Any:
        """Recursively sanitize data structure."""
        if depth > self.config.max_nesting_depth:
            raise ValueError(f"Maximum nesting depth exceeded: {depth}")
        
        if data is None:
            return None
        
        if isinstance(data, bool):
            return data
        
        if isinstance(data, (int, float)):
            return self._sanitize_number(data)
        
        if isinstance(data, str):
            return self._sanitize_string(data, context)
        
        if isinstance(data, list):
            return self._sanitize_list(data, depth, context)
        
        if isinstance(data, dict):
            return self._sanitize_dict(data, depth, context)
        
        if isinstance(data, tuple):
            sanitized_list = self._sanitize_list(list(data), depth, context)
            return tuple(sanitized_list)
        
        if isinstance(data, set):
            sanitized_list = self._sanitize_list(list(data), depth, context)
            return set(sanitized_list)
        
        # For other types, convert to string and sanitize
        self.logger.warning(f"Sanitizing unknown type {type(data)} in context '{context}'")
        return self._sanitize_string(str(data), context)
    
    def _sanitize_string(self, text: str, context: str) -> str:
        """Sanitize string input."""
        if len(text) > self.config.max_string_length:
            self.logger.warning(f"String truncated in context '{context}': {len(text)} chars")
            text = text[:self.config.max_string_length]
        
        # Check for blocked patterns
        for pattern in self.blocked_regex:
            if pattern.search(text):
                threat = pattern.pattern
                self.logger.warning(f"Blocked pattern detected in context '{context}': {threat}")
                if self.config.level == SanitizationLevel.PARANOID:
                    raise ValueError(f"Blocked content detected: {threat}")
                else:
                    # Remove the pattern
                    text = pattern.sub('', text)
        
        # HTML escape if needed
        if self.config.level in [SanitizationLevel.STRICT, SanitizationLevel.PARANOID]:
            text = html.escape(text)
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        return text.strip()
    
    def _sanitize_number(self, num: Union[int, float]) -> Union[int, float]:
        """Sanitize numeric input."""
        if isinstance(num, float):
            if not (-1e10 < num < 1e10):  # Reasonable bounds
                self.logger.warning(f"Float value out of reasonable bounds: {num}")
                return max(-1e10, min(1e10, num))
        elif isinstance(num, int):
            if not (-2**31 < num < 2**31):  # 32-bit int bounds
                self.logger.warning(f"Integer value out of reasonable bounds: {num}")
                return max(-2**31, min(2**31-1, num))
        
        return num
    
    def _sanitize_list(self, data: list, depth: int, context: str) -> list:
        """Sanitize list input."""
        if len(data) > self.config.max_list_items:
            self.logger.warning(f"List truncated in context '{context}': {len(data)} items")
            data = data[:self.config.max_list_items]
        
        return [self._sanitize_recursive(item, depth + 1, f"{context}[{i}]") 
                for i, item in enumerate(data)]
    
    def _sanitize_dict(self, data: dict, depth: int, context: str) -> dict:
        """Sanitize dictionary input."""
        if len(data) > self.config.max_dict_keys:
            self.logger.warning(f"Dict truncated in context '{context}': {len(data)} keys")
            # Keep first N items
            items = list(data.items())[:self.config.max_dict_keys]
            data = dict(items)
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            clean_key = self._sanitize_string(str(key), f"{context}.key")
            # Sanitize value
            clean_value = self._sanitize_recursive(value, depth + 1, f"{context}.{clean_key}")
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def validate_json(self, json_string: str) -> Dict[str, Any]:
        """Validate and sanitize JSON input."""
        try:
            # Parse JSON
            data = json.loads(json_string)
            
            # Sanitize the parsed data
            sanitized = self.sanitize(data, "json_input")
            
            return sanitized
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
    
    def validate_command(self, command: str) -> str:
        """Validate and sanitize command input."""
        # Basic command validation
        command = self._sanitize_string(command, "command")
        
        # Check for shell injection attempts
        dangerous_chars = ['|', '&', ';', '`', '$', '(', ')']
        if any(char in command for char in dangerous_chars):
            if self.config.level == SanitizationLevel.PARANOID:
                raise ValueError("Potentially dangerous command characters detected")
            else:
                # Remove dangerous characters
                for char in dangerous_chars:
                    command = command.replace(char, '')
                self.logger.warning("Removed potentially dangerous command characters")
        
        return command
    
    def validate_file_path(self, path: str) -> str:
        """Validate and sanitize file path."""
        path = self._sanitize_string(path, "file_path")
        
        # Check for path traversal
        if '..' in path or path.startswith('/'):
            if self.config.level == SanitizationLevel.PARANOID:
                raise ValueError("Path traversal attempt detected")
            else:
                # Remove path traversal attempts
                path = path.replace('..', '').lstrip('/')
                self.logger.warning("Removed path traversal attempt")
        
        # Limit to reasonable characters
        allowed_chars = re.compile(r'^[a-zA-Z0-9._/-]+$')
        if not allowed_chars.match(path):
            raise ValueError("File path contains invalid characters")
        
        return path
    
    def validate_coordinates(self, lat: float, lon: float, alt: Optional[float] = None) -> tuple:
        """Validate geographic coordinates."""
        # Latitude bounds
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Invalid latitude: {lat}")
        
        # Longitude bounds  
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Invalid longitude: {lon}")
        
        # Altitude bounds (if provided)
        if alt is not None:
            if not (-1000.0 <= alt <= 50000.0):  # From Dead Sea to upper atmosphere
                raise ValueError(f"Invalid altitude: {alt}")
            return (lat, lon, alt)
        
        return (lat, lon)
    
    def get_sanitization_report(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        return {
            'config': {
                'level': self.config.level.value,
                'max_string_length': self.config.max_string_length,
                'max_list_items': self.config.max_list_items,
                'max_dict_keys': self.config.max_dict_keys,
                'max_nesting_depth': self.config.max_nesting_depth,
                'blocked_patterns_count': len(self.config.blocked_patterns),
            },
            'status': 'active',
            'last_updated': 'system_start'
        }


# Global sanitizer instance
_default_sanitizer = InputSanitizer()

def sanitize_input(data: Any, 
                  context: str = "unknown",
                  config: Optional[SanitizationConfig] = None) -> Any:
    """Convenience function to sanitize input."""
    if config:
        sanitizer = InputSanitizer(config)
    else:
        sanitizer = _default_sanitizer
    
    return sanitizer.sanitize(data, context)


def validate_mission_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate mission input data specifically."""
    config = SanitizationConfig(
        level=SanitizationLevel.STRICT,
        max_string_length=5000,  # Allow longer mission descriptions
        max_dict_keys=50,
    )
    sanitizer = InputSanitizer(config)
    return sanitizer.sanitize(data, "mission_input")


def validate_drone_command(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate drone command data specifically."""
    config = SanitizationConfig(
        level=SanitizationLevel.PARANOID,  # Highest security for commands
        max_string_length=1000,
        max_dict_keys=20,
    )
    sanitizer = InputSanitizer(config)
    return sanitizer.sanitize(data, "drone_command")


def validate_communication_message(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate communication message data."""
    config = SanitizationConfig(
        level=SanitizationLevel.STRICT,
        max_string_length=50000,  # Allow larger message payloads
        max_dict_keys=100,
    )
    sanitizer = InputSanitizer(config)
    return sanitizer.sanitize(data, "communication_message")


def validate_sensor_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sensor data input."""
    config = SanitizationConfig(
        level=SanitizationLevel.BASIC,  # Sensor data is typically numeric
        max_dict_keys=200,  # Many sensor readings
    )
    sanitizer = InputSanitizer(config)
    return sanitizer.sanitize(data, "sensor_data")


def validate_configuration_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration data."""
    config = SanitizationConfig(
        level=SanitizationLevel.STRICT,
        max_string_length=2000,
        max_dict_keys=100,
    )
    sanitizer = InputSanitizer(config)
    return sanitizer.sanitize(data, "configuration_data")