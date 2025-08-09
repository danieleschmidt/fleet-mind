"""Circuit breaker pattern implementation for Fleet-Mind resilience."""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager

from .logging import get_logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Time before attempting recovery (seconds)
    expected_exception: type = Exception # Exception type that counts as failure
    success_threshold: int = 3           # Successes needed to close from half-open


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = get_logger(f"circuit_breaker.{name}")
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
            
        def sync_wrapper(*args, **kwargs):
            return self.call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Call async function through circuit breaker."""
        self.total_calls += 1
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                self.logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            else:
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._record_success()
            return result
            
        except self.config.expected_exception as e:
            self._record_failure()
            raise e
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Call sync function through circuit breaker."""
        self.total_calls += 1
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                self.logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            else:
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except self.config.expected_exception as e:
            self._record_failure()
            raise e
    
    def _record_success(self):
        """Record successful call."""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0  # Reset failure count on success
    
    def _record_failure(self):
        """Record failed call."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.logger.error(f"Circuit breaker {self.name} transitioning to OPEN after {self.failure_count} failures")
            self.state = CircuitState.OPEN
            self.success_count = 0
    
    def force_open(self):
        """Manually open circuit breaker."""
        self.logger.warning(f"Circuit breaker {self.name} forced to OPEN state")
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
    
    def force_close(self):
        """Manually close circuit breaker."""
        self.logger.info(f"Circuit breaker {self.name} forced to CLOSED state")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'total_successes': self.total_successes,
            'failure_rate': self.total_failures / max(self.total_calls, 1),
            'last_failure_time': self.last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
            }
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_logger("circuit_breaker_manager")
    
    def get_breaker(self, 
                   name: str, 
                   config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
            self.logger.info(f"Created circuit breaker: {name}")
        return self.circuit_breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() 
                for name, breaker in self.circuit_breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        for breaker in self.circuit_breakers.values():
            breaker.force_close()
        self.logger.info("Reset all circuit breakers to CLOSED state")


# Global circuit breaker manager
_circuit_manager = CircuitBreakerManager()

def circuit_breaker(name: str, 
                   config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker to function."""
    def decorator(func: Callable) -> Callable:
        breaker = _circuit_manager.get_breaker(name, config)
        return breaker(func)
    return decorator


def get_circuit_breaker(name: str, 
                       config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get circuit breaker instance."""
    return _circuit_manager.get_breaker(name, config)


def get_all_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    return _circuit_manager.get_all_status()