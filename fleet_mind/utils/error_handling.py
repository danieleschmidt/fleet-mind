"""Comprehensive error handling and recovery system for Fleet-Mind."""

import asyncio
import functools
import sys
import traceback
import time
from typing import Dict, List, Any, Optional, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager, asynccontextmanager

from .logging import get_logger, FleetMindLogger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    IGNORE = "ignore"
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    SHUTDOWN = "shutdown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    component: str
    operation: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    drone_id: Optional[str] = None
    mission_id: Optional[str] = None
    user_id: Optional[str] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component,
            'operation': self.operation,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'timestamp': self.timestamp,
            'drone_id': self.drone_id,
            'mission_id': self.mission_id,
            'user_id': self.user_id,
            'stack_trace': self.stack_trace,
            'retry_count': self.retry_count,
            'recovery_strategy': self.recovery_strategy.value if self.recovery_strategy else None,
        }


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    strategy: RecoveryStrategy
    action: Callable
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_action: Optional[Callable] = None
    escalation_target: Optional[str] = None


class FleetMindError(Exception):
    """Base exception class for Fleet-Mind errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context or ErrorContext(
            component="unknown",
            operation="unknown",
            error_type=self.__class__.__name__,
            error_message=message,
            severity=ErrorSeverity.MEDIUM,
        )


class CommunicationError(FleetMindError):
    """Communication-related errors."""
    
    def __init__(self, message: str, drone_id: Optional[str] = None):
        context = ErrorContext(
            component="communication",
            operation="message_transfer",
            error_type="CommunicationError",
            error_message=message,
            severity=ErrorSeverity.HIGH,
            drone_id=drone_id,
        )
        super().__init__(message, context)


class PlanningError(FleetMindError):
    """Mission planning related errors."""
    
    def __init__(self, message: str, mission_id: Optional[str] = None):
        context = ErrorContext(
            component="planning",
            operation="mission_planning",
            error_type="PlanningError",
            error_message=message,
            severity=ErrorSeverity.HIGH,
            mission_id=mission_id,
        )
        super().__init__(message, context)


class DroneError(FleetMindError):
    """Drone-specific errors."""
    
    def __init__(self, message: str, drone_id: str, severity: ErrorSeverity = ErrorSeverity.HIGH):
        context = ErrorContext(
            component="drone",
            operation="drone_operation",
            error_type="DroneError",
            error_message=message,
            severity=severity,
            drone_id=drone_id,
        )
        super().__init__(message, context)


class ValidationError(FleetMindError):
    """Data validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        context = ErrorContext(
            component="validation",
            operation="data_validation",
            error_type="ValidationError",
            error_message=message,
            severity=ErrorSeverity.MEDIUM,
        )
        super().__init__(message, context)


class SecurityError(FleetMindError):
    """Security-related errors."""
    
    def __init__(self, message: str, user_id: Optional[str] = None):
        context = ErrorContext(
            component="security",
            operation="security_check",
            error_type="SecurityError",
            error_message=message,
            severity=ErrorSeverity.CRITICAL,
            user_id=user_id,
        )
        super().__init__(message, context)


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.logger = get_logger("error_handler", component="error_handling")
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Recovery strategies
        self._initialize_recovery_strategies()
        
        # Global exception handler
        self._setup_global_exception_handler()

    def _initialize_recovery_strategies(self) -> None:
        """Initialize default recovery strategies."""
        self.recovery_actions.update({
            'CommunicationError': RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action=self._retry_communication,
                max_retries=3,
                retry_delay=2.0,
                fallback_action=self._fallback_communication,
            ),
            'PlanningError': RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action=self._fallback_planning,
                fallback_action=self._emergency_stop,
            ),
            'DroneError': RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                action=self._handle_drone_error,
                escalation_target="fleet_supervisor",
            ),
            'ValidationError': RecoveryAction(
                strategy=RecoveryStrategy.IGNORE,
                action=self._sanitize_input,
            ),
            'SecurityError': RecoveryAction(
                strategy=RecoveryStrategy.SHUTDOWN,
                action=self._security_shutdown,
                escalation_target="security_team",
            ),
        })

    def _setup_global_exception_handler(self) -> None:
        """Set up global exception handler."""
        def global_exception_handler(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, FleetMindError):
                self.handle_error(exc_value)
            else:
                # Handle unexpected exceptions
                context = ErrorContext(
                    component="system",
                    operation="unknown",
                    error_type=exc_type.__name__,
                    error_message=str(exc_value),
                    severity=ErrorSeverity.CRITICAL,
                    stack_trace=''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
                )
                self.handle_error(FleetMindError(str(exc_value), context))
        
        sys.excepthook = global_exception_handler

    def handle_error(self, error: Union[Exception, ErrorContext]) -> Optional[Any]:
        """Handle error with appropriate recovery strategy.
        
        Args:
            error: Error to handle
            
        Returns:
            Recovery result if applicable
        """
        # Convert exception to ErrorContext if needed
        if isinstance(error, Exception):
            if isinstance(error, FleetMindError):
                context = error.context
            else:
                context = ErrorContext(
                    component="unknown",
                    operation="unknown",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    severity=ErrorSeverity.MEDIUM,
                    stack_trace=traceback.format_exc(),
                )
        else:
            context = error
        
        # Log error
        self._log_error(context)
        
        # Track error
        self._track_error(context)
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(context.component, context.operation):
            self.logger.warning(
                f"Circuit breaker open for {context.component}.{context.operation}",
                component=context.component,
                operation=context.operation,
            )
            return None
        
        # Execute recovery strategy
        return self._execute_recovery(context)

    def _log_error(self, context: ErrorContext) -> None:
        """Log error with appropriate level."""
        log_data = {
            'component': context.component,
            'operation': context.operation,
            'error_type': context.error_type,
            'severity': context.severity.value,
            'drone_id': context.drone_id,
            'mission_id': context.mission_id,
            'user_id': context.user_id,
            'retry_count': context.retry_count,
        }
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(context.error_message, **log_data)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(context.error_message, **log_data)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(context.error_message, **log_data)
        else:
            self.logger.info(context.error_message, **log_data)

    def _track_error(self, context: ErrorContext) -> None:
        """Track error for analysis and circuit breaker logic."""
        # Add to history
        self.error_history.append(context)
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Update error counts
        error_key = f"{context.component}.{context.operation}.{context.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Update circuit breaker
        self._update_circuit_breaker(context)

    def _update_circuit_breaker(self, context: ErrorContext) -> None:
        """Update circuit breaker state based on error."""
        breaker_key = f"{context.component}.{context.operation}"
        
        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure_time': None,
                'threshold': 5,
                'timeout': 60,  # seconds
            }
        
        breaker = self.circuit_breakers[breaker_key]
        breaker['failure_count'] += 1
        breaker['last_failure_time'] = time.time()
        
        # Open circuit breaker if threshold exceeded
        if breaker['failure_count'] >= breaker['threshold']:
            breaker['state'] = 'open'
            self.logger.warning(
                f"Circuit breaker opened for {breaker_key}",
                component=context.component,
                operation=context.operation,
                failure_count=breaker['failure_count'],
            )

    def _is_circuit_breaker_open(self, component: str, operation: str) -> bool:
        """Check if circuit breaker is open."""
        breaker_key = f"{component}.{operation}"
        
        if breaker_key not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[breaker_key]
        
        if breaker['state'] == 'closed':
            return False
        
        # Check if timeout has passed (half-open state)
        if time.time() - breaker['last_failure_time'] > breaker['timeout']:
            breaker['state'] = 'half-open'
            breaker['failure_count'] = 0
            self.logger.info(
                f"Circuit breaker half-open for {breaker_key}",
                component=component,
                operation=operation,
            )
            return False
        
        return breaker['state'] == 'open'

    def _execute_recovery(self, context: ErrorContext) -> Optional[Any]:
        """Execute recovery strategy for error."""
        recovery_action = self.recovery_actions.get(context.error_type)
        
        if not recovery_action:
            # No specific recovery action, use default
            self.logger.warning(f"No recovery action for {context.error_type}")
            return None
        
        context.recovery_strategy = recovery_action.strategy
        
        try:
            if recovery_action.strategy == RecoveryStrategy.RETRY:
                return self._retry_operation(context, recovery_action)
            elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                return self._fallback_operation(context, recovery_action)
            elif recovery_action.strategy == RecoveryStrategy.ESCALATE:
                return self._escalate_error(context, recovery_action)
            elif recovery_action.strategy == RecoveryStrategy.SHUTDOWN:
                return self._shutdown_operation(context, recovery_action)
            else:  # IGNORE
                self.logger.info(f"Ignoring error: {context.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"Recovery action failed: {e}")
            return None

    def _retry_operation(self, context: ErrorContext, recovery_action: RecoveryAction) -> Optional[Any]:
        """Retry failed operation."""
        if context.retry_count >= recovery_action.max_retries:
            self.logger.warning(
                f"Max retries exceeded for {context.operation}",
                retry_count=context.retry_count,
                max_retries=recovery_action.max_retries,
            )
            
            # Try fallback if available
            if recovery_action.fallback_action:
                return recovery_action.fallback_action(context)
            
            return None
        
        context.retry_count += 1
        
        self.logger.info(
            f"Retrying operation {context.operation} (attempt {context.retry_count})",
            component=context.component,
            operation=context.operation,
            retry_count=context.retry_count,
        )
        
        # Wait before retry
        time.sleep(recovery_action.retry_delay * context.retry_count)  # Exponential backoff
        
        try:
            return recovery_action.action(context)
        except Exception as e:
            # Recursive call to handle retry failure
            new_context = ErrorContext(
                component=context.component,
                operation=context.operation,
                error_type=type(e).__name__,
                error_message=str(e),
                severity=context.severity,
                drone_id=context.drone_id,
                mission_id=context.mission_id,
                user_id=context.user_id,
                retry_count=context.retry_count,
            )
            return self._retry_operation(new_context, recovery_action)

    def _fallback_operation(self, context: ErrorContext, recovery_action: RecoveryAction) -> Optional[Any]:
        """Execute fallback operation."""
        self.logger.info(
            f"Executing fallback for {context.operation}",
            component=context.component,
            operation=context.operation,
        )
        
        try:
            return recovery_action.action(context)
        except Exception as e:
            self.logger.error(f"Fallback operation failed: {e}")
            
            # Try secondary fallback
            if recovery_action.fallback_action:
                return recovery_action.fallback_action(context)
            
            return None

    def _escalate_error(self, context: ErrorContext, recovery_action: RecoveryAction) -> Optional[Any]:
        """Escalate error to higher authority."""
        self.logger.critical(
            f"Escalating error: {context.error_message}",
            component=context.component,
            operation=context.operation,
            escalation_target=recovery_action.escalation_target,
        )
        
        # In production, this would notify operators, send alerts, etc.
        # For now, execute the recovery action
        try:
            return recovery_action.action(context)
        except Exception as e:
            self.logger.error(f"Escalation action failed: {e}")
            return None

    def _shutdown_operation(self, context: ErrorContext, recovery_action: RecoveryAction) -> Optional[Any]:
        """Shutdown operation or component."""
        self.logger.critical(
            f"Shutting down due to critical error: {context.error_message}",
            component=context.component,
            operation=context.operation,
        )
        
        try:
            return recovery_action.action(context)
        except Exception as e:
            self.logger.error(f"Shutdown action failed: {e}")
            return None

    # Default recovery actions
    def _retry_communication(self, context: ErrorContext) -> bool:
        """Retry communication operation."""
        # Placeholder for communication retry logic
        self.logger.info(f"Retrying communication for drone {context.drone_id}")
        return True

    def _fallback_communication(self, context: ErrorContext) -> bool:
        """Fallback communication method."""
        self.logger.info(f"Using fallback communication for drone {context.drone_id}")
        return True

    def _fallback_planning(self, context: ErrorContext) -> bool:
        """Fallback planning method."""
        self.logger.info(f"Using fallback planning for mission {context.mission_id}")
        return True

    def _handle_drone_error(self, context: ErrorContext) -> bool:
        """Handle drone-specific error."""
        self.logger.warning(f"Handling drone error for {context.drone_id}")
        return True

    def _sanitize_input(self, context: ErrorContext) -> bool:
        """Sanitize invalid input."""
        self.logger.info("Sanitizing input data")
        return True

    def _emergency_stop(self, context: ErrorContext) -> bool:
        """Execute emergency stop."""
        self.logger.critical("Executing emergency stop")
        return True

    def _security_shutdown(self, context: ErrorContext) -> bool:
        """Security-related shutdown."""
        self.logger.critical(f"Security shutdown triggered by {context.user_id}")
        return True

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics."""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'most_common_errors': [],
                'circuit_breaker_states': {},
                'health_score': 1.0,
            }
        
        # Calculate error rates
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        error_rate = len(recent_errors) / 3600  # Errors per second
        
        # Most common errors
        error_types = {}
        for error in self.error_history:
            key = f"{error.component}.{error.error_type}"
            error_types[key] = error_types.get(key, 0) + 1
        
        most_common = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Circuit breaker states
        breaker_states = {k: v['state'] for k, v in self.circuit_breakers.items()}
        
        # Health score (simple calculation)
        critical_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL])
        high_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.HIGH])
        
        health_score = max(0.0, 1.0 - (critical_errors * 0.1 + high_errors * 0.05))
        
        return {
            'total_errors': total_errors,
            'recent_errors': len(recent_errors),
            'error_rate': error_rate,
            'most_common_errors': most_common,
            'circuit_breaker_states': breaker_states,
            'health_score': health_score,
            'error_counts': dict(self.error_counts),
        }


# Global error handler instance
_global_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


# Decorators for error handling
def handle_errors(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    raise_on_error: bool = False,
):
    """Decorator for automatic error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=severity,
                    stack_trace=traceback.format_exc(),
                )
                
                error_handler = get_error_handler()
                result = error_handler.handle_error(context)
                
                if raise_on_error:
                    raise FleetMindError(str(e), context)
                
                return result
        
        return wrapper
    return decorator


def handle_async_errors(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    raise_on_error: bool = False,
):
    """Decorator for automatic async error handling."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    component=component,
                    operation=operation,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=severity,
                    stack_trace=traceback.format_exc(),
                )
                
                error_handler = get_error_handler()
                result = error_handler.handle_error(context)
                
                if raise_on_error:
                    raise FleetMindError(str(e), context)
                
                return result
        
        return wrapper
    return decorator


# Context managers for error handling
@contextmanager
def error_context(component: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        context = ErrorContext(
            component=component,
            operation=operation,
            error_type=type(e).__name__,
            error_message=str(e),
            severity=severity,
            stack_trace=traceback.format_exc(),
        )
        
        error_handler = get_error_handler()
        error_handler.handle_error(context)


@asynccontextmanager
async def async_error_context(component: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Async context manager for error handling."""
    try:
        yield
    except Exception as e:
        context = ErrorContext(
            component=component,
            operation=operation,
            error_type=type(e).__name__,
            error_message=str(e),
            severity=severity,
            stack_trace=traceback.format_exc(),
        )
        
        error_handler = get_error_handler()
        error_handler.handle_error(context)