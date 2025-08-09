"""Comprehensive logging system for Fleet-Mind."""

import logging
import logging.handlers
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

# Structlog import with fallback handling
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    # Fallback implementation for when structlog is not available
    class MockBoundLogger:
        """Mock bound logger that chains bind calls."""
        def __init__(self, logger, bindings=None):
            self._logger = logger
            self._bindings = bindings or {}
        
        def bind(self, **kwargs):
            """Return new mock logger with additional bindings."""
            new_bindings = self._bindings.copy()
            new_bindings.update(kwargs)
            return MockBoundLogger(self._logger, new_bindings)
        
        def debug(self, message, **kwargs):
            self._logger.debug(f"{message} {self._format_context(kwargs)}")
        
        def info(self, message, **kwargs):
            self._logger.info(f"{message} {self._format_context(kwargs)}")
            
        def warning(self, message, **kwargs):
            self._logger.warning(f"{message} {self._format_context(kwargs)}")
            
        def error(self, message, **kwargs):
            self._logger.error(f"{message} {self._format_context(kwargs)}")
            
        def critical(self, message, **kwargs):
            self._logger.critical(f"{message} {self._format_context(kwargs)}")
        
        def _format_context(self, extra_kwargs=None):
            """Format context for logging."""
            context = self._bindings.copy()
            if extra_kwargs:
                context.update(extra_kwargs)
            if context:
                context_str = " ".join(f"{k}={v}" for k, v in context.items())
                return f"[{context_str}]"
            return ""
    
    class structlog:
        @staticmethod
        def get_logger(name):
            std_logger = logging.getLogger(name)
            return MockBoundLogger(std_logger)
        
        @staticmethod
        def configure(*args, **kwargs):
            pass
        
        class stdlib:
            @staticmethod
            def filter_by_level(*args, **kwargs): pass
            @staticmethod
            def add_logger_name(*args, **kwargs): pass
            @staticmethod
            def add_log_level(*args, **kwargs): pass
            class PositionalArgumentsFormatter: pass
            class LoggerFactory: pass
            class BoundLogger: pass
        
        class processors:
            @staticmethod
            def TimeStamper(*args, **kwargs): pass
            @staticmethod
            def StackInfoRenderer(*args, **kwargs): pass
            @staticmethod
            def format_exc_info(*args, **kwargs): pass
            @staticmethod
            def UnicodeDecoder(*args, **kwargs): pass
            @staticmethod
            def JSONRenderer(*args, **kwargs): pass
        
        class dev:
            @staticmethod
            def ConsoleRenderer(*args, **kwargs): pass
    
    STRUCTLOG_AVAILABLE = False
    print("Warning: structlog not available, using standard logging")


@dataclass
class LogContext:
    """Structured log context."""
    component: str
    drone_id: Optional[str] = None
    mission_id: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FleetMindLogger:
    """Enhanced logger for Fleet-Mind with structured logging."""
    
    def __init__(self, name: str, context: Optional[LogContext] = None):
        self.name = name
        self.context = context or LogContext(component=name)
        
        # Get structured logger
        self.logger = structlog.get_logger(name)
        
        # Bind context
        self.logger = self.logger.bind(
            component=self.context.component,
            timestamp=self.context.timestamp,
        )
        
        if self.context.drone_id:
            self.logger = self.logger.bind(drone_id=self.context.drone_id)
        if self.context.mission_id:
            self.logger = self.logger.bind(mission_id=self.context.mission_id)
        if self.context.operation:
            self.logger = self.logger.bind(operation=self.context.operation)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with optional exception."""
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
            })
        self.logger.error(message, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
            })
        self.logger.critical(message, **kwargs)

    def log_performance(self, operation: str, duration_ms: float, **kwargs) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"Performance: {operation}",
            operation=operation,
            duration_ms=duration_ms,
            performance=True,
            **kwargs
        )

    def log_security_event(self, event_type: str, severity: str, **kwargs) -> None:
        """Log security-related events."""
        self.logger.warning(
            f"Security Event: {event_type}",
            event_type=event_type,
            severity=severity,
            security=True,
            **kwargs
        )

    def log_mission_event(self, event: str, mission_id: str, **kwargs) -> None:
        """Log mission-related events."""
        self.logger.info(
            f"Mission Event: {event}",
            event=event,
            mission_id=mission_id,
            mission=True,
            **kwargs
        )

    def log_drone_event(self, event: str, drone_id: str, **kwargs) -> None:
        """Log drone-related events."""
        self.logger.info(
            f"Drone Event: {event}",
            event=event,
            drone_id=drone_id,
            drone=True,
            **kwargs
        )

    def bind(self, **kwargs) -> 'FleetMindLogger':
        """Create new logger with additional context."""
        new_logger = FleetMindLogger(self.name, self.context)
        new_logger.logger = self.logger.bind(**kwargs)
        return new_logger


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
        }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """Set up comprehensive logging for Fleet-Mind.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to ~/.fleet-mind/logs)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Enable JSON structured logging
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup log files to keep
    """
    
    # Set up log directory
    if log_dir is None:
        log_dir = Path.home() / ".fleet-mind" / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        if enable_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Main log file
        main_log_file = log_dir / "fleet_mind.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        
        if enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(file_handler)
        
        # Error log file (ERROR and CRITICAL only)
        error_log_file = log_dir / "fleet_mind_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        if enable_json:
            error_handler.setFormatter(JSONFormatter())
        else:
            error_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(error_handler)
        
        # Performance log file
        perf_log_file = log_dir / "fleet_mind_performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        
        # Custom filter for performance logs
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                return getattr(record, 'performance', False)
        
        perf_handler.addFilter(PerformanceFilter())
        
        if enable_json:
            perf_handler.setFormatter(JSONFormatter())
        else:
            perf_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(perf_handler)
        
        # Security log file
        security_log_file = log_dir / "fleet_mind_security.log"
        security_handler = logging.handlers.RotatingFileHandler(
            security_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        
        # Custom filter for security logs
        class SecurityFilter(logging.Filter):
            def filter(self, record):
                return getattr(record, 'security', False)
        
        security_handler.addFilter(SecurityFilter())
        
        if enable_json:
            security_handler.setFormatter(JSONFormatter())
        else:
            security_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(security_handler)


def get_logger(
    name: str,
    component: Optional[str] = None,
    drone_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    operation: Optional[str] = None,
) -> FleetMindLogger:
    """Get a configured Fleet-Mind logger.
    
    Args:
        name: Logger name
        component: Component name for context
        drone_id: Drone ID for context
        mission_id: Mission ID for context
        operation: Operation name for context
    
    Returns:
        Configured FleetMindLogger instance
    """
    context = LogContext(
        component=component or name,
        drone_id=drone_id,
        mission_id=mission_id,
        operation=operation,
    )
    
    return FleetMindLogger(name, context)


# Performance timing decorator
def log_performance(logger: FleetMindLogger, operation: str):
    """Decorator to log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, success=False, error=str(e))
                raise
        return wrapper
    return decorator


# Async performance timing decorator
def log_async_performance(logger: FleetMindLogger, operation: str):
    """Decorator to log async function performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.log_performance(operation, duration_ms, success=False, error=str(e))
                raise
        return wrapper
    return decorator


# Context manager for logging operations
class LoggedOperation:
    """Context manager for logging operations with automatic timing."""
    
    def __init__(self, logger: FleetMindLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.log_performance(
                self.operation,
                duration_ms,
                success=True,
                **self.context
            )
            self.logger.info(f"Completed {self.operation}", **self.context)
        else:
            self.logger.log_performance(
                self.operation,
                duration_ms,
                success=False,
                error=str(exc_val),
                **self.context
            )
            self.logger.error(
                f"Failed {self.operation}",
                error=exc_val,
                **self.context
            )


# Audit logging for security-sensitive operations
class AuditLogger:
    """Specialized logger for audit trails."""
    
    def __init__(self, logger: FleetMindLogger):
        self.logger = logger
    
    def log_authentication(self, user: str, success: bool, **kwargs):
        """Log authentication events."""
        self.logger.log_security_event(
            "authentication",
            "high" if not success else "info",
            user=user,
            success=success,
            **kwargs
        )
    
    def log_authorization(self, user: str, resource: str, action: str, granted: bool, **kwargs):
        """Log authorization events."""
        self.logger.log_security_event(
            "authorization",
            "high" if not granted else "info",
            user=user,
            resource=resource,
            action=action,
            granted=granted,
            **kwargs
        )
    
    def log_data_access(self, user: str, resource: str, action: str, **kwargs):
        """Log data access events."""
        self.logger.log_security_event(
            "data_access",
            "medium",
            user=user,
            resource=resource,
            action=action,
            **kwargs
        )
    
    def log_configuration_change(self, user: str, component: str, changes: Dict[str, Any], **kwargs):
        """Log configuration changes."""
        self.logger.log_security_event(
            "configuration_change",
            "high",
            user=user,
            component=component,
            changes=changes,
            **kwargs
        )
    
    def log_emergency_action(self, user: str, action: str, reason: str, **kwargs):
        """Log emergency actions."""
        self.logger.log_security_event(
            "emergency_action",
            "critical",
            user=user,
            action=action,
            reason=reason,
            **kwargs
        )