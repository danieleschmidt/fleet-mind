"""Advanced logging system with structured logging, metrics, and monitoring for Fleet-Mind."""

import asyncio
import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TextIO
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
import threading
from queue import Queue, Empty

# Optional dependencies with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


class LogLevel(Enum):
    """Extended log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60


class LogCategory(Enum):
    """Log categories for classification."""
    SYSTEM = "system"
    COMMUNICATION = "communication"
    COORDINATION = "coordination"
    PLANNING = "planning"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    MISSION = "mission"
    ERROR = "error"
    USER = "user"


@dataclass
class LogMetrics:
    """Log metrics tracking."""
    total_logs: int = 0
    logs_by_level: Dict[str, int] = field(default_factory=dict)
    logs_by_component: Dict[str, int] = field(default_factory=dict)
    logs_by_category: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    warning_rate: float = 0.0
    avg_log_size: float = 0.0
    peak_logs_per_second: float = 0.0
    start_time: float = field(default_factory=time.time)


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    message: str
    component: str
    category: LogCategory
    correlation_id: Optional[str] = None
    drone_id: Optional[str] = None
    mission_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp_iso'] = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()
        data['category'] = self.category.value
        return data


class FleetMindLogger:
    """Advanced logger with structured logging and metrics."""
    
    def __init__(
        self,
        name: str,
        component: str = "unknown",
        log_level: LogLevel = LogLevel.INFO,
        log_format: str = "json",
        output_file: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
        enable_metrics: bool = True,
        enable_async_logging: bool = True,
        buffer_size: int = 1000,
    ):
        """Initialize advanced logger.
        
        Args:
            name: Logger name
            component: Component name for categorization
            log_level: Minimum log level
            log_format: Log format (json, text, structured)
            output_file: Output file path (None for console only)
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            enable_metrics: Enable logging metrics collection
            enable_async_logging: Enable asynchronous logging
            buffer_size: Buffer size for async logging
        """
        self.name = name
        self.component = component
        self.log_level = log_level
        self.log_format = log_format
        self.output_file = output_file
        self.enable_metrics = enable_metrics
        self.enable_async_logging = enable_async_logging
        
        # Logging infrastructure
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.value)
        self._setup_handlers(max_file_size, backup_count)
        
        # Metrics collection
        self.metrics = LogMetrics() if enable_metrics else None
        self._log_buffer: List[LogEntry] = []
        self._log_times: List[float] = []
        
        # Async logging
        if enable_async_logging:
            self._log_queue: Queue = Queue(maxsize=buffer_size)
            self._stop_async_logging = threading.Event()
            self._async_thread = threading.Thread(
                target=self._async_logging_worker,
                daemon=True
            )
            self._async_thread.start()
        else:
            self._log_queue = None
            self._async_thread = None
        
        # Context tracking
        self._context_stack: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._last_metrics_update = time.time()
        self._metrics_update_interval = 60.0  # 1 minute
        
        self.info(f"Logger {name} initialized", category=LogCategory.SYSTEM)
    
    def _setup_handlers(self, max_file_size: int, backup_count: int):
        """Setup log handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level.value)
        
        if self.log_format == "json" and JSON_LOGGER_AVAILABLE:
            console_formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.output_file:
            from logging.handlers import RotatingFileHandler
            
            # Ensure directory exists
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.output_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(self.log_level.value)
            
            if self.log_format == "json":
                if JSON_LOGGER_AVAILABLE:
                    file_formatter = jsonlogger.JsonFormatter()
                else:
                    # Fallback to manual JSON formatting
                    file_formatter = logging.Formatter(
                        '{"timestamp": "%(asctime)s", "name": "%(name)s", '
                        '"level": "%(levelname)s", "message": "%(message)s"}'
                    )
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _async_logging_worker(self):
        """Worker thread for async logging."""
        while not self._stop_async_logging.is_set():
            try:
                # Get log entry from queue with timeout
                log_entry = self._log_queue.get(timeout=1.0)
                self._write_log_entry(log_entry)
                self._log_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                # Fallback to sync logging for this error
                print(f"Async logging error: {e}", file=sys.stderr)
    
    def _write_log_entry(self, entry: LogEntry):
        """Write log entry to handlers."""
        # Create log record
        record = logging.LogRecord(
            name=self.logger.name,
            level=getattr(logging, entry.level.upper(), logging.INFO),
            pathname="",
            lineno=0,
            msg=self._format_message(entry),
            args=(),
            exc_info=None
        )
        
        # Add custom attributes
        for key, value in entry.extra_data.items():
            setattr(record, key, value)
        
        # Process through handlers
        self.logger.handle(record)
        
        # Update metrics
        if self.metrics:
            self._update_metrics(entry)
    
    def _format_message(self, entry: LogEntry) -> str:
        """Format log message based on format type."""
        if self.log_format == "json":
            return json.dumps(entry.to_dict())
        elif self.log_format == "structured":
            return (f"[{entry.component}:{entry.category.value}] "
                   f"{entry.message} "
                   f"(drone={entry.drone_id}, mission={entry.mission_id})")
        else:
            return entry.message
    
    def _create_log_entry(
        self,
        level: str,
        message: str,
        category: LogCategory,
        **kwargs
    ) -> LogEntry:
        """Create structured log entry."""
        # Get current context
        context = self._get_current_context()
        
        # Create entry
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            component=self.component,
            category=category,
            correlation_id=context.get('correlation_id'),
            drone_id=kwargs.get('drone_id', context.get('drone_id')),
            mission_id=kwargs.get('mission_id', context.get('mission_id')),
            user_id=kwargs.get('user_id', context.get('user_id')),
            session_id=kwargs.get('session_id', context.get('session_id')),
            thread_id=str(threading.current_thread().ident),
            process_id=os.getpid() if hasattr(__builtins__, 'os') else None,
            extra_data={k: v for k, v in kwargs.items() 
                       if k not in ['drone_id', 'mission_id', 'user_id', 'session_id']}
        )
        
        return entry
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        context = {}
        for ctx in self._context_stack:
            context.update(ctx)
        return context
    
    def _log(self, level: str, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Internal logging method."""
        # Check log level
        level_value = getattr(LogLevel, level.upper(), LogLevel.INFO).value
        if level_value < self.log_level.value:
            return
        
        # Create log entry
        entry = self._create_log_entry(level, message, category, **kwargs)
        
        # Log asynchronously or synchronously
        if self.enable_async_logging and self._log_queue:
            try:
                self._log_queue.put_nowait(entry)
            except:
                # Queue full, fall back to sync logging
                self._write_log_entry(entry)
        else:
            self._write_log_entry(entry)
        
        # Store in buffer for metrics
        if self.metrics:
            self._log_buffer.append(entry)
            self._log_times.append(entry.timestamp)
            
            # Periodic metrics update
            if time.time() - self._last_metrics_update > self._metrics_update_interval:
                self._update_periodic_metrics()
    
    def _update_metrics(self, entry: LogEntry):
        """Update logging metrics."""
        if not self.metrics:
            return
        
        self.metrics.total_logs += 1
        
        # Update by level
        level = entry.level.lower()
        self.metrics.logs_by_level[level] = self.metrics.logs_by_level.get(level, 0) + 1
        
        # Update by component
        component = entry.component
        self.metrics.logs_by_component[component] = self.metrics.logs_by_component.get(component, 0) + 1
        
        # Update by category
        category = entry.category.value
        self.metrics.logs_by_category[category] = self.metrics.logs_by_category.get(category, 0) + 1
        
        # Calculate rates
        total_logs = self.metrics.total_logs
        error_logs = self.metrics.logs_by_level.get('error', 0) + self.metrics.logs_by_level.get('critical', 0)
        warning_logs = self.metrics.logs_by_level.get('warning', 0)
        
        self.metrics.error_rate = error_logs / total_logs if total_logs > 0 else 0
        self.metrics.warning_rate = warning_logs / total_logs if total_logs > 0 else 0
        
        # Calculate average log size (approximate)
        message_size = len(entry.message) + len(json.dumps(entry.extra_data))
        if self.metrics.total_logs == 1:
            self.metrics.avg_log_size = message_size
        else:
            # Running average
            self.metrics.avg_log_size = (
                (self.metrics.avg_log_size * (total_logs - 1) + message_size) / total_logs
            )
    
    def _update_periodic_metrics(self):
        """Update periodic metrics like logs per second."""
        if not self.metrics or not self._log_times:
            return
        
        current_time = time.time()
        window_size = 60.0  # 1 minute window
        
        # Count logs in last minute
        recent_logs = [t for t in self._log_times if current_time - t <= window_size]
        logs_per_second = len(recent_logs) / window_size
        
        if logs_per_second > self.metrics.peak_logs_per_second:
            self.metrics.peak_logs_per_second = logs_per_second
        
        # Clean old log times
        self._log_times = recent_logs
        
        # Clean old log buffer (keep last 1000 entries)
        if len(self._log_buffer) > 1000:
            self._log_buffer = self._log_buffer[-1000:]
        
        self._last_metrics_update = current_time
    
    # Public logging methods
    def trace(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log trace message."""
        self._log("trace", message, category, **kwargs)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log debug message."""
        self._log("debug", message, category, **kwargs)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log info message."""
        self._log("info", message, category, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log warning message."""
        self._log("warning", message, category, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs):
        """Log error message."""
        self._log("error", message, category, **kwargs)
    
    def critical(self, message: str, category: LogCategory = LogCategory.ERROR, **kwargs):
        """Log critical message."""
        self._log("critical", message, category, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log security event."""
        self._log("warning", f"[SECURITY] {message}", LogCategory.SECURITY, **kwargs)
    
    def audit(self, message: str, **kwargs):
        """Log audit event.""" 
        self._log("info", f"[AUDIT] {message}", LogCategory.AUDIT, **kwargs)
    
    def performance(self, message: str, duration_ms: Optional[float] = None, **kwargs):
        """Log performance metric."""
        if duration_ms is not None:
            message = f"{message} (took {duration_ms:.2f}ms)"
        self._log("info", message, LogCategory.PERFORMANCE, **kwargs)
    
    def mission_log(self, message: str, mission_id: str, **kwargs):
        """Log mission-related event."""
        self._log("info", message, LogCategory.MISSION, mission_id=mission_id, **kwargs)
    
    def user_action(self, message: str, user_id: str, **kwargs):
        """Log user action."""
        self._log("info", message, LogCategory.USER, user_id=user_id, **kwargs)
    
    # Context management
    def push_context(self, **context):
        """Push logging context."""
        self._context_stack.append(context)
    
    def pop_context(self):
        """Pop logging context."""
        if self._context_stack:
            self._context_stack.pop()
    
    def with_context(self, **context):
        """Context manager for logging context."""
        return LoggingContext(self, context)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging metrics."""
        if not self.metrics:
            return {}
        
        self._update_periodic_metrics()
        
        # Calculate uptime
        uptime = time.time() - self.metrics.start_time
        
        # System metrics (if available)
        system_metrics = {}
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                system_metrics = {
                    'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
                }
            except:
                pass
        
        return {
            'logger_name': self.name,
            'component': self.component,
            'uptime_seconds': uptime,
            'total_logs': self.metrics.total_logs,
            'logs_by_level': dict(self.metrics.logs_by_level),
            'logs_by_component': dict(self.metrics.logs_by_component),
            'logs_by_category': dict(self.metrics.logs_by_category),
            'error_rate': self.metrics.error_rate,
            'warning_rate': self.metrics.warning_rate,
            'avg_log_size_bytes': self.metrics.avg_log_size,
            'peak_logs_per_second': self.metrics.peak_logs_per_second,
            'current_logs_per_second': len(self._log_times) / 60.0 if self._log_times else 0,
            'system_metrics': system_metrics,
        }
    
    def shutdown(self):
        """Shutdown logger and cleanup resources."""
        if self.enable_async_logging and self._async_thread:
            self._stop_async_logging.set()
            
            # Wait for queue to empty
            if self._log_queue:
                try:
                    self._log_queue.join()
                except:
                    pass
            
            # Wait for thread to finish
            self._async_thread.join(timeout=5.0)
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.info("Logger shutdown complete", category=LogCategory.SYSTEM)


class LoggingContext:
    """Context manager for logging context."""
    
    def __init__(self, logger: FleetMindLogger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        self.logger.push_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.pop_context()


class LoggerFactory:
    """Factory for creating configured loggers."""
    
    _loggers: Dict[str, FleetMindLogger] = {}
    _default_config = {
        'log_level': LogLevel.INFO,
        'log_format': 'json',
        'enable_metrics': True,
        'enable_async_logging': True,
    }
    
    @classmethod
    def configure_defaults(cls, **config):
        """Configure default logger settings."""
        cls._default_config.update(config)
    
    @classmethod
    def get_logger(cls, name: str, component: str = "unknown", **config) -> FleetMindLogger:
        """Get or create logger with specified configuration."""
        logger_key = f"{name}:{component}"
        
        if logger_key not in cls._loggers:
            # Merge with defaults
            logger_config = {**cls._default_config, **config}
            
            cls._loggers[logger_key] = FleetMindLogger(
                name=name,
                component=component,
                **logger_config
            )
        
        return cls._loggers[logger_key]
    
    @classmethod
    def shutdown_all(cls):
        """Shutdown all loggers."""
        for logger in cls._loggers.values():
            logger.shutdown()
        cls._loggers.clear()


# Convenience functions
def get_logger(name: str, component: str = "unknown", **config) -> FleetMindLogger:
    """Get configured logger instance."""
    return LoggerFactory.get_logger(name, component, **config)


def configure_logging(
    log_level: LogLevel = LogLevel.INFO,
    log_format: str = "json",
    output_dir: Optional[str] = None,
    enable_metrics: bool = True,
    enable_async_logging: bool = True,
):
    """Configure global logging settings."""
    config = {
        'log_level': log_level,
        'log_format': log_format,
        'enable_metrics': enable_metrics,
        'enable_async_logging': enable_async_logging,
    }
    
    if output_dir:
        config['output_file'] = f"{output_dir}/fleet_mind.log"
    
    LoggerFactory.configure_defaults(**config)


def shutdown_logging():
    """Shutdown all logging."""
    LoggerFactory.shutdown_all()


# Performance logging decorator
def log_performance(logger: FleetMindLogger, operation: str):
    """Decorator for performance logging."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    logger.performance(f"{operation} completed", duration_ms=duration)
                    return result
                except Exception as e:
                    duration = (time.time() - start_time) * 1000
                    logger.performance(f"{operation} failed", duration_ms=duration, error=str(e))
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    logger.performance(f"{operation} completed", duration_ms=duration)
                    return result
                except Exception as e:
                    duration = (time.time() - start_time) * 1000
                    logger.performance(f"{operation} failed", duration_ms=duration, error=str(e))
                    raise
            return sync_wrapper
    return decorator


import os  # Add this import at the top