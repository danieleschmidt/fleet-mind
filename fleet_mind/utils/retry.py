"""Retry mechanisms with exponential backoff for Fleet-Mind."""

import asyncio
import time
import random
from functools import wraps
from typing import Callable, Any, Optional, List, Type, Union
from dataclasses import dataclass

from .logging import get_logger


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    initial_delay: float = 1.0          # Initial delay in seconds
    max_delay: float = 60.0             # Maximum delay in seconds
    exponential_base: float = 2.0       # Exponential backoff base
    jitter: bool = True                 # Add random jitter
    exceptions: tuple = (Exception,)    # Exceptions to retry on


class RetryError(Exception):
    """Exception raised when all retry attempts fail."""
    def __init__(self, message: str, last_exception: Exception, attempt_count: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempt_count = attempt_count


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for given attempt."""
    delay = config.initial_delay * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        # Add up to 10% jitter
        jitter_amount = delay * 0.1 * random.random()
        delay += jitter_amount
    
    return delay


def retry_sync(config: RetryConfig):
    """Synchronous retry decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f"retry.{func.__name__}")
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt}: {e}")
                    
                    if attempt < config.max_attempts:
                        delay = calculate_delay(attempt, config)
                        logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {config.max_attempts} attempts")
                        raise RetryError(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts",
                            last_exception,
                            attempt
                        ) from last_exception
            
            # Should not reach here
            raise RetryError(f"Unexpected retry failure for {func.__name__}", last_exception, config.max_attempts)
        
        return wrapper
    return decorator


def retry_async(config: RetryConfig):
    """Asynchronous retry decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f"retry.{func.__name__}")
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt}: {e}")
                    
                    if attempt < config.max_attempts:
                        delay = calculate_delay(attempt, config)
                        logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {config.max_attempts} attempts")
                        raise RetryError(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts",
                            last_exception,
                            attempt
                        ) from last_exception
            
            # Should not reach here
            raise RetryError(f"Unexpected retry failure for {func.__name__}", last_exception, config.max_attempts)
        
        return wrapper
    return decorator


def retry(max_attempts: int = 3,
         initial_delay: float = 1.0,
         max_delay: float = 60.0,
         exponential_base: float = 2.0,
         jitter: bool = True,
         exceptions: tuple = (Exception,)):
    """Retry decorator that works with both sync and async functions."""
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        exceptions=exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            return retry_async(config)(func)
        else:
            return retry_sync(config)(func)
    
    return decorator


class RetryableOperation:
    """Class for creating retryable operations programmatically."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = get_logger("retryable_operation")
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                if attempt > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt}")
                return result
                
            except self.config.exceptions as e:
                last_exception = e
                self.logger.warning(f"Operation failed on attempt {attempt}: {e}")
                
                if attempt < self.config.max_attempts:
                    delay = calculate_delay(attempt, self.config)
                    self.logger.info(f"Retrying operation in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Operation failed after {self.config.max_attempts} attempts")
                    raise RetryError(
                        f"Operation failed after {self.config.max_attempts} attempts",
                        last_exception,
                        attempt
                    ) from last_exception
        
        raise RetryError(f"Unexpected retry failure", last_exception, self.config.max_attempts)
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt}")
                return result
                
            except self.config.exceptions as e:
                last_exception = e
                self.logger.warning(f"Operation failed on attempt {attempt}: {e}")
                
                if attempt < self.config.max_attempts:
                    delay = calculate_delay(attempt, self.config)
                    self.logger.info(f"Retrying operation in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Operation failed after {self.config.max_attempts} attempts")
                    raise RetryError(
                        f"Operation failed after {self.config.max_attempts} attempts",
                        last_exception,
                        attempt
                    ) from last_exception
        
        raise RetryError(f"Unexpected retry failure", last_exception, self.config.max_attempts)


# Common retry configurations
NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    exceptions=(ConnectionError, TimeoutError, OSError)
)

LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=60.0,
    exceptions=(Exception,)  # Broad exception catching for external APIs
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=4,
    initial_delay=1.0,
    max_delay=45.0,
    exceptions=(ConnectionError, TimeoutError)
)