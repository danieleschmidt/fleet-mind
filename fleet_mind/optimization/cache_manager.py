"""Advanced caching system for Fleet-Mind performance optimization."""

import asyncio
import time
import json
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import threading

# Redis imports with fallback
try:
    import redis
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    # Mock Redis classes
    class redis:
        @staticmethod
        def Redis(*args, **kwargs):
            return MockRedis()
    
    class aioredis:
        @staticmethod
        def from_url(*args, **kwargs):
            return MockAsyncRedis()
    
    REDIS_AVAILABLE = False
    print("Warning: Redis not available, using in-memory caching only")

class MockRedis:
    def __init__(self):
        self._data = {}
    def get(self, key): return self._data.get(key)
    def set(self, key, value, ex=None): self._data[key] = value
    def delete(self, key): self._data.pop(key, None)
    def flushdb(self): self._data.clear()
    def exists(self, key): return key in self._data
    def ping(self): return True

class MockAsyncRedis:
    def __init__(self):
        self._data = {}
    async def get(self, key): return self._data.get(key)
    async def set(self, key, value, ex=None): self._data[key] = value
    async def delete(self, key): self._data.pop(key, None)
    async def flushdb(self): self._data.clear()
    async def exists(self, key): return key in self._data
    async def ping(self): return True

from ..utils.logging import get_logger


T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""
    L1_MEMORY = "l1_memory"  # In-memory cache
    L2_REDIS = "l2_redis"    # Redis cache
    L3_DISK = "l3_disk"      # Disk cache


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access information."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


class MemoryCache:
    """High-performance in-memory cache with multiple eviction strategies."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 strategy: CacheStrategy = CacheStrategy.LRU,
                 default_ttl: Optional[float] = None):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            strategy: Eviction strategy
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = default_ttl
        
        self.logger = get_logger("memory_cache", component="cache")
        
        # Storage
        if strategy == CacheStrategy.LRU:
            self._storage = OrderedDict()
        else:
            self._storage = {}
        
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 60.0  # seconds

    async def start(self) -> None:
        """Start cache background tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Memory cache started")

    async def stop(self) -> None:
        """Stop cache background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Memory cache stopped")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._entries:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
            
            entry = self._entries[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
            
            # Update access info
            entry.touch()
            
            # Update LRU order
            if self.strategy == CacheStrategy.LRU:
                self._storage.move_to_end(key)
            
            self.stats.hits += 1
            self.stats.update_hit_rate()
            
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for cache entry
        """
        with self._lock:
            # Calculate value size (approximate)
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default estimate
            
            # Check memory limits
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return
            
            # Remove existing entry if present
            if key in self._entries:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes,
                tags=tags or [],
            )
            
            # Ensure capacity
            self._ensure_capacity(size_bytes)
            
            # Store entry
            self._entries[key] = entry
            self._storage[key] = value
            
            # Update statistics
            self.stats.entry_count += 1
            self.stats.size_bytes += size_bytes

    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._storage.clear()
            self._entries.clear()
            self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return self.stats

    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries by tags.
        
        Args:
            tags: Tags to match
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._entries.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)

    def _ensure_capacity(self, new_entry_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check size limit
        while (self.stats.entry_count >= self.max_size or 
               self.stats.size_bytes + new_entry_size > self.max_memory_bytes):
            
            if not self._entries:
                break
            
            # Evict based on strategy
            key_to_evict = self._select_eviction_candidate()
            if key_to_evict:
                self._remove_entry(key_to_evict)
                self.stats.evictions += 1
            else:
                break

    def _select_eviction_candidate(self) -> Optional[str]:
        """Select candidate for eviction based on strategy."""
        if not self._entries:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # Return least recently used (first in OrderedDict)
            return next(iter(self._storage))
        
        elif self.strategy == CacheStrategy.LFU:
            # Return least frequently used
            return min(self._entries.keys(), key=lambda k: self._entries[k].access_count)
        
        elif self.strategy == CacheStrategy.TTL:
            # Return entry closest to expiration
            return min(self._entries.keys(), 
                      key=lambda k: self._entries[k].timestamp + (self._entries[k].ttl or float('inf')))
        
        else:  # FIFO
            # Return oldest entry
            return min(self._entries.keys(), key=lambda k: self._entries[k].timestamp)

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._entries:
            entry = self._entries[key]
            self.stats.entry_count -= 1
            self.stats.size_bytes -= entry.size_bytes
            del self._entries[key]
        
        if key in self._storage:
            del self._storage[key]

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                self._cleanup_expired()
        except asyncio.CancelledError:
            pass

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._entries.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "fleet_mind:"):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for namespace isolation
        """
        self.redis_url = redis_url
        self.prefix = prefix
        
        self.logger = get_logger("redis_cache", component="cache")
        
        # Connections
        self._sync_redis: Optional[redis.Redis] = None
        self._async_redis: Optional[aioredis.Redis] = None
        
        # Statistics
        self.stats = CacheStats()

    async def start(self) -> None:
        """Initialize Redis connections."""
        try:
            # Async connection
            self._async_redis = aioredis.from_url(self.redis_url, decode_responses=False)
            
            # Sync connection
            self._sync_redis = redis.from_url(self.redis_url, decode_responses=False)
            
            # Test connections
            await self._async_redis.ping()
            self._sync_redis.ping()
            
            self.logger.info("Redis cache connected")
            
        except Exception as e:
            self.logger.error(f"Redis cache connection failed: {e}")
            raise

    async def stop(self) -> None:
        """Close Redis connections."""
        if self._async_redis:
            await self._async_redis.close()
        
        if self._sync_redis:
            self._sync_redis.close()
        
        self.logger.info("Redis cache disconnected")

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._async_redis:
            return None
        
        try:
            prefixed_key = self._make_key(key)
            data = await self._async_redis.get(prefixed_key)
            
            if data is None:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
            
            # Deserialize
            value = pickle.loads(data)
            
            self.stats.hits += 1
            self.stats.update_hit_rate()
            
            return value
            
        except Exception as e:
            self.logger.error(f"Redis get error for key {key}: {e}")
            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in Redis cache."""
        if not self._async_redis:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            data = pickle.dumps(value)
            
            if ttl:
                await self._async_redis.setex(prefixed_key, int(ttl), data)
            else:
                await self._async_redis.set(prefixed_key, data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self._async_redis:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            result = await self._async_redis.delete(prefixed_key)
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Redis delete error for key {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        if not self._async_redis:
            return False
        
        try:
            pattern = f"{self.prefix}*"
            keys = await self._async_redis.keys(pattern)
            
            if keys:
                await self._async_redis.delete(*keys)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")
            return False


class CacheManager:
    """Multi-level cache manager with hierarchical caching."""
    
    def __init__(self, 
                 enable_l1: bool = True,
                 enable_l2: bool = True,
                 l1_config: Optional[Dict[str, Any]] = None,
                 l2_config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager.
        
        Args:
            enable_l1: Enable L1 memory cache
            enable_l2: Enable L2 Redis cache
            l1_config: L1 cache configuration
            l2_config: L2 Redis configuration
        """
        self.enable_l1 = enable_l1
        self.enable_l2 = enable_l2
        
        self.logger = get_logger("cache_manager", component="cache")
        
        # Initialize caches
        self.l1_cache: Optional[MemoryCache] = None
        self.l2_cache: Optional[RedisCache] = None
        
        if enable_l1:
            l1_config = l1_config or {}
            self.l1_cache = MemoryCache(**l1_config)
        
        if enable_l2:
            l2_config = l2_config or {}
            self.l2_cache = RedisCache(**l2_config)
        
        # Cache policies
        self.write_through = True  # Write to all levels
        self.read_through = True   # Try all levels on miss

    async def start(self) -> None:
        """Start all cache levels."""
        if self.l1_cache:
            await self.l1_cache.start()
        
        if self.l2_cache:
            await self.l2_cache.start()
        
        self.logger.info("Cache manager started")

    async def stop(self) -> None:
        """Stop all cache levels."""
        if self.l1_cache:
            await self.l1_cache.stop()
        
        if self.l2_cache:
            await self.l2_cache.stop()
        
        self.logger.info("Cache manager stopped")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        # Try L1 first
        if self.l1_cache:
            value = self.l1_cache.get(key)
            if value is not None:
                return value
        
        # Try L2 if L1 miss
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Populate L1 with L2 value
                if self.l1_cache:
                    self.l1_cache.set(key, value)
                return value
        
        return None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> None:
        """Set value in cache hierarchy."""
        # Set in L1
        if self.l1_cache:
            self.l1_cache.set(key, value, ttl, tags)
        
        # Set in L2 if write-through enabled
        if self.l2_cache and self.write_through:
            await self.l2_cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        deleted = False
        
        if self.l1_cache:
            deleted = self.l1_cache.delete(key) or deleted
        
        if self.l2_cache:
            deleted = await self.l2_cache.delete(key) or deleted
        
        return deleted

    async def clear(self) -> None:
        """Clear all cache levels."""
        if self.l1_cache:
            self.l1_cache.clear()
        
        if self.l2_cache:
            await self.l2_cache.clear()

    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries by tags (L1 only for now)."""
        if self.l1_cache:
            return self.l1_cache.invalidate_by_tags(tags)
        return 0

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics from all cache levels."""
        stats = {}
        
        if self.l1_cache:
            stats['l1'] = self.l1_cache.get_stats()
        
        if self.l2_cache:
            stats['l2'] = self.l2_cache.stats
        
        return stats

    # Decorators for caching function results
    def cache_result(self, key_func: Optional[Callable] = None, ttl: Optional[float] = None, tags: Optional[List[str]] = None):
        """Decorator to cache function results.
        
        Args:
            key_func: Function to generate cache key
            ttl: Cache TTL in seconds
            tags: Cache tags
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl, tags)
                
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


# Convenience decorators using global cache manager
def cached(key_func: Optional[Callable] = None, ttl: Optional[float] = None, tags: Optional[List[str]] = None):
    """Decorator to cache function results using global cache manager."""
    cache_manager = get_cache_manager()
    return cache_manager.cache_result(key_func, ttl, tags)