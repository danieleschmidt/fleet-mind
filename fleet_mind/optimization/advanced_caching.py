"""Advanced caching system with intelligent cache management for Fleet-Mind."""

import asyncio
import time
import hashlib
import pickle
import json
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import threading
from collections import OrderedDict

from ..utils.logging import get_logger


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    tags: List[str]
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class IntelligentCache:
    """High-performance cache with multiple eviction policies and analytics."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 100.0,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 default_ttl: Optional[float] = 3600.0):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.policy = policy
        self.default_ttl = default_ttl
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.size_tracker = 0
        self.lock = threading.RLock()
        self.logger = get_logger("intelligent_cache")
        
        # Analytics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.created_at = time.time()
        
        # Adaptive policy state
        self.access_patterns: Dict[str, List[float]] = {}
        self.last_cleanup = time.time()
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if hasattr(obj, '__len__'):
                if isinstance(obj, (str, bytes)):
                    return len(obj)
                elif isinstance(obj, (list, tuple, dict)):
                    # Rough estimation
                    return len(pickle.dumps(obj))
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            return 64  # Default size
    
    def _generate_key(self, key_parts: Tuple[Any, ...]) -> str:
        """Generate consistent cache key from parts."""
        key_str = str(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _should_evict(self) -> bool:
        """Check if cache needs eviction."""
        return (len(self.cache) >= self.max_size or 
                self.size_tracker >= self.max_memory_bytes)
    
    def _evict_entries(self):
        """Evict entries based on policy."""
        if not self.cache:
            return
        
        current_time = time.time()
        entries_to_remove = []
        
        # First, remove expired entries
        for key, entry in self.cache.items():
            if entry.is_expired():
                entries_to_remove.append(key)
        
        # Remove expired entries
        for key in entries_to_remove:
            self._remove_entry(key)
        
        # If still need space, apply eviction policy
        while self._should_evict() and self.cache:
            if self.policy == CachePolicy.LRU:
                key = next(iter(self.cache))  # First (oldest) item
            elif self.policy == CachePolicy.LFU:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            elif self.policy == CachePolicy.TTL:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            elif self.policy == CachePolicy.ADAPTIVE:
                key = self._adaptive_evict()
            else:
                key = next(iter(self.cache))
            
            self._remove_entry(key)
            self.evictions += 1
    
    def _adaptive_evict(self) -> str:
        """Intelligent eviction based on access patterns."""
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            # Calculate adaptive score based on:
            # - Recency (how recently accessed)
            # - Frequency (how often accessed)
            # - Trend (increasing or decreasing access)
            
            recency_score = 1.0 / max(current_time - entry.last_accessed, 1.0)
            frequency_score = entry.access_count / max(current_time - entry.created_at, 1.0)
            
            # Trend analysis
            trend_score = 0.0
            if key in self.access_patterns:
                recent_accesses = [t for t in self.access_patterns[key] if current_time - t < 3600]
                if len(recent_accesses) > 1:
                    # Simple trend calculation
                    early_half = len([t for t in recent_accesses if t < recent_accesses[len(recent_accesses)//2]])
                    late_half = len(recent_accesses) - early_half
                    trend_score = (late_half - early_half) / len(recent_accesses)
            
            # Combined score (higher = more valuable, less likely to evict)
            scores[key] = recency_score * 0.4 + frequency_score * 0.4 + trend_score * 0.2
        
        # Return key with lowest score
        return min(scores.keys(), key=scores.get)
    
    def _remove_entry(self, key: str):
        """Remove entry and update size tracker."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.size_tracker -= entry.size_bytes
            
            # Clean up access patterns
            if key in self.access_patterns:
                del self.access_patterns[key]
    
    def _update_access_pattern(self, key: str):
        """Update access patterns for adaptive caching."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # Keep only last 100 access times
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Store value in cache."""
        with self.lock:
            current_time = time.time()
            size_bytes = self._calculate_size(value)
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Evict if necessary
            self._evict_entries()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            self.cache[key] = entry
            self.size_tracker += size_bytes
            
            self.logger.debug(f"Cached item {key} ({size_bytes} bytes)")
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access metadata
            entry.touch()
            self._update_access_pattern(key)
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear_by_tags(self, tags: List[str]):
        """Clear all entries with specified tags."""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            self.logger.info(f"Cleared {len(keys_to_remove)} entries with tags: {tags}")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_patterns.clear()
            self.size_tracker = 0
            self.logger.info("Cache cleared")
    
    def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            current_time = time.time()
            
            # Only cleanup if enough time has passed
            if current_time - self.last_cleanup < 60:  # 1 minute
                return
            
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            self.last_cleanup = current_time
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "memory_used_mb": self.size_tracker / 1024 / 1024,
                "memory_limit_mb": self.max_memory_bytes / 1024 / 1024,
                "memory_utilization": self.size_tracker / self.max_memory_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "policy": self.policy.value,
                "uptime_seconds": time.time() - self.created_at,
            }
    
    def get_entry_stats(self) -> List[Dict[str, Any]]:
        """Get detailed entry statistics."""
        with self.lock:
            stats = []
            for key, entry in self.cache.items():
                stats.append({
                    "key": key[:50] + "..." if len(key) > 50 else key,
                    "size_bytes": entry.size_bytes,
                    "age_seconds": time.time() - entry.created_at,
                    "last_accessed_seconds_ago": time.time() - entry.last_accessed,
                    "access_count": entry.access_count,
                    "tags": entry.tags,
                    "expired": entry.is_expired()
                })
            
            # Sort by access count descending
            return sorted(stats, key=lambda x: x["access_count"], reverse=True)


class DistributedCache:
    """Distributed cache across multiple nodes."""
    
    def __init__(self, node_id: str = "local"):
        self.node_id = node_id
        self.local_cache = IntelligentCache(max_size=500, max_memory_mb=50.0)
        self.remote_nodes: Dict[str, str] = {}  # node_id -> address
        self.logger = get_logger("distributed_cache")
        
        # Consistent hashing for distribution
        self.hash_ring: List[Tuple[int, str]] = []
        self._rebuild_hash_ring()
    
    def add_node(self, node_id: str, address: str):
        """Add remote cache node."""
        self.remote_nodes[node_id] = address
        self._rebuild_hash_ring()
        self.logger.info(f"Added cache node: {node_id} at {address}")
    
    def remove_node(self, node_id: str):
        """Remove remote cache node."""
        if node_id in self.remote_nodes:
            del self.remote_nodes[node_id]
            self._rebuild_hash_ring()
            self.logger.info(f"Removed cache node: {node_id}")
    
    def _rebuild_hash_ring(self):
        """Rebuild consistent hash ring."""
        self.hash_ring.clear()
        
        # Add local node
        hash_val = hash(self.node_id) & 0x7FFFFFFF
        self.hash_ring.append((hash_val, self.node_id))
        
        # Add remote nodes
        for node_id in self.remote_nodes:
            hash_val = hash(node_id) & 0x7FFFFFFF
            self.hash_ring.append((hash_val, node_id))
        
        # Sort by hash value
        self.hash_ring.sort()
    
    def _get_node_for_key(self, key: str) -> str:
        """Get responsible node for key using consistent hashing."""
        if not self.hash_ring:
            return self.node_id
        
        key_hash = hash(key) & 0x7FFFFFFF
        
        # Find first node with hash >= key_hash
        for hash_val, node_id in self.hash_ring:
            if hash_val >= key_hash:
                return node_id
        
        # Wrap around to first node
        return self.hash_ring[0][1]
    
    async def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Store value in distributed cache."""
        responsible_node = self._get_node_for_key(key)
        
        if responsible_node == self.node_id:
            return self.local_cache.put(key, value, ttl, tags)
        else:
            # In real implementation, would send to remote node
            # For now, store locally as backup
            self.logger.debug(f"Key {key} should be on {responsible_node}, storing locally")
            return self.local_cache.put(key, value, ttl, tags)
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from distributed cache."""
        responsible_node = self._get_node_for_key(key)
        
        if responsible_node == self.node_id:
            return self.local_cache.get(key)
        else:
            # In real implementation, would fetch from remote node
            # For now, check local cache
            result = self.local_cache.get(key)
            if result is None:
                self.logger.debug(f"Key {key} not found locally, should check {responsible_node}")
            return result
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get distributed cache cluster statistics."""
        return {
            "local_node": self.node_id,
            "remote_nodes": len(self.remote_nodes),
            "hash_ring_size": len(self.hash_ring),
            "local_cache_stats": self.local_cache.get_stats()
        }


# Cache decorators
def smart_cache(ttl: Optional[float] = 3600.0, 
               max_size: int = 1000,
               tags: List[str] = None,
               key_generator: Optional[Callable] = None):
    """Smart caching decorator with automatic key generation."""
    cache = IntelligentCache(max_size=max_size, default_ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result, ttl, tags)
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result, ttl, tags)
            
            return result
        
        # Attach cache management methods
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        wrapper.cache_cleanup = cache.cleanup_expired
        
        return wrapper
    
    return decorator


# Global cache instances
_mission_cache = IntelligentCache(max_size=200, max_memory_mb=20.0, policy=CachePolicy.ADAPTIVE)
_plan_cache = IntelligentCache(max_size=500, max_memory_mb=50.0, policy=CachePolicy.LRU)
_distributed_cache: Optional[DistributedCache] = None

def get_mission_cache() -> IntelligentCache:
    """Get global mission cache."""
    return _mission_cache

def get_plan_cache() -> IntelligentCache:
    """Get global plan cache."""
    return _plan_cache

def get_distributed_cache() -> DistributedCache:
    """Get or create distributed cache."""
    global _distributed_cache
    if _distributed_cache is None:
        _distributed_cache = DistributedCache()
    return _distributed_cache

def cleanup_all_caches():
    """Cleanup expired entries in all caches."""
    _mission_cache.cleanup_expired()
    _plan_cache.cleanup_expired()
    if _distributed_cache:
        _distributed_cache.local_cache.cleanup_expired()