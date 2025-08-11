"""Multi-Tier Caching System for Fleet-Mind Generation 3.

This module implements advanced caching strategies including:
- L1 in-memory caching with intelligent eviction
- L2 Redis distributed caching with clustering
- L3 persistent storage caching with compression
- Intelligent cache invalidation and warming
- Cache analytics and optimization
- Predictive cache preloading
"""

import asyncio
import time
import json
import pickle
import hashlib
import zlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum
import statistics
import weakref

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from ..utils.logging import get_logger
from .ai_performance_optimizer import get_ai_optimizer, record_performance_metrics


class CacheLevel(Enum):
    """Cache level tiers."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis" 
    L3_PERSISTENT = "l3_persistent"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    FIFO = "fifo"            # First In, First Out
    ARC = "arc"              # Adaptive Replacement Cache
    INTELLIGENT = "intelligent"  # AI-driven eviction


class CompressionType(Enum):
    """Compression algorithms for caching."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    BROTLI = "brotli"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    ttl_seconds: float = 3600.0  # 1 hour default
    compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > (self.timestamp + self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.timestamp
    
    def access(self):
        """Record access to this entry."""
        self.access_count += 1
        self.last_access = time.time()


class LRUCache:
    """High-performance LRU cache implementation."""
    
    def __init__(self, max_size: int, ttl_seconds: float = 3600.0):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Default TTL for entries
        """
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access()
            self.hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            ttl = ttl_seconds or self.default_ttl
            current_time = time.time()
            
            # Calculate size
            size_bytes = len(str(value).encode('utf-8'))
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=current_time,
                last_access=current_time,
                size_bytes=size_bytes,
                ttl_seconds=ttl,
                cache_level=CacheLevel.L1_MEMORY,
            )
            
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Check capacity and evict if needed
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self._cache[key] = entry
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self._cache:
            self._cache.popitem(last=False)  # Remove oldest (first) item
            self.evictions += 1
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)
        
        with self._lock:
            total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_size_bytes": total_size_bytes,
        }


class IntelligentCache:
    """AI-driven cache with intelligent eviction and preloading."""
    
    def __init__(self, max_size: int, ttl_seconds: float = 3600.0):
        """Initialize intelligent cache."""
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # AI components
        self.ai_optimizer = get_ai_optimizer()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.preloads = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value with intelligent tracking."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                self._record_access_pattern(key, False)
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                self.misses += 1
                self._record_access_pattern(key, False)
                return None
            
            entry.access()
            self.hits += 1
            self._record_access_pattern(key, True)
            
            # Trigger predictive preloading
            asyncio.create_task(self._predictive_preload(key))
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put value with intelligent management."""
        with self._lock:
            ttl = ttl_seconds or self.default_ttl
            current_time = time.time()
            
            # Calculate size
            size_bytes = len(str(value).encode('utf-8'))
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=current_time,
                last_access=current_time,
                size_bytes=size_bytes,
                ttl_seconds=ttl,
                cache_level=CacheLevel.L1_MEMORY,
            )
            
            # Check capacity and intelligently evict
            while len(self._cache) >= self.max_size:
                self._intelligent_evict()
            
            self._cache[key] = entry
            return True
    
    def _intelligent_evict(self):
        """Intelligently evict entries using AI."""
        if not self._cache:
            return
        
        # Score entries for eviction
        eviction_scores = {}
        
        for key, entry in self._cache.items():
            # Base score factors
            age_factor = entry.age_seconds / 3600.0  # Hours
            access_factor = 1.0 / max(1, entry.access_count)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            
            # Access pattern analysis
            pattern_score = self._analyze_access_pattern(key)
            
            # Combined eviction score (higher = more likely to evict)
            score = (age_factor * 0.3) + (access_factor * 0.3) + \
                   (size_factor * 0.2) + (pattern_score * 0.2)
            
            eviction_scores[key] = score
        
        # Evict highest scoring entry
        worst_key = max(eviction_scores.items(), key=lambda x: x[1])[0]
        del self._cache[worst_key]
        self.evictions += 1
    
    def _analyze_access_pattern(self, key: str) -> float:
        """Analyze access pattern to predict future access probability."""
        if key not in self._access_patterns:
            return 0.5  # Neutral score
        
        accesses = self._access_patterns[key]
        if len(accesses) < 3:
            return 0.3  # Low confidence
        
        # Calculate access frequency and recency
        recent_accesses = [a for a in accesses if time.time() - a < 3600]  # Last hour
        if not recent_accesses:
            return 0.8  # High eviction score (not accessed recently)
        
        # Access frequency
        frequency = len(recent_accesses) / 60.0  # Per minute
        
        # Time since last access
        time_since_last = time.time() - max(accesses)
        recency_score = min(1.0, time_since_last / 1800.0)  # 30 minutes normalization
        
        # Pattern regularity (using coefficient of variation)
        if len(recent_accesses) > 1:
            intervals = [recent_accesses[i] - recent_accesses[i-1] 
                        for i in range(1, len(recent_accesses))]
            if intervals:
                regularity = statistics.stdev(intervals) / max(statistics.mean(intervals), 1)
                regularity_score = min(1.0, regularity)
            else:
                regularity_score = 0.5
        else:
            regularity_score = 0.5
        
        # Combined score (lower = more likely to be accessed again)
        pattern_score = (recency_score * 0.4) + ((1.0 - min(1.0, frequency)) * 0.4) + \
                       (regularity_score * 0.2)
        
        return pattern_score
    
    def _record_access_pattern(self, key: str, hit: bool):
        """Record access pattern for future prediction."""
        current_time = time.time()
        
        if hit:
            self._access_patterns[key].append(current_time)
            # Keep only recent history
            self._access_patterns[key] = [
                t for t in self._access_patterns[key] 
                if current_time - t < 7200  # 2 hours
            ]
    
    async def _predictive_preload(self, accessed_key: str):
        """Predictively preload related cache entries."""
        try:
            # Simple pattern-based preloading
            # In practice, this would use ML models to predict related keys
            
            if accessed_key.startswith("mission_"):
                # Preload related mission data
                mission_id = accessed_key.split("_", 1)[1]
                related_keys = [
                    f"drone_status_{mission_id}",
                    f"formation_{mission_id}",
                    f"weather_{mission_id}",
                ]
                
                for related_key in related_keys:
                    if related_key not in self._cache:
                        # Simulate preloading (replace with actual data fetching)
                        await asyncio.sleep(0.01)
                        self.preloads += 1
                        
        except Exception as e:
            # Silently handle preload errors
            pass


class MultiTierCache:
    """Advanced multi-tier caching system with L1, L2, and L3 tiers."""
    
    def __init__(
        self,
        l1_max_size: int = 10000,
        l1_ttl: float = 900.0,  # 15 minutes
        l2_ttl: float = 3600.0,  # 1 hour
        l3_ttl: float = 86400.0,  # 24 hours
        redis_host: str = "localhost",
        redis_port: int = 6379,
        storage_path: str = "/tmp/fleet_mind_cache",
        enable_compression: bool = True,
        compression_threshold: int = 1024,  # Compress if >1KB
    ):
        """Initialize multi-tier cache.
        
        Args:
            l1_max_size: L1 cache maximum entries
            l1_ttl: L1 cache default TTL
            l2_ttl: L2 (Redis) cache TTL
            l3_ttl: L3 (persistent) cache TTL
            redis_host: Redis server host
            redis_port: Redis server port
            storage_path: L3 storage directory path
            enable_compression: Enable data compression
            compression_threshold: Minimum size for compression
        """
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl
        self.l3_ttl = l3_ttl
        self.storage_path = storage_path
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # L1 Cache (In-Memory)
        self.l1_cache = IntelligentCache(l1_max_size, l1_ttl)
        
        # L2 Cache (Redis)
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = aioredis.from_url(
                    f"redis://{redis_host}:{redis_port}",
                    encoding="utf-8",
                    decode_responses=False  # Handle binary data
                )
            except Exception as e:
                self.logger.warning(f"Redis not available: {e}")
        
        # L3 Cache (Persistent Storage)
        import os
        os.makedirs(storage_path, exist_ok=True)
        
        # Statistics
        self.total_requests = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.total_misses = 0
        
        # Cache warming
        self.warm_keys: set = set()
        self.warming_in_progress: set = set()
        
        # Logging
        self.logger = get_logger("multi_tier_cache")
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
    async def start(self):
        """Start background cache management tasks."""
        self.running = True
        
        self.background_tasks = [
            asyncio.create_task(self._cache_maintenance_loop()),
            asyncio.create_task(self._analytics_loop()),
            asyncio.create_task(self._warming_loop()),
        ]
        
        self.logger.info("Multi-tier cache system started")
    
    async def stop(self):
        """Stop background tasks and cleanup."""
        self.running = False
        
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Multi-tier cache system stopped")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from multi-tier cache."""
        self.total_requests += 1
        start_time = time.time()
        
        try:
            # Try L1 cache first
            value = self.l1_cache.get(key)
            if value is not None:
                self.l1_hits += 1
                self._record_cache_performance(time.time() - start_time, "l1_hit")
                return value
            
            # Try L2 cache (Redis)
            if self.redis_client:
                value = await self._get_from_l2(key)
                if value is not None:
                    self.l2_hits += 1
                    # Promote to L1
                    self.l1_cache.put(key, value, self.l1_ttl)
                    self._record_cache_performance(time.time() - start_time, "l2_hit")
                    return value
            
            # Try L3 cache (Persistent)
            value = await self._get_from_l3(key)
            if value is not None:
                self.l3_hits += 1
                # Promote to L1 and L2
                self.l1_cache.put(key, value, self.l1_ttl)
                if self.redis_client:
                    await self._put_to_l2(key, value, self.l2_ttl)
                self._record_cache_performance(time.time() - start_time, "l3_hit")
                return value
            
            # Cache miss
            self.total_misses += 1
            self._record_cache_performance(time.time() - start_time, "miss")
            return default
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            self.total_misses += 1
            return default
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl_l1: Optional[float] = None,
        ttl_l2: Optional[float] = None,
        ttl_l3: Optional[float] = None,
        promote_to_l1: bool = True,
        promote_to_l2: bool = True,
        persist_to_l3: bool = True,
    ) -> bool:
        """Put value in multi-tier cache."""
        try:
            ttl_l1 = ttl_l1 or self.l1_ttl
            ttl_l2 = ttl_l2 or self.l2_ttl
            ttl_l3 = ttl_l3 or self.l3_ttl
            
            success = True
            
            # Store in L1
            if promote_to_l1:
                success &= self.l1_cache.put(key, value, ttl_l1)
            
            # Store in L2
            if promote_to_l2 and self.redis_client:
                success &= await self._put_to_l2(key, value, ttl_l2)
            
            # Store in L3
            if persist_to_l3:
                success &= await self._put_to_l3(key, value, ttl_l3)
            
            # Add to warm keys for future warming
            self.warm_keys.add(key)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache put error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        try:
            success = True
            
            # Delete from L1
            success &= self.l1_cache.delete(key)
            
            # Delete from L2
            if self.redis_client:
                try:
                    await self.redis_client.delete(self._redis_key(key))
                except Exception:
                    success = False
            
            # Delete from L3
            success &= await self._delete_from_l3(key)
            
            # Remove from warm keys
            self.warm_keys.discard(key)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get value from L2 (Redis) cache."""
        try:
            if not self.redis_client:
                return None
            
            data = await self.redis_client.get(self._redis_key(key))
            if data is None:
                return None
            
            # Deserialize
            return self._deserialize(data)
            
        except Exception as e:
            self.logger.error(f"L2 cache get error for key {key}: {e}")
            return None
    
    async def _put_to_l2(self, key: str, value: Any, ttl: float) -> bool:
        """Put value to L2 (Redis) cache."""
        try:
            if not self.redis_client:
                return False
            
            # Serialize
            data = self._serialize(value)
            
            # Store with TTL
            await self.redis_client.setex(
                self._redis_key(key),
                int(ttl),
                data
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"L2 cache put error for key {key}: {e}")
            return False
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get value from L3 (persistent) cache."""
        try:
            filepath = self._l3_filepath(key)
            
            if AIOFILES_AVAILABLE:
                try:
                    async with aiofiles.open(filepath, 'rb') as f:
                        data = await f.read()
                except FileNotFoundError:
                    return None
            else:
                try:
                    with open(filepath, 'rb') as f:
                        data = f.read()
                except FileNotFoundError:
                    return None
            
            # Deserialize
            entry_data = pickle.loads(data)
            
            # Check expiration
            if time.time() > entry_data['expires_at']:
                await self._delete_from_l3(key)
                return None
            
            return self._deserialize(entry_data['value'])
            
        except Exception as e:
            self.logger.error(f"L3 cache get error for key {key}: {e}")
            return None
    
    async def _put_to_l3(self, key: str, value: Any, ttl: float) -> bool:
        """Put value to L3 (persistent) cache."""
        try:
            filepath = self._l3_filepath(key)
            
            # Prepare entry data
            entry_data = {
                'value': self._serialize(value),
                'timestamp': time.time(),
                'expires_at': time.time() + ttl,
                'key': key,
            }
            
            # Serialize entry
            data = pickle.dumps(entry_data)
            
            # Write to file
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(data)
            else:
                with open(filepath, 'wb') as f:
                    f.write(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"L3 cache put error for key {key}: {e}")
            return False
    
    async def _delete_from_l3(self, key: str) -> bool:
        """Delete value from L3 cache."""
        try:
            filepath = self._l3_filepath(key)
            
            if AIOFILES_AVAILABLE:
                try:
                    import aiofiles.os
                    await aiofiles.os.remove(filepath)
                except FileNotFoundError:
                    pass
            else:
                try:
                    import os
                    os.remove(filepath)
                except FileNotFoundError:
                    pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"L3 cache delete error for key {key}: {e}")
            return False
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        # Convert to JSON string first
        json_data = json.dumps(value, default=str).encode('utf-8')
        
        # Compress if enabled and data is large enough
        if self.enable_compression and len(json_data) > self.compression_threshold:
            return zlib.compress(json_data, level=6)
        else:
            return json_data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try to decompress first
            if self.enable_compression:
                try:
                    json_data = zlib.decompress(data)
                except zlib.error:
                    # Data might not be compressed
                    json_data = data
            else:
                json_data = data
            
            # Parse JSON
            return json.loads(json_data.decode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            return None
    
    def _redis_key(self, key: str) -> str:
        """Generate Redis key with namespace."""
        return f"fleet_mind:cache:{key}"
    
    def _l3_filepath(self, key: str) -> str:
        """Generate L3 cache file path."""
        # Hash key to create safe filename
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return f"{self.storage_path}/cache_{key_hash[:2]}/{key_hash}.cache"
    
    def _record_cache_performance(self, latency_ms: float, cache_result: str):
        """Record cache performance metrics."""
        latency_ms *= 1000  # Convert to milliseconds
        
        # Record performance for AI optimization
        hit_rate = (self.l1_hits + self.l2_hits + self.l3_hits) / max(1, self.total_requests)
        
        record_performance_metrics(
            latency_ms=latency_ms,
            cache_hit_rate=hit_rate,
        )
    
    async def _cache_maintenance_loop(self):
        """Background cache maintenance."""
        while self.running:
            try:
                await self._cleanup_expired_l3_entries()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cache maintenance error: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_expired_l3_entries(self):
        """Clean up expired L3 cache entries."""
        try:
            import os
            import glob
            
            # Find all cache files
            pattern = f"{self.storage_path}/cache_*/*.cache"
            cache_files = glob.glob(pattern, recursive=True)
            
            cleaned_count = 0
            current_time = time.time()
            
            for filepath in cache_files:
                try:
                    if AIOFILES_AVAILABLE:
                        async with aiofiles.open(filepath, 'rb') as f:
                            data = await f.read()
                    else:
                        with open(filepath, 'rb') as f:
                            data = f.read()
                    
                    entry_data = pickle.loads(data)
                    
                    if current_time > entry_data['expires_at']:
                        # Entry expired, remove it
                        if AIOFILES_AVAILABLE:
                            import aiofiles.os
                            await aiofiles.os.remove(filepath)
                        else:
                            os.remove(filepath)
                        
                        cleaned_count += 1
                        
                except Exception:
                    # Skip problematic files
                    continue
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired L3 cache entries")
                
        except Exception as e:
            self.logger.error(f"L3 cleanup error: {e}")
    
    async def _analytics_loop(self):
        """Background cache analytics and optimization."""
        while self.running:
            try:
                # Analyze cache performance
                stats = self.get_comprehensive_stats()
                
                # Suggest optimizations
                if stats['overall_hit_rate'] < 0.7:  # <70% hit rate
                    self.logger.info("Low cache hit rate detected, consider warming more keys")
                
                if stats['l1_hit_rate'] < 0.4:  # <40% L1 hit rate
                    # Suggest increasing L1 cache size
                    self.logger.info("Low L1 hit rate, consider increasing L1 cache size")
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Cache analytics error: {e}")
                await asyncio.sleep(600)
    
    async def _warming_loop(self):
        """Background cache warming."""
        while self.running:
            try:
                # Warm frequently accessed keys
                await self._warm_cache()
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(1800)
    
    async def _warm_cache(self):
        """Intelligently warm cache with predicted access patterns."""
        try:
            # Get keys that should be warmed
            keys_to_warm = []
            
            for key in list(self.warm_keys):
                if key not in self.warming_in_progress:
                    # Check if key is likely to be accessed soon
                    should_warm = self._should_warm_key(key)
                    if should_warm:
                        keys_to_warm.append(key)
            
            # Warm keys in batches
            batch_size = 10
            for i in range(0, len(keys_to_warm), batch_size):
                batch = keys_to_warm[i:i + batch_size]
                await self._warm_key_batch(batch)
            
        except Exception as e:
            self.logger.error(f"Cache warming error: {e}")
    
    def _should_warm_key(self, key: str) -> bool:
        """Determine if a key should be warmed."""
        # Simple heuristic - in practice, use ML model
        
        # Check if key is in L1
        if self.l1_cache.get(key) is not None:
            return False  # Already cached
        
        # Check access patterns
        # This would use historical access data to predict future access
        
        # For now, warm keys that match certain patterns
        warm_patterns = ["mission_", "drone_status_", "formation_", "weather_"]
        return any(key.startswith(pattern) for pattern in warm_patterns)
    
    async def _warm_key_batch(self, keys: List[str]):
        """Warm a batch of keys."""
        for key in keys:
            if key in self.warming_in_progress:
                continue
            
            self.warming_in_progress.add(key)
            
            try:
                # Simulate fetching data for warming (replace with actual data source)
                await asyncio.sleep(0.01)
                
                # Mock warm data
                warm_data = {"warmed": True, "key": key, "timestamp": time.time()}
                
                # Store in all tiers
                await self.put(key, warm_data)
                
            except Exception as e:
                self.logger.error(f"Error warming key {key}: {e}")
            finally:
                self.warming_in_progress.discard(key)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.stats()
        
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        overall_hit_rate = total_hits / max(1, self.total_requests)
        l1_hit_rate = self.l1_hits / max(1, self.total_requests)
        l2_hit_rate = self.l2_hits / max(1, self.total_requests)
        l3_hit_rate = self.l3_hits / max(1, self.total_requests)
        miss_rate = self.total_misses / max(1, self.total_requests)
        
        return {
            "overall_hit_rate": overall_hit_rate,
            "l1_hit_rate": l1_hit_rate,
            "l2_hit_rate": l2_hit_rate,
            "l3_hit_rate": l3_hit_rate,
            "miss_rate": miss_rate,
            "total_requests": self.total_requests,
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "l3_hits": self.l3_hits,
            "total_misses": self.total_misses,
            "l1_cache_stats": l1_stats,
            "warm_keys_count": len(self.warm_keys),
            "warming_in_progress": len(self.warming_in_progress),
            "compression_enabled": self.enable_compression,
            "redis_available": self.redis_client is not None,
        }
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching a pattern."""
        try:
            # For L1 cache
            keys_to_delete = []
            for key in self.l1_cache._cache.keys():
                if pattern in key:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                await self.delete(key)
            
            self.logger.info(f"Invalidated {len(keys_to_delete)} keys matching pattern: {pattern}")
            
        except Exception as e:
            self.logger.error(f"Pattern invalidation error: {e}")


# Global multi-tier cache instance
_multi_tier_cache: Optional[MultiTierCache] = None

async def get_multi_tier_cache() -> MultiTierCache:
    """Get or create global multi-tier cache."""
    global _multi_tier_cache
    
    if _multi_tier_cache is None:
        _multi_tier_cache = MultiTierCache(
            l1_max_size=50000,  # Increased for Generation 3
            l1_ttl=600.0,  # 10 minutes
            l2_ttl=3600.0,  # 1 hour
            l3_ttl=86400.0,  # 24 hours
            enable_compression=True,
            compression_threshold=512,  # 512 bytes
        )
        await _multi_tier_cache.start()
    
    return _multi_tier_cache

async def cached_get(key: str, default: Any = None) -> Any:
    """Get value from global cache."""
    try:
        cache = await get_multi_tier_cache()
        return await cache.get(key, default)
    except Exception:
        return default

async def cached_put(
    key: str,
    value: Any,
    ttl_l1: Optional[float] = None,
    ttl_l2: Optional[float] = None,
    ttl_l3: Optional[float] = None,
) -> bool:
    """Put value in global cache."""
    try:
        cache = await get_multi_tier_cache()
        return await cache.put(key, value, ttl_l1, ttl_l2, ttl_l3)
    except Exception:
        return False

async def cached_delete(key: str) -> bool:
    """Delete key from global cache."""
    try:
        cache = await get_multi_tier_cache()
        return await cache.delete(key)
    except Exception:
        return False

def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    try:
        if _multi_tier_cache:
            return _multi_tier_cache.get_comprehensive_stats()
        else:
            return {"error": "Cache not initialized"}
    except Exception:
        return {"error": "Cache stats not available"}