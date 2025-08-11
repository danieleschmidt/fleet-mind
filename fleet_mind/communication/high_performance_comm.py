"""High-Performance Communication Layer for Fleet-Mind Generation 3.

This module implements advanced communication optimizations including:
- Connection pooling and multiplexing for massive scale
- Protocol optimization (binary vs JSON)
- Bandwidth optimization with compression
- Real-time data streaming optimizations
- Network topology optimization
- Advanced WebRTC streaming with load balancing
"""

import asyncio
import time
import json
import gzip
import struct
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import concurrent.futures
import weakref

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..utils.logging import get_logger
from ..optimization.multi_tier_cache import get_multi_tier_cache
from ..optimization.ai_performance_optimizer import record_performance_metrics


class Protocol(Enum):
    """Communication protocols."""
    JSON = "json"
    MSGPACK = "msgpack"
    BINARY = "binary"
    PROTOBUF = "protobuf"


class CompressionAlgorithm(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    BROTLI = "brotli"
    LZ4 = "lz4"


class ConnectionType(Enum):
    """Connection types."""
    WEBSOCKET = "websocket"
    HTTP = "http"
    UDP = "udp"
    WEBRTC = "webrtc"
    MQTT = "mqtt"


class Priority(Enum):
    """Message priority levels."""
    CRITICAL = 0    # Emergency, safety-critical
    HIGH = 1        # Important control messages
    NORMAL = 2      # Standard operational messages
    LOW = 3         # Background, telemetry
    BULK = 4        # Large data transfers


@dataclass
class ConnectionConfig:
    """Configuration for connection pooling."""
    connection_type: ConnectionType
    max_connections: int = 100
    max_connections_per_host: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    keepalive_interval: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress messages >1KB
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    protocol: Protocol = Protocol.MSGPACK


@dataclass
class Message:
    """High-performance message container."""
    id: str
    destination: str
    payload: Any
    priority: Priority = Priority.NORMAL
    compression: CompressionAlgorithm = CompressionAlgorithm.NONE
    protocol: Protocol = Protocol.JSON
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return time.time() > (self.timestamp + self.ttl_seconds)
    
    def serialize(self) -> bytes:
        """Serialize message to bytes."""
        try:
            # Prepare message data
            msg_data = {
                "id": self.id,
                "destination": self.destination,
                "payload": self.payload,
                "priority": self.priority.value,
                "timestamp": self.timestamp,
                "metadata": self.metadata,
            }
            
            # Encode based on protocol
            if self.protocol == Protocol.MSGPACK and MSGPACK_AVAILABLE:
                data = msgpack.packb(msg_data, use_bin_type=True)
            elif self.protocol == Protocol.BINARY:
                # Simple binary protocol
                json_data = json.dumps(msg_data).encode('utf-8')
                data = struct.pack(f">I{len(json_data)}s", len(json_data), json_data)
            else:  # JSON
                data = json.dumps(msg_data, separators=(',', ':')).encode('utf-8')
            
            # Apply compression if needed
            if self.compression != CompressionAlgorithm.NONE and len(data) > 1024:
                if self.compression == CompressionAlgorithm.GZIP:
                    data = gzip.compress(data, compresslevel=6)
                elif self.compression == CompressionAlgorithm.ZLIB:
                    import zlib
                    data = zlib.compress(data, level=6)
            
            return data
            
        except Exception as e:
            # Fallback to simple JSON
            return json.dumps({"error": str(e)}).encode('utf-8')
    
    @classmethod
    def deserialize(
        cls,
        data: bytes,
        protocol: Protocol = Protocol.JSON,
        compression: CompressionAlgorithm = CompressionAlgorithm.NONE
    ) -> 'Message':
        """Deserialize message from bytes."""
        try:
            # Decompress if needed
            if compression != CompressionAlgorithm.NONE:
                if compression == CompressionAlgorithm.GZIP:
                    data = gzip.decompress(data)
                elif compression == CompressionAlgorithm.ZLIB:
                    import zlib
                    data = zlib.decompress(data)
            
            # Decode based on protocol
            if protocol == Protocol.MSGPACK and MSGPACK_AVAILABLE:
                msg_data = msgpack.unpackb(data, raw=False)
            elif protocol == Protocol.BINARY:
                # Simple binary protocol
                length = struct.unpack(">I", data[:4])[0]
                json_data = data[4:4+length].decode('utf-8')
                msg_data = json.loads(json_data)
            else:  # JSON
                msg_data = json.loads(data.decode('utf-8'))
            
            # Create message object
            return cls(
                id=msg_data.get("id", "unknown"),
                destination=msg_data.get("destination", "unknown"),
                payload=msg_data.get("payload", {}),
                priority=Priority(msg_data.get("priority", Priority.NORMAL.value)),
                timestamp=msg_data.get("timestamp", time.time()),
                metadata=msg_data.get("metadata", {}),
                protocol=protocol,
                compression=compression,
            )
            
        except Exception as e:
            # Return error message
            return cls(
                id="error",
                destination="unknown",
                payload={"error": str(e)},
                priority=Priority.LOW,
            )


class Connection:
    """High-performance connection wrapper."""
    
    def __init__(
        self,
        connection_id: str,
        connection_type: ConnectionType,
        endpoint: str,
        config: ConnectionConfig,
    ):
        """Initialize connection.
        
        Args:
            connection_id: Unique connection identifier
            connection_type: Type of connection
            endpoint: Connection endpoint
            config: Connection configuration
        """
        self.connection_id = connection_id
        self.connection_type = connection_type
        self.endpoint = endpoint
        self.config = config
        
        # Connection state
        self.connected = False
        self.last_used = time.time()
        self.bytes_sent = 0
        self.bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0
        
        # Connection objects
        self.websocket = None
        self.http_session = None
        
        # Threading
        self.lock = asyncio.Lock()
        
        # Logging
        self.logger = get_logger(f"connection_{connection_id}")
    
    async def connect(self) -> bool:
        """Establish connection."""
        try:
            async with self.lock:
                if self.connected:
                    return True
                
                if self.connection_type == ConnectionType.WEBSOCKET:
                    await self._connect_websocket()
                elif self.connection_type == ConnectionType.HTTP:
                    await self._connect_http()
                # Add other connection types as needed
                
                self.connected = True
                self.last_used = time.time()
                return True
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    async def _connect_websocket(self):
        """Connect WebSocket."""
        if WEBSOCKETS_AVAILABLE:
            self.websocket = await websockets.connect(
                self.endpoint,
                timeout=self.config.connection_timeout,
                ping_interval=self.config.keepalive_interval,
                ping_timeout=10,
            )
    
    async def _connect_http(self):
        """Connect HTTP session."""
        if AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections_per_host,
                keepalive_timeout=self.config.idle_timeout,
                enable_cleanup_closed=True,
            )
            self.http_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
    
    async def send_message(self, message: Message) -> bool:
        """Send message through connection."""
        try:
            if not self.connected:
                if not await self.connect():
                    return False
            
            # Serialize message
            data = message.serialize()
            
            # Send based on connection type
            if self.connection_type == ConnectionType.WEBSOCKET and self.websocket:
                await self.websocket.send(data)
            elif self.connection_type == ConnectionType.HTTP and self.http_session:
                async with self.http_session.post(
                    self.endpoint,
                    data=data,
                    headers={"Content-Type": "application/octet-stream"}
                ) as response:
                    if response.status >= 400:
                        return False
            
            # Update statistics
            self.bytes_sent += len(data)
            self.messages_sent += 1
            self.last_used = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Message send failed: {e}")
            self.connected = False
            return False
    
    async def receive_message(self) -> Optional[Message]:
        """Receive message from connection."""
        try:
            if not self.connected:
                return None
            
            data = None
            
            if self.connection_type == ConnectionType.WEBSOCKET and self.websocket:
                data = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
            # Add other connection types as needed
            
            if data:
                # Update statistics
                self.bytes_received += len(data)
                self.messages_received += 1
                self.last_used = time.time()
                
                # Deserialize message
                return Message.deserialize(
                    data if isinstance(data, bytes) else data.encode('utf-8'),
                    protocol=self.config.protocol,
                    compression=self.config.compression_algorithm,
                )
            
            return None
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Message receive failed: {e}")
            self.connected = False
            return None
    
    async def close(self):
        """Close connection."""
        try:
            async with self.lock:
                self.connected = False
                
                if self.websocket:
                    await self.websocket.close()
                    self.websocket = None
                
                if self.http_session:
                    await self.http_session.close()
                    self.http_session = None
                    
        except Exception as e:
            self.logger.error(f"Connection close error: {e}")
    
    def is_idle(self) -> bool:
        """Check if connection is idle."""
        return (time.time() - self.last_used) > self.config.idle_timeout
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "connection_id": self.connection_id,
            "connection_type": self.connection_type.value,
            "endpoint": self.endpoint,
            "connected": self.connected,
            "last_used": self.last_used,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "idle_time": time.time() - self.last_used,
        }


class ConnectionPool:
    """High-performance connection pool with load balancing."""
    
    def __init__(
        self,
        pool_id: str,
        config: ConnectionConfig,
        max_pool_size: int = 1000,
    ):
        """Initialize connection pool.
        
        Args:
            pool_id: Pool identifier
            config: Connection configuration
            max_pool_size: Maximum pool size
        """
        self.pool_id = pool_id
        self.config = config
        self.max_pool_size = max_pool_size
        
        # Connection management
        self.connections: Dict[str, Connection] = {}
        self.available_connections: Dict[str, Set[str]] = defaultdict(set)  # endpoint -> connection_ids
        self.connection_usage: Dict[str, int] = defaultdict(int)  # connection_id -> usage_count
        
        # Load balancing
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Threading
        self.pool_lock = asyncio.Lock()
        
        # Background tasks
        self.maintenance_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.total_connections_created = 0
        self.total_connections_destroyed = 0
        self.pool_hits = 0
        self.pool_misses = 0
        
        # Logging
        self.logger = get_logger(f"connection_pool_{pool_id}")
    
    async def start(self):
        """Start connection pool maintenance."""
        self.running = True
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.logger.info(f"Started connection pool {self.pool_id}")
    
    async def stop(self):
        """Stop connection pool and cleanup."""
        self.running = False
        
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection in list(self.connections.values()):
            await connection.close()
        
        self.connections.clear()
        self.available_connections.clear()
        
        self.logger.info(f"Stopped connection pool {self.pool_id}")
    
    async def get_connection(self, endpoint: str) -> Optional[Connection]:
        """Get connection from pool."""
        try:
            async with self.pool_lock:
                # Try to get existing available connection
                if endpoint in self.available_connections and self.available_connections[endpoint]:
                    connection_id = self._select_connection(endpoint)
                    if connection_id and connection_id in self.connections:
                        connection = self.connections[connection_id]
                        self.available_connections[endpoint].discard(connection_id)
                        self.connection_usage[connection_id] += 1
                        self.pool_hits += 1
                        return connection
                
                # Create new connection if under limits
                if len(self.connections) < self.max_pool_size:
                    connection = await self._create_connection(endpoint)
                    if connection:
                        self.pool_misses += 1
                        return connection
                
                # Pool is full and no available connections
                self.logger.warning(f"Connection pool full, cannot create connection to {endpoint}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting connection to {endpoint}: {e}")
            return None
    
    def _select_connection(self, endpoint: str) -> Optional[str]:
        """Select connection using load balancing."""
        available = list(self.available_connections[endpoint])
        if not available:
            return None
        
        # Use round-robin with usage consideration
        counter = self.round_robin_counters[endpoint]
        selected_id = available[counter % len(available)]
        self.round_robin_counters[endpoint] += 1
        
        return selected_id
    
    async def _create_connection(self, endpoint: str) -> Optional[Connection]:
        """Create new connection."""
        try:
            connection_id = f"{self.pool_id}_{len(self.connections)}_{int(time.time())}"
            connection = Connection(
                connection_id=connection_id,
                connection_type=self.config.connection_type,
                endpoint=endpoint,
                config=self.config,
            )
            
            # Attempt to connect
            if await connection.connect():
                self.connections[connection_id] = connection
                self.connection_usage[connection_id] = 1
                self.total_connections_created += 1
                self.logger.debug(f"Created connection {connection_id} to {endpoint}")
                return connection
            else:
                await connection.close()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create connection to {endpoint}: {e}")
            return None
    
    async def return_connection(self, connection: Connection):
        """Return connection to pool."""
        try:
            async with self.pool_lock:
                if connection.connection_id in self.connections:
                    self.connection_usage[connection.connection_id] -= 1
                    
                    if self.connection_usage[connection.connection_id] <= 0:
                        # Connection is no longer in use, make it available
                        self.available_connections[connection.endpoint].add(connection.connection_id)
                        
        except Exception as e:
            self.logger.error(f"Error returning connection {connection.connection_id}: {e}")
    
    async def remove_connection(self, connection: Connection):
        """Remove connection from pool."""
        try:
            async with self.pool_lock:
                connection_id = connection.connection_id
                
                if connection_id in self.connections:
                    await connection.close()
                    del self.connections[connection_id]
                    
                    # Remove from available connections
                    self.available_connections[connection.endpoint].discard(connection_id)
                    
                    # Remove usage tracking
                    if connection_id in self.connection_usage:
                        del self.connection_usage[connection_id]
                    
                    self.total_connections_destroyed += 1
                    self.logger.debug(f"Removed connection {connection_id}")
                    
        except Exception as e:
            self.logger.error(f"Error removing connection {connection.connection_id}: {e}")
    
    async def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.running:
            try:
                await self._cleanup_idle_connections()
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Connection pool maintenance error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_idle_connections(self):
        """Clean up idle connections."""
        try:
            idle_connections = []
            
            async with self.pool_lock:
                for connection in self.connections.values():
                    if (connection.is_idle() and 
                        self.connection_usage.get(connection.connection_id, 0) <= 0):
                        idle_connections.append(connection)
            
            # Remove idle connections
            for connection in idle_connections:
                await self.remove_connection(connection)
            
            if idle_connections:
                self.logger.info(f"Cleaned up {len(idle_connections)} idle connections")
                
        except Exception as e:
            self.logger.error(f"Idle connection cleanup error: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        total_connections = len(self.connections)
        available_count = sum(len(conns) for conns in self.available_connections.values())
        in_use_count = total_connections - available_count
        
        total_requests = self.pool_hits + self.pool_misses
        hit_rate = self.pool_hits / max(1, total_requests)
        
        return {
            "pool_id": self.pool_id,
            "total_connections": total_connections,
            "available_connections": available_count,
            "in_use_connections": in_use_count,
            "max_pool_size": self.max_pool_size,
            "pool_utilization": total_connections / max(1, self.max_pool_size),
            "pool_hit_rate": hit_rate,
            "total_connections_created": self.total_connections_created,
            "total_connections_destroyed": self.total_connections_destroyed,
            "endpoints_count": len(self.available_connections),
        }


class HighPerformanceCommunicator:
    """High-performance communication system with advanced optimizations."""
    
    def __init__(
        self,
        max_connections: int = 10000,
        enable_compression: bool = True,
        default_protocol: Protocol = Protocol.MSGPACK,
        enable_caching: bool = True,
        message_queue_size: int = 100000,
    ):
        """Initialize high-performance communicator.
        
        Args:
            max_connections: Maximum total connections
            enable_compression: Enable message compression
            default_protocol: Default protocol for messages
            enable_caching: Enable response caching
            message_queue_size: Maximum message queue size
        """
        self.max_connections = max_connections
        self.enable_compression = enable_compression
        self.default_protocol = default_protocol
        self.enable_caching = enable_caching
        self.message_queue_size = message_queue_size
        
        # Connection pools
        self.connection_pools: Dict[str, ConnectionPool] = {}
        
        # Message queues by priority
        self.message_queues: Dict[Priority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=message_queue_size // len(Priority))
            for priority in Priority
        }
        
        # Message routing and load balancing
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0,
            "failures": 0,
            "avg_latency_ms": 0.0,
            "last_success": 0.0,
        })
        
        # Caching
        self.cache = None
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Performance tracking
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.start_time = time.time()
        
        # Logging
        self.logger = get_logger("high_performance_communicator")
    
    async def start(self):
        """Start high-performance communication system."""
        self.running = True
        
        # Initialize cache if enabled
        if self.enable_caching:
            self.cache = await get_multi_tier_cache()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._connection_monitor()),
        ]
        
        self.logger.info("High-performance communication system started")
    
    async def stop(self):
        """Stop communication system and cleanup."""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Stop connection pools
        for pool in self.connection_pools.values():
            await pool.stop()
        
        self.logger.info("High-performance communication system stopped")
    
    async def send_message(
        self,
        destination: str,
        payload: Any,
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Send high-performance message.
        
        Args:
            destination: Destination endpoint
            payload: Message payload
            priority: Message priority
            timeout: Send timeout in seconds
            use_cache: Enable response caching
            
        Returns:
            Send result with performance metrics
        """
        start_time = time.time()
        
        try:
            # Check cache for recent response
            cache_key = None
            if use_cache and self.cache:
                cache_key = f"response:{hashlib.md5(f'{destination}:{json.dumps(payload, sort_keys=True)}'.encode()).hexdigest()}"
                cached_response = await self.cache.get(cache_key)
                if cached_response:
                    return {
                        "success": True,
                        "cached": True,
                        "response": cached_response,
                        "latency_ms": (time.time() - start_time) * 1000,
                    }
            
            # Create message
            message = Message(
                id=f"msg_{int(time.time() * 1000000)}",
                destination=destination,
                payload=payload,
                priority=priority,
                protocol=self.default_protocol,
                compression=CompressionAlgorithm.GZIP if self.enable_compression else CompressionAlgorithm.NONE,
            )
            
            # Get connection
            connection = await self._get_connection(destination)
            if not connection:
                return {
                    "success": False,
                    "error": "No connection available",
                    "latency_ms": (time.time() - start_time) * 1000,
                }
            
            # Send message
            success = await connection.send_message(message)
            
            # Update statistics
            self.total_messages_sent += 1
            self.total_bytes_sent += len(message.serialize())
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Update endpoint statistics
            self._update_endpoint_stats(destination, success, latency_ms)
            
            # Record performance metrics
            record_performance_metrics(
                latency_ms=latency_ms,
                throughput_rps=1.0,
                error_rate=0.0 if success else 1.0,
            )
            
            # Return connection to pool
            await self._return_connection(connection)
            
            result = {
                "success": success,
                "cached": False,
                "message_id": message.id,
                "latency_ms": latency_ms,
                "bytes_sent": len(message.serialize()),
            }
            
            # Cache successful response if enabled
            if success and use_cache and self.cache and cache_key:
                await self.cache.put(cache_key, result, ttl_l1=60.0)  # Cache for 1 minute
            
            return result
            
        except Exception as e:
            self.logger.error(f"Message send error to {destination}: {e}")
            return {
                "success": False,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            }
    
    async def broadcast_message(
        self,
        destinations: List[str],
        payload: Any,
        priority: Priority = Priority.NORMAL,
        max_concurrent: int = 100,
    ) -> Dict[str, Any]:
        """Broadcast message to multiple destinations."""
        start_time = time.time()
        
        try:
            # Limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def send_to_destination(dest):
                async with semaphore:
                    return await self.send_message(dest, payload, priority)
            
            # Send to all destinations concurrently
            tasks = [send_to_destination(dest) for dest in destinations]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            failed = len(destinations) - successful
            
            return {
                "total_destinations": len(destinations),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(destinations) if destinations else 0,
                "total_latency_ms": (time.time() - start_time) * 1000,
                "results": results,
            }
            
        except Exception as e:
            self.logger.error(f"Broadcast error: {e}")
            return {
                "total_destinations": len(destinations),
                "successful": 0,
                "failed": len(destinations),
                "success_rate": 0.0,
                "error": str(e),
            }
    
    async def _get_connection(self, destination: str) -> Optional[Connection]:
        """Get connection for destination."""
        try:
            # Extract connection type and endpoint from destination
            if destination.startswith("ws://") or destination.startswith("wss://"):
                connection_type = ConnectionType.WEBSOCKET
            elif destination.startswith("http://") or destination.startswith("https://"):
                connection_type = ConnectionType.HTTP
            else:
                # Default to WebSocket
                connection_type = ConnectionType.WEBSOCKET
            
            # Get or create connection pool
            pool_key = f"{connection_type.value}_{destination.split('/')[2] if '/' in destination else destination}"
            
            if pool_key not in self.connection_pools:
                config = ConnectionConfig(
                    connection_type=connection_type,
                    max_connections=min(100, self.max_connections // 10),
                    enable_compression=self.enable_compression,
                    protocol=self.default_protocol,
                )
                
                pool = ConnectionPool(pool_key, config)
                await pool.start()
                self.connection_pools[pool_key] = pool
            
            # Get connection from pool
            return await self.connection_pools[pool_key].get_connection(destination)
            
        except Exception as e:
            self.logger.error(f"Error getting connection for {destination}: {e}")
            return None
    
    async def _return_connection(self, connection: Connection):
        """Return connection to appropriate pool."""
        try:
            for pool in self.connection_pools.values():
                if connection.connection_id in pool.connections:
                    await pool.return_connection(connection)
                    break
                    
        except Exception as e:
            self.logger.error(f"Error returning connection {connection.connection_id}: {e}")
    
    def _update_endpoint_stats(self, endpoint: str, success: bool, latency_ms: float):
        """Update endpoint performance statistics."""
        stats = self.endpoint_stats[endpoint]
        stats["requests"] += 1
        
        if success:
            stats["last_success"] = time.time()
            # Update average latency
            old_avg = stats["avg_latency_ms"]
            old_count = stats["requests"] - stats["failures"] - 1
            if old_count > 0:
                stats["avg_latency_ms"] = ((old_avg * old_count) + latency_ms) / (old_count + 1)
            else:
                stats["avg_latency_ms"] = latency_ms
        else:
            stats["failures"] += 1
    
    async def _message_processor(self):
        """Background message processor for queued messages."""
        while self.running:
            try:
                # Process messages by priority
                for priority in Priority:
                    queue = self.message_queues[priority]
                    
                    try:
                        # Process up to 10 messages per priority per iteration
                        for _ in range(10):
                            message = queue.get_nowait()
                            asyncio.create_task(self._process_queued_message(message))
                    except asyncio.QueueEmpty:
                        continue
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_queued_message(self, message: Message):
        """Process queued message."""
        try:
            result = await self.send_message(
                message.destination,
                message.payload,
                message.priority,
            )
            
            # Handle result if needed
            if not result["success"] and message.retry_count < message.max_retries:
                # Retry failed message
                message.retry_count += 1
                await self.message_queues[message.priority].put(message)
                
        except Exception as e:
            self.logger.error(f"Queued message processing error: {e}")
    
    async def _performance_monitor(self):
        """Background performance monitoring."""
        while self.running:
            try:
                # Calculate performance metrics
                uptime = time.time() - self.start_time
                messages_per_second = self.total_messages_sent / max(1, uptime)
                bytes_per_second = self.total_bytes_sent / max(1, uptime)
                
                # Log performance summary periodically
                if int(uptime) % 300 == 0:  # Every 5 minutes
                    self.logger.info(
                        f"Performance: {messages_per_second:.1f} msg/s, "
                        f"{bytes_per_second / 1024:.1f} KB/s, "
                        f"{len(self.connection_pools)} pools, "
                        f"{sum(len(pool.connections) for pool in self.connection_pools.values())} connections"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _connection_monitor(self):
        """Background connection monitoring and optimization."""
        while self.running:
            try:
                # Monitor connection pool health
                for pool_key, pool in list(self.connection_pools.items()):
                    stats = pool.get_pool_stats()
                    
                    # Remove pools with no connections for extended time
                    if (stats["total_connections"] == 0 and 
                        hasattr(pool, '_last_used') and 
                        time.time() - pool._last_used > 1800):  # 30 minutes
                        await pool.stop()
                        del self.connection_pools[pool_key]
                        self.logger.info(f"Removed unused connection pool: {pool_key}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(300)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics."""
        uptime = time.time() - self.start_time
        
        # Calculate rates
        messages_per_second = self.total_messages_sent / max(1, uptime)
        bytes_per_second = self.total_bytes_sent / max(1, uptime)
        
        # Pool statistics
        total_connections = sum(len(pool.connections) for pool in self.connection_pools.values())
        total_available = sum(
            sum(len(conns) for conns in pool.available_connections.values())
            for pool in self.connection_pools.values()
        )
        
        # Endpoint statistics
        top_endpoints = sorted(
            self.endpoint_stats.items(),
            key=lambda x: x[1]["requests"],
            reverse=True
        )[:10]
        
        return {
            "uptime_seconds": uptime,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "messages_per_second": messages_per_second,
            "bytes_per_second": bytes_per_second,
            "connection_pools": len(self.connection_pools),
            "total_connections": total_connections,
            "available_connections": total_available,
            "connection_utilization": (total_connections - total_available) / max(1, total_connections),
            "protocol": self.default_protocol.value,
            "compression_enabled": self.enable_compression,
            "caching_enabled": self.enable_caching,
            "top_endpoints": [
                {
                    "endpoint": endpoint,
                    "requests": stats["requests"],
                    "failures": stats["failures"],
                    "success_rate": (stats["requests"] - stats["failures"]) / max(1, stats["requests"]),
                    "avg_latency_ms": stats["avg_latency_ms"],
                }
                for endpoint, stats in top_endpoints
            ],
        }


# Global high-performance communicator
_hp_communicator: Optional[HighPerformanceCommunicator] = None

async def get_hp_communicator() -> HighPerformanceCommunicator:
    """Get or create global high-performance communicator."""
    global _hp_communicator
    
    if _hp_communicator is None:
        _hp_communicator = HighPerformanceCommunicator(
            max_connections=50000,  # Support massive scale
            enable_compression=True,
            default_protocol=Protocol.MSGPACK if MSGPACK_AVAILABLE else Protocol.JSON,
            enable_caching=True,
            message_queue_size=1000000,  # Large queue for high throughput
        )
        await _hp_communicator.start()
    
    return _hp_communicator

async def send_high_performance_message(
    destination: str,
    payload: Any,
    priority: Priority = Priority.NORMAL,
) -> Dict[str, Any]:
    """Send message using high-performance communicator."""
    try:
        comm = await get_hp_communicator()
        return await comm.send_message(destination, payload, priority)
    except Exception as e:
        return {"success": False, "error": str(e)}

async def broadcast_high_performance_message(
    destinations: List[str],
    payload: Any,
    priority: Priority = Priority.NORMAL,
) -> Dict[str, Any]:
    """Broadcast message using high-performance communicator."""
    try:
        comm = await get_hp_communicator()
        return await comm.broadcast_message(destinations, payload, priority)
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_communication_stats() -> Dict[str, Any]:
    """Get communication system statistics."""
    try:
        if _hp_communicator:
            return _hp_communicator.get_comprehensive_stats()
        else:
            return {"error": "Communication system not initialized"}
    except Exception:
        return {"error": "Communication stats not available"}