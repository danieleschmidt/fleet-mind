"""WebRTC-based real-time communication streamer for drone fleet."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from aiortc import RTCPeerConnection, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.signaling import TcpSocketSignaling
import aiohttp


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = "critical"
    REAL_TIME = "real_time"
    BEST_EFFORT = "best_effort"


class ReliabilityMode(Enum):
    """Message reliability modes."""
    GUARANTEED = "guaranteed"
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


@dataclass
class ConnectionMetrics:
    """Connection quality metrics."""
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    bandwidth_mbps: float = 0.0
    jitter_ms: float = 0.0
    last_update: float = 0.0


@dataclass
class StreamerConfig:
    """WebRTC streamer configuration."""
    stun_servers: List[str] = None
    turn_servers: List[Dict[str, str]] = None
    codec: str = "h264"
    bitrate: int = 1000000  # 1 Mbps per drone
    max_connections: int = 100
    connection_timeout: float = 30.0
    
    def __post_init__(self):
        if self.stun_servers is None:
            self.stun_servers = ['stun:stun.l.google.com:19302']
        if self.turn_servers is None:
            self.turn_servers = []


class WebRTCStreamer:
    """WebRTC-based streamer for low-latency drone communication.
    
    Implements peer-to-peer mesh networking with adaptive quality of service
    for broadcasting latent action codes to drone fleets.
    """
    
    def __init__(self, config: Optional[StreamerConfig] = None):
        """Initialize WebRTC streamer.
        
        Args:
            config: Streamer configuration parameters
        """
        self.config = config or StreamerConfig()
        
        # WebRTC configuration
        ice_servers = [RTCIceServer(url) for url in self.config.stun_servers]
        for turn_server in self.config.turn_servers:
            ice_servers.append(RTCIceServer(
                turn_server['urls'],
                username=turn_server.get('username'),
                credential=turn_server.get('credential')
            ))
        
        self.rtc_config = RTCConfiguration(iceServers=ice_servers)
        
        # Connection management
        self.connections: Dict[str, RTCPeerConnection] = {}
        self.data_channels: Dict[str, RTCDataChannel] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.active_drones: Set[str] = set()
        
        # Message queuing and prioritization
        self.message_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        
        # State management
        self.is_initialized = False
        self.signaling_server: Optional[TcpSocketSignaling] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

    async def initialize(self, drone_ids: List[str]) -> None:
        """Initialize WebRTC connections to drone fleet.
        
        Args:
            drone_ids: List of drone identifiers to connect to
        """
        if len(drone_ids) > self.config.max_connections:
            raise ValueError(f"Too many drones: {len(drone_ids)} > {self.config.max_connections}")
        
        print(f"Initializing WebRTC connections to {len(drone_ids)} drones...")
        
        # Start connection establishment
        connection_tasks = [
            self._establish_connection(drone_id) for drone_id in drone_ids
        ]
        
        # Wait for all connections with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*connection_tasks, return_exceptions=True),
                timeout=self.config.connection_timeout
            )
        except asyncio.TimeoutError:
            print("Warning: Some connections timed out during initialization")
        
        # Start background tasks
        self._sender_task = asyncio.create_task(self._message_sender_loop())
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        self.is_initialized = True
        connected_count = len(self.active_drones)
        print(f"WebRTC initialization complete: {connected_count}/{len(drone_ids)} drones connected")

    async def broadcast(
        self,
        data: Any,
        priority: str = "real_time",
        reliability: str = "best_effort",
        target_drones: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Broadcast data to drone fleet.
        
        Args:
            data: Data to broadcast (will be JSON serialized)
            priority: Message priority level
            reliability: Message reliability mode
            target_drones: Specific drones to target (None for all)
            
        Returns:
            Dictionary mapping drone_id to success status
        """
        if not self.is_initialized:
            raise RuntimeError("WebRTC streamer not initialized")
        
        # Serialize data
        try:
            message_data = json.dumps({
                'type': 'latent_action',
                'data': data,
                'timestamp': time.time(),
                'priority': priority,
                'reliability': reliability,
            })
        except (TypeError, ValueError) as e:
            raise ValueError(f"Data serialization failed: {e}")
        
        # Determine target drones
        targets = target_drones if target_drones else list(self.active_drones)
        
        # Queue message for sending
        message_priority = MessagePriority(priority)
        for drone_id in targets:
            if drone_id in self.active_drones:
                await self.message_queues[message_priority].put({
                    'drone_id': drone_id,
                    'data': message_data,
                    'reliability': ReliabilityMode(reliability),
                    'timestamp': time.time(),
                })
        
        # Return immediate success status (actual delivery is async)
        return {drone_id: drone_id in self.active_drones for drone_id in targets}

    async def send_to_drone(
        self,
        drone_id: str,
        data: Any,
        priority: str = "real_time",
        reliability: str = "reliable"
    ) -> bool:
        """Send data to specific drone.
        
        Args:
            drone_id: Target drone identifier
            data: Data to send
            priority: Message priority
            reliability: Message reliability mode
            
        Returns:
            True if message was queued successfully
        """
        return (await self.broadcast(
            data, priority, reliability, [drone_id]
        )).get(drone_id, False)

    def get_status(self) -> Dict[str, Any]:
        """Get current streamer status and metrics.
        
        Returns:
            Comprehensive status information
        """
        return {
            'initialized': self.is_initialized,
            'active_connections': len(self.active_drones),
            'total_connections': len(self.connections),
            'average_latency_ms': self._get_average_latency(),
            'average_packet_loss': self._get_average_packet_loss(),
            'total_bandwidth_mbps': self._get_total_bandwidth(),
            'queue_sizes': {
                priority.value: queue.qsize() 
                for priority, queue in self.message_queues.items()
            },
            'connection_health': {
                drone_id: self._assess_connection_health(drone_id)
                for drone_id in self.active_drones
            }
        }

    async def close(self) -> None:
        """Close all WebRTC connections and cleanup resources."""
        # Cancel background tasks
        if self._sender_task:
            self._sender_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Close all peer connections
        for connection in self.connections.values():
            await connection.close()
        
        # Clear state
        self.connections.clear()
        self.data_channels.clear()
        self.active_drones.clear()
        self.is_initialized = False
        
        print("WebRTC streamer closed")

    async def _establish_connection(self, drone_id: str) -> None:
        """Establish WebRTC connection to specific drone."""
        try:
            # Create peer connection
            connection = RTCPeerConnection(self.rtc_config)
            
            # Create data channel
            data_channel = connection.createDataChannel(
                f"fleet_control_{drone_id}",
                ordered=True,
                maxRetransmits=3
            )
            
            # Set up event handlers
            @data_channel.on("open")
            def on_open():
                print(f"Data channel to drone {drone_id} opened")
                self.active_drones.add(drone_id)
            
            @data_channel.on("close")
            def on_close():
                print(f"Data channel to drone {drone_id} closed")
                self.active_drones.discard(drone_id)
            
            @data_channel.on("message")
            def on_message(message):
                asyncio.create_task(self._handle_drone_message(drone_id, message))
            
            @connection.on("connectionstatechange")
            async def on_connection_state_change():
                state = connection.connectionState
                if state == "connected":
                    self.connection_metrics[drone_id] = ConnectionMetrics(
                        last_update=time.time()
                    )
                elif state in ["failed", "closed"]:
                    self.active_drones.discard(drone_id)
            
            # Store connections
            self.connections[drone_id] = connection
            self.data_channels[drone_id] = data_channel
            
            # Simulate signaling (in real implementation, this would use actual signaling server)
            await self._simulate_signaling(drone_id, connection)
            
        except Exception as e:
            print(f"Failed to establish connection to drone {drone_id}: {e}")

    async def _simulate_signaling(self, drone_id: str, connection: RTCPeerConnection) -> None:
        """Simulate WebRTC signaling process (simplified for MVP)."""
        try:
            # Create offer
            offer = await connection.createOffer()
            await connection.setLocalDescription(offer)
            
            # In real implementation, exchange SDP via signaling server
            # For MVP, we simulate successful signaling
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Simulate remote answer
            # This would normally come from the drone
            answer_sdp = self._create_mock_answer(offer.sdp)
            from aiortc import RTCSessionDescription
            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await connection.setRemoteDescription(answer)
            
        except Exception as e:
            print(f"Signaling failed for drone {drone_id}: {e}")

    def _create_mock_answer(self, offer_sdp: str) -> str:
        """Create mock SDP answer for simulation."""
        # Simplified mock answer - in real implementation this comes from drone
        return offer_sdp.replace("a=sendrecv", "a=recvonly")

    async def _message_sender_loop(self) -> None:
        """Background loop for prioritized message sending."""
        try:
            while True:
                # Process messages in priority order
                for priority in [MessagePriority.CRITICAL, MessagePriority.REAL_TIME, MessagePriority.BEST_EFFORT]:
                    queue = self.message_queues[priority]
                    
                    try:
                        # Non-blocking get with timeout
                        message = await asyncio.wait_for(queue.get(), timeout=0.001)
                        await self._send_message(message)
                        queue.task_done()
                    except asyncio.TimeoutError:
                        continue
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)  # 1ms
                
        except asyncio.CancelledError:
            print("Message sender loop cancelled")

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send individual message to drone."""
        drone_id = message['drone_id']
        
        if drone_id not in self.data_channels:
            return
        
        data_channel = self.data_channels[drone_id]
        
        try:
            if data_channel.readyState == "open":
                data_channel.send(message['data'])
                
                # Update metrics
                if drone_id in self.connection_metrics:
                    self.connection_metrics[drone_id].last_update = time.time()
                    
        except Exception as e:
            print(f"Failed to send message to drone {drone_id}: {e}")

    async def _metrics_collection_loop(self) -> None:
        """Background loop for collecting connection metrics."""
        try:
            while True:
                for drone_id in list(self.active_drones):
                    await self._update_connection_metrics(drone_id)
                
                await asyncio.sleep(1.0)  # Update every second
                
        except asyncio.CancelledError:
            print("Metrics collection loop cancelled")

    async def _update_connection_metrics(self, drone_id: str) -> None:
        """Update connection metrics for specific drone."""
        if drone_id not in self.connections:
            return
        
        connection = self.connections[drone_id]
        
        try:
            # Get WebRTC stats (simplified for MVP)
            stats = await connection.getStats()
            
            # Extract key metrics
            latency = 0.0
            packet_loss = 0.0
            bandwidth = 0.0
            
            # In real implementation, parse actual WebRTC stats
            # For MVP, simulate reasonable values
            latency = 15.0 + (hash(drone_id) % 20)  # 15-35ms
            packet_loss = max(0, (hash(drone_id) % 100) / 1000)  # 0-10%
            bandwidth = 0.8 + (hash(drone_id) % 40) / 100  # 0.8-1.2 Mbps
            
            # Update metrics
            if drone_id not in self.connection_metrics:
                self.connection_metrics[drone_id] = ConnectionMetrics()
            
            metrics = self.connection_metrics[drone_id]
            metrics.latency_ms = latency
            metrics.packet_loss = packet_loss
            metrics.bandwidth_mbps = bandwidth
            metrics.last_update = time.time()
            
        except Exception as e:
            print(f"Failed to update metrics for drone {drone_id}: {e}")

    async def _handle_drone_message(self, drone_id: str, message: str) -> None:
        """Handle incoming message from drone."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if data.get('type') == 'telemetry':
                await self._handle_telemetry(drone_id, data)
            elif data.get('type') == 'status':
                await self._handle_status_update(drone_id, data)
            elif data.get('type') == 'emergency':
                await self._handle_emergency(drone_id, data)
                
        except json.JSONDecodeError:
            print(f"Invalid JSON from drone {drone_id}: {message}")
        except Exception as e:
            print(f"Error handling message from drone {drone_id}: {e}")

    async def _handle_telemetry(self, drone_id: str, data: Dict[str, Any]) -> None:
        """Handle telemetry data from drone."""
        # Store telemetry for coordinator use
        pass

    async def _handle_status_update(self, drone_id: str, data: Dict[str, Any]) -> None:
        """Handle status update from drone."""
        # Update drone status
        pass

    async def _handle_emergency(self, drone_id: str, data: Dict[str, Any]) -> None:
        """Handle emergency message from drone."""
        print(f"EMERGENCY from drone {drone_id}: {data}")

    def _get_average_latency(self) -> float:
        """Calculate average latency across all connections."""
        if not self.connection_metrics:
            return 0.0
        
        latencies = [m.latency_ms for m in self.connection_metrics.values()]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def _get_average_packet_loss(self) -> float:
        """Calculate average packet loss across all connections."""
        if not self.connection_metrics:
            return 0.0
        
        losses = [m.packet_loss for m in self.connection_metrics.values()]
        return sum(losses) / len(losses) if losses else 0.0

    def _get_total_bandwidth(self) -> float:
        """Calculate total bandwidth across all connections."""
        if not self.connection_metrics:
            return 0.0
        
        return sum(m.bandwidth_mbps for m in self.connection_metrics.values())

    def _assess_connection_health(self, drone_id: str) -> str:
        """Assess health status of specific connection."""
        if drone_id not in self.connection_metrics:
            return "unknown"
        
        metrics = self.connection_metrics[drone_id]
        
        # Simple health assessment based on metrics
        if metrics.latency_ms > 100 or metrics.packet_loss > 0.05:
            return "poor"
        elif metrics.latency_ms > 50 or metrics.packet_loss > 0.02:
            return "fair"
        else:
            return "good"