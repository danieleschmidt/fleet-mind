"""Tests for WebRTC Streamer functionality."""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock

from fleet_mind.communication.webrtc_streamer import (
    WebRTCStreamer, 
    StreamerConfig, 
    MessagePriority, 
    ReliabilityMode,
    ConnectionMetrics
)


@pytest.fixture
def streamer_config():
    """Test streamer configuration."""
    return StreamerConfig(
        stun_servers=['stun:test.example.com:19302'],
        turn_servers=[],
        codec='h264',
        bitrate=500000,
        max_connections=10,
        connection_timeout=5.0,
    )


@pytest.fixture
async def streamer(streamer_config):
    """Create WebRTC streamer instance."""
    streamer = WebRTCStreamer(streamer_config)
    yield streamer
    # Cleanup
    await streamer.close()


class TestWebRTCStreamer:
    """Test cases for WebRTC Streamer."""
    
    def test_initialization(self, streamer, streamer_config):
        """Test streamer initialization."""
        assert streamer.config == streamer_config
        assert not streamer.is_initialized
        assert len(streamer.connections) == 0
        assert len(streamer.active_drones) == 0

    @pytest.mark.asyncio
    async def test_initialize_with_drones(self, streamer):
        """Test initialization with drone list."""
        drone_ids = ['drone_1', 'drone_2', 'drone_3']
        
        # Mock connection establishment
        with patch.object(streamer, '_establish_connection', new=AsyncMock()) as mock_establish:
            await streamer.initialize(drone_ids)
            
            assert mock_establish.call_count == len(drone_ids)
            assert streamer.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_too_many_drones(self, streamer):
        """Test initialization with too many drones."""
        # Create more drones than max_connections
        drone_ids = [f'drone_{i}' for i in range(streamer.config.max_connections + 1)]
        
        with pytest.raises(ValueError, match="Too many drones"):
            await streamer.initialize(drone_ids)

    @pytest.mark.asyncio
    async def test_broadcast_message(self, streamer):
        """Test message broadcasting."""
        # Setup
        drone_ids = ['drone_1', 'drone_2']
        streamer.active_drones = set(drone_ids)
        streamer.is_initialized = True
        
        test_data = {"action": "takeoff", "altitude": 50}
        
        # Mock message queuing
        with patch.object(streamer.message_queues[MessagePriority.REAL_TIME], 'put', new=AsyncMock()) as mock_put:
            result = await streamer.broadcast(
                test_data,
                priority="real_time",
                reliability="best_effort"
            )
            
            assert mock_put.call_count == len(drone_ids)
            assert all(result.values())

    @pytest.mark.asyncio
    async def test_broadcast_not_initialized(self, streamer):
        """Test broadcast when not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await streamer.broadcast({"test": "data"})

    @pytest.mark.asyncio
    async def test_broadcast_invalid_data(self, streamer):
        """Test broadcast with invalid data."""
        streamer.is_initialized = True
        
        # Test with non-serializable data
        invalid_data = {"func": lambda x: x}  # Lambda functions are not JSON serializable
        
        with pytest.raises(ValueError, match="Data serialization failed"):
            await streamer.broadcast(invalid_data)

    @pytest.mark.asyncio
    async def test_send_to_drone(self, streamer):
        """Test sending message to specific drone."""
        drone_id = 'drone_1'
        streamer.active_drones = {drone_id}
        streamer.is_initialized = True
        
        test_data = {"command": "land"}
        
        with patch.object(streamer, 'broadcast', new=AsyncMock(return_value={drone_id: True})) as mock_broadcast:
            result = await streamer.send_to_drone(drone_id, test_data)
            
            assert result is True
            mock_broadcast.assert_called_once_with(
                test_data, "real_time", "reliable", [drone_id]
            )

    def test_get_status(self, streamer):
        """Test status retrieval."""
        # Setup test state
        streamer.is_initialized = True
        streamer.active_drones = {'drone_1', 'drone_2'}
        streamer.connections = {'drone_1': Mock(), 'drone_2': Mock()}
        streamer.connection_metrics = {
            'drone_1': ConnectionMetrics(latency_ms=25.0, packet_loss=0.01),
            'drone_2': ConnectionMetrics(latency_ms=30.0, packet_loss=0.02),
        }
        
        status = streamer.get_status()
        
        assert status['initialized'] is True
        assert status['active_connections'] == 2
        assert status['total_connections'] == 2
        assert 'average_latency_ms' in status
        assert 'queue_sizes' in status
        assert 'connection_health' in status

    @pytest.mark.asyncio
    async def test_connection_establishment(self, streamer):
        """Test WebRTC connection establishment."""
        drone_id = 'test_drone'
        
        # Mock WebRTC components
        mock_connection = Mock()
        mock_data_channel = Mock()
        
        with patch('fleet_mind.communication.webrtc_streamer.RTCPeerConnection', return_value=mock_connection):
            with patch.object(mock_connection, 'createDataChannel', return_value=mock_data_channel):
                with patch.object(streamer, '_simulate_signaling', new=AsyncMock()):
                    await streamer._establish_connection(drone_id)
                    
                    assert drone_id in streamer.connections
                    assert drone_id in streamer.data_channels

    @pytest.mark.asyncio
    async def test_message_sender_loop(self, streamer):
        """Test message sender loop."""
        # Setup
        drone_id = 'test_drone'
        test_message = {
            'drone_id': drone_id,
            'data': '{"test": "data"}',
            'reliability': ReliabilityMode.BEST_EFFORT,
            'timestamp': time.time(),
        }
        
        # Mock data channel
        mock_channel = Mock()
        mock_channel.readyState = "open"
        streamer.data_channels[drone_id] = mock_channel
        
        # Add message to queue
        await streamer.message_queues[MessagePriority.REAL_TIME].put(test_message)
        
        # Run sender loop for short time
        sender_task = asyncio.create_task(streamer._message_sender_loop())
        await asyncio.sleep(0.1)
        sender_task.cancel()
        
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        
        # Verify message was sent
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_collection_loop(self, streamer):
        """Test metrics collection loop."""
        drone_id = 'test_drone'
        streamer.active_drones = {drone_id}
        
        # Mock connection
        mock_connection = Mock()
        streamer.connections[drone_id] = mock_connection
        
        with patch.object(streamer, '_update_connection_metrics', new=AsyncMock()) as mock_update:
            # Run metrics loop for short time
            metrics_task = asyncio.create_task(streamer._metrics_collection_loop())
            await asyncio.sleep(0.1)
            metrics_task.cancel()
            
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
            
            # Verify metrics were updated
            mock_update.assert_called()

    @pytest.mark.asyncio
    async def test_connection_metrics_update(self, streamer):
        """Test connection metrics updating."""
        drone_id = 'test_drone'
        
        # Mock connection with stats
        mock_connection = Mock()
        mock_stats = AsyncMock(return_value=[])
        mock_connection.getStats = mock_stats
        
        streamer.connections[drone_id] = mock_connection
        
        await streamer._update_connection_metrics(drone_id)
        
        # Verify stats were collected
        mock_stats.assert_called_once()
        
        # Check metrics were created
        assert drone_id in streamer.connection_metrics

    @pytest.mark.asyncio
    async def test_handle_drone_message(self, streamer):
        """Test drone message handling."""
        drone_id = 'test_drone'
        
        # Test telemetry message
        telemetry_msg = json.dumps({
            'type': 'telemetry',
            'data': {'battery': 85, 'position': [10, 20, 30]}
        })
        
        with patch.object(streamer, '_handle_telemetry', new=AsyncMock()) as mock_handle:
            await streamer._handle_drone_message(drone_id, telemetry_msg)
            mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_invalid_message(self, streamer):
        """Test handling of invalid JSON messages."""
        drone_id = 'test_drone'
        invalid_msg = "not json"
        
        # Should not raise exception
        await streamer._handle_drone_message(drone_id, invalid_msg)

    @pytest.mark.asyncio
    async def test_emergency_message_handling(self, streamer):
        """Test emergency message handling."""
        drone_id = 'test_drone'
        
        emergency_msg = json.dumps({
            'type': 'emergency',
            'data': {'reason': 'low_battery', 'severity': 'critical'}
        })
        
        with patch.object(streamer, '_handle_emergency', new=AsyncMock()) as mock_handle:
            await streamer._handle_drone_message(drone_id, emergency_msg)
            mock_handle.assert_called_once()

    def test_average_latency_calculation(self, streamer):
        """Test average latency calculation."""
        # Setup metrics
        streamer.connection_metrics = {
            'drone_1': ConnectionMetrics(latency_ms=20.0),
            'drone_2': ConnectionMetrics(latency_ms=30.0),
            'drone_3': ConnectionMetrics(latency_ms=40.0),
        }
        
        avg_latency = streamer._get_average_latency()
        assert avg_latency == 30.0

    def test_average_latency_no_metrics(self, streamer):
        """Test average latency with no metrics."""
        streamer.connection_metrics = {}
        avg_latency = streamer._get_average_latency()
        assert avg_latency == 0.0

    def test_connection_health_assessment(self, streamer):
        """Test connection health assessment."""
        # Good connection
        streamer.connection_metrics['drone_1'] = ConnectionMetrics(
            latency_ms=25.0,
            packet_loss=0.01
        )
        health = streamer._assess_connection_health('drone_1')
        assert health == "good"
        
        # Poor connection
        streamer.connection_metrics['drone_2'] = ConnectionMetrics(
            latency_ms=150.0,
            packet_loss=0.08
        )
        health = streamer._assess_connection_health('drone_2')
        assert health == "poor"
        
        # Unknown connection
        health = streamer._assess_connection_health('nonexistent')
        assert health == "unknown"

    @pytest.mark.asyncio
    async def test_close_cleanup(self, streamer):
        """Test proper cleanup on close."""
        # Setup some state
        streamer.active_drones = {'drone_1', 'drone_2'}
        streamer.is_initialized = True
        
        # Mock tasks
        streamer._sender_task = Mock()
        streamer._metrics_task = Mock()
        
        # Mock connections
        mock_conn1 = AsyncMock()
        mock_conn2 = AsyncMock()
        streamer.connections = {'drone_1': mock_conn1, 'drone_2': mock_conn2}
        
        await streamer.close()
        
        # Verify cleanup
        assert len(streamer.connections) == 0
        assert len(streamer.active_drones) == 0
        assert not streamer.is_initialized
        
        # Verify connections were closed
        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_called_once()

    def test_message_priority_enum(self):
        """Test message priority enumeration."""
        assert MessagePriority.CRITICAL.value == "critical"
        assert MessagePriority.REAL_TIME.value == "real_time"
        assert MessagePriority.BEST_EFFORT.value == "best_effort"

    def test_reliability_mode_enum(self):
        """Test reliability mode enumeration."""
        assert ReliabilityMode.GUARANTEED.value == "guaranteed"
        assert ReliabilityMode.RELIABLE.value == "reliable"
        assert ReliabilityMode.BEST_EFFORT.value == "best_effort"

    @pytest.mark.asyncio
    async def test_broadcast_with_target_drones(self, streamer):
        """Test broadcast to specific target drones."""
        # Setup
        all_drones = {'drone_1', 'drone_2', 'drone_3'}
        target_drones = ['drone_1', 'drone_3']
        
        streamer.active_drones = all_drones
        streamer.is_initialized = True
        
        test_data = {"command": "hover"}
        
        with patch.object(streamer.message_queues[MessagePriority.REAL_TIME], 'put', new=AsyncMock()) as mock_put:
            result = await streamer.broadcast(
                test_data,
                target_drones=target_drones
            )
            
            # Should only queue messages for target drones
            assert mock_put.call_count == len(target_drones)
            assert result['drone_1'] is True
            assert result['drone_3'] is True
            assert 'drone_2' not in result

    @pytest.mark.asyncio
    async def test_connection_timeout(self, streamer):
        """Test connection establishment timeout."""
        drone_ids = ['drone_1']
        
        # Mock slow connection establishment
        async def slow_establish(drone_id):
            await asyncio.sleep(10)  # Longer than timeout
        
        with patch.object(streamer, '_establish_connection', side_effect=slow_establish):
            # This should complete due to timeout, not wait indefinitely
            start_time = time.time()
            await streamer.initialize(drone_ids)
            elapsed = time.time() - start_time
            
            # Should complete within reasonable time (timeout + buffer)
            assert elapsed < streamer.config.connection_timeout + 2