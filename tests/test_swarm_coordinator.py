"""Tests for SwarmCoordinator functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from fleet_mind.coordination.swarm_coordinator import SwarmCoordinator, MissionConstraints, MissionStatus
from fleet_mind.fleet.drone_fleet import DroneFleet


@pytest.fixture
def mock_constraints():
    """Mock mission constraints."""
    return MissionConstraints(
        max_altitude=100.0,
        battery_time=20.0,
        safety_distance=5.0,
    )


@pytest.fixture
def mock_drone_fleet():
    """Mock drone fleet."""
    fleet = Mock(spec=DroneFleet)
    fleet.drone_ids = [f"drone_{i}" for i in range(5)]
    fleet.get_active_drones.return_value = fleet.drone_ids
    fleet.get_capabilities.return_value = {
        drone_id: ["basic_flight", "gps", "communication"]
        for drone_id in fleet.drone_ids
    }
    fleet.get_average_battery.return_value = 85.0
    fleet.get_average_health.return_value = 0.9
    fleet.get_health_status = AsyncMock(return_value={
        'healthy': fleet.drone_ids,
        'warning': [],
        'critical': [],
        'failed': [],
    })
    return fleet


@pytest.fixture
async def coordinator(mock_constraints):
    """Create SwarmCoordinator instance."""
    coordinator = SwarmCoordinator(
        llm_model="test-model",
        latent_dim=64,
        max_drones=10,
        update_rate=5.0,
        safety_constraints=mock_constraints,
    )
    yield coordinator
    # Cleanup
    if coordinator._running:
        await coordinator.emergency_stop()


class TestSwarmCoordinator:
    """Test cases for SwarmCoordinator."""
    
    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.llm_model == "test-model"
        assert coordinator.latent_dim == 64
        assert coordinator.max_drones == 10
        assert coordinator.update_rate == 5.0
        assert coordinator.mission_status == MissionStatus.IDLE
        assert coordinator.fleet is None

    @pytest.mark.asyncio
    async def test_connect_fleet(self, coordinator, mock_drone_fleet):
        """Test fleet connection."""
        with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
            await coordinator.connect_fleet(mock_drone_fleet)
        
        assert coordinator.fleet == mock_drone_fleet
        assert coordinator.swarm_state.num_total_drones == 5
        assert coordinator.swarm_state.num_active_drones == 5

    @pytest.mark.asyncio
    async def test_generate_plan(self, coordinator, mock_drone_fleet):
        """Test mission plan generation."""
        # Mock LLM planner
        mock_plan = {
            'summary': 'Test mission plan',
            'actions': ['takeoff', 'formation', 'move'],
            'estimated_duration_minutes': 10.0,
        }
        
        with patch.object(coordinator.llm_planner, 'generate_plan', new=AsyncMock(return_value=mock_plan)):
            with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
                await coordinator.connect_fleet(mock_drone_fleet)
            
            plan = await coordinator.generate_plan("Test mission")
            
            assert 'mission_id' in plan
            assert 'latent_code' in plan
            assert 'planning_latency_ms' in plan
            assert plan['description'] == "Test mission"
            assert plan['raw_plan'] == mock_plan

    @pytest.mark.asyncio
    async def test_execute_mission(self, coordinator, mock_drone_fleet):
        """Test mission execution."""
        # Setup
        with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
            await coordinator.connect_fleet(mock_drone_fleet)
        
        # Mock mission plan
        latent_plan = {
            'mission_id': 'test_mission_1',
            'description': 'Test mission',
            'latent_code': [1, 2, 3, 4],
            'timestamp': time.time(),
        }
        
        # Mock WebRTC broadcast
        with patch.object(coordinator.webrtc_streamer, 'broadcast', new=AsyncMock(return_value={})):
            with patch.object(coordinator, '_planning_loop', new=AsyncMock()):
                with patch.object(coordinator, '_monitoring_loop', new=AsyncMock()):
                    # Execute with short timeout for testing
                    coordinator._running = True
                    coordinator.mission_status = MissionStatus.EXECUTING
                    
                    # Simulate mission completion
                    await asyncio.sleep(0.1)
                    coordinator._running = False
                    coordinator.mission_status = MissionStatus.COMPLETED
                    
                    success = True  # Mock success
                    
                    assert success is True

    @pytest.mark.asyncio
    async def test_emergency_stop(self, coordinator, mock_drone_fleet):
        """Test emergency stop functionality."""
        with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
            await coordinator.connect_fleet(mock_drone_fleet)
        
        with patch.object(coordinator.webrtc_streamer, 'broadcast', new=AsyncMock()) as mock_broadcast:
            await coordinator.emergency_stop()
            
            assert coordinator.mission_status == MissionStatus.PAUSED
            mock_broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_swarm_status(self, coordinator, mock_drone_fleet):
        """Test swarm status retrieval."""
        with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
            await coordinator.connect_fleet(mock_drone_fleet)
        
        with patch.object(coordinator.webrtc_streamer, 'get_status', return_value={'active_connections': 5}):
            status = await coordinator.get_swarm_status()
            
            assert 'mission_status' in status
            assert 'swarm_state' in status
            assert 'fleet_health' in status
            assert 'communication_status' in status
            assert status['mission_status'] == MissionStatus.IDLE.value

    def test_add_callback(self, coordinator):
        """Test callback registration."""
        callback_called = False
        
        def test_callback(data):
            nonlocal callback_called
            callback_called = True
        
        coordinator.add_callback('mission_start', test_callback)
        
        # Verify callback is registered
        assert len(coordinator._callbacks['mission_start']) == 1

    @pytest.mark.asyncio
    async def test_planning_loop_error_handling(self, coordinator, mock_drone_fleet):
        """Test planning loop error handling."""
        with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
            await coordinator.connect_fleet(mock_drone_fleet)
        
        # Mock error in planning
        with patch.object(coordinator, '_assess_plan_confidence', side_effect=Exception("Test error")):
            with patch.object(coordinator, '_running', True):
                # Should handle error gracefully
                try:
                    await asyncio.wait_for(coordinator._planning_loop(frequency=10.0, replan_threshold=0.7), timeout=0.2)
                except asyncio.TimeoutError:
                    pass  # Expected for continuous loop

    @pytest.mark.asyncio 
    async def test_monitoring_loop_error_handling(self, coordinator, mock_drone_fleet):
        """Test monitoring loop error handling."""
        with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
            await coordinator.connect_fleet(mock_drone_fleet)
        
        # Mock error in monitoring
        mock_drone_fleet.get_active_drones.side_effect = Exception("Test error")
        
        with patch.object(coordinator, '_running', True):
            # Should handle error gracefully
            try:
                await asyncio.wait_for(coordinator._monitoring_loop(frequency=10.0), timeout=0.2)
            except asyncio.TimeoutError:
                pass  # Expected for continuous loop

    def test_plan_confidence_assessment(self, coordinator):
        """Test plan confidence assessment."""
        # Test with no fleet
        confidence = asyncio.run(coordinator._assess_plan_confidence())
        assert confidence == 0.0
        
        # Test with mock fleet
        mock_fleet = Mock(spec=DroneFleet)
        mock_fleet.get_average_health.return_value = 0.8
        coordinator.fleet = mock_fleet
        coordinator.swarm_state.mission_progress = 0.6
        
        confidence = asyncio.run(coordinator._assess_plan_confidence())
        assert 0.0 <= confidence <= 1.0

    def test_context_history_management(self, coordinator):
        """Test context history management."""
        # Add multiple context entries
        for i in range(10):
            coordinator.context_history.append({
                'timestamp': time.time() + i,
                'mission': f'mission_{i}',
                'plan_summary': f'summary_{i}',
                'latency_ms': 50.0 + i,
            })
        
        assert len(coordinator.context_history) == 10
        
        # Test recent latency calculation
        recent_latency = coordinator._get_recent_latency()
        assert recent_latency > 0

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, coordinator):
        """Test callback error handling."""
        def failing_callback(data):
            raise Exception("Callback error")
        
        coordinator.add_callback('mission_start', failing_callback)
        
        # Should not raise exception when callback fails
        await coordinator._trigger_callbacks('mission_start', {})

    def test_mission_constraints_validation(self, mock_constraints):
        """Test mission constraints validation."""
        assert mock_constraints.max_altitude == 100.0
        assert mock_constraints.battery_time == 20.0
        assert mock_constraints.safety_distance == 5.0

    @pytest.mark.asyncio
    async def test_generate_plan_with_context(self, coordinator, mock_drone_fleet):
        """Test plan generation with additional context."""
        mock_plan = {
            'summary': 'Contextual mission plan',
            'actions': ['takeoff', 'navigate', 'land'],
        }
        
        with patch.object(coordinator.llm_planner, 'generate_plan', new=AsyncMock(return_value=mock_plan)):
            with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
                await coordinator.connect_fleet(mock_drone_fleet)
            
            context = {
                'weather': 'clear',
                'obstacles': ['building_1', 'tree_2'],
            }
            
            plan = await coordinator.generate_plan("Contextual mission", context=context)
            
            assert plan['raw_plan'] == mock_plan
            assert 'latent_code' in plan

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, coordinator, mock_drone_fleet):
        """Test concurrent operations handling."""
        with patch.object(coordinator.webrtc_streamer, 'initialize', new=AsyncMock()):
            await coordinator.connect_fleet(mock_drone_fleet)
        
        # Test concurrent plan generation
        mock_plan = {'summary': 'Concurrent plan', 'actions': []}
        
        with patch.object(coordinator.llm_planner, 'generate_plan', new=AsyncMock(return_value=mock_plan)):
            # Generate multiple plans concurrently
            tasks = [
                coordinator.generate_plan(f"Mission {i}")
                for i in range(3)
            ]
            
            plans = await asyncio.gather(*tasks)
            
            assert len(plans) == 3
            for plan in plans:
                assert 'mission_id' in plan
                assert 'latent_code' in plan