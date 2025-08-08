#!/usr/bin/env python3
"""Basic Fleet-Mind mission example - demonstrates core functionality."""

import asyncio
import time
from fleet_mind import (
    SwarmCoordinator, 
    DroneFleet, 
    MissionConstraints,
    DroneCapability,
    performance_monitor
)

async def basic_mission_example():
    """Demonstrate basic mission planning and execution."""
    print("🚁 Fleet-Mind Basic Mission Example")
    print("=" * 50)
    
    # Initialize fleet
    print("\n1. Initializing drone fleet...")
    drone_ids = [f"drone_{i}" for i in range(5)]
    fleet = DroneFleet(
        drone_ids=drone_ids,
        communication_protocol="webrtc",
        topology="mesh"
    )
    
    # Add capabilities to drones
    for drone_id in drone_ids[:3]:  # First 3 drones get cameras
        fleet.add_drone_capability(drone_id, DroneCapability.CAMERA)
    for drone_id in drone_ids[1:4]:  # Middle 3 get thermal sensors
        fleet.add_drone_capability(drone_id, DroneCapability.THERMAL)
    
    print(f"✓ Fleet initialized with {len(drone_ids)} drones")
    
    # Initialize coordinator
    print("\n2. Setting up swarm coordinator...")
    constraints = MissionConstraints(
        max_altitude=100.0,
        battery_time=25.0,
        safety_distance=8.0
    )
    
    coordinator = SwarmCoordinator(
        llm_model="gpt-4o",
        latent_dim=512,
        max_drones=10,
        update_rate=10.0,
        safety_constraints=constraints
    )
    
    # Connect fleet to coordinator
    await coordinator.connect_fleet(fleet)
    await fleet.start_monitoring()
    print("✓ Coordinator connected and monitoring started")
    
    # Define mission
    mission_description = """
    Search and rescue mission: Survey a 200x200m area looking for survivors.
    Form a grid pattern at 50m altitude, maintain 20m spacing between drones.
    Use thermal cameras to detect heat signatures. If survivors found, mark location and alert ground team.
    """
    
    print(f"\n3. Mission: {mission_description.strip()}")
    
    # Generate mission plan
    print("\n4. Generating mission plan...")
    start_time = time.time()
    
    plan = await coordinator.generate_plan(
        mission_description,
        constraints=constraints,
        context={
            'environment': {'weather': 'clear', 'wind_speed': 5, 'visibility': 'good'},
            'urgency': 'high'
        }
    )
    
    planning_time = (time.time() - start_time) * 1000
    print(f"✓ Plan generated in {planning_time:.1f}ms")
    
    # Display plan summary
    print(f"\nMission Plan Summary:")
    print(f"  Mission ID: {plan['mission_id']}")
    print(f"  Description: {plan['raw_plan'].get('summary', 'N/A')}")
    print(f"  Estimated Duration: {plan['raw_plan'].get('estimated_duration_minutes', 0):.1f} minutes")
    print(f"  Objectives: {len(plan['raw_plan'].get('objectives', []))}")
    print(f"  Action Sequences: {len(plan['raw_plan'].get('action_sequences', []))}")
    
    # Show plan objectives
    objectives = plan['raw_plan'].get('objectives', [])
    if objectives:
        print(f"\nObjectives:")
        for i, obj in enumerate(objectives, 1):
            print(f"  {i}. {obj}")
    
    # Execute mission (simulation)
    print(f"\n5. Executing mission...")
    
    # Start execution
    execution_task = asyncio.create_task(
        coordinator.execute_mission(
            plan, 
            monitor_frequency=5.0,
            replan_threshold=0.7
        )
    )
    
    # Monitor progress
    monitor_duration = 10  # Monitor for 10 seconds
    for i in range(monitor_duration):
        await asyncio.sleep(1)
        
        # Get status
        status = await coordinator.get_swarm_status()
        fleet_status = fleet.get_fleet_status()
        
        print(f"  [{i+1:2d}s] Mission: {status['mission_status']:10} | "
              f"Active: {fleet_status['active_drones']:2d} | "
              f"Battery: {fleet_status['average_battery']:5.1f}% | "
              f"Health: {fleet_status['average_health']:4.2f}")
    
    # Stop execution for demo
    await coordinator.emergency_stop()
    execution_task.cancel()
    
    print(f"✓ Mission demonstration completed")
    
    # Show performance statistics
    print(f"\n6. Performance Statistics:")
    perf_stats = coordinator.get_comprehensive_stats()
    
    print(f"  System Health:")
    sys_health = perf_stats.get('system_health', {})
    print(f"    Memory Usage: {sys_health.get('memory_usage_mb', 0):.1f} MB")
    print(f"    Active Tasks: {sys_health.get('active_tasks', 0)}")
    print(f"    Error Rate: {sys_health.get('error_rate', 0):.2%}")
    
    print(f"  Communication:")
    comm_status = coordinator.webrtc_streamer.get_status()
    print(f"    Active Connections: {comm_status['active_connections']}")
    print(f"    Average Latency: {comm_status['average_latency_ms']:.1f}ms")
    print(f"    Total Bandwidth: {comm_status['total_bandwidth_mbps']:.2f} Mbps")
    
    # Cleanup
    await fleet.stop_monitoring()
    print(f"\n✓ Mission example completed successfully!")

@performance_monitor
async def monitored_mission():
    """Wrapper to demonstrate performance monitoring."""
    await basic_mission_example()

if __name__ == "__main__":
    try:
        asyncio.run(monitored_mission())
    except KeyboardInterrupt:
        print("\n\nMission interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")