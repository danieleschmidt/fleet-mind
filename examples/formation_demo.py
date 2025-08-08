#!/usr/bin/env python3
"""Formation flying demonstration with Fleet-Mind."""

import asyncio
import time
import math
from fleet_mind import (
    SwarmCoordinator,
    DroneFleet, 
    MissionConstraints,
    DroneCapability,
    PlanningLevel
)

async def formation_demo():
    """Demonstrate various formation flying patterns."""
    print("🛩️  Fleet-Mind Formation Flying Demo")
    print("=" * 50)
    
    # Initialize larger fleet for formations
    print("\n1. Initializing formation fleet...")
    num_drones = 12
    drone_ids = [f"formation_drone_{i}" for i in range(num_drones)]
    
    fleet = DroneFleet(
        drone_ids=drone_ids,
        communication_protocol="webrtc",
        topology="mesh"
    )
    
    # Add formation capabilities to all drones
    for drone_id in drone_ids:
        fleet.add_drone_capability(drone_id, DroneCapability.FORMATION_FLIGHT)
        fleet.add_drone_capability(drone_id, DroneCapability.PRECISION_HOVER)
        fleet.add_drone_capability(drone_id, DroneCapability.OBSTACLE_AVOIDANCE)
    
    print(f"✓ Formation fleet ready: {num_drones} drones")
    
    # Initialize coordinator with tactical focus
    print("\n2. Setting up tactical coordinator...")
    constraints = MissionConstraints(
        max_altitude=80.0,
        battery_time=20.0,
        safety_distance=3.0  # Tighter formation
    )
    
    coordinator = SwarmCoordinator(
        llm_model="gpt-4o",
        latent_dim=256,  # Smaller for faster tactical responses
        max_drones=num_drones,
        update_rate=20.0,  # Higher frequency for formation control
        safety_constraints=constraints
    )
    
    await coordinator.connect_fleet(fleet)
    await fleet.start_monitoring()
    print("✓ Tactical coordinator ready")
    
    # Test different formations
    formations_to_test = [
        {
            'name': 'Grid Formation',
            'description': 'Arrange drones in a 3x4 grid pattern with 10m spacing',
            'formation_type': 'grid',
            'spacing': 10.0,
            'mission': 'Form a precise 3x4 grid formation at 60m altitude for area surveillance'
        },
        {
            'name': 'V-Formation', 
            'description': 'Classic V formation for efficient flight',
            'formation_type': 'v_formation',
            'spacing': 8.0,
            'mission': 'Arrange in V formation like migrating birds for long-distance efficient flight'
        },
        {
            'name': 'Line Formation',
            'description': 'Single line formation for perimeter patrol',
            'formation_type': 'line',
            'spacing': 15.0,
            'mission': 'Form a single line formation for perimeter security patrol'
        },
        {
            'name': 'Circle Formation',
            'description': 'Circular formation around a central point',
            'formation_type': 'circle',
            'spacing': 12.0,
            'mission': 'Arrange in circular formation around target area for 360-degree surveillance'
        }
    ]
    
    for i, formation in enumerate(formations_to_test, 1):
        print(f"\n{i}. Testing {formation['name']}")
        print(f"   {formation['description']}")
        
        # Generate formation plan
        print("   Generating formation plan...")
        plan_start = time.time()
        
        plan = await coordinator.llm_planner.generate_plan(
            context={
                'mission': formation['mission'],
                'num_drones': num_drones,
                'constraints': constraints.__dict__,
                'drone_capabilities': fleet.get_capabilities(),
                'current_state': fleet.get_fleet_status(),
                'formation_requirements': {
                    'type': formation['formation_type'],
                    'spacing_meters': formation['spacing'],
                    'precision_required': True
                }
            },
            planning_level=PlanningLevel.TACTICAL,
            custom_instructions=f"Focus on precise {formation['formation_type']} formation control"
        )
        
        planning_time = (time.time() - plan_start) * 1000
        print(f"   ✓ Formation plan generated in {planning_time:.1f}ms")
        
        # Check formation candidates
        candidates = fleet.get_formation_candidates(
            formation['formation_type'],
            min_drones=num_drones,
            required_capabilities=[DroneCapability.FORMATION_FLIGHT]
        )
        
        print(f"   Formation-ready drones: {len(candidates)}/{num_drones}")
        
        # Calculate current formation quality
        target_formation = {
            'formation_type': formation['formation_type'],
            'spacing_meters': formation['spacing'],
            'orientation_degrees': 0.0
        }
        
        initial_quality = fleet.get_formation_quality_score(target_formation)
        print(f"   Initial formation quality: {initial_quality:.2f} ({initial_quality*100:.1f}%)")
        
        # Simulate formation execution
        print("   Executing formation...")
        
        # Update drone positions to simulate formation (simplified)
        await simulate_formation_execution(fleet, formation, duration=3)
        
        # Measure final quality
        final_quality = fleet.get_formation_quality_score(target_formation)
        improvement = final_quality - initial_quality
        
        print(f"   Final formation quality: {final_quality:.2f} ({final_quality*100:.1f}%)")
        print(f"   Improvement: {improvement:+.2f} ({improvement*100:+.1f}%)")
        
        if final_quality > 0.8:
            print("   ✅ Formation EXCELLENT")
        elif final_quality > 0.6:
            print("   ✅ Formation GOOD")  
        elif final_quality > 0.4:
            print("   ⚠️  Formation FAIR")
        else:
            print("   ❌ Formation POOR")
    
    # Formation transition demonstration
    print(f"\n5. Formation Transition Demo")
    print("   Demonstrating dynamic formation changes...")
    
    transition_mission = """
    Dynamic formation sequence: Start in line formation, transition to V formation,
    then to grid formation for area coverage. Maintain smooth transitions with
    collision avoidance and optimal timing coordination.
    """
    
    transition_plan = await coordinator.generate_plan(
        transition_mission,
        context={
            'formation_sequence': ['line', 'v_formation', 'grid'],
            'transition_time': 30,  # seconds per formation
            'coordination_priority': 'safety_first'
        }
    )
    
    print(f"   ✓ Transition plan: {len(transition_plan['raw_plan'].get('action_sequences', []))} phases")
    
    # Simulate quick transitions
    formations = ['line', 'v_formation', 'grid']
    for j, form_type in enumerate(formations):
        print(f"   Phase {j+1}: Transitioning to {form_type} formation...")
        await simulate_formation_execution(fleet, {'formation_type': form_type, 'spacing': 10.0}, duration=2)
        
        quality = fleet.get_formation_quality_score({
            'formation_type': form_type,
            'spacing_meters': 10.0
        })
        print(f"   Quality: {quality:.2f} - {'✅' if quality > 0.7 else '⚠️'}")
    
    # Performance analysis
    print(f"\n6. Formation Performance Analysis")
    perf_stats = coordinator.get_comprehensive_stats()
    
    print(f"   Tactical Performance:")
    print(f"     Average Planning Time: {perf_stats.get('performance_stats', {}).get('avg_execution_time', 0):.1f}ms")
    print(f"     Formation Response Rate: {perf_stats['swarm_status']['recent_latency_ms']:.1f}ms")
    
    formation_candidates = fleet.get_formation_candidates('grid')
    print(f"     Available Formation Drones: {len(formation_candidates)}")
    
    fleet_health = await fleet.get_health_status()
    print(f"     Healthy Drones: {len(fleet_health['healthy'])}")
    print(f"     Warning Drones: {len(fleet_health['warning'])}")
    
    # Cleanup
    await fleet.stop_monitoring()
    print(f"\n✅ Formation demonstration completed!")

async def simulate_formation_execution(fleet: DroneFleet, formation: dict, duration: int):
    """Simulate formation execution by updating drone positions."""
    formation_type = formation.get('formation_type', 'grid')
    spacing = formation.get('spacing', 10.0)
    
    # Get active drones
    active_drones = fleet.get_active_drones()
    
    # Generate target positions based on formation type
    positions = generate_formation_positions(formation_type, len(active_drones), spacing)
    
    # Simulate smooth transition over duration
    steps = duration * 10  # 10 steps per second
    for step in range(steps):
        progress = step / steps  # 0 to 1
        
        # Update each drone's position towards target
        for i, drone_id in enumerate(active_drones):
            if i < len(positions):
                current_state = fleet.get_drone_state(drone_id)
                if current_state:
                    # Interpolate towards target position
                    current_pos = current_state.position
                    target_pos = positions[i]
                    
                    new_pos = (
                        current_pos[0] + (target_pos[0] - current_pos[0]) * progress * 0.1,
                        current_pos[1] + (target_pos[1] - current_pos[1]) * progress * 0.1,
                        current_pos[2] + (target_pos[2] - current_pos[2]) * progress * 0.1
                    )
                    
                    # Update drone state
                    fleet.update_drone_state(
                        drone_id,
                        position=new_pos,
                        velocity=(
                            (target_pos[0] - current_pos[0]) * 0.1,
                            (target_pos[1] - current_pos[1]) * 0.1,
                            (target_pos[2] - current_pos[2]) * 0.1
                        )
                    )
        
        await asyncio.sleep(0.1)  # 10 Hz update rate

def generate_formation_positions(formation_type: str, num_drones: int, spacing: float):
    """Generate target positions for formation."""
    positions = []
    center_x, center_y, center_z = 0.0, 0.0, 60.0  # Formation center
    
    if formation_type == 'grid':
        # Grid formation
        cols = int(math.ceil(math.sqrt(num_drones)))
        rows = int(math.ceil(num_drones / cols))
        
        for i in range(num_drones):
            row = i // cols
            col = i % cols
            x = center_x + (col - cols/2) * spacing
            y = center_y + (row - rows/2) * spacing
            positions.append((x, y, center_z))
    
    elif formation_type == 'line':
        # Line formation
        for i in range(num_drones):
            x = center_x + (i - num_drones/2) * spacing
            positions.append((x, center_y, center_z))
    
    elif formation_type == 'v_formation':
        # V formation
        positions.append((center_x, center_y, center_z))  # Leader
        for i in range(1, num_drones):
            side = 1 if i % 2 else -1
            offset = (i + 1) // 2
            x = center_x + side * offset * spacing * 0.7
            y = center_y - offset * spacing
            positions.append((x, y, center_z))
    
    elif formation_type == 'circle':
        # Circular formation
        radius = spacing * num_drones / (2 * math.pi)
        for i in range(num_drones):
            angle = 2 * math.pi * i / num_drones
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y, center_z))
    
    else:
        # Default: random spread
        for i in range(num_drones):
            x = center_x + (i % 5 - 2) * spacing
            y = center_y + (i // 5 - 1) * spacing
            positions.append((x, y, center_z))
    
    return positions

if __name__ == "__main__":
    try:
        asyncio.run(formation_demo())
    except KeyboardInterrupt:
        print("\n\nFormation demo interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()