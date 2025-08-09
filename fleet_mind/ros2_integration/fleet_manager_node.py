"""ROS 2 Fleet Manager Node for Fleet-Mind coordination."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any

# ROS 2 imports with fallback
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String, Float32
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import OccupancyGrid
    from sensor_msgs.msg import PointCloud2, Image
    from std_srvs.srv import SetBool, Trigger
    from rclpy.callback_groups import ReentrantCallbackGroup
    ROS2_AVAILABLE = True
except ImportError:
    # Mock ROS 2 classes for environments without ROS 2
    class Node:
        def __init__(self, name): 
            self.name = name
            print(f"Mock ROS 2 Node created: {name}")
        def get_logger(self): 
            import logging
            return logging.getLogger(self.name)
        def create_publisher(self, msg_type, topic, qos): 
            return MockPublisher(topic)
        def create_subscription(self, msg_type, topic, callback, qos): 
            return MockSubscription(topic)
        def create_service(self, srv_type, name, callback): 
            return MockService(name)
        def create_timer(self, period, callback): 
            return MockTimer(period, callback)
        def destroy_node(self): pass
    
    class MockPublisher:
        def __init__(self, topic): self.topic = topic
        def publish(self, msg): pass
    
    class MockSubscription:
        def __init__(self, topic): self.topic = topic
    
    class MockService:
        def __init__(self, name): self.name = name
    
    class MockTimer:
        def __init__(self, period, callback): 
            self.period = period
            self.callback = callback
    
    class QoSProfile:
        def __init__(self, **kwargs): pass
    
    class ReliabilityPolicy:
        RELIABLE = "reliable"
        BEST_EFFORT = "best_effort"
    
    class DurabilityPolicy:
        TRANSIENT_LOCAL = "transient_local"
    
    class ReentrantCallbackGroup: pass
    
    class String: 
        def __init__(self): self.data = ""
    
    class Float32:
        def __init__(self): self.data = 0.0
    
    # Mock message classes
    class PoseStamped: 
        def __init__(self): 
            self.header = type('Header', (), {'stamp': 0, 'frame_id': ''})()
            self.pose = type('Pose', (), {'position': type('Point', (), {'x': 0, 'y': 0, 'z': 0})(), 'orientation': type('Quaternion', (), {'x': 0, 'y': 0, 'z': 0, 'w': 1})()})()
    
    class Twist:
        def __init__(self):
            self.linear = type('Vector3', (), {'x': 0, 'y': 0, 'z': 0})()
            self.angular = type('Vector3', (), {'x': 0, 'y': 0, 'z': 0})()
    
    class OccupancyGrid: pass
    class PointCloud2: pass  
    class Image: pass
    class SetBool: pass
    class Trigger: pass
    
    class rclpy:
        @staticmethod
        def init(): pass
        @staticmethod
        def shutdown(): pass
        @staticmethod
        def spin(node): pass
        @staticmethod
        def spin_once(node, timeout_sec=None): pass
    
    ROS2_AVAILABLE = False
    print("Warning: ROS 2 not available, using mock implementation")

from ..coordination.swarm_coordinator import SwarmCoordinator, MissionConstraints
from ..fleet.drone_fleet import DroneFleet, DroneStatus
from ..communication.webrtc_streamer import WebRTCStreamer


class FleetManagerNode(Node):
    """ROS 2 node for Fleet-Mind swarm coordination.
    
    Integrates Fleet-Mind with ROS 2 ecosystem, providing:
    - Mission planning and execution services
    - Real-time fleet status publishing
    - Latent action code distribution
    - Sensor data aggregation
    - Emergency stop functionality
    """
    
    def __init__(self):
        super().__init__('fleet_mind_coordinator')
        
        # Initialize callback group for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # ROS 2 QoS profiles
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # Node parameters
        self.declare_parameter('llm_model', 'gpt-4o')
        self.declare_parameter('num_drones', 10)
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('latent_dim', 512)
        self.declare_parameter('max_altitude', 120.0)
        self.declare_parameter('safety_distance', 5.0)
        
        # Get parameters
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        self.num_drones = self.get_parameter('num_drones').get_parameter_value().integer_value
        self.update_rate = self.get_parameter('update_rate').get_parameter_value().double_value
        self.latent_dim = self.get_parameter('latent_dim').get_parameter_value().integer_value
        self.max_altitude = self.get_parameter('max_altitude').get_parameter_value().double_value
        self.safety_distance = self.get_parameter('safety_distance').get_parameter_value().double_value
        
        self.get_logger().info(f'Initializing Fleet-Mind with {self.num_drones} drones')
        
        # Initialize Fleet-Mind components
        self._initialize_fleet_mind()
        
        # ROS 2 Publishers
        self._setup_publishers()
        
        # ROS 2 Subscribers
        self._setup_subscribers()
        
        # ROS 2 Services
        self._setup_services()
        
        # ROS 2 Timers
        self._setup_timers()
        
        # State tracking
        self.current_mission: Optional[str] = None
        self.mission_start_time: Optional[float] = None
        self.total_missions = 0
        
        self.get_logger().info('Fleet-Mind ROS 2 node initialized successfully')

    def _initialize_fleet_mind(self) -> None:
        """Initialize Fleet-Mind core components."""
        # Create constraints from parameters
        constraints = MissionConstraints(
            max_altitude=self.max_altitude,
            safety_distance=self.safety_distance,
        )
        
        # Initialize swarm coordinator
        self.coordinator = SwarmCoordinator(
            llm_model=self.llm_model,
            latent_dim=self.latent_dim,
            max_drones=self.num_drones,
            update_rate=self.update_rate,
            safety_constraints=constraints,
        )
        
        # Initialize drone fleet
        drone_ids = [f"drone_{i}" for i in range(self.num_drones)]
        self.fleet = DroneFleet(
            drone_ids=drone_ids,
            communication_protocol="webrtc",
            topology="mesh"
        )
        
        # Connect fleet to coordinator (async operation)
        asyncio.create_task(self._connect_fleet_async())

    async def _connect_fleet_async(self) -> None:
        """Asynchronously connect fleet to coordinator."""
        try:
            await self.coordinator.connect_fleet(self.fleet)
            await self.fleet.start_monitoring()
            self.get_logger().info('Fleet connected to coordinator successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to connect fleet: {e}')

    def _setup_publishers(self) -> None:
        """Set up ROS 2 publishers."""
        # Latent action codes
        self.latent_plan_pub = self.create_publisher(
            String,
            '/fleet/latent_plan',
            self.reliable_qos
        )
        
        # Fleet status
        self.fleet_status_pub = self.create_publisher(
            String,
            '/fleet/status',
            self.best_effort_qos
        )
        
        # Mission progress
        self.mission_progress_pub = self.create_publisher(
            Float32,
            '/fleet/mission_progress',
            self.best_effort_qos
        )
        
        # Individual drone commands
        self.drone_cmd_pubs = {}
        for i in range(self.num_drones):
            topic = f'/drone_{i}/cmd_vel'
            self.drone_cmd_pubs[f'drone_{i}'] = self.create_publisher(
                Twist,
                topic,
                self.reliable_qos
            )
        
        # Emergency alerts
        self.emergency_pub = self.create_publisher(
            String,
            '/fleet/emergency',
            self.reliable_qos
        )

    def _setup_subscribers(self) -> None:
        """Set up ROS 2 subscribers."""
        # Drone telemetry
        self.telemetry_subs = {}
        for i in range(self.num_drones):
            topic = f'/drone_{i}/pose'
            self.telemetry_subs[f'drone_{i}'] = self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, drone_id=f'drone_{i}': self.telemetry_callback(msg, drone_id),
                self.best_effort_qos
            )
        
        # Mission commands
        self.mission_sub = self.create_subscription(
            String,
            '/fleet/mission_command',
            self.mission_command_callback,
            self.reliable_qos
        )
        
        # External sensor data
        self.sensor_subs = [
            self.create_subscription(
                PointCloud2,
                '/sensors/lidar',
                self.lidar_callback,
                self.best_effort_qos
            ),
            self.create_subscription(
                Image,
                '/sensors/camera',
                self.camera_callback,
                self.best_effort_qos
            ),
            self.create_subscription(
                OccupancyGrid,
                '/sensors/occupancy_grid',
                self.occupancy_grid_callback,
                self.best_effort_qos
            ),
        ]

    def _setup_services(self) -> None:
        """Set up ROS 2 services."""
        # Mission control services
        self.start_mission_srv = self.create_service(
            SetBool,
            '/fleet/start_mission',
            self.start_mission_callback,
            callback_group=self.callback_group
        )
        
        self.stop_mission_srv = self.create_service(
            Trigger,
            '/fleet/stop_mission',
            self.stop_mission_callback,
            callback_group=self.callback_group
        )
        
        self.emergency_stop_srv = self.create_service(
            Trigger,
            '/fleet/emergency_stop',
            self.emergency_stop_callback,
            callback_group=self.callback_group
        )
        
        # Fleet management services
        self.get_status_srv = self.create_service(
            Trigger,
            '/fleet/get_status',
            self.get_status_callback,
            callback_group=self.callback_group
        )

    def _setup_timers(self) -> None:
        """Set up ROS 2 timers."""
        # Main planning timer
        planning_period = 1.0 / self.update_rate
        self.planning_timer = self.create_timer(
            planning_period,
            self.planning_callback,
            callback_group=self.callback_group
        )
        
        # Status publishing timer
        self.status_timer = self.create_timer(
            1.0,  # 1 Hz
            self.status_callback
        )
        
        # Health monitoring timer
        self.health_timer = self.create_timer(
            5.0,  # 0.2 Hz
            self.health_callback
        )

    def telemetry_callback(self, msg: PoseStamped, drone_id: str) -> None:
        """Handle drone telemetry updates."""
        try:
            # Extract position from pose
            pos = msg.pose.position
            position = (pos.x, pos.y, pos.z)
            
            # Update drone state in fleet
            self.fleet.update_drone_state(
                drone_id=drone_id,
                position=position,
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing telemetry for {drone_id}: {e}')

    def mission_command_callback(self, msg: String) -> None:
        """Handle incoming mission commands."""
        try:
            command_data = json.loads(msg.data)
            mission_description = command_data.get('mission', '')
            
            if mission_description:
                self.get_logger().info(f'Received mission: {mission_description}')
                asyncio.create_task(self._execute_mission_async(mission_description))
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid mission command JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing mission command: {e}')

    async def _execute_mission_async(self, mission: str) -> None:
        """Execute mission asynchronously."""
        try:
            # Generate mission plan
            self.get_logger().info('Generating mission plan...')
            plan = await self.coordinator.generate_plan(mission)
            
            # Publish latent plan
            latent_msg = String()
            latent_msg.data = json.dumps({
                'mission_id': plan['mission_id'],
                'latent_code': plan['latent_code'].tolist(),
                'timestamp': time.time(),
            })
            self.latent_plan_pub.publish(latent_msg)
            
            # Execute mission
            self.current_mission = mission
            self.mission_start_time = time.time()
            
            success = await self.coordinator.execute_mission(plan)
            
            if success:
                self.get_logger().info('Mission completed successfully')
                self.total_missions += 1
            else:
                self.get_logger().error('Mission execution failed')
            
            self.current_mission = None
            self.mission_start_time = None
            
        except Exception as e:
            self.get_logger().error(f'Mission execution error: {e}')

    def planning_callback(self) -> None:
        """Main planning loop callback."""
        try:
            # Get current swarm status
            status = asyncio.run_coroutine_threadsafe(
                self.coordinator.get_swarm_status(),
                asyncio.get_event_loop()
            ).result(timeout=1.0)
            
            # Update mission progress if active
            if self.current_mission and self.mission_start_time:
                elapsed = time.time() - self.mission_start_time
                # Simple progress estimation (real implementation would be more sophisticated)
                progress = min(elapsed / 300.0, 1.0)  # 5-minute missions
                
                progress_msg = Float32()
                progress_msg.data = progress
                self.mission_progress_pub.publish(progress_msg)
            
        except Exception as e:
            self.get_logger().error(f'Planning callback error: {e}')

    def status_callback(self) -> None:
        """Publish fleet status."""
        try:
            fleet_status = self.fleet.get_fleet_status()
            
            status_msg = String()
            status_msg.data = json.dumps({
                'timestamp': time.time(),
                'fleet_status': fleet_status,
                'current_mission': self.current_mission,
                'total_missions': self.total_missions,
                'node_status': 'operational',
            })
            
            self.fleet_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Status callback error: {e}')

    def health_callback(self) -> None:
        """Monitor fleet health."""
        try:
            health_status = asyncio.run_coroutine_threadsafe(
                self.fleet.get_health_status(),
                asyncio.get_event_loop()
            ).result(timeout=2.0)
            
            # Check for critical issues
            if health_status['critical'] or health_status['failed']:
                warning_msg = String()
                warning_data = {
                    'type': 'health_warning',
                    'critical_drones': health_status['critical'],
                    'failed_drones': health_status['failed'],
                    'timestamp': time.time(),
                }
                warning_msg.data = json.dumps(warning_data)
                self.emergency_pub.publish(warning_msg)
                
                self.get_logger().warn(
                    f"Health warning: {len(health_status['critical'])} critical, "
                    f"{len(health_status['failed'])} failed drones"
                )
            
        except Exception as e:
            self.get_logger().error(f'Health callback error: {e}')

    def start_mission_callback(self, request, response) -> None:
        """Handle start mission service call."""
        try:
            if request.data:  # If True, start default mission
                default_mission = "Perform area survey in grid formation maintaining 5m spacing"
                asyncio.create_task(self._execute_mission_async(default_mission))
                response.success = True
                response.message = "Mission started successfully"
            else:
                response.success = False
                response.message = "Mission start cancelled"
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to start mission: {e}"
        
        return response

    def stop_mission_callback(self, request, response) -> None:
        """Handle stop mission service call."""
        try:
            if self.current_mission:
                asyncio.create_task(self.coordinator.emergency_stop())
                self.current_mission = None
                self.mission_start_time = None
                
                response.success = True
                response.message = "Mission stopped successfully"
            else:
                response.success = False
                response.message = "No active mission to stop"
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to stop mission: {e}"
        
        return response

    def emergency_stop_callback(self, request, response) -> None:
        """Handle emergency stop service call."""
        try:
            asyncio.create_task(self.coordinator.emergency_stop())
            
            # Publish emergency alert
            emergency_msg = String()
            emergency_msg.data = json.dumps({
                'type': 'emergency_stop',
                'timestamp': time.time(),
                'reason': 'Manual emergency stop requested',
            })
            self.emergency_pub.publish(emergency_msg)
            
            response.success = True
            response.message = "Emergency stop executed"
            
        except Exception as e:
            response.success = False
            response.message = f"Emergency stop failed: {e}"
        
        return response

    def get_status_callback(self, request, response) -> None:
        """Handle get status service call."""
        try:
            status = asyncio.run_coroutine_threadsafe(
                self.coordinator.get_swarm_status(),
                asyncio.get_event_loop()
            ).result(timeout=2.0)
            
            response.success = True
            response.message = json.dumps(status, indent=2)
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to get status: {e}"
        
        return response

    def lidar_callback(self, msg: PointCloud2) -> None:
        """Handle LiDAR sensor data."""
        # In real implementation, process point cloud for obstacle detection
        pass

    def camera_callback(self, msg: Image) -> None:
        """Handle camera sensor data."""
        # In real implementation, process image for visual navigation
        pass

    def occupancy_grid_callback(self, msg: OccupancyGrid) -> None:
        """Handle occupancy grid updates."""
        # In real implementation, update world model for path planning
        pass

    def destroy_node(self) -> None:
        """Clean shutdown of the node."""
        try:
            # Stop Fleet-Mind components
            asyncio.create_task(self.fleet.stop_monitoring())
            self.get_logger().info('Fleet-Mind node shutting down...')
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {e}')
        finally:
            super().destroy_node()


def main(args=None):
    """Main entry point for Fleet-Mind ROS 2 node."""
    rclpy.init(args=args)
    
    try:
        node = FleetManagerNode()
        
        # Use MultiThreadedExecutor for async callbacks
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor(num_threads=4)
        
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            executor.shutdown()
            
    except Exception as e:
        print(f"Failed to start Fleet-Mind node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()