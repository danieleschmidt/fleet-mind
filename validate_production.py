#!/usr/bin/env python3
"""Production validation script for Fleet-Mind system."""

import sys
import asyncio
import time
sys.path.append('/root/repo')

async def validate_production_readiness():
    """Comprehensive production readiness validation."""
    print("🚀 Fleet-Mind Production Validation")
    print("=" * 50)
    
    validation_results = {
        "core_modules": False,
        "security_system": False,
        "health_monitoring": False,
        "mission_planning": False,
        "performance_optimization": False,
        "auto_scaling": False,
        "deployment_ready": False
    }
    
    try:
        # Test 1: Core Module Import
        print("1. Testing core module imports...")
        from fleet_mind.coordination.swarm_coordinator import SwarmCoordinator
        from fleet_mind.fleet.drone_fleet import DroneFleet
        from fleet_mind.security.security_manager import SecurityLevel
        from fleet_mind.monitoring.health_monitor import HealthMonitor
        from fleet_mind.optimization.performance_optimizer import PerformanceOptimizer
        validation_results["core_modules"] = True
        print("   ✅ All core modules imported successfully")
        
        # Test 2: Security System
        print("2. Testing security system...")
        coordinator = SwarmCoordinator(
            llm_model="gpt-4o-mini",
            max_drones=5,
            security_level=SecurityLevel.HIGH,
            enable_health_monitoring=True
        )
        
        drone_fleet = DroneFleet(['prod_drone_0', 'prod_drone_1'])
        await coordinator.connect_fleet(drone_fleet)
        
        # Verify security manager initialized
        assert coordinator.security_manager is not None
        assert coordinator.security_manager.security_level == SecurityLevel.HIGH
        
        # Test authentication
        drone_id = 'prod_drone_0'
        invalid_auth = coordinator.security_manager.authenticate_drone(drone_id, "invalid_token")
        assert invalid_auth is False
        
        validation_results["security_system"] = True
        print("   ✅ Security system operational (HIGH level)")
        
        # Test 3: Health Monitoring  
        print("3. Testing health monitoring...")
        assert coordinator.health_monitor is not None
        
        # Update and get health metrics
        await coordinator.update_health_metrics()
        health_status = await coordinator.get_health_status()
        
        assert "overall_status" in health_status
        assert health_status["monitored_components"] >= 4
        
        validation_results["health_monitoring"] = True
        print(f"   ✅ Health monitoring active ({health_status['monitored_components']} components)")
        
        # Test 4: Mission Planning
        print("4. Testing mission planning...")
        start_time = time.time()
        plan = await coordinator.generate_plan("Production validation mission")
        planning_time = (time.time() - start_time) * 1000
        
        assert "mission_id" in plan
        assert "action_sequences" in plan
        assert planning_time < 2000  # Should be under 2 seconds
        
        validation_results["mission_planning"] = True
        print(f"   ✅ Mission planning functional ({planning_time:.1f}ms latency)")
        
        # Test 5: Performance Optimization
        print("5. Testing performance optimization...")
        from fleet_mind.optimization.performance_optimizer import OptimizationStrategy
        
        optimizer = PerformanceOptimizer(
            strategy=OptimizationStrategy.BALANCED,
            enable_auto_optimization=False
        )
        await optimizer.start()
        
        # Collect metrics
        metrics = await optimizer.collect_metrics()
        assert metrics.timestamp > 0
        assert metrics.cpu_usage_percent >= 0
        assert metrics.memory_usage_percent >= 0
        
        # Test bottleneck analysis
        bottlenecks = optimizer.analyze_bottlenecks(metrics)
        assert isinstance(bottlenecks, list)
        
        await optimizer.stop()
        validation_results["performance_optimization"] = True
        print(f"   ✅ Performance optimization active (CPU: {metrics.cpu_usage_percent:.1f}%, Memory: {metrics.memory_usage_percent:.1f}%)")
        
        # Test 6: Auto-scaling
        print("6. Testing auto-scaling decisions...")
        scaling_decision = optimizer.decide_scaling_action(metrics)
        assert scaling_decision.action in ["scale_up", "scale_down", "maintain"]
        assert 0 <= scaling_decision.confidence <= 1
        
        validation_results["auto_scaling"] = True
        print(f"   ✅ Auto-scaling functional (Decision: {scaling_decision.action}, Confidence: {scaling_decision.confidence:.2f})")
        
        # Test 7: Deployment Readiness
        print("7. Validating deployment configuration...")
        
        # Check comprehensive stats
        stats = coordinator.get_comprehensive_stats()
        required_stats = ["swarm_status", "fleet_stats", "performance_stats", "system_health"]
        
        for stat in required_stats:
            assert stat in stats, f"Missing required stat: {stat}"
        
        # Check security status
        security_status = await coordinator.get_security_status()
        assert "security_level" in security_status
        assert "threat_level" in security_status
        
        validation_results["deployment_ready"] = True
        print("   ✅ Deployment configuration validated")
        
        # Cleanup
        await drone_fleet.stop_monitoring()
        if coordinator.health_monitor:
            await coordinator.health_monitor.stop_monitoring()
            
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return False, validation_results
    
    return True, validation_results

def print_validation_report(success, results):
    """Print comprehensive validation report."""
    print("\n" + "=" * 50)
    print("🎯 PRODUCTION VALIDATION REPORT")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    # Individual test results
    test_names = {
        "core_modules": "Core Module Imports",
        "security_system": "Security System (HIGH level)",
        "health_monitoring": "Health Monitoring",
        "mission_planning": "Mission Planning",
        "performance_optimization": "Performance Optimization", 
        "auto_scaling": "Auto-scaling System",
        "deployment_ready": "Deployment Configuration"
    }
    
    for key, name in test_names.items():
        status = "✅ PASS" if results[key] else "❌ FAIL"
        print(f"{name:.<35} {status}")
    
    print()
    
    if success and passed_tests == total_tests:
        print("🚀 PRODUCTION STATUS: READY")
        print("🔒 SECURITY LEVEL: HIGH")
        print("📊 MONITORING: ACTIVE") 
        print("⚡ OPTIMIZATION: ENABLED")
        print("📈 AUTO-SCALING: OPERATIONAL")
        print("🌐 DEPLOYMENT: VALIDATED")
        print()
        print("Fleet-Mind is ready for production deployment!")
    else:
        print("❌ PRODUCTION STATUS: NOT READY")
        print("Please address the failed validations before deployment.")
    
    print("=" * 50)

async def main():
    """Main validation execution."""
    success, results = await validate_production_readiness()
    print_validation_report(success, results)
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)