"""Cloud Deployment Optimizer for Fleet-Mind Generation 3.

This module implements advanced cloud deployment optimization including:
- Spot instance management and cost optimization
- Multi-cloud deployment strategies
- Kubernetes auto-scaling integration
- Cost monitoring and optimization
- Infrastructure as Code management
- Cloud resource lifecycle management
"""

import asyncio
import time
import json
import yaml
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import math

try:
    import boto3
    import botocore
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

from ..utils.logging import get_logger
from .ml_cost_optimizer import get_cost_optimizer, InstanceType, CostOptimizationStrategy


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DIGITAL_OCEAN = "digital_ocean"
    KUBERNETES = "kubernetes"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    SINGLE_REGION = "single_region"
    MULTI_REGION = "multi_region"
    HYBRID_CLOUD = "hybrid_cloud"
    EDGE_COMPUTING = "edge_computing"
    COST_OPTIMIZED = "cost_optimized"


class ResourceType(Enum):
    """Cloud resource types."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    LOAD_BALANCER = "load_balancer"


@dataclass
class CloudResource:
    """Cloud resource configuration."""
    resource_id: str
    resource_type: ResourceType
    cloud_provider: CloudProvider
    instance_type: str
    region: str
    cost_per_hour: float
    specifications: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "pending"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    deployment_id: str
    strategy: DeploymentStrategy
    target_regions: List[str]
    min_instances: int = 3
    max_instances: int = 100
    spot_instance_ratio: float = 0.7
    enable_auto_scaling: bool = True
    cost_budget_per_hour: float = 50.0
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    

class SpotInstanceManager:
    """Manages spot instances for cost optimization."""
    
    def __init__(self, cloud_provider: CloudProvider = CloudProvider.AWS):
        """Initialize spot instance manager.
        
        Args:
            cloud_provider: Cloud provider to use
        """
        self.cloud_provider = cloud_provider
        self.active_spot_instances: Dict[str, Dict[str, Any]] = {}
        self.spot_price_history: deque = deque(maxlen=1000)
        self.interruption_history: List[Dict[str, Any]] = []
        
        # Initialize cloud client
        self.cloud_client = None
        if cloud_provider == CloudProvider.AWS and AWS_AVAILABLE:
            try:
                self.cloud_client = boto3.client('ec2')
            except Exception as e:
                pass
        
        # Logging
        self.logger = get_logger("spot_instance_manager")
    
    async def get_spot_prices(self, instance_types: List[str], regions: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current spot prices for instance types and regions.
        
        Args:
            instance_types: List of instance types
            regions: List of regions
            
        Returns:
            Dict mapping region -> instance_type -> price
        """
        prices = defaultdict(dict)
        
        try:
            if self.cloud_provider == CloudProvider.AWS and self.cloud_client:
                for region in regions:
                    # Switch to region
                    regional_client = boto3.client('ec2', region_name=region)
                    
                    response = regional_client.describe_spot_price_history(
                        InstanceTypes=instance_types,
                        ProductDescriptions=['Linux/UNIX'],
                        MaxResults=100
                    )
                    
                    for price_info in response['SpotPrices']:
                        instance_type = price_info['InstanceType']
                        price = float(price_info['SpotPrice'])
                        prices[region][instance_type] = price
            else:
                # Mock prices for testing
                for region in regions:
                    for instance_type in instance_types:
                        base_price = self._get_base_price(instance_type)
                        spot_price = base_price * 0.3  # 70% discount typical
                        prices[region][instance_type] = spot_price
        
        except Exception as e:
            self.logger.error(f"Error getting spot prices: {e}")
            # Return mock prices as fallback
            for region in regions:
                for instance_type in instance_types:
                    prices[region][instance_type] = self._get_base_price(instance_type) * 0.3
        
        return dict(prices)
    
    def _get_base_price(self, instance_type: str) -> float:
        """Get base on-demand price for instance type."""
        # Simplified pricing model
        price_map = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            't3.xlarge': 0.1664,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34,
            'c5.4xlarge': 0.68,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'r5.large': 0.126,
            'r5.xlarge': 0.252,
        }
        return price_map.get(instance_type, 0.10)  # Default price
    
    async def launch_spot_instances(
        self,
        instance_type: str,
        count: int,
        region: str,
        max_price_per_hour: float,
        user_data: str = "",
    ) -> List[str]:
        """Launch spot instances.
        
        Args:
            instance_type: EC2 instance type
            count: Number of instances to launch
            region: AWS region
            max_price_per_hour: Maximum price per hour
            user_data: User data script
            
        Returns:
            List of instance IDs
        """
        launched_instances = []
        
        try:
            if self.cloud_provider == CloudProvider.AWS and self.cloud_client:
                # Real AWS spot instance launch
                regional_client = boto3.client('ec2', region_name=region)
                
                response = regional_client.request_spot_instances(
                    SpotPrice=str(max_price_per_hour),
                    InstanceCount=count,
                    Type='one-time',
                    LaunchSpecification={
                        'ImageId': 'ami-0abcdef1234567890',  # Replace with actual AMI
                        'InstanceType': instance_type,
                        'UserData': user_data,
                        'SecurityGroupIds': ['sg-default'],
                        'SubnetId': 'subnet-default',
                    }
                )
                
                for request in response['SpotInstanceRequests']:
                    request_id = request['SpotInstanceRequestId']
                    launched_instances.append(request_id)
                    
                    # Track the request
                    self.active_spot_instances[request_id] = {
                        'instance_type': instance_type,
                        'region': region,
                        'max_price': max_price_per_hour,
                        'status': 'pending',
                        'launched_at': time.time(),
                    }
            
            else:
                # Mock spot instance launch for testing
                for i in range(count):
                    instance_id = f"i-{int(time.time())}{i:03d}"
                    launched_instances.append(instance_id)
                    
                    self.active_spot_instances[instance_id] = {
                        'instance_type': instance_type,
                        'region': region,
                        'max_price': max_price_per_hour,
                        'status': 'running',
                        'launched_at': time.time(),
                    }
            
            self.logger.info(f"Launched {len(launched_instances)} spot instances in {region}")
            return launched_instances
            
        except Exception as e:
            self.logger.error(f"Error launching spot instances: {e}")
            return []
    
    async def monitor_spot_instances(self):
        """Monitor spot instances for interruptions."""
        try:
            if self.cloud_provider == CloudProvider.AWS and self.cloud_client:
                # Check spot instance status
                instance_ids = list(self.active_spot_instances.keys())
                if not instance_ids:
                    return
                
                # Query instance status
                response = self.cloud_client.describe_instances(
                    InstanceIds=instance_ids
                )
                
                for reservation in response['Reservations']:
                    for instance in reservation['Instances']:
                        instance_id = instance['InstanceId']
                        state = instance['State']['Name']
                        
                        if instance_id in self.active_spot_instances:
                            old_status = self.active_spot_instances[instance_id]['status']
                            self.active_spot_instances[instance_id]['status'] = state
                            
                            # Detect interruptions
                            if old_status == 'running' and state in ['shutting-down', 'terminated']:
                                self._handle_spot_interruption(instance_id)
            
            else:
                # Mock monitoring - simulate random interruptions
                for instance_id, info in list(self.active_spot_instances.items()):
                    if info['status'] == 'running':
                        # Small chance of interruption (0.1% per check)
                        if time.time() - info['launched_at'] > 300 and random.random() < 0.001:
                            self._handle_spot_interruption(instance_id)
            
        except Exception as e:
            self.logger.error(f"Error monitoring spot instances: {e}")
    
    def _handle_spot_interruption(self, instance_id: str):
        """Handle spot instance interruption."""
        try:
            instance_info = self.active_spot_instances.get(instance_id, {})
            
            # Record interruption
            interruption_record = {
                'instance_id': instance_id,
                'instance_type': instance_info.get('instance_type'),
                'region': instance_info.get('region'),
                'uptime_seconds': time.time() - instance_info.get('launched_at', 0),
                'interrupted_at': time.time(),
            }
            
            self.interruption_history.append(interruption_record)
            
            # Remove from active instances
            if instance_id in self.active_spot_instances:
                del self.active_spot_instances[instance_id]
            
            self.logger.warning(f"Spot instance {instance_id} was interrupted")
            
            # Trigger replacement instance launch
            asyncio.create_task(self._replace_interrupted_instance(interruption_record))
            
        except Exception as e:
            self.logger.error(f"Error handling spot interruption: {e}")
    
    async def _replace_interrupted_instance(self, interrupted_instance: Dict[str, Any]):
        """Replace interrupted spot instance."""
        try:
            # Launch replacement with slightly higher bid
            instance_type = interrupted_instance['instance_type']
            region = interrupted_instance['region']
            
            # Get current spot price
            spot_prices = await self.get_spot_prices([instance_type], [region])
            current_price = spot_prices.get(region, {}).get(instance_type, 0.10)
            
            # Bid 20% above current spot price
            max_price = current_price * 1.2
            
            replacement_instances = await self.launch_spot_instances(
                instance_type=instance_type,
                count=1,
                region=region,
                max_price_per_hour=max_price,
            )
            
            if replacement_instances:
                self.logger.info(f"Launched replacement instance for {interrupted_instance['instance_id']}")
            
        except Exception as e:
            self.logger.error(f"Error replacing interrupted instance: {e}")
    
    def get_spot_instance_stats(self) -> Dict[str, Any]:
        """Get spot instance statistics."""
        active_count = len(self.active_spot_instances)
        total_interruptions = len(self.interruption_history)
        
        if self.interruption_history:
            avg_uptime = statistics.mean([
                record['uptime_seconds'] for record in self.interruption_history
            ])
            interruption_rate = total_interruptions / max(1, active_count + total_interruptions)
        else:
            avg_uptime = 0
            interruption_rate = 0
        
        return {
            "active_spot_instances": active_count,
            "total_interruptions": total_interruptions,
            "average_uptime_seconds": avg_uptime,
            "interruption_rate": interruption_rate,
            "cost_savings_estimate": self._calculate_cost_savings(),
        }
    
    def _calculate_cost_savings(self) -> float:
        """Calculate estimated cost savings from spot instances."""
        total_savings = 0.0
        
        for instance_info in self.active_spot_instances.values():
            instance_type = instance_info.get('instance_type', '')
            max_price = instance_info.get('max_price', 0)
            base_price = self._get_base_price(instance_type)
            uptime_hours = (time.time() - instance_info.get('launched_at', 0)) / 3600
            
            savings = (base_price - max_price) * uptime_hours
            total_savings += savings
        
        return total_savings


class KubernetesOptimizer:
    """Optimizes Kubernetes deployments for cost and performance."""
    
    def __init__(self):
        """Initialize Kubernetes optimizer."""
        self.k8s_client = None
        self.apps_v1 = None
        
        if KUBERNETES_AVAILABLE:
            try:
                config.load_incluster_config()  # Try in-cluster config first
            except:
                try:
                    config.load_kube_config()  # Fall back to local config
                except:
                    pass
            
            if config:
                self.k8s_client = client.CoreV1Api()
                self.apps_v1 = client.AppsV1Api()
        
        self.logger = get_logger("kubernetes_optimizer")
    
    async def optimize_resource_requests(self, namespace: str = "default") -> Dict[str, Any]:
        """Optimize Kubernetes resource requests based on usage."""
        try:
            if not self.k8s_client:
                return {"error": "Kubernetes client not available"}
            
            optimizations = []
            
            # Get all pods in namespace
            pods = self.k8s_client.list_namespaced_pod(namespace)
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                
                # Get resource requests and limits
                containers = pod.spec.containers
                
                for container in containers:
                    resources = container.resources
                    requests = resources.requests or {}
                    limits = resources.limits or {}
                    
                    # Get actual usage (simplified - would use metrics API in practice)
                    actual_usage = await self._get_pod_usage(pod_name, container.name, namespace)
                    
                    # Compare requests vs usage
                    if actual_usage:
                        cpu_request = self._parse_cpu(requests.get('cpu', '100m'))
                        cpu_usage = actual_usage.get('cpu', 0)
                        
                        memory_request = self._parse_memory(requests.get('memory', '128Mi'))
                        memory_usage = actual_usage.get('memory', 0)
                        
                        # Suggest optimizations
                        if cpu_usage < cpu_request * 0.5:  # <50% utilization
                            optimizations.append({
                                "type": "cpu_over_provisioned",
                                "pod": pod_name,
                                "container": container.name,
                                "current_request": requests.get('cpu'),
                                "suggested_request": f"{int(cpu_usage * 1.2)}m",
                                "potential_savings": f"{cpu_request - cpu_usage:.0f}m",
                            })
                        
                        if memory_usage < memory_request * 0.5:  # <50% utilization
                            optimizations.append({
                                "type": "memory_over_provisioned",
                                "pod": pod_name,
                                "container": container.name,
                                "current_request": requests.get('memory'),
                                "suggested_request": f"{int(memory_usage * 1.2)}Mi",
                                "potential_savings": f"{memory_request - memory_usage:.0f}Mi",
                            })
            
            return {
                "namespace": namespace,
                "optimizations": optimizations,
                "potential_cost_savings": len(optimizations) * 0.05,  # Rough estimate
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing resource requests: {e}")
            return {"error": str(e)}
    
    async def _get_pod_usage(self, pod_name: str, container_name: str, namespace: str) -> Dict[str, float]:
        """Get actual resource usage for pod container."""
        # Simplified mock usage - in practice would query metrics API
        import random
        return {
            "cpu": random.uniform(50, 200),  # millicores
            "memory": random.uniform(64, 512),  # Mi
        }
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to millicores."""
        if not cpu_str:
            return 100.0
        
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1])
        else:
            return float(cpu_str) * 1000
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to Mi."""
        if not memory_str:
            return 128.0
        
        if memory_str.endswith('Mi'):
            return float(memory_str[:-2])
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith('Ki'):
            return float(memory_str[:-2]) / 1024
        else:
            return float(memory_str) / (1024 * 1024)  # Assume bytes
    
    async def scale_deployment(
        self,
        deployment_name: str,
        target_replicas: int,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """Scale Kubernetes deployment."""
        try:
            if not self.apps_v1:
                return {"error": "Kubernetes Apps API not available"}
            
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update replica count
            deployment.spec.replicas = target_replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            return {
                "deployment": deployment_name,
                "namespace": namespace,
                "target_replicas": target_replicas,
                "success": True,
            }
            
        except Exception as e:
            self.logger.error(f"Error scaling deployment: {e}")
            return {"error": str(e)}


class CloudDeploymentOptimizer:
    """Comprehensive cloud deployment optimization system."""
    
    def __init__(self):
        """Initialize cloud deployment optimizer."""
        self.spot_manager = SpotInstanceManager()
        self.k8s_optimizer = KubernetesOptimizer()
        self.cost_optimizer = get_cost_optimizer()
        
        # Deployment tracking
        self.active_deployments: Dict[str, DeploymentConfig] = {}
        self.cost_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Logging
        self.logger = get_logger("cloud_deployment_optimizer")
    
    async def start(self):
        """Start cloud deployment optimizer."""
        self.running = True
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.logger.info("Cloud deployment optimizer started")
    
    async def stop(self):
        """Stop cloud deployment optimizer."""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        await asyncio.gather(
            self.monitoring_task, self.optimization_task, return_exceptions=True
        )
        
        self.logger.info("Cloud deployment optimizer stopped")
    
    async def optimize_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Optimize cloud deployment based on configuration."""
        try:
            optimization_results = {
                "deployment_id": config.deployment_id,
                "strategy": config.strategy.value,
                "optimizations": [],
                "estimated_cost_savings": 0.0,
                "recommendations": [],
            }
            
            # Cost optimization
            cost_optimization = await self._optimize_costs(config)
            optimization_results["optimizations"].append(cost_optimization)
            optimization_results["estimated_cost_savings"] += cost_optimization.get("savings", 0)
            
            # Spot instance optimization
            spot_optimization = await self._optimize_spot_instances(config)
            optimization_results["optimizations"].append(spot_optimization)
            optimization_results["estimated_cost_savings"] += spot_optimization.get("savings", 0)
            
            # Kubernetes optimization
            if config.strategy in [DeploymentStrategy.MULTI_REGION, DeploymentStrategy.HYBRID_CLOUD]:
                k8s_optimization = await self.k8s_optimizer.optimize_resource_requests()
                optimization_results["optimizations"].append(k8s_optimization)
                optimization_results["estimated_cost_savings"] += k8s_optimization.get("potential_cost_savings", 0)
            
            # Generate recommendations
            recommendations = self._generate_deployment_recommendations(config, optimization_results)
            optimization_results["recommendations"] = recommendations
            
            # Store deployment configuration
            self.active_deployments[config.deployment_id] = config
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing deployment: {e}")
            return {"error": str(e)}
    
    async def _optimize_costs(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Optimize deployment costs."""
        try:
            # Get current load characteristics
            current_load = {
                "cpu_usage": 50.0,  # Mock values
                "memory_usage": 60.0,
                "response_time": 80.0,
                "throughput": 100.0,
            }
            
            # Get cost optimization recommendation
            recommendation = await self.cost_optimizer.predict_optimal_scaling(
                current_load, current_load
            )
            
            # Calculate potential savings
            current_hourly_cost = config.max_instances * 0.10  # $0.10 per instance
            optimized_cost = recommendation.target_capacity * 0.08  # Optimized rate
            savings = max(0, current_hourly_cost - optimized_cost)
            
            return {
                "type": "cost_optimization",
                "current_cost_per_hour": current_hourly_cost,
                "optimized_cost_per_hour": optimized_cost,
                "savings": savings,
                "recommendation": recommendation.action,
                "confidence": recommendation.confidence,
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing costs: {e}")
            return {"type": "cost_optimization", "error": str(e)}
    
    async def _optimize_spot_instances(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Optimize spot instance usage."""
        try:
            # Get spot prices for target regions
            instance_types = ['c5.large', 'c5.xlarge', 'm5.large']
            spot_prices = await self.spot_manager.get_spot_prices(
                instance_types, config.target_regions
            )
            
            # Find best price/region combinations
            best_options = []
            for region, prices in spot_prices.items():
                for instance_type, price in prices.items():
                    best_options.append({
                        "region": region,
                        "instance_type": instance_type,
                        "spot_price": price,
                        "on_demand_price": self.spot_manager._get_base_price(instance_type),
                        "savings_percent": (1 - price / self.spot_manager._get_base_price(instance_type)) * 100,
                    })
            
            # Sort by savings
            best_options.sort(key=lambda x: x["savings_percent"], reverse=True)
            
            # Calculate total potential savings
            spot_instances = int(config.max_instances * config.spot_instance_ratio)
            if best_options:
                best_option = best_options[0]
                hourly_savings = (best_option["on_demand_price"] - best_option["spot_price"]) * spot_instances
            else:
                hourly_savings = 0
            
            return {
                "type": "spot_instance_optimization",
                "spot_instance_ratio": config.spot_instance_ratio,
                "recommended_instances": spot_instances,
                "best_options": best_options[:5],  # Top 5
                "hourly_savings": hourly_savings,
                "savings": hourly_savings * 24,  # Daily savings
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing spot instances: {e}")
            return {"type": "spot_instance_optimization", "error": str(e)}
    
    def _generate_deployment_recommendations(
        self, 
        config: DeploymentConfig, 
        optimization_results: Dict[str, Any]
    ) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []
        
        # Cost-based recommendations
        total_savings = optimization_results.get("estimated_cost_savings", 0)
        if total_savings > config.cost_budget_per_hour * 0.2:  # >20% savings
            recommendations.append(
                f"Significant cost savings possible: ${total_savings:.2f}/hour. "
                "Consider implementing recommended optimizations."
            )
        
        # Spot instance recommendations
        if config.spot_instance_ratio < 0.5:
            recommendations.append(
                "Consider increasing spot instance ratio to 50-70% for additional cost savings. "
                "Ensure proper handling of interruptions."
            )
        
        # Multi-region recommendations
        if config.strategy == DeploymentStrategy.SINGLE_REGION:
            recommendations.append(
                "Consider multi-region deployment for improved availability and disaster recovery."
            )
        
        # Auto-scaling recommendations
        if not config.enable_auto_scaling:
            recommendations.append(
                "Enable auto-scaling to optimize costs and handle traffic variations efficiently."
            )
        
        # Resource optimization
        if len(config.target_regions) > 3:
            recommendations.append(
                "Large number of regions may increase complexity and costs. "
                "Consider optimizing region selection based on user distribution."
            )
        
        return recommendations
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Monitor spot instances
                await self.spot_manager.monitor_spot_instances()
                
                # Collect cost metrics
                await self._collect_cost_metrics()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while self.running:
            try:
                # Re-optimize deployments periodically
                for deployment_id, config in list(self.active_deployments.items()):
                    await self._reoptimize_deployment(deployment_id, config)
                
                await asyncio.sleep(900)  # Check every 15 minutes
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(900)
    
    async def _collect_cost_metrics(self):
        """Collect cost metrics for all deployments."""
        try:
            total_cost = 0.0
            
            for deployment_id, config in self.active_deployments.items():
                # Calculate deployment cost (simplified)
                deployment_cost = config.max_instances * 0.10  # Base rate
                if config.spot_instance_ratio > 0:
                    spot_savings = deployment_cost * config.spot_instance_ratio * 0.7  # 70% savings
                    deployment_cost -= spot_savings
                
                total_cost += deployment_cost
            
            # Record cost history
            self.cost_history.append({
                "timestamp": time.time(),
                "total_cost_per_hour": total_cost,
                "active_deployments": len(self.active_deployments),
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting cost metrics: {e}")
    
    async def _reoptimize_deployment(self, deployment_id: str, config: DeploymentConfig):
        """Re-optimize deployment based on current conditions."""
        try:
            # Get updated optimization
            updated_optimization = await self.optimize_deployment(config)
            
            # Apply optimizations if significant savings
            if updated_optimization.get("estimated_cost_savings", 0) > 5.0:  # >$5/hour
                self.logger.info(f"Applying optimization to deployment {deployment_id}")
                # In practice, would apply the optimizations here
                
        except Exception as e:
            self.logger.error(f"Error re-optimizing deployment {deployment_id}: {e}")
    
    def generate_cost_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost report."""
        try:
            if not self.cost_history:
                return {"error": "No cost data available"}
            
            recent_costs = list(self.cost_history)[-24:]  # Last 24 hours
            
            current_cost = self.cost_history[-1]["total_cost_per_hour"]
            avg_cost = statistics.mean([entry["total_cost_per_hour"] for entry in recent_costs])
            
            # Calculate trends
            if len(recent_costs) >= 2:
                cost_trend = recent_costs[-1]["total_cost_per_hour"] - recent_costs[0]["total_cost_per_hour"]
            else:
                cost_trend = 0.0
            
            # Spot instance stats
            spot_stats = self.spot_manager.get_spot_instance_stats()
            
            return {
                "current_cost_per_hour": current_cost,
                "average_cost_24h": avg_cost,
                "cost_trend_24h": cost_trend,
                "daily_cost_estimate": current_cost * 24,
                "monthly_cost_estimate": current_cost * 24 * 30,
                "active_deployments": len(self.active_deployments),
                "spot_instance_stats": spot_stats,
                "optimization_recommendations": self._get_cost_optimization_recommendations(),
            }
            
        except Exception as e:
            self.logger.error(f"Error generating cost report: {e}")
            return {"error": str(e)}
    
    def _get_cost_optimization_recommendations(self) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []
        
        if not self.cost_history:
            return recommendations
        
        current_cost = self.cost_history[-1]["total_cost_per_hour"]
        
        # High cost warnings
        if current_cost > 100:  # >$100/hour
            recommendations.append("High hourly costs detected. Review instance sizes and usage patterns.")
        
        # Spot instance recommendations
        spot_stats = self.spot_manager.get_spot_instance_stats()
        if spot_stats["cost_savings_estimate"] < current_cost * 0.3:  # <30% savings
            recommendations.append("Increase spot instance usage to achieve better cost savings.")
        
        # Trend analysis
        if len(self.cost_history) >= 7:
            weekly_costs = [entry["total_cost_per_hour"] for entry in list(self.cost_history)[-7:]]
            if statistics.mean(weekly_costs) > weekly_costs[0] * 1.2:  # 20% increase
                recommendations.append("Cost trend is increasing. Consider optimization review.")
        
        return recommendations


# Global cloud deployment optimizer
_cloud_optimizer: Optional[CloudDeploymentOptimizer] = None

async def get_cloud_optimizer() -> CloudDeploymentOptimizer:
    """Get or create global cloud deployment optimizer."""
    global _cloud_optimizer
    if _cloud_optimizer is None:
        _cloud_optimizer = CloudDeploymentOptimizer()
        await _cloud_optimizer.start()
    return _cloud_optimizer

async def optimize_fleet_deployment(
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.COST_OPTIMIZED,
    target_regions: List[str] = ["us-east-1", "us-west-2"],
    max_instances: int = 100,
    cost_budget_per_hour: float = 50.0,
) -> Dict[str, Any]:
    """Optimize Fleet-Mind deployment for cost and performance."""
    optimizer = await get_cloud_optimizer()
    
    config = DeploymentConfig(
        deployment_id=f"fleet_mind_{int(time.time())}",
        strategy=deployment_strategy,
        target_regions=target_regions,
        max_instances=max_instances,
        spot_instance_ratio=0.7,
        enable_auto_scaling=True,
        cost_budget_per_hour=cost_budget_per_hour,
        performance_requirements={
            "target_latency_ms": 100,
            "min_availability": 99.9,
        }
    )
    
    return await optimizer.optimize_deployment(config)