"""Service Mesh Coordinator for Fleet-Mind Generation 3.

This module implements advanced distributed computing architecture including:
- Service mesh management for 1000+ drone coordination
- Advanced load balancing with service discovery
- Edge computing integration for reduced latency
- Microservices decomposition and orchestration
- Inter-service communication optimization
"""

import asyncio
import time
import hashlib
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import concurrent.futures

from ..utils.logging import get_logger
from ..utils.circuit_breaker import circuit_breaker, CircuitBreakerConfig
from ..utils.retry import retry
from .ai_performance_optimizer import get_ai_optimizer, record_performance_metrics


class ServiceType(Enum):
    """Types of services in the mesh."""
    COORDINATOR = "coordinator"
    PLANNER = "planner"
    EXECUTOR = "executor"
    MONITOR = "monitor"
    GATEWAY = "gateway"
    CACHE = "cache"
    STORAGE = "storage"
    ANALYTICS = "analytics"


class ServiceStatus(Enum):
    """Service status in the mesh."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    PROXIMITY_BASED = "proximity_based"
    LATENCY_BASED = "latency_based"
    AI_OPTIMIZED = "ai_optimized"


@dataclass
class ServiceEndpoint:
    """Service endpoint in the mesh."""
    service_id: str
    service_type: ServiceType
    host: str
    port: int
    protocol: str = "http"
    region: str = "default"
    zone: str = "default"
    weight: float = 1.0
    max_connections: int = 1000
    current_connections: int = 0
    status: ServiceStatus = ServiceStatus.STARTING
    last_health_check: float = 0.0
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        """Get full service address."""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor."""
        if self.max_connections == 0:
            return 0.0
        return self.current_connections / self.max_connections


@dataclass
class ServiceRequest:
    """Service request in the mesh."""
    request_id: str
    source_service: str
    target_service_type: ServiceType
    method: str
    payload: Any
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    priority: int = 0  # Higher number = higher priority
    created_at: float = field(default_factory=time.time)
    
    
@dataclass
class ServiceResponse:
    """Service response in the mesh."""
    request_id: str
    status_code: int
    payload: Any
    headers: Dict[str, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    served_by: str = ""
    error: Optional[str] = None


class ServiceMeshCoordinator:
    """Advanced service mesh coordinator for distributed Fleet-Mind architecture."""
    
    def __init__(
        self,
        mesh_id: str = None,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.AI_OPTIMIZED,
        health_check_interval: float = 10.0,
        service_discovery_interval: float = 30.0,
        max_concurrent_requests: int = 10000,
    ):
        """Initialize service mesh coordinator.
        
        Args:
            mesh_id: Unique mesh identifier
            load_balancing_strategy: Load balancing strategy to use
            health_check_interval: Health check frequency in seconds
            service_discovery_interval: Service discovery frequency in seconds
            max_concurrent_requests: Maximum concurrent requests
        """
        self.mesh_id = mesh_id or f"mesh_{uuid.uuid4().hex[:8]}"
        self.load_balancing_strategy = load_balancing_strategy
        self.health_check_interval = health_check_interval
        self.service_discovery_interval = service_discovery_interval
        self.max_concurrent_requests = max_concurrent_requests
        
        # Service registry
        self.services: Dict[str, ServiceEndpoint] = {}
        self.services_by_type: Dict[ServiceType, List[ServiceEndpoint]] = defaultdict(list)
        
        # Request tracking
        self.active_requests: Dict[str, ServiceRequest] = {}
        self.request_history: deque = deque(maxlen=10000)
        
        # Load balancing state
        self.round_robin_counters: Dict[ServiceType, int] = defaultdict(int)
        self.consistent_hash_ring: Dict[int, str] = {}
        
        # Performance tracking
        self.service_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Circuit breakers for services
        self.circuit_breakers: Dict[str, Any] = {}
        
        # Threading and async
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=50)
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Logging
        self.logger = get_logger("service_mesh_coordinator")
        
        # Edge computing nodes
        self.edge_nodes: Dict[str, Dict[str, Any]] = {}
        
        # Service dependency graph
        self.service_dependencies: Dict[ServiceType, Set[ServiceType]] = {
            ServiceType.COORDINATOR: {ServiceType.PLANNER, ServiceType.MONITOR},
            ServiceType.PLANNER: {ServiceType.CACHE, ServiceType.ANALYTICS},
            ServiceType.EXECUTOR: {ServiceType.MONITOR, ServiceType.STORAGE},
            ServiceType.GATEWAY: {ServiceType.COORDINATOR},
        }
    
    async def start(self):
        """Start the service mesh coordinator."""
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._service_discovery_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._request_cleanup_loop()),
        ]
        
        self.logger.info(f"Started service mesh coordinator {self.mesh_id}")
    
    async def stop(self):
        """Stop the service mesh coordinator."""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info(f"Stopped service mesh coordinator {self.mesh_id}")
    
    def register_service(self, endpoint: ServiceEndpoint) -> bool:
        """Register a service endpoint in the mesh.
        
        Args:
            endpoint: Service endpoint to register
            
        Returns:
            True if registration successful
        """
        try:
            # Validate endpoint
            if not endpoint.service_id or not endpoint.host or endpoint.port <= 0:
                raise ValueError("Invalid service endpoint")
            
            # Register service
            self.services[endpoint.service_id] = endpoint
            self.services_by_type[endpoint.service_type].append(endpoint)
            
            # Initialize metrics
            self.service_metrics[endpoint.service_id] = {
                "requests_total": 0,
                "requests_success": 0,
                "requests_failed": 0,
                "avg_response_time_ms": 0.0,
                "last_seen": time.time(),
            }
            
            # Create circuit breaker
            self.circuit_breakers[endpoint.service_id] = CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                expected_exception=Exception
            )
            
            # Update consistent hash ring
            self._update_consistent_hash_ring()
            
            self.logger.info(f"Registered service: {endpoint.service_id} "
                           f"({endpoint.service_type.value}) at {endpoint.address}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service {endpoint.service_id}: {e}")
            return False
    
    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from the mesh.
        
        Args:
            service_id: Service ID to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            if service_id not in self.services:
                return False
            
            endpoint = self.services[service_id]
            
            # Remove from registry
            del self.services[service_id]
            self.services_by_type[endpoint.service_type] = [
                svc for svc in self.services_by_type[endpoint.service_type]
                if svc.service_id != service_id
            ]
            
            # Clean up
            if service_id in self.service_metrics:
                del self.service_metrics[service_id]
            if service_id in self.circuit_breakers:
                del self.circuit_breakers[service_id]
            
            # Update hash ring
            self._update_consistent_hash_ring()
            
            self.logger.info(f"Unregistered service: {service_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister service {service_id}: {e}")
            return False
    
    async def route_request(self, request: ServiceRequest) -> ServiceResponse:
        """Route request through the service mesh.
        
        Args:
            request: Service request to route
            
        Returns:
            Service response
        """
        start_time = time.time()
        
        try:
            # Check request limits
            if len(self.active_requests) >= self.max_concurrent_requests:
                return ServiceResponse(
                    request_id=request.request_id,
                    status_code=503,
                    payload=None,
                    error="Service mesh overloaded"
                )
            
            # Track active request
            self.active_requests[request.request_id] = request
            
            # Select target service
            target_service = await self._select_target_service(request)
            if not target_service:
                return ServiceResponse(
                    request_id=request.request_id,
                    status_code=503,
                    payload=None,
                    error=f"No available {request.target_service_type.value} services"
                )
            
            # Execute request with circuit breaker
            response = await self._execute_request_with_circuit_breaker(request, target_service)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time
            response.served_by = target_service.service_id
            
            await self._update_service_metrics(target_service.service_id, response)
            
            # Record performance metrics for AI optimization
            record_performance_metrics(
                latency_ms=processing_time,
                throughput_rps=1.0,
                error_rate=1.0 if response.status_code >= 400 else 0.0,
            )
            
            return response
            
        except Exception as e:
            error_response = ServiceResponse(
                request_id=request.request_id,
                status_code=500,
                payload=None,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            self.logger.error(f"Request routing failed: {e}")
            return error_response
            
        finally:
            # Clean up active request
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            # Store in history
            self.request_history.append((request, time.time()))
    
    async def _select_target_service(self, request: ServiceRequest) -> Optional[ServiceEndpoint]:
        """Select target service based on load balancing strategy."""
        available_services = [
            svc for svc in self.services_by_type[request.target_service_type]
            if svc.status == ServiceStatus.HEALTHY
        ]
        
        if not available_services:
            return None
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(request.target_service_type, available_services)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(available_services)
        elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(available_services)
        elif self.load_balancing_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._select_consistent_hash(request, available_services)
        elif self.load_balancing_strategy == LoadBalancingStrategy.PROXIMITY_BASED:
            return self._select_proximity_based(request, available_services)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LATENCY_BASED:
            return self._select_latency_based(available_services)
        elif self.load_balancing_strategy == LoadBalancingStrategy.AI_OPTIMIZED:
            return await self._select_ai_optimized(request, available_services)
        else:
            return available_services[0] if available_services else None
    
    def _select_round_robin(
        self, 
        service_type: ServiceType, 
        services: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Select service using round-robin."""
        index = self.round_robin_counters[service_type] % len(services)
        self.round_robin_counters[service_type] += 1
        return services[index]
    
    def _select_least_connections(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select service with least connections."""
        return min(services, key=lambda s: s.current_connections)
    
    def _select_weighted_round_robin(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select service using weighted round-robin."""
        total_weight = sum(s.weight for s in services)
        if total_weight == 0:
            return services[0]
        
        # Simple weighted selection
        weights = [s.weight / total_weight for s in services]
        cumulative = 0.0
        random_val = hash(str(time.time())) % 1000 / 1000.0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if random_val <= cumulative:
                return services[i]
        
        return services[-1]
    
    def _select_consistent_hash(
        self, 
        request: ServiceRequest, 
        services: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Select service using consistent hashing."""
        # Hash request key
        key = f"{request.source_service}:{request.method}"
        hash_value = hash(key) % (2**32)
        
        # Find closest service in hash ring
        available_hashes = [
            h for h, service_id in self.consistent_hash_ring.items()
            if service_id in [s.service_id for s in services]
        ]
        
        if not available_hashes:
            return services[0]
        
        # Find next hash value
        next_hash = min(h for h in available_hashes if h >= hash_value) if \
                   any(h >= hash_value for h in available_hashes) else min(available_hashes)
        
        service_id = self.consistent_hash_ring[next_hash]
        return next(s for s in services if s.service_id == service_id)
    
    def _select_proximity_based(
        self, 
        request: ServiceRequest, 
        services: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Select service based on geographic proximity."""
        # Simple proximity based on region/zone
        source_region = request.headers.get("region", "default")
        source_zone = request.headers.get("zone", "default")
        
        # Prefer same region/zone
        same_region = [s for s in services if s.region == source_region]
        if same_region:
            same_zone = [s for s in same_region if s.zone == source_zone]
            if same_zone:
                return min(same_zone, key=lambda s: s.current_connections)
            return min(same_region, key=lambda s: s.current_connections)
        
        return min(services, key=lambda s: s.current_connections)
    
    def _select_latency_based(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select service with lowest latency."""
        return min(services, key=lambda s: s.response_time_ms)
    
    async def _select_ai_optimized(
        self, 
        request: ServiceRequest, 
        services: List[ServiceEndpoint]
    ) -> ServiceEndpoint:
        """Select service using AI optimization."""
        try:
            ai_optimizer = get_ai_optimizer()
            
            # Score services based on multiple factors
            service_scores = []
            
            for service in services:
                # Calculate composite score
                latency_score = 1000.0 / max(1.0, service.response_time_ms)
                load_score = 1.0 - service.load_factor
                health_score = 1.0 if service.status == ServiceStatus.HEALTHY else 0.5
                
                # Historical performance
                metrics = self.service_metrics.get(service.service_id, {})
                success_rate = metrics.get("requests_success", 0) / max(1, metrics.get("requests_total", 1))
                
                # Composite score
                composite_score = (
                    latency_score * 0.3 +
                    load_score * 0.3 +
                    health_score * 0.2 +
                    success_rate * 0.2
                )
                
                service_scores.append((service, composite_score))
            
            # Select best service
            best_service = max(service_scores, key=lambda x: x[1])[0]
            return best_service
            
        except Exception as e:
            self.logger.error(f"AI optimization selection failed: {e}")
            return self._select_least_connections(services)
    
    @circuit_breaker("service_request", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0))
    async def _execute_request_with_circuit_breaker(
        self, 
        request: ServiceRequest, 
        target_service: ServiceEndpoint
    ) -> ServiceResponse:
        """Execute request with circuit breaker protection."""
        try:
            # Update connection count
            target_service.current_connections += 1
            
            # Simulate request execution (replace with actual service call)
            await asyncio.sleep(0.01 + (target_service.response_time_ms / 1000.0))
            
            # Simulate response based on service type
            if request.target_service_type == ServiceType.PLANNER:
                response_payload = {
                    "plan": {"actions": ["navigate", "coordinate"], "duration": 300},
                    "confidence": 0.85
                }
            elif request.target_service_type == ServiceType.MONITOR:
                response_payload = {
                    "status": "healthy",
                    "metrics": {"cpu": 45.2, "memory": 67.8},
                    "alerts": []
                }
            elif request.target_service_type == ServiceType.CACHE:
                response_payload = {
                    "cache_hit": True,
                    "data": f"cached_data_for_{request.request_id}"
                }
            else:
                response_payload = {"result": "success", "data": f"processed_{request.request_id}"}
            
            return ServiceResponse(
                request_id=request.request_id,
                status_code=200,
                payload=response_payload
            )
            
        except Exception as e:
            return ServiceResponse(
                request_id=request.request_id,
                status_code=500,
                payload=None,
                error=str(e)
            )
        
        finally:
            # Update connection count
            target_service.current_connections -= 1
    
    async def _update_service_metrics(self, service_id: str, response: ServiceResponse):
        """Update service performance metrics."""
        if service_id not in self.service_metrics:
            return
        
        metrics = self.service_metrics[service_id]
        metrics["requests_total"] += 1
        
        if response.status_code < 400:
            metrics["requests_success"] += 1
        else:
            metrics["requests_failed"] += 1
        
        # Update average response time
        old_avg = metrics.get("avg_response_time_ms", 0.0)
        old_count = metrics["requests_total"] - 1
        
        if old_count > 0:
            new_avg = ((old_avg * old_count) + response.processing_time_ms) / metrics["requests_total"]
        else:
            new_avg = response.processing_time_ms
        
        metrics["avg_response_time_ms"] = new_avg
        metrics["last_seen"] = time.time()
        
        # Update service endpoint response time
        if service_id in self.services:
            self.services[service_id].response_time_ms = new_avg
        
        # Store latency history
        self.latency_history[service_id].append(response.processing_time_ms)
    
    def _update_consistent_hash_ring(self):
        """Update consistent hash ring for load balancing."""
        self.consistent_hash_ring.clear()
        
        # Add virtual nodes for each service
        virtual_nodes_per_service = 100
        
        for service_id in self.services:
            for i in range(virtual_nodes_per_service):
                hash_key = f"{service_id}:{i}"
                hash_value = hash(hash_key) % (2**32)
                self.consistent_hash_ring[hash_value] = service_id
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.running:
            try:
                for service in self.services.values():
                    await self._check_service_health(service)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_service_health(self, service: ServiceEndpoint):
        """Check health of individual service."""
        try:
            # Simple health check (replace with actual health endpoint call)
            current_time = time.time()
            
            # Check if service has been active recently
            metrics = self.service_metrics.get(service.service_id, {})
            last_seen = metrics.get("last_seen", 0)
            
            if current_time - last_seen > 60:  # No activity for 60 seconds
                if service.status == ServiceStatus.HEALTHY:
                    service.status = ServiceStatus.DEGRADED
                    self.logger.warning(f"Service {service.service_id} marked as degraded")
            else:
                if service.status == ServiceStatus.DEGRADED:
                    service.status = ServiceStatus.HEALTHY
                    self.logger.info(f"Service {service.service_id} recovered to healthy")
            
            service.last_health_check = current_time
            
        except Exception as e:
            service.status = ServiceStatus.UNHEALTHY
            self.logger.error(f"Health check failed for {service.service_id}: {e}")
    
    async def _service_discovery_loop(self):
        """Background service discovery loop."""
        while self.running:
            try:
                # Auto-discover services in the mesh
                await self._discover_edge_services()
                
                await asyncio.sleep(self.service_discovery_interval)
                
            except Exception as e:
                self.logger.error(f"Service discovery loop error: {e}")
                await asyncio.sleep(self.service_discovery_interval)
    
    async def _discover_edge_services(self):
        """Discover edge computing services."""
        # Simulate edge service discovery
        for edge_id, edge_info in self.edge_nodes.items():
            if edge_info.get("auto_register", False):
                # Check if edge services are already registered
                edge_service_id = f"edge_{edge_id}_executor"
                
                if edge_service_id not in self.services:
                    # Register edge executor service
                    edge_endpoint = ServiceEndpoint(
                        service_id=edge_service_id,
                        service_type=ServiceType.EXECUTOR,
                        host=edge_info["host"],
                        port=edge_info["port"],
                        region=edge_info.get("region", "edge"),
                        zone=edge_info.get("zone", "edge"),
                        weight=edge_info.get("weight", 0.8),  # Lower weight for edge
                        metadata={"edge_node": True, "capabilities": edge_info.get("capabilities", [])}
                    )
                    
                    self.register_service(edge_endpoint)
                    self.logger.info(f"Auto-registered edge service: {edge_service_id}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.running:
            try:
                # Collect and analyze service mesh metrics
                await self._collect_mesh_metrics()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_mesh_metrics(self):
        """Collect comprehensive service mesh metrics."""
        try:
            # Calculate mesh-wide metrics
            total_services = len(self.services)
            healthy_services = len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY])
            
            avg_response_time = 0.0
            total_requests = 0
            
            for metrics in self.service_metrics.values():
                total_requests += metrics.get("requests_total", 0)
                avg_response_time += metrics.get("avg_response_time_ms", 0.0)
            
            if total_services > 0:
                avg_response_time /= total_services
            
            # Record mesh performance
            record_performance_metrics(
                latency_ms=avg_response_time,
                throughput_rps=len(self.active_requests),
                cpu_usage_percent=(len(self.active_requests) / self.max_concurrent_requests) * 100,
            )
            
            self.logger.debug(f"Mesh metrics: {total_services} services, "
                            f"{healthy_services} healthy, "
                            f"{avg_response_time:.1f}ms avg response time")
        
        except Exception as e:
            self.logger.error(f"Error collecting mesh metrics: {e}")
    
    async def _request_cleanup_loop(self):
        """Clean up old requests and perform maintenance."""
        while self.running:
            try:
                current_time = time.time()
                
                # Clean up expired requests
                expired_requests = [
                    req_id for req_id, req in self.active_requests.items()
                    if current_time - req.created_at > req.timeout_seconds
                ]
                
                for req_id in expired_requests:
                    del self.active_requests[req_id]
                    self.logger.warning(f"Cleaned up expired request: {req_id}")
                
                # Optimize load balancer based on AI feedback
                if self.load_balancing_strategy == LoadBalancingStrategy.AI_OPTIMIZED:
                    await self._optimize_load_balancing()
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Request cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_load_balancing(self):
        """Optimize load balancing using AI feedback."""
        try:
            ai_optimizer = get_ai_optimizer()
            
            # Analyze service performance patterns
            for service_type, services in self.services_by_type.items():
                if not services:
                    continue
                
                # Find best performing service
                best_service = min(services, 
                                 key=lambda s: s.response_time_ms + (s.load_factor * 100))
                
                # Adjust weights based on performance
                for service in services:
                    metrics = self.service_metrics.get(service.service_id, {})
                    success_rate = metrics.get("requests_success", 0) / max(1, metrics.get("requests_total", 1))
                    
                    # Increase weight for better performing services
                    if success_rate > 0.95 and service.response_time_ms < best_service.response_time_ms * 1.2:
                        service.weight = min(2.0, service.weight * 1.1)
                    elif success_rate < 0.8 or service.response_time_ms > best_service.response_time_ms * 2.0:
                        service.weight = max(0.1, service.weight * 0.9)
        
        except Exception as e:
            self.logger.error(f"Load balancing optimization error: {e}")
    
    def add_edge_node(
        self, 
        edge_id: str, 
        host: str, 
        port: int, 
        capabilities: List[str] = None
    ):
        """Add edge computing node to the mesh.
        
        Args:
            edge_id: Edge node identifier
            host: Edge node host
            port: Edge node port
            capabilities: Edge node capabilities
        """
        self.edge_nodes[edge_id] = {
            "host": host,
            "port": port,
            "capabilities": capabilities or ["basic_execution"],
            "auto_register": True,
            "region": "edge",
            "zone": f"edge_{edge_id}",
            "weight": 0.8,
        }
        
        self.logger.info(f"Added edge node: {edge_id} at {host}:{port}")
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get comprehensive mesh status."""
        total_services = len(self.services)
        healthy_services = len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY])
        
        service_types_count = {}
        for service_type in ServiceType:
            count = len(self.services_by_type[service_type])
            service_types_count[service_type.value] = count
        
        avg_response_time = 0.0
        if self.service_metrics:
            avg_response_time = sum(
                metrics.get("avg_response_time_ms", 0.0) 
                for metrics in self.service_metrics.values()
            ) / len(self.service_metrics)
        
        return {
            "mesh_id": self.mesh_id,
            "total_services": total_services,
            "healthy_services": healthy_services,
            "service_availability": healthy_services / max(1, total_services),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "active_requests": len(self.active_requests),
            "max_concurrent_requests": self.max_concurrent_requests,
            "service_types": service_types_count,
            "edge_nodes": len(self.edge_nodes),
            "avg_response_time_ms": avg_response_time,
            "request_history_size": len(self.request_history),
            "uptime_seconds": time.time() - getattr(self, "_start_time", time.time()),
        }
    
    def get_service_metrics(self, service_id: str = None) -> Dict[str, Any]:
        """Get detailed service metrics."""
        if service_id:
            return self.service_metrics.get(service_id, {})
        else:
            return dict(self.service_metrics)
    
    async def scale_service_type(
        self, 
        service_type: ServiceType, 
        target_instances: int
    ) -> Dict[str, Any]:
        """Scale services of a specific type.
        
        Args:
            service_type: Service type to scale
            target_instances: Target number of instances
            
        Returns:
            Scaling operation result
        """
        current_instances = len(self.services_by_type[service_type])
        
        result = {
            "service_type": service_type.value,
            "current_instances": current_instances,
            "target_instances": target_instances,
            "scaling_action": "none",
            "success": False,
        }
        
        try:
            if target_instances > current_instances:
                # Scale up - simulate creating new service instances
                for i in range(target_instances - current_instances):
                    new_service_id = f"{service_type.value}_{uuid.uuid4().hex[:8]}"
                    new_endpoint = ServiceEndpoint(
                        service_id=new_service_id,
                        service_type=service_type,
                        host="auto-scaled-host",
                        port=8000 + i,
                        metadata={"auto_scaled": True}
                    )
                    
                    self.register_service(new_endpoint)
                
                result["scaling_action"] = "scale_up"
                result["success"] = True
                
            elif target_instances < current_instances:
                # Scale down - remove excess instances
                services_to_remove = self.services_by_type[service_type][target_instances:]
                
                for service in services_to_remove:
                    self.unregister_service(service.service_id)
                
                result["scaling_action"] = "scale_down"
                result["success"] = True
            
            result["final_instances"] = len(self.services_by_type[service_type])
            
            self.logger.info(f"Scaled {service_type.value} from {current_instances} "
                           f"to {result['final_instances']} instances")
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Service scaling failed: {e}")
        
        return result


# Global service mesh coordinator
_service_mesh: Optional[ServiceMeshCoordinator] = None

async def get_service_mesh() -> ServiceMeshCoordinator:
    """Get or create global service mesh coordinator."""
    global _service_mesh
    
    if _service_mesh is None:
        _service_mesh = ServiceMeshCoordinator(
            load_balancing_strategy=LoadBalancingStrategy.AI_OPTIMIZED,
            max_concurrent_requests=50000,  # Support high concurrency
        )
        await _service_mesh.start()
    
    return _service_mesh

async def route_service_request(
    source_service: str,
    target_service_type: ServiceType,
    method: str,
    payload: Any,
    timeout_seconds: float = 30.0,
) -> ServiceResponse:
    """Route a request through the service mesh."""
    try:
        mesh = await get_service_mesh()
        
        request = ServiceRequest(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            source_service=source_service,
            target_service_type=target_service_type,
            method=method,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        
        return await mesh.route_request(request)
        
    except Exception as e:
        return ServiceResponse(
            request_id="error",
            status_code=500,
            payload=None,
            error=str(e)
        )