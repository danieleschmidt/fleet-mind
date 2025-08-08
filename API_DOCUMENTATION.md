# Fleet-Mind API Documentation

## Overview

Fleet-Mind provides a comprehensive RESTful API for real-time drone swarm coordination with WebRTC streaming capabilities. The API supports mission planning, fleet management, security controls, and performance monitoring with <100ms end-to-end latency.

**Base URL**: `https://api.fleet-mind.ai/v1`
**Authentication**: Bearer Token (JWT)
**Rate Limiting**: 1000 requests/minute per API key

## Authentication

### JWT Token Authentication
```http
Authorization: Bearer <your-jwt-token>
```

### API Key Authentication (Alternative)
```http
X-API-Key: <your-api-key>
```

### Get Access Token
```http
POST /auth/token
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh-token-here"
}
```

## Core API Endpoints

### Fleet Management

#### Get Fleet Status
```http
GET /fleet/status
```

**Response:**
```json
{
  "total_drones": 50,
  "active_drones": 47,
  "failed_drones": 2,
  "maintenance_drones": 1,
  "average_battery": 78.5,
  "average_health": 0.92,
  "communication_quality": 0.98,
  "missions_completed": 156,
  "total_flight_time_hours": 234.7,
  "last_updated": "2025-01-18T15:30:00Z"
}
```

#### List All Drones
```http
GET /fleet/drones
```

**Query Parameters:**
- `status` (optional): Filter by drone status
- `capability` (optional): Filter by capability
- `page` (optional): Page number (default: 1)
- `limit` (optional): Results per page (default: 50)

**Response:**
```json
{
  "drones": [
    {
      "drone_id": "drone_001",
      "status": "active",
      "position": {
        "x": 123.45,
        "y": 67.89,
        "z": 50.0
      },
      "velocity": {
        "vx": 2.1,
        "vy": 0.0,
        "vz": 0.0
      },
      "battery_percent": 85.2,
      "health_score": 0.95,
      "capabilities": [
        "basic_flight",
        "formation_flight",
        "camera",
        "thermal"
      ],
      "mission_id": "mission_abc123",
      "last_update": "2025-01-18T15:29:55Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 50,
    "has_next": false
  }
}
```

#### Get Specific Drone
```http
GET /fleet/drones/{drone_id}
```

**Response:**
```json
{
  "drone_id": "drone_001",
  "status": "active",
  "position": {
    "x": 123.45,
    "y": 67.89,
    "z": 50.0
  },
  "velocity": {
    "vx": 2.1,
    "vy": 0.0,
    "vz": 0.0
  },
  "orientation": {
    "roll": 0.1,
    "pitch": 0.05,
    "yaw": 45.0
  },
  "battery_percent": 85.2,
  "health_score": 0.95,
  "communication_quality": 0.98,
  "mission_progress": 67.3,
  "capabilities": [
    "basic_flight",
    "formation_flight",
    "camera",
    "thermal"
  ],
  "sensor_data": {
    "gps_quality": 0.95,
    "imu_status": "healthy",
    "camera_status": "recording",
    "thermal_temp": 23.5
  },
  "last_update": "2025-01-18T15:29:55Z"
}
```

#### Update Drone State
```http
PATCH /fleet/drones/{drone_id}
```

**Request:**
```json
{
  "position": {
    "x": 125.0,
    "y": 70.0,
    "z": 55.0
  },
  "battery_percent": 84.8,
  "sensor_data": {
    "camera_status": "idle"
  }
}
```

**Response:**
```json
{
  "success": true,
  "updated_fields": ["position", "battery_percent", "sensor_data"],
  "timestamp": "2025-01-18T15:30:10Z"
}
```

### Mission Management

#### List Missions
```http
GET /missions
```

**Query Parameters:**
- `status` (optional): Filter by mission status
- `start_date` (optional): Filter missions after date (ISO 8601)
- `end_date` (optional): Filter missions before date (ISO 8601)

**Response:**
```json
{
  "missions": [
    {
      "mission_id": "mission_abc123",
      "title": "Search and Rescue - Grid Pattern",
      "status": "executing",
      "created_at": "2025-01-18T14:00:00Z",
      "started_at": "2025-01-18T14:05:00Z",
      "estimated_completion": "2025-01-18T15:00:00Z",
      "progress": 67.3,
      "assigned_drones": ["drone_001", "drone_002", "drone_003"],
      "constraints": {
        "max_altitude": 100.0,
        "safety_distance": 8.0,
        "battery_reserve": 25.0
      }
    }
  ]
}
```

#### Create Mission
```http
POST /missions
```

**Request:**
```json
{
  "title": "Perimeter Security Patrol",
  "description": "Patrol the facility perimeter using 4 drones in formation",
  "mission_type": "patrol",
  "constraints": {
    "max_altitude": 80.0,
    "safety_distance": 10.0,
    "battery_reserve": 30.0,
    "geofence": [
      {"lat": 37.7749, "lng": -122.4194},
      {"lat": 37.7849, "lng": -122.4094}
    ]
  },
  "required_capabilities": ["formation_flight", "camera"],
  "priority": "medium",
  "estimated_duration_minutes": 45
}
```

**Response:**
```json
{
  "mission_id": "mission_def456",
  "status": "planning",
  "created_at": "2025-01-18T15:30:00Z",
  "planning_latency_ms": 127.3,
  "assigned_drones": ["drone_004", "drone_005", "drone_006", "drone_007"],
  "plan_summary": {
    "total_waypoints": 24,
    "formations": ["line", "diamond"],
    "estimated_duration": 48.5
  }
}
```

#### Get Mission Details
```http
GET /missions/{mission_id}
```

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "title": "Search and Rescue - Grid Pattern",
  "description": "Search 2x2km area for survivors using thermal imaging",
  "status": "executing",
  "created_at": "2025-01-18T14:00:00Z",
  "started_at": "2025-01-18T14:05:00Z",
  "progress": 67.3,
  "assigned_drones": ["drone_001", "drone_002", "drone_003"],
  "mission_plan": {
    "plan_id": "plan_789",
    "objectives": [
      "Survey designated search area",
      "Maintain grid formation",
      "Report thermal anomalies"
    ],
    "action_sequences": [
      {
        "sequence_id": "seq_001",
        "actions": [
          {
            "type": "takeoff",
            "altitude": 50,
            "duration": 30
          },
          {
            "type": "formation",
            "pattern": "grid",
            "spacing": 100
          }
        ],
        "duration_seconds": 180,
        "priority": "high"
      }
    ],
    "formations": [
      {
        "formation_type": "grid",
        "spacing_meters": 100.0,
        "orientation_degrees": 0.0
      }
    ]
  },
  "performance_metrics": {
    "area_covered_km2": 1.35,
    "targets_detected": 2,
    "formation_quality": 0.87,
    "communication_latency_ms": 45.2
  }
}
```

#### Execute Mission
```http
POST /missions/{mission_id}/execute
```

**Request:**
```json
{
  "start_immediately": true,
  "monitor_frequency": 5.0,
  "replan_threshold": 0.7
}
```

**Response:**
```json
{
  "mission_id": "mission_abc123",
  "execution_started": true,
  "started_at": "2025-01-18T15:30:15Z",
  "monitoring_frequency": 5.0,
  "initial_status": "executing"
}
```

#### Pause/Resume Mission
```http
POST /missions/{mission_id}/pause
POST /missions/{mission_id}/resume
```

#### Cancel Mission
```http
DELETE /missions/{mission_id}
```

### Real-time Communication

#### WebRTC Connection Status
```http
GET /communication/status
```

**Response:**
```json
{
  "active_connections": 47,
  "total_bandwidth_mbps": 12.5,
  "average_latency_ms": 23.1,
  "packet_loss_rate": 0.001,
  "connection_quality": 0.98,
  "protocols_active": ["webrtc", "websocket"],
  "last_updated": "2025-01-18T15:30:00Z"
}
```

#### Send Command to Drone
```http
POST /communication/drones/{drone_id}/command
```

**Request:**
```json
{
  "command": "move_to",
  "parameters": {
    "position": {
      "x": 150.0,
      "y": 75.0,
      "z": 60.0
    },
    "speed": 5.0,
    "heading": 90.0
  },
  "priority": "high",
  "timeout_seconds": 30.0
}
```

**Response:**
```json
{
  "command_id": "cmd_123456",
  "status": "sent",
  "drone_id": "drone_001",
  "sent_at": "2025-01-18T15:30:10Z",
  "expected_completion": "2025-01-18T15:30:40Z",
  "transmission_latency_ms": 18.5
}
```

#### Broadcast Message
```http
POST /communication/broadcast
```

**Request:**
```json
{
  "message": {
    "type": "formation_update",
    "data": {
      "formation_type": "diamond",
      "spacing": 15.0
    }
  },
  "target_drones": ["drone_001", "drone_002", "drone_003"],
  "priority": "medium",
  "reliability_mode": "best_effort"
}
```

### Security and Compliance

#### Security Status
```http
GET /security/status
```

**Response:**
```json
{
  "security_level": "high",
  "active_threats": 0,
  "blocked_sources": ["192.168.1.100"],
  "recent_events": [
    {
      "event_id": "sec_001",
      "timestamp": "2025-01-18T15:25:00Z",
      "type": "failed_authentication",
      "source": "192.168.1.50",
      "action_taken": "rate_limited",
      "severity": "medium"
    }
  ],
  "threat_detection_enabled": true,
  "last_key_rotation": "2025-01-18T12:00:00Z"
}
```

#### Generate Drone Credentials
```http
POST /security/drones/{drone_id}/credentials
```

**Request:**
```json
{
  "permissions": [
    "basic_flight",
    "telemetry",
    "emergency_response"
  ],
  "expiry_days": 30
}
```

**Response:**
```json
{
  "drone_id": "drone_001",
  "credentials": {
    "public_key": "-----BEGIN PUBLIC KEY-----\n...",
    "certificate": "-----BEGIN CERTIFICATE-----\n...",
    "issued_at": "2025-01-18T15:30:00Z",
    "expires_at": "2025-02-17T15:30:00Z"
  },
  "permissions": [
    "basic_flight",
    "telemetry", 
    "emergency_response"
  ]
}
```

#### Compliance Audit
```http
POST /compliance/audit
```

**Request:**
```json
{
  "standard": "GDPR",
  "scope": "full",
  "include_recommendations": true
}
```

**Response:**
```json
{
  "audit_id": "audit_gdpr_20250118",
  "standard": "GDPR",
  "timestamp": "2025-01-18T15:30:00Z",
  "compliance_rate": 0.95,
  "violations": [
    {
      "requirement_id": "gdpr_data_retention",
      "description": "Some data exceeds retention period",
      "severity": "medium",
      "recommendation": "Implement automated data purging"
    }
  ],
  "recommendations": [
    "Update privacy policy with data retention schedules",
    "Implement consent withdrawal mechanism",
    "Add data portability export feature"
  ],
  "next_audit_date": "2025-01-25T15:30:00Z"
}
```

### Performance and Monitoring

#### System Health
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-18T15:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "components": {
    "coordinator": {
      "status": "healthy",
      "response_time_ms": 12.5,
      "memory_usage_mb": 245.8,
      "cpu_usage_percent": 15.2
    },
    "communication": {
      "status": "healthy", 
      "active_connections": 47,
      "average_latency_ms": 23.1
    },
    "database": {
      "status": "healthy",
      "connection_pool_size": 10,
      "query_response_time_ms": 5.8
    }
  }
}
```

#### Performance Metrics
```http
GET /metrics
```

**Response:**
```json
{
  "timestamp": "2025-01-18T15:30:00Z",
  "system_metrics": {
    "cpu_usage_percent": 15.2,
    "memory_usage_mb": 245.8,
    "disk_usage_gb": 12.5,
    "network_throughput_mbps": 8.7
  },
  "application_metrics": {
    "requests_per_second": 42.1,
    "average_response_time_ms": 127.3,
    "error_rate": 0.001,
    "active_missions": 3,
    "total_drones_managed": 50
  },
  "mission_metrics": {
    "missions_completed_today": 7,
    "average_mission_duration_minutes": 35.2,
    "success_rate": 0.98,
    "formation_quality_score": 0.87
  }
}
```

#### Get Performance Statistics
```http
GET /performance/stats
```

**Query Parameters:**
- `timeframe` (optional): `1h`, `24h`, `7d`, `30d` (default: `1h`)
- `metric_type` (optional): `latency`, `throughput`, `errors`

**Response:**
```json
{
  "timeframe": "24h",
  "summary": {
    "total_requests": 48250,
    "average_latency_ms": 89.3,
    "p95_latency_ms": 156.7,
    "p99_latency_ms": 298.1,
    "error_rate": 0.002,
    "peak_throughput_rps": 125.8
  },
  "timeseries": [
    {
      "timestamp": "2025-01-18T14:00:00Z",
      "latency_ms": 87.2,
      "throughput_rps": 45.3,
      "error_count": 0
    }
  ]
}
```

## WebSocket API

### Real-time Mission Updates
```javascript
const ws = new WebSocket('wss://api.fleet-mind.ai/v1/ws/missions/{mission_id}');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log('Mission update:', update);
};
```

**Message Format:**
```json
{
  "type": "mission_progress",
  "mission_id": "mission_abc123",
  "progress": 72.5,
  "active_drones": 3,
  "current_phase": "formation_flight",
  "timestamp": "2025-01-18T15:30:00Z"
}
```

### Real-time Drone Telemetry
```javascript
const ws = new WebSocket('wss://api.fleet-mind.ai/v1/ws/drones/{drone_id}/telemetry');

ws.onmessage = function(event) {
  const telemetry = JSON.parse(event.data);
  console.log('Telemetry:', telemetry);
};
```

## Error Handling

### HTTP Status Codes
- `200 OK` - Successful request
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - System maintenance

### Error Response Format
```json
{
  "error": {
    "code": "DRONE_NOT_FOUND",
    "message": "Drone with ID 'drone_999' not found",
    "details": {
      "requested_drone_id": "drone_999",
      "available_drones": 50
    },
    "timestamp": "2025-01-18T15:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes
- `INVALID_DRONE_ID` - Invalid drone identifier
- `MISSION_NOT_FOUND` - Mission does not exist
- `INSUFFICIENT_PERMISSIONS` - Missing required permissions
- `DRONE_BUSY` - Drone assigned to another mission
- `COMMUNICATION_ERROR` - Unable to communicate with drone
- `VALIDATION_ERROR` - Request validation failed
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `SYSTEM_OVERLOAD` - System at capacity

## SDK Examples

### Python SDK
```python
from fleet_mind_sdk import FleetMindClient

# Initialize client
client = FleetMindClient(
    base_url="https://api.fleet-mind.ai/v1",
    api_key="your-api-key"
)

# Get fleet status
status = client.fleet.get_status()
print(f"Active drones: {status.active_drones}")

# Create and execute mission
mission = client.missions.create({
    "title": "Test Mission",
    "description": "Formation flight test",
    "constraints": {
        "max_altitude": 100.0,
        "safety_distance": 5.0
    }
})

# Execute mission
client.missions.execute(mission.mission_id)

# Monitor progress
for update in client.missions.stream_progress(mission.mission_id):
    print(f"Progress: {update.progress}%")
    if update.status == "completed":
        break
```

### JavaScript SDK
```javascript
import FleetMind from '@fleet-mind/sdk';

const client = new FleetMind({
  baseUrl: 'https://api.fleet-mind.ai/v1',
  apiKey: 'your-api-key'
});

// Get fleet status
const status = await client.fleet.getStatus();
console.log(`Active drones: ${status.activeDrones}`);

// Create mission
const mission = await client.missions.create({
  title: 'Test Mission',
  description: 'Formation flight test',
  constraints: {
    maxAltitude: 100.0,
    safetyDistance: 5.0
  }
});

// Execute mission
await client.missions.execute(mission.missionId);

// Stream real-time updates
client.missions.streamProgress(mission.missionId, (update) => {
  console.log(`Progress: ${update.progress}%`);
});
```

## Rate Limiting

### Limits
- **Standard Tier**: 1,000 requests/hour
- **Professional Tier**: 10,000 requests/hour  
- **Enterprise Tier**: 100,000 requests/hour

### Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642521600
```

## Webhooks

### Configuration
```http
POST /webhooks
```

**Request:**
```json
{
  "url": "https://your-app.com/webhooks/fleet-mind",
  "events": [
    "mission.completed",
    "drone.low_battery",
    "security.threat_detected"
  ],
  "secret": "your-webhook-secret"
}
```

### Webhook Events
- `mission.started` - Mission execution started
- `mission.completed` - Mission completed successfully
- `mission.failed` - Mission execution failed
- `drone.connected` - Drone connected to fleet
- `drone.disconnected` - Drone disconnected
- `drone.low_battery` - Drone battery below threshold
- `security.threat_detected` - Security threat identified
- `system.health_warning` - System health issue

### Webhook Payload
```json
{
  "event": "mission.completed",
  "timestamp": "2025-01-18T15:30:00Z",
  "data": {
    "mission_id": "mission_abc123",
    "duration_seconds": 1847,
    "drones_used": 4,
    "success_metrics": {
      "area_covered_km2": 2.1,
      "formation_quality": 0.89
    }
  },
  "webhook_id": "wh_123456"
}
```

## Support

### Documentation
- **API Reference**: https://docs.fleet-mind.ai/api
- **SDK Documentation**: https://docs.fleet-mind.ai/sdks
- **Tutorials**: https://docs.fleet-mind.ai/tutorials

### Support Channels
- **Technical Support**: api-support@terragon.ai
- **Bug Reports**: https://github.com/terragon/fleet-mind/issues
- **Feature Requests**: https://feedback.fleet-mind.ai

### Status Page
Monitor API status and incidents: https://status.fleet-mind.ai

---

**API Version**: v1.0.0  
**Last Updated**: 2025-01-18  
**OpenAPI Specification**: Available at `/v1/openapi.json`