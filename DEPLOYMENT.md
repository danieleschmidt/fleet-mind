# Fleet-Mind Production Deployment Guide

## Overview
Fleet-Mind is a real-time LLM-powered drone swarm coordination platform with <100ms end-to-end latency, supporting 100+ drones with advanced security, monitoring, and global compliance features.

## System Requirements

### Minimum Hardware Requirements
- **CPU**: 8-core processor (Intel Xeon or AMD EPYC recommended)
- **Memory**: 16GB RAM (32GB recommended for production)
- **Storage**: 100GB SSD (NVMe recommended)
- **Network**: 1Gbps connection with low latency

### Recommended Production Hardware
- **CPU**: 16-core processor with hyper-threading
- **Memory**: 64GB RAM
- **Storage**: 500GB NVMe SSD
- **Network**: 10Gbps connection
- **GPU**: NVIDIA GPU for ML acceleration (optional but recommended)

## Deployment Options

### 1. Docker Deployment (Recommended)

#### Prerequisites
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### Quick Start
```bash
# Clone repository
git clone https://github.com/your-org/fleet-mind.git
cd fleet-mind

# Configure environment
cp .env.example .env
vim .env  # Configure your settings

# Start services
docker-compose up -d

# Verify deployment
docker-compose ps
docker-compose logs -f coordinator
```

#### Production Docker Compose Configuration
```yaml
version: '3.8'

services:
  coordinator:
    image: fleet-mind:latest
    ports:
      - "8080:8080"
      - "8443:8443"
    environment:
      - FLEET_MIND_ENV=production
      - MAX_DRONES=100
      - LOG_LEVEL=INFO
      - SECURITY_LEVEL=high
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  grafana_data:
```

### 2. Kubernetes Deployment

#### Namespace and ConfigMap
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: fleet-mind

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fleet-mind-config
  namespace: fleet-mind
data:
  config.json: |
    {
      "max_drones": 100,
      "security_level": "high",
      "log_level": "INFO",
      "enable_metrics": true
    }
```

#### Deployment Configuration
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fleet-mind-coordinator
  namespace: fleet-mind
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fleet-mind-coordinator
  template:
    metadata:
      labels:
        app: fleet-mind-coordinator
    spec:
      containers:
      - name: coordinator
        image: fleet-mind:latest
        ports:
        - containerPort: 8080
        - containerPort: 8443
        env:
        - name: FLEET_MIND_ENV
          value: "production"
        - name: MAX_DRONES
          value: "100"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: config
        configMap:
          name: fleet-mind-config
      - name: logs
        emptyDir: {}

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fleet-mind-service
  namespace: fleet-mind
spec:
  selector:
    app: fleet-mind-coordinator
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: https
    port: 443
    targetPort: 8443
  type: LoadBalancer

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fleet-mind-ingress
  namespace: fleet-mind
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - fleet-mind.yourdomain.com
    secretName: fleet-mind-tls
  rules:
  - host: fleet-mind.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fleet-mind-service
            port:
              number: 80
```

### 3. Cloud Provider Deployments

#### AWS ECS Deployment
```json
{
  "family": "fleet-mind-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "fleet-mind-coordinator",
      "image": "your-repo/fleet-mind:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLEET_MIND_ENV",
          "value": "production"
        },
        {
          "name": "MAX_DRONES",
          "value": "100"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fleet-mind",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8080/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Google Cloud Run Deployment
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: fleet-mind
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "8Gi"
        run.googleapis.com/cpu: "4"
    spec:
      containerConcurrency: 1000
      containers:
      - image: gcr.io/your-project/fleet-mind:latest
        ports:
        - containerPort: 8080
        env:
        - name: FLEET_MIND_ENV
          value: production
        - name: MAX_DRONES
          value: "100"
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
```

## Configuration

### Environment Variables
```bash
# Core Configuration
FLEET_MIND_ENV=production
MAX_DRONES=100
UPDATE_RATE=10.0
LATENT_DIM=512

# Security Configuration
SECURITY_LEVEL=high
ENABLE_TLS=true
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# Database Configuration
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/fleetmind

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_OUTPUT_DIR=/app/logs

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=true

# External Services
OPENAI_API_KEY=your-openai-key
WEATHER_API_KEY=your-weather-key
```

### Configuration File (config.json)
```json
{
  "coordinator": {
    "max_drones": 100,
    "update_rate": 10.0,
    "latent_dim": 512,
    "llm_model": "gpt-4o",
    "safety_constraints": {
      "max_altitude": 120.0,
      "safety_distance": 5.0,
      "battery_warning_threshold": 20.0,
      "battery_critical_threshold": 10.0
    }
  },
  "communication": {
    "protocol": "webrtc",
    "topology": "mesh",
    "compression": "learned_vqvae",
    "enable_encryption": true,
    "timeout_seconds": 30.0
  },
  "security": {
    "level": "high",
    "enable_threat_detection": true,
    "key_rotation_interval": 3600,
    "max_failed_attempts": 5,
    "lockout_duration": 300
  },
  "monitoring": {
    "enable_health_checks": true,
    "check_interval": 30.0,
    "alert_thresholds": {
      "cpu_usage": 80.0,
      "memory_usage": 85.0,
      "error_rate": 0.05,
      "response_time": 1000.0
    }
  },
  "compliance": {
    "enabled_standards": ["GDPR", "CCPA", "FAA_PART_107"],
    "data_retention_days": 90,
    "audit_frequency": "daily"
  },
  "performance": {
    "enable_auto_scaling": true,
    "min_workers": 2,
    "max_workers": 16,
    "enable_caching": true,
    "cache_ttl": 3600
  }
}
```

## Security Hardening

### 1. Network Security
```bash
# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 8080/tcp # Fleet-Mind API
sudo ufw deny 6379/tcp  # Redis (internal only)
```

### 2. TLS Configuration
```bash
# Generate TLS certificates
openssl req -x509 -newkey rsa:4096 -keyout private-key.pem -out cert.pem -days 365 -nodes

# Or use Let's Encrypt
certbot certonly --standalone -d fleet-mind.yourdomain.com
```

### 3. Access Control
```yaml
# rbac.yaml (Kubernetes)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: fleet-mind
  name: fleet-mind-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: fleet-mind-rolebinding
  namespace: fleet-mind
roleRef:
  kind: Role
  name: fleet-mind-role
  apiGroup: rbac.authorization.k8s.io
subjects:
- kind: ServiceAccount
  name: fleet-mind-service-account
  namespace: fleet-mind
```

## Monitoring and Observability

### 1. Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fleet-mind'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### 2. Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Fleet-Mind Monitoring",
    "panels": [
      {
        "title": "Active Drones",
        "type": "graph",
        "targets": [
          {
            "expr": "fleet_mind_active_drones",
            "legendFormat": "Active Drones"
          }
        ]
      },
      {
        "title": "Mission Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "fleet_mind_mission_latency_ms",
            "legendFormat": "Latency (ms)"
          }
        ]
      },
      {
        "title": "System Health",
        "type": "singlestat",
        "targets": [
          {
            "expr": "fleet_mind_health_score",
            "legendFormat": "Health Score"
          }
        ]
      }
    ]
  }
}
```

### 3. Log Aggregation (ELK Stack)
```yaml
# logstash.conf
input {
  file {
    path => "/app/logs/fleet-mind.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
  if [level] == "ERROR" {
    mutate { add_tag => [ "error" ] }
  }
  
  if [component] == "security" {
    mutate { add_tag => [ "security" ] }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "fleet-mind-%{+YYYY.MM.dd}"
  }
}
```

## High Availability Setup

### 1. Load Balancer Configuration (NGINX)
```nginx
upstream fleet_mind_backend {
    server 10.0.1.10:8080 weight=3;
    server 10.0.1.11:8080 weight=2;
    server 10.0.1.12:8080 weight=1 backup;
}

server {
    listen 80;
    listen 443 ssl;
    server_name fleet-mind.yourdomain.com;

    ssl_certificate /etc/ssl/certs/fleet-mind.crt;
    ssl_certificate_key /etc/ssl/private/fleet-mind.key;

    location / {
        proxy_pass http://fleet_mind_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebRTC support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /health {
        proxy_pass http://fleet_mind_backend/health;
        access_log off;
    }
}
```

### 2. Database Clustering (Redis Sentinel)
```bash
# redis-sentinel.conf
port 26379
sentinel monitor fleet-mind-master 127.0.0.1 6379 2
sentinel down-after-milliseconds fleet-mind-master 5000
sentinel failover-timeout fleet-mind-master 10000
sentinel parallel-syncs fleet-mind-master 1
```

## Backup and Disaster Recovery

### 1. Data Backup Strategy
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/fleet-mind"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup Redis data
redis-cli --rdb "$BACKUP_DIR/$DATE/dump.rdb"

# Backup configuration
cp -r /app/config "$BACKUP_DIR/$DATE/"

# Backup logs
tar -czf "$BACKUP_DIR/$DATE/logs.tar.gz" /app/logs/

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR/$DATE"
```

### 2. Disaster Recovery Plan
```yaml
# disaster-recovery.yml
recovery_procedures:
  data_corruption:
    - stop_services
    - restore_from_backup
    - validate_data_integrity
    - restart_services
    - verify_functionality
  
  service_outage:
    - check_health_endpoints
    - restart_failed_components
    - scale_up_if_needed
    - notify_operations_team
  
  security_breach:
    - isolate_affected_systems
    - rotate_all_credentials
    - audit_access_logs
    - patch_vulnerabilities
    - restore_from_clean_backup
```

## Performance Tuning

### 1. System Optimization
```bash
# /etc/sysctl.conf
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 5000
vm.swappiness = 1
fs.file-max = 2097152

# Apply changes
sudo sysctl -p
```

### 2. Container Optimization
```dockerfile
# Production Dockerfile optimizations
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash fleetmind
USER fleetmind

# Expose ports
EXPOSE 8080 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD ["python", "-m", "fleet_mind.cli", "start", "--config", "/app/config/production.json"]
```

## Troubleshooting

### Common Issues

1. **High Latency**
   ```bash
   # Check network connectivity
   ping drone-fleet-1.local
   
   # Monitor network statistics
   ss -tuln | grep :8080
   
   # Check system resources
   htop
   iostat -x 1
   ```

2. **Memory Leaks**
   ```bash
   # Monitor memory usage
   docker stats fleet-mind-coordinator
   
   # Check for memory leaks
   pmap -d $(pgrep -f fleet-mind)
   
   # Generate heap dump (if using debug build)
   kill -USR1 $(pgrep -f fleet-mind)
   ```

3. **Connection Issues**
   ```bash
   # Test WebRTC connectivity
   curl -I http://localhost:8080/webrtc/status
   
   # Check certificate validity
   openssl x509 -in cert.pem -text -noout
   
   # Verify DNS resolution
   nslookup fleet-mind.yourdomain.com
   ```

### Log Analysis
```bash
# Real-time log monitoring
tail -f /app/logs/fleet-mind.log | jq '.'

# Error analysis
grep -E "(ERROR|CRITICAL)" /app/logs/fleet-mind.log | tail -20

# Performance analysis
grep "latency_ms" /app/logs/fleet-mind.log | awk '{print $NF}' | sort -n
```

## Scaling Guidelines

### Horizontal Scaling
- Deploy multiple coordinator instances behind load balancer
- Use Redis cluster for distributed caching
- Implement drone affinity for consistent connections

### Vertical Scaling
- Monitor CPU and memory usage patterns
- Scale based on drone count and mission complexity
- Consider GPU acceleration for ML workloads

### Auto-scaling Configuration
```yaml
# hpa.yaml (Kubernetes)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fleet-mind-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fleet-mind-coordinator
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Support and Maintenance

### Regular Maintenance Tasks
- Daily: Log analysis and error monitoring
- Weekly: Performance metrics review and capacity planning
- Monthly: Security updates and vulnerability scans
- Quarterly: Disaster recovery testing and backup validation

### Support Contacts
- **Technical Support**: support@terragon.ai
- **Security Issues**: security@terragon.ai
- **Emergency Hotline**: +1-XXX-XXX-XXXX

### Documentation Updates
This deployment guide is maintained at: https://docs.fleet-mind.ai/deployment

Last updated: 2025-01-18
Version: 1.0.0