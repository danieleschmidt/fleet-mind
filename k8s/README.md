# Fleet-Mind Kubernetes Deployment

This directory contains Kubernetes manifests and deployment scripts for running Fleet-Mind in production.

## Architecture Overview

Fleet-Mind is deployed as a microservices architecture on Kubernetes with the following components:

- **Coordinator**: Main application pods running the SwarmCoordinator
- **Redis**: Caching and session storage
- **Monitoring**: Prometheus and Grafana for observability  
- **Ingress**: NGINX ingress controller for external access
- **Security**: NetworkPolicies, RBAC, and PodSecurityPolicies

## Quick Start

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- NGINX Ingress Controller (will be installed if not present)
- OpenAI API key for production deployments

### Deploy to Development

```bash
# Quick deployment with defaults
./scripts/deploy-k8s.sh dev deploy

# Check status
./scripts/deploy-k8s.sh dev status
```

### Deploy to Production

```bash
# Interactive production deployment (will prompt for secrets)
./scripts/deploy-k8s.sh prod deploy
```

## Deployment Files

### Core Components

- `namespace.yaml` - Fleet-Mind namespace
- `coordinator-deployment.yaml` - Main application deployment
- `redis-deployment.yaml` - Redis cache deployment
- `configmap.yaml` - Application configuration

### Networking & Security

- `ingress.yaml` - Ingress configuration with TLS
- `security.yaml` - RBAC, ServiceAccounts, NetworkPolicies
- `production.yaml` - Production-specific configurations

### Monitoring & Operations

- `monitoring.yaml` - Prometheus and Grafana stack
- `production.yaml` - Backup jobs, resource quotas, alerts

## Configuration

### Environment Variables

Key configuration is managed through ConfigMaps and Secrets:

```yaml
# ConfigMap - Non-sensitive configuration
fleet-mind-config:
  - ENVIRONMENT: "production"  
  - LOG_LEVEL: "INFO"
  - REDIS_HOST: "fleet-mind-redis"
  - METRICS_PORT: "8082"

# Secret - Sensitive configuration  
fleet-mind-secrets:
  - openai-api-key: "your-openai-key"
  - jwt-secret: "auto-generated"
  - redis-password: "auto-generated"
```

### Resource Requirements

#### Development Environment
- **Coordinator**: 1 replica, 500m CPU, 1Gi memory
- **Redis**: 1 replica, 250m CPU, 512Mi memory
- **Total**: ~1 CPU, ~2Gi memory

#### Production Environment  
- **Coordinator**: 3-20 replicas (auto-scaling), 1 CPU, 2Gi memory each
- **Redis**: 1 replica, 500m CPU, 1Gi memory
- **Monitoring**: 500m CPU, 1Gi memory
- **Total**: 3-25 CPUs, 8-50Gi memory

## Monitoring & Observability

### Prometheus Metrics

Fleet-Mind exposes metrics on port 8082:

- `fleet_mind_missions_total` - Total missions executed
- `fleet_mind_drones_connected` - Currently connected drones
- `fleet_mind_latency_seconds` - Request latency
- `fleet_mind_errors_total` - Error counts by type

### Grafana Dashboards

Access monitoring at `http://your-domain/grafana`:

- Fleet-Mind Operations Dashboard
- Resource Usage Dashboard  
- Error Analysis Dashboard

Default credentials: `admin / admin123` (change in production)

### Alerting

Production deployment includes AlertManager rules for:

- High CPU/memory usage
- Pod crash loops
- Service outages
- High error rates
- Redis connectivity issues

## Security

### Network Security

- **NetworkPolicies**: Restrict pod-to-pod communication
- **Ingress**: TLS termination with proper certificates
- **RBAC**: Minimal permissions for service accounts

### Application Security

- **Secrets**: Encrypted storage of sensitive data
- **Non-root**: All containers run as non-root users
- **Security Context**: Drop all capabilities, read-only filesystem

### Production Hardening

- **PodSecurityPolicy**: Enforce security standards
- **Resource Quotas**: Prevent resource exhaustion
- **Backup**: Automated daily backups to persistent storage

## Scaling

### Horizontal Pod Autoscaler

Automatic scaling based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Custom metrics (request rate, queue depth)

```yaml
minReplicas: 3
maxReplicas: 20
```

### Manual Scaling

```bash
# Scale coordinator pods
kubectl scale deployment fleet-mind-coordinator --replicas=5 -n fleet-mind

# Scale Redis (single instance recommended)
kubectl scale deployment fleet-mind-redis --replicas=1 -n fleet-mind
```

## Operations

### Deployment Commands

```bash
# Deploy
./scripts/deploy-k8s.sh [env] deploy

# Update configuration
./scripts/deploy-k8s.sh [env] update  

# Check status
./scripts/deploy-k8s.sh [env] status

# View logs
./scripts/deploy-k8s.sh [env] logs [component]

# Delete deployment  
./scripts/deploy-k8s.sh [env] delete
```

### Troubleshooting

#### Pod Not Starting

```bash
# Check pod status
kubectl get pods -n fleet-mind

# Describe pod for events
kubectl describe pod <pod-name> -n fleet-mind

# Check logs
kubectl logs <pod-name> -n fleet-mind
```

#### Service Not Accessible

```bash  
# Check service endpoints
kubectl get endpoints -n fleet-mind

# Check ingress
kubectl describe ingress fleet-mind-ingress -n fleet-mind

# Test internal connectivity
kubectl run debug --image=busybox -i --tty --rm -- /bin/sh
```

#### Resource Issues

```bash
# Check resource usage
kubectl top pods -n fleet-mind
kubectl top nodes

# Check resource quotas
kubectl describe quota -n fleet-mind
```

### Backup & Recovery

#### Automated Backups

Daily backups run at 2 AM UTC:
- Redis data dump
- Configuration snapshots
- Logs archive

#### Manual Backup

```bash
# Create Redis backup
kubectl exec deployment/fleet-mind-redis -n fleet-mind -- redis-cli save
kubectl cp fleet-mind/redis-pod:/data/dump.rdb ./backup-$(date +%Y%m%d).rdb
```

#### Disaster Recovery

```bash
# Restore from backup
kubectl cp ./backup-20231201.rdb fleet-mind/redis-pod:/data/dump.rdb  
kubectl rollout restart deployment/fleet-mind-redis -n fleet-mind
```

## TLS Configuration

### Development (Self-Signed)

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=fleet-mind.yourdomain.com"

# Create TLS secret
kubectl create secret tls fleet-mind-tls \
  --cert=tls.crt --key=tls.key -n fleet-mind
```

### Production (Let's Encrypt)

Install cert-manager for automatic TLS:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Configure Let's Encrypt issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer  
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Performance Tuning

### JVM Options (if using Java components)

```yaml
env:
- name: JAVA_OPTS
  value: "-Xms1g -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### Redis Tuning

```yaml
args:
- redis-server
- --maxmemory
- 512mb  
- --maxmemory-policy
- allkeys-lru
- --save
- 900 1
```

### Network Optimization

```yaml
# Ingress annotations for WebRTC
nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
nginx.ingress.kubernetes.io/proxy-send-timeout: "3600" 
nginx.ingress.kubernetes.io/server-snippets: |
  location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
  }
```

## Compliance & Governance

### GDPR Compliance

- Data encryption at rest and in transit
- Configurable data retention periods
- User data deletion capabilities
- Audit logging enabled

### SOC 2 Controls

- Access control (RBAC)
- Monitoring and alerting
- Backup and recovery procedures
- Security policy enforcement

For additional support, see the main Fleet-Mind documentation or contact the development team.