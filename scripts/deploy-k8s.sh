#!/bin/bash
set -e

# Fleet-Mind Kubernetes Deployment Script
# Usage: ./deploy-k8s.sh [environment] [action]
# Environment: dev, staging, prod (default: dev)
# Action: deploy, update, delete, status (default: deploy)

ENVIRONMENT=${1:-dev}
ACTION=${2:-deploy}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_ROOT/k8s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if we can access the cluster
    kubectl get nodes &> /dev/null || {
        log_error "Cannot access Kubernetes nodes. Please check permissions."
        exit 1
    }
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace..."
    kubectl apply -f "$K8S_DIR/namespace.yaml"
    log_success "Namespace created/updated"
}

# Deploy secrets (with user input for sensitive data)
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets already exist
    if kubectl get secret fleet-mind-secrets -n fleet-mind &> /dev/null; then
        log_warning "Secrets already exist. Skipping secret creation."
        log_warning "To update secrets, delete them first: kubectl delete secret fleet-mind-secrets -n fleet-mind"
        return
    fi
    
    # Create secrets interactively for production
    if [ "$ENVIRONMENT" = "prod" ]; then
        log_warning "Production deployment detected. Please provide sensitive values:"
        
        read -sp "Enter OpenAI API Key: " OPENAI_API_KEY
        echo
        read -sp "Enter JWT Secret (or press enter for auto-generated): " JWT_SECRET
        echo
        read -sp "Enter Redis Password (or press enter for auto-generated): " REDIS_PASSWORD
        echo
        
        # Generate random values if not provided
        JWT_SECRET=${JWT_SECRET:-$(openssl rand -base64 64)}
        REDIS_PASSWORD=${REDIS_PASSWORD:-$(openssl rand -base64 32)}
        
        # Create secret with actual values
        kubectl create secret generic fleet-mind-secrets \
            --from-literal=openai-api-key="$OPENAI_API_KEY" \
            --from-literal=jwt-secret="$JWT_SECRET" \
            --from-literal=redis-password="$REDIS_PASSWORD" \
            --from-literal=db-username="fleet-mind" \
            --from-literal=db-password="$(openssl rand -base64 32)" \
            -n fleet-mind
        
        log_success "Production secrets created"
    else
        # For dev/staging, use the template
        kubectl apply -f "$K8S_DIR/security.yaml"
        log_success "Development secrets created"
    fi
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    log_success "Configuration deployed"
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis..."
    kubectl apply -f "$K8S_DIR/redis-deployment.yaml"
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/fleet-mind-redis -n fleet-mind
    log_success "Redis deployed and ready"
}

# Deploy main application
deploy_app() {
    log_info "Deploying Fleet-Mind coordinator..."
    kubectl apply -f "$K8S_DIR/coordinator-deployment.yaml"
    
    # Wait for deployment to be ready
    log_info "Waiting for coordinator to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/fleet-mind-coordinator -n fleet-mind
    log_success "Fleet-Mind coordinator deployed and ready"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    kubectl apply -f "$K8S_DIR/monitoring.yaml"
    
    # Wait for monitoring to be ready
    log_info "Waiting for monitoring to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n fleet-mind
    kubectl wait --for=condition=available --timeout=300s deployment/grafana -n fleet-mind
    log_success "Monitoring stack deployed and ready"
}

# Deploy ingress
deploy_ingress() {
    log_info "Deploying ingress..."
    
    # Check if NGINX ingress controller is installed
    if ! kubectl get ingressclass nginx &> /dev/null; then
        log_warning "NGINX ingress controller not found. Installing..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
        
        # Wait for ingress controller
        kubectl wait --namespace ingress-nginx \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/component=controller \
            --timeout=300s
    fi
    
    kubectl apply -f "$K8S_DIR/ingress.yaml"
    log_success "Ingress deployed"
}

# Deploy production extras
deploy_production() {
    if [ "$ENVIRONMENT" = "prod" ]; then
        log_info "Deploying production configuration..."
        kubectl apply -f "$K8S_DIR/production.yaml"
        log_success "Production configuration deployed"
    fi
}

# Main deployment function
deploy() {
    log_info "Starting Fleet-Mind deployment to $ENVIRONMENT environment..."
    
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_config
    
    # Apply security configurations
    kubectl apply -f "$K8S_DIR/security.yaml"
    
    deploy_redis
    deploy_app
    deploy_monitoring
    deploy_ingress
    deploy_production
    
    log_success "Deployment completed successfully!"
    
    # Show status
    show_status
}

# Update deployment
update() {
    log_info "Updating Fleet-Mind deployment..."
    
    check_prerequisites
    
    # Update configuration
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    
    # Rolling update of the main application
    kubectl rollout restart deployment/fleet-mind-coordinator -n fleet-mind
    kubectl rollout status deployment/fleet-mind-coordinator -n fleet-mind
    
    log_success "Update completed successfully!"
}

# Delete deployment
delete() {
    log_warning "This will delete the entire Fleet-Mind deployment. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "Deleting Fleet-Mind deployment..."
        
        # Delete in reverse order
        kubectl delete -f "$K8S_DIR/ingress.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/monitoring.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/coordinator-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/redis-deployment.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/production.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/security.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/configmap.yaml" --ignore-not-found=true
        kubectl delete -f "$K8S_DIR/namespace.yaml" --ignore-not-found=true
        
        log_success "Deployment deleted successfully!"
    else
        log_info "Deletion cancelled."
    fi
}

# Show deployment status
show_status() {
    log_info "Fleet-Mind Deployment Status:"
    echo
    
    # Namespace status
    echo "Namespace:"
    kubectl get namespace fleet-mind -o wide 2>/dev/null || echo "  Namespace not found"
    echo
    
    # Pod status
    echo "Pods:"
    kubectl get pods -n fleet-mind -o wide 2>/dev/null || echo "  No pods found"
    echo
    
    # Service status
    echo "Services:"
    kubectl get services -n fleet-mind -o wide 2>/dev/null || echo "  No services found"
    echo
    
    # Ingress status
    echo "Ingress:"
    kubectl get ingress -n fleet-mind -o wide 2>/dev/null || echo "  No ingress found"
    echo
    
    # PVC status
    echo "Persistent Volume Claims:"
    kubectl get pvc -n fleet-mind 2>/dev/null || echo "  No PVCs found"
    echo
    
    # Get external IP/hostname
    EXTERNAL_IP=$(kubectl get service -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Not available")
    if [ "$EXTERNAL_IP" != "Not available" ]; then
        echo "External Access:"
        echo "  Fleet-Mind API: http://$EXTERNAL_IP/api"
        echo "  WebRTC Endpoint: ws://$EXTERNAL_IP/webrtc"
        echo "  Metrics: http://$EXTERNAL_IP/metrics"
    fi
}

# Show logs
show_logs() {
    COMPONENT=${3:-coordinator}
    case $COMPONENT in
        coordinator)
            kubectl logs -f deployment/fleet-mind-coordinator -n fleet-mind
            ;;
        redis)
            kubectl logs -f deployment/fleet-mind-redis -n fleet-mind
            ;;
        prometheus)
            kubectl logs -f deployment/prometheus -n fleet-mind
            ;;
        grafana)
            kubectl logs -f deployment/grafana -n fleet-mind
            ;;
        *)
            log_error "Unknown component: $COMPONENT"
            log_info "Available components: coordinator, redis, prometheus, grafana"
            exit 1
            ;;
    esac
}

# Main script logic
case $ACTION in
    deploy)
        deploy
        ;;
    update)
        update
        ;;
    delete)
        delete
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 [environment] [action]"
        echo "Environment: dev, staging, prod (default: dev)"
        echo "Actions:"
        echo "  deploy  - Deploy Fleet-Mind to Kubernetes"
        echo "  update  - Update existing deployment"
        echo "  delete  - Delete the deployment"
        echo "  status  - Show deployment status"
        echo "  logs    - Show logs (specify component as 3rd argument)"
        exit 1
        ;;
esac