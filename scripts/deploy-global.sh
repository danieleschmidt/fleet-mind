#!/bin/bash
set -e

# Fleet-Mind Global Deployment Script
# Deploys Fleet-Mind to multiple regions with compliance and localization
# Usage: ./deploy-global.sh [region] [action]

REGION=${1:-all}
ACTION=${2:-deploy}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
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

# Regional configurations
declare -A REGIONS=(
    ["europe"]="eu-west-1 fleet-mind-eu"
    ["north_america"]="us-east-1 fleet-mind-na"
    ["asia_pacific"]="ap-southeast-1 fleet-mind-apac"
)

# Load global configuration
load_global_config() {
    if [ ! -f "$CONFIG_DIR/global-deployment.yaml" ]; then
        log_error "Global deployment configuration not found"
        exit 1
    fi
    
    log_info "Global deployment configuration loaded"
}

# Check prerequisites for global deployment
check_global_prerequisites() {
    log_info "Checking global deployment prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "yq" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check cloud provider CLIs
    local cloud_tools=("gcloud" "aws" "az")
    local available_clouds=0
    for tool in "${cloud_tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            ((available_clouds++))
        fi
    done
    
    if [ $available_clouds -eq 0 ]; then
        log_error "At least one cloud provider CLI must be installed (gcloud, aws, or az)"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup kubectl context for region
setup_kubectl_context() {
    local region=$1
    local cluster_info=(${REGIONS[$region]})
    local aws_region=${cluster_info[0]}
    local cluster_name=${cluster_info[1]}
    
    log_info "Setting up kubectl context for $region ($cluster_name)"
    
    # Try different cloud providers
    if command -v gcloud &> /dev/null; then
        # Google Cloud
        if gcloud container clusters get-credentials "$cluster_name" --region="$aws_region" --project="${GCP_PROJECT_ID:-fleet-mind-prod}" 2>/dev/null; then
            log_success "Connected to GKE cluster: $cluster_name"
            return 0
        fi
    fi
    
    if command -v aws &> /dev/null; then
        # AWS EKS
        if aws eks update-kubeconfig --region "$aws_region" --name "$cluster_name" 2>/dev/null; then
            log_success "Connected to EKS cluster: $cluster_name"
            return 0
        fi
    fi
    
    if command -v az &> /dev/null; then
        # Azure AKS
        local resource_group="fleet-mind-${region}"
        if az aks get-credentials --resource-group "$resource_group" --name "$cluster_name" 2>/dev/null; then
            log_success "Connected to AKS cluster: $cluster_name"
            return 0
        fi
    fi
    
    log_error "Could not connect to cluster $cluster_name in region $region"
    return 1
}

# Deploy compliance configurations
deploy_compliance() {
    local region=$1
    
    log_info "Deploying compliance configurations for $region"
    
    # Create compliance namespace
    kubectl create namespace fleet-mind-compliance --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy region-specific compliance configmap
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: compliance-config-$region
  namespace: fleet-mind-compliance
data:
  region: "$region"
  gdpr_enabled: "$([ "$region" = "europe" ] && echo "true" || echo "false")"
  ccpa_enabled: "$([ "$region" = "north_america" ] && echo "true" || echo "false")"
  pdpa_enabled: "$([ "$region" = "asia_pacific" ] && echo "true" || echo "false")"
  data_residency_required: "$([ "$region" != "north_america" ] && echo "true" || echo "false")"
  audit_retention_years: "7"
  consent_management: "$([ "$region" = "europe" ] && echo "strict" || echo "standard")"
EOF
    
    # Deploy data retention job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: data-retention-cleanup-$region
  namespace: fleet-mind-compliance
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM local time
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: compliance-cleanup
            image: fleet-mind/compliance:latest
            command: ["python", "/app/cleanup_expired_data.py"]
            env:
            - name: REGION
              value: "$region"
            - name: DRY_RUN
              value: "false"
          restartPolicy: OnFailure
EOF
    
    log_success "Compliance configurations deployed for $region"
}

# Deploy localization configurations
deploy_localization() {
    local region=$1
    
    log_info "Deploying localization configurations for $region"
    
    # Get region-specific languages from global config
    local languages
    case $region in
        "europe")
            languages="en,de,fr,es,it,nl"
            timezone="Europe/London"
            currency="EUR"
            ;;
        "north_america")
            languages="en,es,fr"
            timezone="America/New_York"
            currency="USD"
            ;;
        "asia_pacific")
            languages="en,ja,ko,zh-CN,zh-TW,hi"
            timezone="Asia/Singapore"
            currency="USD"
            ;;
        *)
            languages="en"
            timezone="UTC"
            currency="USD"
            ;;
    esac
    
    # Create localization configmap
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: localization-config-$region
  namespace: fleet-mind
data:
  supported_languages: "$languages"
  default_language: "en"
  timezone: "$timezone"
  currency: "$currency"
  date_format: "YYYY-MM-DD"
  distance_unit: "$([ "$region" = "north_america" ] && echo "feet" || echo "meters")"
  coordinate_system: "WGS84"
EOF
    
    log_success "Localization configurations deployed for $region"
}

# Deploy regional monitoring
deploy_regional_monitoring() {
    local region=$1
    
    log_info "Deploying regional monitoring for $region"
    
    # Deploy region-specific Prometheus configuration
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config-$region
  namespace: fleet-mind
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'fleet-mind-$region'
        region: '$region'
    
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    scrape_configs:
    - job_name: 'fleet-mind-coordinator-$region'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: fleet-mind-coordinator
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - replacement: '$region'
        target_label: region
    
    remote_write:
    - url: https://prometheus-global.fleet-mind.com/api/v1/write
      basic_auth:
        username: ${PROMETHEUS_REMOTE_USER}
        password: ${PROMETHEUS_REMOTE_PASSWORD}
EOF
    
    log_success "Regional monitoring deployed for $region"
}

# Deploy network policies for region
deploy_network_policies() {
    local region=$1
    
    log_info "Deploying network policies for $region"
    
    # Create region-specific network policy
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fleet-mind-regional-policy-$region
  namespace: fleet-mind
spec:
  podSelector:
    matchLabels:
      app: fleet-mind-coordinator
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  - from:
    - podSelector:
        matchLabels:
          app: fleet-mind-coordinator
  egress:
  # Allow DNS
  - to: []
    ports:
    - protocol: UDP
      port: 53
  # Allow HTTPS for external APIs
  - to: []
    ports:
    - protocol: TCP
      port: 443
  # Allow inter-region communication (for data replication)
  - to: []
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 8081
EOF
    
    log_success "Network policies deployed for $region"
}

# Deploy to a specific region
deploy_region() {
    local region=$1
    
    log_info "Deploying Fleet-Mind to region: $region"
    
    # Setup kubectl context
    if ! setup_kubectl_context "$region"; then
        log_error "Failed to setup kubectl context for $region"
        return 1
    fi
    
    # Deploy base Fleet-Mind components
    log_info "Deploying base components..."
    "$SCRIPT_DIR/deploy-k8s.sh" prod deploy
    
    # Deploy region-specific configurations
    deploy_compliance "$region"
    deploy_localization "$region"
    deploy_regional_monitoring "$region"
    deploy_network_policies "$region"
    
    # Update deployment with region-specific settings
    kubectl patch deployment fleet-mind-coordinator -n fleet-mind -p "{
        \"spec\": {
            \"template\": {
                \"spec\": {
                    \"containers\": [{
                        \"name\": \"coordinator\",
                        \"env\": [
                            {\"name\": \"DEPLOYMENT_REGION\", \"value\": \"$region\"},
                            {\"name\": \"COMPLIANCE_CONFIG\", \"valueFrom\": {\"configMapKeyRef\": {\"name\": \"compliance-config-$region\", \"key\": \"region\"}}},
                            {\"name\": \"LOCALIZATION_CONFIG\", \"valueFrom\": {\"configMapKeyRef\": {\"name\": \"localization-config-$region\", \"key\": \"supported_languages\"}}}
                        ]
                    }]
                }
            }
        }
    }"
    
    # Wait for rollout
    kubectl rollout status deployment/fleet-mind-coordinator -n fleet-mind --timeout=300s
    
    log_success "Fleet-Mind deployed successfully to $region"
}

# Deploy global load balancer
deploy_global_load_balancer() {
    log_info "Deploying global load balancer configuration"
    
    # This would typically be done through cloud provider CLI or Terraform
    # For demonstration, we'll create a placeholder configuration
    
    cat <<EOF > /tmp/global-lb-config.yaml
# Global Load Balancer Configuration
# This would be deployed using cloud provider-specific tools
global_load_balancer:
  name: fleet-mind-global
  dns_name: fleet-mind.com
  
  backends:
    - name: europe
      region: eu-west-1
      health_check: https://eu.fleet-mind.com/health
      weight: 30
      
    - name: north_america
      region: us-east-1
      health_check: https://na.fleet-mind.com/health
      weight: 50
      
    - name: asia_pacific
      region: ap-southeast-1
      health_check: https://apac.fleet-mind.com/health
      weight: 20
  
  routing_rules:
    - condition: "origin_country in ['DE', 'FR', 'IT', 'ES', 'NL', 'GB']"
      backend: europe
    - condition: "origin_country in ['US', 'CA', 'MX']"
      backend: north_america
    - condition: "origin_country in ['JP', 'KR', 'CN', 'IN', 'SG', 'AU']"
      backend: asia_pacific
    - default: north_america
EOF
    
    log_info "Global load balancer configuration created (manual deployment required)"
    log_warning "Please deploy the global load balancer using your cloud provider's tools"
    
    log_success "Global load balancer configuration prepared"
}

# Setup global monitoring
setup_global_monitoring() {
    log_info "Setting up global monitoring dashboard"
    
    # Create global monitoring namespace in a central cluster
    kubectl create namespace fleet-mind-global --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy global Grafana dashboard
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: global-dashboard-config
  namespace: fleet-mind-global
data:
  global-overview.json: |
    {
      "dashboard": {
        "title": "Fleet-Mind Global Overview",
        "panels": [
          {
            "title": "Global Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "sum by (region) (rate(fleet_mind_requests_total[5m]))",
                "legendFormat": "{{region}}"
              }
            ]
          },
          {
            "title": "Regional Health Status",
            "type": "stat",
            "targets": [
              {
                "expr": "up{job=\"fleet-mind-coordinator\"}",
                "legendFormat": "{{region}}"
              }
            ]
          },
          {
            "title": "Compliance Metrics",
            "type": "table",
            "targets": [
              {
                "expr": "fleet_mind_gdpr_requests_total",
                "legendFormat": "GDPR Requests"
              },
              {
                "expr": "fleet_mind_data_retention_cleaned_total",
                "legendFormat": "Data Cleaned"
              }
            ]
          }
        ]
      }
    }
EOF
    
    log_success "Global monitoring dashboard configured"
}

# Main deployment function
deploy_all_regions() {
    log_info "Starting global Fleet-Mind deployment..."
    
    check_global_prerequisites
    load_global_config
    
    # Deploy to each region
    for region in "${!REGIONS[@]}"; do
        log_info "Deploying to region: $region"
        if deploy_region "$region"; then
            log_success "Successfully deployed to $region"
        else
            log_error "Failed to deploy to $region"
            exit 1
        fi
    done
    
    # Setup global components
    deploy_global_load_balancer
    setup_global_monitoring
    
    log_success "Global deployment completed successfully!"
    
    # Show global status
    show_global_status
}

# Show global deployment status
show_global_status() {
    log_info "Global Fleet-Mind Deployment Status:"
    echo
    
    for region in "${!REGIONS[@]}"; do
        echo "Region: $region"
        
        if setup_kubectl_context "$region" &>/dev/null; then
            echo "  Cluster: Connected"
            echo "  Pods:"
            kubectl get pods -n fleet-mind -o wide 2>/dev/null | grep -E "(NAME|fleet-mind)" | sed 's/^/    /'
            echo "  Services:"
            kubectl get services -n fleet-mind 2>/dev/null | grep -E "(NAME|fleet-mind)" | sed 's/^/    /'
            echo
        else
            echo "  Cluster: Not accessible"
            echo
        fi
    done
    
    echo "Global Endpoints:"
    echo "  Main: https://fleet-mind.com"
    echo "  Europe: https://eu.fleet-mind.com"
    echo "  North America: https://na.fleet-mind.com"
    echo "  Asia Pacific: https://apac.fleet-mind.com"
    echo "  Global Monitoring: https://monitoring.fleet-mind.com"
}

# Cleanup global deployment
cleanup_global() {
    log_warning "This will delete the entire global Fleet-Mind deployment. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "Cleaning up global deployment..."
        
        for region in "${!REGIONS[@]}"; do
            log_info "Cleaning up region: $region"
            if setup_kubectl_context "$region" &>/dev/null; then
                "$SCRIPT_DIR/deploy-k8s.sh" prod delete
            fi
        done
        
        log_success "Global deployment cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Main script logic
case $ACTION in
    deploy)
        if [ "$REGION" = "all" ]; then
            deploy_all_regions
        elif [[ " ${!REGIONS[@]} " =~ " $REGION " ]]; then
            check_global_prerequisites
            load_global_config
            deploy_region "$REGION"
        else
            log_error "Invalid region: $REGION"
            echo "Available regions: ${!REGIONS[@]} all"
            exit 1
        fi
        ;;
    status)
        show_global_status
        ;;
    cleanup)
        cleanup_global
        ;;
    *)
        echo "Usage: $0 [region] [action]"
        echo "Regions: ${!REGIONS[@]} all"
        echo "Actions:"
        echo "  deploy  - Deploy Fleet-Mind globally or to specific region"
        echo "  status  - Show global deployment status"
        echo "  cleanup - Clean up global deployment"
        exit 1
        ;;
esac