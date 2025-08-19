# Fleet-Mind Generation 7 Production Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "fleet-mind-production"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.28"
}

variable "db_password" {
  description = "Password for the RDS database"
  type        = string
  sensitive   = true
}

variable "redis_auth_token" {
  description = "Auth token for Redis cluster"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key for LLM integration"
  type        = string
  sensitive   = true
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "fleet-mind"
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "terragon-labs"
}

# Scaling configuration
variable "max_drones" {
  description = "Maximum number of drones to support"
  type        = number
  default     = 10000000  # 10 million drones
}

variable "coordination_nodes_min" {
  description = "Minimum number of coordination nodes"
  type        = number
  default     = 100
}

variable "coordination_nodes_max" {
  description = "Maximum number of coordination nodes"
  type        = number
  default     = 1000
}

variable "processing_nodes_min" {
  description = "Minimum number of processing nodes"
  type        = number
  default     = 50
}

variable "processing_nodes_max" {
  description = "Maximum number of processing nodes"
  type        = number
  default     = 500
}

variable "storage_nodes_min" {
  description = "Minimum number of storage nodes"
  type        = number
  default     = 20
}

variable "storage_nodes_max" {
  description = "Maximum number of storage nodes"
  type        = number
  default     = 200
}

variable "edge_nodes_min" {
  description = "Minimum number of edge nodes"
  type        = number
  default     = 200
}

variable "edge_nodes_max" {
  description = "Maximum number of edge nodes"
  type        = number
  default     = 2000
}

# Database configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.24xlarge"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS in GB"
  type        = number
  default     = 10000  # 10TB
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS in GB"
  type        = number
  default     = 50000  # 50TB
}

variable "db_backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 30
}

# Redis configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r7g.24xlarge"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters in the Redis replication group"
  type        = number
  default     = 6
}

variable "redis_snapshot_retention_limit" {
  description = "Number of days for which ElastiCache retains automatic snapshots"
  type        = number
  default     = 7
}

# Network configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/8"  # Large CIDR for massive scale
}

variable "availability_zones_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3
}

# Security configuration
variable "enable_cluster_encryption" {
  description = "Enable EKS cluster encryption"
  type        = bool
  default     = true
}

variable "enable_rds_encryption" {
  description = "Enable RDS encryption at rest"
  type        = bool
  default     = true
}

variable "enable_redis_encryption" {
  description = "Enable Redis encryption at rest and in transit"
  type        = bool
  default     = true
}

# Monitoring configuration
variable "enable_performance_insights" {
  description = "Enable RDS Performance Insights"
  type        = bool
  default     = true
}

variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

# Cost optimization
variable "use_spot_instances" {
  description = "Use spot instances for edge nodes"
  type        = bool
  default     = true
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

# Generation 7 specific configuration
variable "quantum_coherence_target" {
  description = "Target quantum coherence level"
  type        = number
  default     = 0.95
}

variable "consciousness_threshold" {
  description = "Consciousness threshold for convergence"
  type        = number
  default     = 0.9
}

variable "transcendence_goal" {
  description = "Transcendence goal for ultimate convergence"
  type        = number
  default     = 0.95
}

variable "byzantine_fault_tolerance" {
  description = "Byzantine fault tolerance threshold"
  type        = number
  default     = 0.33
}

variable "reality_coherence_minimum" {
  description = "Minimum reality coherence level"
  type        = number
  default     = 0.9
}

# Performance targets
variable "target_latency_ms" {
  description = "Target latency in milliseconds"
  type        = number
  default     = 100
}

variable "target_throughput_ops_sec" {
  description = "Target throughput in operations per second"
  type        = number
  default     = 10000
}

variable "max_error_rate" {
  description = "Maximum acceptable error rate"
  type        = number
  default     = 0.01  # 1%
}

# Resource limits
variable "max_cpu_per_node" {
  description = "Maximum CPU cores per node"
  type        = number
  default     = 128
}

variable "max_memory_per_node_gb" {
  description = "Maximum memory per node in GB"
  type        = number
  default     = 2048  # 2TB
}

variable "max_gpu_per_node" {
  description = "Maximum GPUs per node"
  type        = number
  default     = 8
}

# Backup and disaster recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = true
}

variable "backup_region" {
  description = "Region for cross-region backups"
  type        = string
  default     = "us-east-1"
}

variable "disaster_recovery_rto_minutes" {
  description = "Recovery Time Objective in minutes"
  type        = number
  default     = 60
}

variable "disaster_recovery_rpo_minutes" {
  description = "Recovery Point Objective in minutes"
  type        = number
  default     = 15
}

# Compliance and governance
variable "enable_compliance_monitoring" {
  description = "Enable compliance monitoring"
  type        = bool
  default     = true
}

variable "enable_audit_logging" {
  description = "Enable comprehensive audit logging"
  type        = bool
  default     = true
}

variable "data_residency_requirements" {
  description = "Data residency requirements (e.g., 'US', 'EU', 'GLOBAL')"
  type        = string
  default     = "GLOBAL"
}

# Operational configuration
variable "maintenance_window" {
  description = "Maintenance window for RDS and ElastiCache"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "backup_window" {
  description = "Backup window for RDS"
  type        = string
  default     = "03:00-04:00"
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = true
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default = {
    Generation = "7"
    System     = "fleet-mind"
    Terraform  = "true"
  }
}