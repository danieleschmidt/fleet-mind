# Fleet-Mind Generation 7 Production Infrastructure
# Terraform configuration for massive-scale deployment

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket = "fleet-mind-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-west-2"
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "Fleet-Mind"
      Environment = "production"
      Generation  = "7"
      Owner       = "terragon-labs"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  cluster_name = "fleet-mind-production"
  
  # Multi-AZ deployment for high availability
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
  
  # Massive scale node groups
  node_groups = {
    coordination_nodes = {
      instance_types = ["c6i.32xlarge", "c7i.48xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size      = 100
      max_size      = 1000
      desired_size  = 200
      
      k8s_labels = {
        role = "coordination"
        generation = "7"
      }
      
      taints = [
        {
          key    = "fleet-mind.com/coordination"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    processing_nodes = {
      instance_types = ["p4d.24xlarge", "p5.48xlarge"]  # GPU instances
      capacity_type  = "ON_DEMAND"
      min_size      = 50
      max_size      = 500
      desired_size  = 100
      
      k8s_labels = {
        role = "processing"
        generation = "7"
        gpu = "true"
      }
      
      taints = [
        {
          key    = "fleet-mind.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    storage_nodes = {
      instance_types = ["r6i.32xlarge", "x2idn.32xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size      = 20
      max_size      = 200
      desired_size  = 50
      
      k8s_labels = {
        role = "storage"
        generation = "7"
      }
    }
    
    edge_nodes = {
      instance_types = ["c6i.16xlarge", "c7i.24xlarge"]
      capacity_type  = "SPOT"
      min_size      = 200
      max_size      = 2000
      desired_size  = 500
      
      k8s_labels = {
        role = "edge"
        generation = "7"
      }
    }
  }
  
  # Massive scale configuration
  scaling_config = {
    max_drones = 10000000  # 10 million drones
    coordination_nodes = 1000
    processing_nodes = 500
    storage_nodes = 200
    edge_nodes = 2000
  }
}

# VPC for massive scale
resource "aws_vpc" "fleet_mind_vpc" {
  cidr_block           = "10.0.0.0/8"  # Large CIDR for massive scale
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${local.cluster_name}-vpc"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "fleet_mind_igw" {
  vpc_id = aws_vpc.fleet_mind_vpc.id
  
  tags = {
    Name = "${local.cluster_name}-igw"
  }
}

# Public subnets for load balancers
resource "aws_subnet" "public" {
  count = length(local.azs)
  
  vpc_id                  = aws_vpc.fleet_mind_vpc.id
  cidr_block              = "10.${count.index}.0.0/16"
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "${local.cluster_name}-public-${local.azs[count.index]}"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb" = "1"
  }
}

# Private subnets for worker nodes
resource "aws_subnet" "private" {
  count = length(local.azs) * 4  # 4 subnets per AZ for different node types
  
  vpc_id            = aws_vpc.fleet_mind_vpc.id
  cidr_block        = "10.${10 + count.index}.0.0/16"
  availability_zone = local.azs[count.index % length(local.azs)]
  
  tags = {
    Name = "${local.cluster_name}-private-${count.index}"
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# NAT Gateways for outbound internet access
resource "aws_eip" "nat" {
  count = length(local.azs)
  
  domain = "vpc"
  
  tags = {
    Name = "${local.cluster_name}-nat-${local.azs[count.index]}"
  }
}

resource "aws_nat_gateway" "fleet_mind_nat" {
  count = length(local.azs)
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = {
    Name = "${local.cluster_name}-nat-${local.azs[count.index]}"
  }
  
  depends_on = [aws_internet_gateway.fleet_mind_igw]
}

# Route tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.fleet_mind_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.fleet_mind_igw.id
  }
  
  tags = {
    Name = "${local.cluster_name}-public"
  }
}

resource "aws_route_table" "private" {
  count = length(local.azs)
  
  vpc_id = aws_vpc.fleet_mind_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.fleet_mind_nat[count.index].id
  }
  
  tags = {
    Name = "${local.cluster_name}-private-${local.azs[count.index]}"
  }
}

# Route table associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index % length(local.azs)].id
}

# Security groups
resource "aws_security_group" "cluster_security_group" {
  name_prefix = "${local.cluster_name}-cluster-"
  vpc_id      = aws_vpc.fleet_mind_vpc.id
  
  # Allow all internal communication
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }
  
  # HTTPS access for API server
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # WebRTC communication
  ingress {
    from_port   = 3478
    to_port     = 3478
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 10000
    to_port     = 20000
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${local.cluster_name}-cluster-sg"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "fleet_mind_cluster" {
  name     = local.cluster_name
  role_arn = aws_iam_role.cluster_role.arn
  version  = "1.28"
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
    security_group_ids      = [aws_security_group.cluster_security_group.id]
  }
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.cluster_encryption.arn
    }
    resources = ["secrets"]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
    aws_iam_role_policy_attachment.vpc_resource_controller,
  ]
  
  tags = {
    Name = local.cluster_name
  }
}

# KMS key for cluster encryption
resource "aws_kms_key" "cluster_encryption" {
  description             = "EKS Cluster ${local.cluster_name} Encryption Key"
  deletion_window_in_days = 7
  
  tags = {
    Name = "${local.cluster_name}-encryption-key"
  }
}

resource "aws_kms_alias" "cluster_encryption" {
  name          = "alias/${local.cluster_name}-encryption-key"
  target_key_id = aws_kms_key.cluster_encryption.key_id
}

# IAM roles and policies
resource "aws_iam_role" "cluster_role" {
  name = "${local.cluster_name}-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster_role.name
}

resource "aws_iam_role_policy_attachment" "vpc_resource_controller" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.cluster_role.name
}

# Node group IAM role
resource "aws_iam_role" "node_group_role" {
  name = "${local.cluster_name}-node-group-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node_group_role.name
}

resource "aws_iam_role_policy_attachment" "cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node_group_role.name
}

resource "aws_iam_role_policy_attachment" "registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node_group_role.name
}

# Additional policy for auto-scaling
resource "aws_iam_role_policy_attachment" "autoscaling_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AutoScalingFullAccess"
  role       = aws_iam_role.node_group_role.name
}

# EKS Node Groups
resource "aws_eks_node_group" "node_groups" {
  for_each = local.node_groups
  
  cluster_name    = aws_eks_cluster.fleet_mind_cluster.name
  node_group_name = each.key
  node_role_arn   = aws_iam_role.node_group_role.arn
  subnet_ids      = aws_subnet.private[*].id
  
  capacity_type  = each.value.capacity_type
  instance_types = each.value.instance_types
  
  scaling_config {
    desired_size = each.value.desired_size
    max_size     = each.value.max_size
    min_size     = each.value.min_size
  }
  
  update_config {
    max_unavailable_percentage = 25
  }
  
  # Launch template for advanced configuration
  launch_template {
    name    = aws_launch_template.node_template[each.key].name
    version = aws_launch_template.node_template[each.key].latest_version
  }
  
  labels = each.value.k8s_labels
  
  dynamic "taint" {
    for_each = lookup(each.value, "taints", [])
    content {
      key    = taint.value.key
      value  = taint.value.value
      effect = taint.value.effect
    }
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.worker_node_policy,
    aws_iam_role_policy_attachment.cni_policy,
    aws_iam_role_policy_attachment.registry_policy,
  ]
  
  tags = {
    Name = "${local.cluster_name}-${each.key}"
  }
}

# Launch templates for node groups
resource "aws_launch_template" "node_template" {
  for_each = local.node_groups
  
  name_prefix = "${local.cluster_name}-${each.key}-"
  
  vpc_security_group_ids = [aws_security_group.cluster_security_group.id]
  
  user_data = base64encode(templatefile("${path.module}/user-data.sh", {
    cluster_name = local.cluster_name
    node_group   = each.key
  }))
  
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 500  # Large volumes for data processing
      volume_type = "gp3"
      iops        = 10000
      throughput  = 1000
      encrypted   = true
    }
  }
  
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
    http_put_response_hop_limit = 2
  }
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${local.cluster_name}-${each.key}"
    }
  }
}

# RDS for metadata storage
resource "aws_db_subnet_group" "fleet_mind_db_subnet_group" {
  name       = "${local.cluster_name}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  
  tags = {
    Name = "${local.cluster_name}-db-subnet-group"
  }
}

resource "aws_security_group" "rds_security_group" {
  name_prefix = "${local.cluster_name}-rds-"
  vpc_id      = aws_vpc.fleet_mind_vpc.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.cluster_security_group.id]
  }
  
  tags = {
    Name = "${local.cluster_name}-rds-sg"
  }
}

resource "aws_db_instance" "fleet_mind_db" {
  identifier = "${local.cluster_name}-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.24xlarge"  # Large instance for metadata
  
  allocated_storage     = 10000  # 10TB
  max_allocated_storage = 50000  # 50TB max
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "fleet_mind"
  username = "fleet_mind_admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds_security_group.id]
  db_subnet_group_name   = aws_db_subnet_group.fleet_mind_db_subnet_group.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  multi_az               = true
  publicly_accessible    = false
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring_role.arn
  
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${local.cluster_name}-db-final-snapshot"
  
  tags = {
    Name = "${local.cluster_name}-db"
  }
}

# RDS monitoring role
resource "aws_iam_role" "rds_monitoring_role" {
  name = "${local.cluster_name}-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring_policy" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
  role       = aws_iam_role.rds_monitoring_role.name
}

# ElastiCache Redis cluster for caching
resource "aws_elasticache_subnet_group" "fleet_mind_cache_subnet_group" {
  name       = "${local.cluster_name}-cache-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "redis_security_group" {
  name_prefix = "${local.cluster_name}-redis-"
  vpc_id      = aws_vpc.fleet_mind_vpc.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.cluster_security_group.id]
  }
  
  tags = {
    Name = "${local.cluster_name}-redis-sg"
  }
}

resource "aws_elasticache_replication_group" "fleet_mind_redis" {
  replication_group_id       = "${local.cluster_name}-redis"
  description                = "Redis cluster for Fleet-Mind caching"
  
  node_type               = "cache.r7g.24xlarge"  # Large instances for massive caching
  num_cache_clusters      = 6
  port                    = 6379
  parameter_group_name    = "default.redis7"
  
  subnet_group_name       = aws_elasticache_subnet_group.fleet_mind_cache_subnet_group.name
  security_group_ids      = [aws_security_group.redis_security_group.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  tags = {
    Name = "${local.cluster_name}-redis"
  }
}

# Application Load Balancer
resource "aws_lb" "fleet_mind_alb" {
  name               = "${local.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_security_group.id]
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = true
  
  tags = {
    Name = "${local.cluster_name}-alb"
  }
}

resource "aws_security_group" "alb_security_group" {
  name_prefix = "${local.cluster_name}-alb-"
  vpc_id      = aws_vpc.fleet_mind_vpc.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${local.cluster_name}-alb-sg"
  }
}

# S3 bucket for artifacts and backups
resource "aws_s3_bucket" "fleet_mind_artifacts" {
  bucket = "${local.cluster_name}-artifacts-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name = "${local.cluster_name}-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "fleet_mind_artifacts_versioning" {
  bucket = aws_s3_bucket.fleet_mind_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "fleet_mind_artifacts_encryption" {
  bucket = aws_s3_bucket.fleet_mind_artifacts.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# CloudWatch Log Group for cluster logs
resource "aws_cloudwatch_log_group" "cluster_logs" {
  name              = "/aws/eks/${local.cluster_name}/cluster"
  retention_in_days = 30
  
  tags = {
    Name = "${local.cluster_name}-cluster-logs"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.fleet_mind_cluster.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.fleet_mind_cluster.vpc_config[0].cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_eks_cluster.fleet_mind_cluster.role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.fleet_mind_cluster.certificate_authority[0].data
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = aws_eks_cluster.fleet_mind_cluster.name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.fleet_mind_db.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.fleet_mind_redis.primary_endpoint_address
  sensitive   = true
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.fleet_mind_alb.dns_name
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for artifacts"
  value       = aws_s3_bucket.fleet_mind_artifacts.bucket
}