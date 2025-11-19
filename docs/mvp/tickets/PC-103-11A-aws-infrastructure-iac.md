# PC-103-11A: AWS Infrastructure as Code

**Ticket ID:** PC-103-11A  
**Epic:** [Epic-11A: Documentation & Infrastructure](../epics/Epic-11A.md)  
**User Story:** US-11A.3 - Infrastructure as Code  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Create Infrastructure as Code (IaC) using AWS CDK or Terraform to provision and manage all AWS resources for PrevCarga, including VPC, ECS/Fargate, S3 buckets, IAM roles, and Secrets Manager with proper security controls.

**As a** platform engineer  
**I want** declarative infrastructure configuration  
**So that** I can provision and manage AWS resources consistently and reproducibly

---

## ✅ Acceptance Criteria

- [ ] IaC code created (AWS CDK or Terraform)
- [ ] VPC with public and private subnets configured
- [ ] Security groups with least privilege rules
- [ ] IAM roles and policies following least privilege
- [ ] S3 buckets for data and models configured
- [ ] Secrets Manager for credentials
- [ ] ECS cluster and task definitions (if using ECS)
- [ ] Parameter Store for configuration
- [ ] CloudWatch Logs configuration
- [ ] Infrastructure validation and testing
- [ ] Cost estimation documented

---

## 🔧 Implementation Tasks

### 1. Choose IaC Tool and Setup
- [ ] Select AWS CDK or Terraform
- [ ] Initialize IaC project structure
- [ ] Configure state management (S3 backend)
- [ ] Set up development environment

### 2. Network Infrastructure
- [ ] VPC with CIDR planning
- [ ] Public and private subnets across AZs
- [ ] Internet Gateway
- [ ] NAT Gateway
- [ ] Route tables
- [ ] Network ACLs

### 3. Security Configuration
- [ ] Security groups for application
- [ ] Security groups for databases (if needed)
- [ ] IAM roles for ECS tasks
- [ ] IAM roles for EC2 (if used)
- [ ] IAM policies with least privilege
- [ ] KMS keys for encryption

### 4. Storage Configuration
- [ ] S3 bucket for raw data
- [ ] S3 bucket for processed data
- [ ] S3 bucket for models
- [ ] S3 bucket policies
- [ ] Lifecycle policies
- [ ] Versioning and encryption

### 5. Secrets and Configuration
- [ ] Secrets Manager for credentials
- [ ] Parameter Store for configuration
- [ ] Secure parameter encryption

### 6. Compute Infrastructure (Optional)
- [ ] ECS cluster
- [ ] Task definitions
- [ ] Service definitions
- [ ] Load balancer (if needed)
- [ ] Auto-scaling configuration

### 7. Monitoring and Logging
- [ ] CloudWatch Log Groups
- [ ] CloudWatch Alarms
- [ ] SNS topics for alerts
- [ ] CloudWatch Dashboard

---

## 📂 Files to Create

```
infrastructure/
├── cdk/                        # AWS CDK option
│   ├── app.py
│   ├── cdk.json
│   ├── requirements.txt
│   └── stacks/
│       ├── network_stack.py
│       ├── storage_stack.py
│       ├── security_stack.py
│       ├── compute_stack.py
│       └── monitoring_stack.py
└── terraform/                  # Terraform option
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    ├── backend.tf
    ├── modules/
    │   ├── network/
    │   ├── storage/
    │   ├── security/
    │   └── monitoring/
    └── environments/
        ├── dev/
        ├── staging/
        └── production/
```

---

## 🔧 Technical Implementation

### AWS CDK Implementation

#### Main Application (`infrastructure/cdk/app.py`)

```python
#!/usr/bin/env python3
from aws_cdk import App, Environment
from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack
from stacks.security_stack import SecurityStack
from stacks.monitoring_stack import MonitoringStack

app = App()

# Environment configuration
env = Environment(
    account=app.node.try_get_context("account"),
    region=app.node.try_get_context("region") or "us-east-1"
)

# Network stack
network_stack = NetworkStack(
    app, "PrevCargaNetwork",
    env=env,
    description="PrevCarga VPC and networking resources"
)

# Security stack
security_stack = SecurityStack(
    app, "PrevCargaSecurity",
    vpc=network_stack.vpc,
    env=env,
    description="PrevCarga IAM roles, security groups, and secrets"
)

# Storage stack
storage_stack = StorageStack(
    app, "PrevCargaStorage",
    task_role=security_stack.task_role,
    env=env,
    description="PrevCarga S3 buckets and storage resources"
)

# Monitoring stack
monitoring_stack = MonitoringStack(
    app, "PrevCargaMonitoring",
    env=env,
    description="PrevCarga CloudWatch and monitoring resources"
)

app.synth()
```

#### Network Stack (`infrastructure/cdk/stacks/network_stack.py`)

```python
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    Tags
)
from constructs import Construct

class NetworkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        
        # VPC Configuration
        self.vpc = ec2.Vpc(
            self, "PrevCargaVPC",
            vpc_name="prevcarga-vpc",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                )
            ],
            enable_dns_hostnames=True,
            enable_dns_support=True
        )
        
        # VPC Flow Logs
        ec2.FlowLog(
            self, "VPCFlowLog",
            resource_type=ec2.FlowLogResourceType.from_vpc(self.vpc),
            destination=ec2.FlowLogDestination.to_cloud_watch_logs()
        )
        
        # Tags
        Tags.of(self.vpc).add("Project", "PrevCarga")
        Tags.of(self.vpc).add("Environment", self.node.try_get_context("environment") or "dev")
```

#### Storage Stack (`infrastructure/cdk/stacks/storage_stack.py`)

```python
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    RemovalPolicy,
    Duration
)
from constructs import Construct

class StorageStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, task_role: iam.Role, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        
        # Data bucket
        self.data_bucket = s3.Bucket(
            self, "DataBucket",
            bucket_name=f"prevcarga-data-{self.account}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ArchiveOldData",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ],
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Models bucket
        self.model_bucket = s3.Bucket(
            self, "ModelBucket",
            bucket_name=f"prevcarga-models-{self.account}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Grant permissions to task role
        self.data_bucket.grant_read_write(task_role)
        self.model_bucket.grant_read_write(task_role)
```

#### Security Stack (`infrastructure/cdk/stacks/security_stack.py`)

```python
from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_secretsmanager as secretsmanager
)
from constructs import Construct

class SecurityStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, vpc: ec2.Vpc, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        
        # Task execution role
        self.execution_role = iam.Role(
            self, "TaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ]
        )
        
        # Task role
        self.task_role = iam.Role(
            self, "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Role for PrevCarga ECS tasks"
        )
        
        # Security group for application
        self.app_security_group = ec2.SecurityGroup(
            self, "AppSecurityGroup",
            vpc=vpc,
            description="Security group for PrevCarga application",
            allow_all_outbound=True
        )
        
        # Secrets Manager
        self.app_secrets = secretsmanager.Secret(
            self, "AppSecrets",
            secret_name="prevcarga/production/config",
            description="Configuration secrets for PrevCarga"
        )
        
        # Grant read access to secrets
        self.app_secrets.grant_read(self.task_role)
```

### Terraform Implementation

#### Main Configuration (`infrastructure/terraform/main.tf`)

```hcl
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "prevcarga-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "prevcarga-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "PrevCarga"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC Module
module "vpc" {
  source = "./modules/network"
  
  vpc_cidr            = var.vpc_cidr
  availability_zones  = var.availability_zones
  environment         = var.environment
}

# Storage Module
module "storage" {
  source = "./modules/storage"
  
  environment = var.environment
  task_role_arn = module.security.task_role_arn
}

# Security Module
module "security" {
  source = "./modules/security"
  
  vpc_id      = module.vpc.vpc_id
  environment = var.environment
}

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"
  
  environment = var.environment
}
```

#### Variables (`infrastructure/terraform/variables.tf`)

```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production"
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}
```

---

## 🧪 Testing & Validation

### AWS CDK

```bash
# Install dependencies
cd infrastructure/cdk
pip install -r requirements.txt

# Synthesize CloudFormation
cdk synth

# Validate stacks
cdk doctor

# Deploy to dev environment
cdk deploy --all --context environment=dev

# Destroy infrastructure
cdk destroy --all
```

### Terraform

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Validate configuration
terraform validate

# Plan changes
terraform plan -var-file=environments/dev/terraform.tfvars

# Apply changes
terraform apply -var-file=environments/dev/terraform.tfvars

# Destroy infrastructure
terraform destroy -var-file=environments/dev/terraform.tfvars
```

---

## 📝 Technical Notes

- Use separate AWS accounts for dev/staging/production
- Store Terraform state in S3 with DynamoDB locking
- Use CDK context or Terraform workspaces for environments
- Tag all resources consistently
- Enable deletion protection on production resources
- Document estimated monthly costs
- Follow AWS Well-Architected Framework

---

## 🔗 Dependencies

**Depends On:**
- AWS account setup and credentials
- Required permissions for resource creation

**Blocks:**
- PC-104-11A: Deployment and Validation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] IaC code created and tested
- [ ] Infrastructure validates successfully
- [ ] Dev environment deployed successfully
- [ ] Security review completed
- [ ] Cost estimation documented
- [ ] Documentation complete
- [ ] Committed to repository

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-19 | 1.0.0 | Initial ticket created from Epic-11A | System |

---

**Next Ticket:** [PC-104-11A: Infrastructure Deployment and Validation](PC-104-11A-deployment-validation.md)
