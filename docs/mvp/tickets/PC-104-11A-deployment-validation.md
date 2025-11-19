# PC-104-11A: Infrastructure Deployment and Validation

**Ticket ID:** PC-104-11A  
**Epic:** [Epic-11A: Documentation & Infrastructure](../epics/Epic-11A.md)  
**User Story:** US-11A.3 - Infrastructure as Code  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Deploy and validate the complete infrastructure in AWS, including smoke tests, integration tests, health checks, and operational runbooks to ensure the system is production-ready.

**As a** platform engineer  
**I want** validated and tested infrastructure deployment  
**So that** I can confidently run PrevCarga in production

---

## ✅ Acceptance Criteria

- [ ] Infrastructure deployed successfully in dev environment
- [ ] All AWS resources created and configured
- [ ] Network connectivity validated
- [ ] S3 bucket access verified
- [ ] IAM permissions tested
- [ ] Container deployment successful
- [ ] Health checks passing
- [ ] End-to-end integration test passing
- [ ] Monitoring and logging verified
- [ ] Runbooks created for common operations
- [ ] Rollback procedure documented
- [ ] Production deployment checklist created

---

## 🔧 Implementation Tasks

### 1. Deploy Infrastructure
- [ ] Deploy network stack
- [ ] Deploy security stack
- [ ] Deploy storage stack
- [ ] Deploy compute stack (if applicable)
- [ ] Deploy monitoring stack
- [ ] Verify all stacks created successfully

### 2. Smoke Tests
- [ ] VPC and subnets accessible
- [ ] Security groups configured correctly
- [ ] S3 buckets created and accessible
- [ ] IAM roles and policies attached
- [ ] Secrets accessible from application
- [ ] CloudWatch logs functioning

### 3. Integration Tests
- [ ] Deploy test container
- [ ] Test S3 data access
- [ ] Test model loading/saving
- [ ] Test prediction workflow
- [ ] Test training workflow
- [ ] Test monitoring integration

### 4. Performance Validation
- [ ] Network latency tests
- [ ] S3 throughput tests
- [ ] Container startup time
- [ ] Resource utilization check
- [ ] Cost validation

### 5. Create Operational Documentation
- [ ] Deployment runbook
- [ ] Rollback procedures
- [ ] Troubleshooting guide
- [ ] Monitoring guide
- [ ] Scaling procedures

### 6. Production Readiness
- [ ] Security review checklist
- [ ] Compliance validation
- [ ] Disaster recovery plan
- [ ] Backup and restore procedures
- [ ] Production deployment checklist

---

## 📂 Files to Create

```
docs/
├── operations/
│   ├── deployment-runbook.md
│   ├── rollback-procedures.md
│   ├── troubleshooting.md
│   ├── monitoring-guide.md
│   └── disaster-recovery.md
└── tests/
    └── integration/
        ├── test_infrastructure.py
        ├── test_deployment.py
        └── test_smoke.py
```

---

## 🔧 Technical Implementation

### Integration Test Script (`tests/integration/test_infrastructure.py`)

```python
"""Infrastructure integration tests."""
import boto3
import pytest
from typing import Dict

class TestInfrastructure:
    """Test AWS infrastructure setup."""
    
    @pytest.fixture(scope="session")
    def aws_clients(self) -> Dict:
        """Create AWS clients."""
        return {
            's3': boto3.client('s3'),
            'iam': boto3.client('iam'),
            'ec2': boto3.client('ec2'),
            'ecs': boto3.client('ecs'),
            'secretsmanager': boto3.client('secretsmanager'),
            'logs': boto3.client('logs')
        }
    
    def test_vpc_exists(self, aws_clients):
        """Test VPC is created."""
        ec2 = aws_clients['ec2']
        vpcs = ec2.describe_vpcs(
            Filters=[{'Name': 'tag:Project', 'Values': ['PrevCarga']}]
        )
        assert len(vpcs['Vpcs']) > 0, "VPC not found"
    
    def test_s3_buckets_exist(self, aws_clients):
        """Test S3 buckets are created."""
        s3 = aws_clients['s3']
        
        # Check data bucket
        try:
            s3.head_bucket(Bucket='prevcarga-data-production')
        except Exception as e:
            pytest.fail(f"Data bucket not accessible: {e}")
        
        # Check models bucket
        try:
            s3.head_bucket(Bucket='prevcarga-models-production')
        except Exception as e:
            pytest.fail(f"Models bucket not accessible: {e}")
    
    def test_s3_bucket_encryption(self, aws_clients):
        """Test S3 buckets have encryption enabled."""
        s3 = aws_clients['s3']
        
        encryption = s3.get_bucket_encryption(
            Bucket='prevcarga-data-production'
        )
        assert 'Rules' in encryption['ServerSideEncryptionConfiguration']
    
    def test_iam_roles_exist(self, aws_clients):
        """Test IAM roles are created."""
        iam = aws_clients['iam']
        
        try:
            iam.get_role(RoleName='PrevCargaTaskRole')
            iam.get_role(RoleName='PrevCargaExecutionRole')
        except iam.exceptions.NoSuchEntityException:
            pytest.fail("Required IAM roles not found")
    
    def test_secrets_accessible(self, aws_clients):
        """Test Secrets Manager secrets are accessible."""
        sm = aws_clients['secretsmanager']
        
        try:
            sm.describe_secret(SecretId='prevcarga/production/config')
        except sm.exceptions.ResourceNotFoundException:
            pytest.fail("Secrets not found")
    
    def test_cloudwatch_logs(self, aws_clients):
        """Test CloudWatch log groups exist."""
        logs = aws_clients['logs']
        
        log_groups = logs.describe_log_groups(
            logGroupNamePrefix='/prevcarga/'
        )
        assert len(log_groups['logGroups']) > 0

class TestDeployment:
    """Test container deployment."""
    
    def test_container_can_access_s3(self):
        """Test container can access S3."""
        s3 = boto3.client('s3')
        
        # Test read access
        try:
            s3.list_objects_v2(
                Bucket='prevcarga-data-production',
                MaxKeys=1
            )
        except Exception as e:
            pytest.fail(f"Cannot access S3: {e}")
    
    def test_container_health(self):
        """Test container health check."""
        # This would test actual container health endpoint
        import requests
        
        response = requests.get('http://localhost:8000/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
```

### Deployment Runbook (`docs/operations/deployment-runbook.md`)

```markdown
# Deployment Runbook

## Pre-Deployment Checklist

- [ ] AWS credentials configured
- [ ] IaC code reviewed and approved
- [ ] Cost estimation reviewed
- [ ] Security review completed
- [ ] Backup of current state taken
- [ ] Rollback plan documented
- [ ] Stakeholders notified

## Deployment Steps

### 1. Validate Infrastructure Code

```bash
# AWS CDK
cd infrastructure/cdk
cdk synth
cdk diff --context environment=production

# Terraform
cd infrastructure/terraform
terraform init
terraform plan -var-file=environments/production/terraform.tfvars
```

### 2. Deploy Network Stack

```bash
# CDK
cdk deploy PrevCargaNetwork --context environment=production

# Terraform
terraform apply -target=module.vpc -var-file=environments/production/terraform.tfvars
```

### 3. Deploy Security Stack

```bash
# CDK
cdk deploy PrevCargaSecurity --context environment=production

# Terraform
terraform apply -target=module.security -var-file=environments/production/terraform.tfvars
```

### 4. Deploy Storage Stack

```bash
# CDK
cdk deploy PrevCargaStorage --context environment=production

# Terraform
terraform apply -target=module.storage -var-file=environments/production/terraform.tfvars
```

### 5. Deploy Monitoring Stack

```bash
# CDK
cdk deploy PrevCargaMonitoring --context environment=production

# Terraform
terraform apply -target=module.monitoring -var-file=environments/production/terraform.tfvars
```

### 6. Verify Deployment

```bash
# Run integration tests
pytest tests/integration/

# Check AWS console
# - VPC created
# - S3 buckets accessible
# - IAM roles configured
# - Secrets created
```

### 7. Deploy Application Container

```bash
# Push Docker image
docker tag prevcarga:latest $ECR_REGISTRY/prevcarga:production
docker push $ECR_REGISTRY/prevcarga:production

# Deploy to ECS/Fargate (if using)
aws ecs update-service \
    --cluster prevcarga-production \
    --service prevcarga \
    --force-new-deployment
```

### 8. Smoke Tests

```bash
# Test prediction
docker run --rm \
    -e AWS_PROFILE=production \
    $ECR_REGISTRY/prevcarga:production \
    prevcarga predict --model lgbm --area area001 --date 2024-01-01 --dry-run

# Check logs
aws logs tail /prevcarga/production --follow
```

## Post-Deployment Verification

- [ ] All resources created successfully
- [ ] Health checks passing
- [ ] Smoke tests passed
- [ ] Monitoring dashboards showing data
- [ ] Alerts configured and working
- [ ] Documentation updated
- [ ] Team notified of completion

## Rollback Procedure

If deployment fails:

```bash
# CDK
cdk destroy --all --context environment=production

# Terraform
terraform destroy -var-file=environments/production/terraform.tfvars

# Restore from backup if needed
```
```

### Monitoring Guide (`docs/operations/monitoring-guide.md`)

```markdown
# Monitoring Guide

## CloudWatch Dashboards

Access dashboards:
- PrevCarga Overview: https://console.aws.amazon.com/cloudwatch/dashboards/PrevCarga
- ECS Metrics: https://console.aws.amazon.com/ecs/

## Key Metrics

### Infrastructure Metrics
- VPC Flow Logs
- NAT Gateway traffic
- S3 request metrics
- S3 error rates

### Application Metrics
- Container CPU utilization
- Container memory utilization
- Request count
- Error rate
- Prediction latency
- Training duration

### Alarms

Configured alarms:
- High CPU utilization (>80%)
- High memory utilization (>85%)
- Error rate threshold (>5%)
- S3 access failures

## Log Analysis

```bash
# View recent logs
aws logs tail /prevcarga/production --follow

# Search for errors
aws logs filter-pattern --log-group-name /prevcarga/production \
    --filter-pattern "ERROR"

# View specific time range
aws logs tail /prevcarga/production \
    --since 1h \
    --format short
```

## Troubleshooting

### High CPU/Memory
1. Check running tasks
2. Review application logs
3. Scale up resources if needed

### S3 Access Issues
1. Verify IAM permissions
2. Check bucket policies
3. Review VPC endpoints

### Networking Issues
1. Check security group rules
2. Verify route tables
3. Test connectivity from container
```

---

## 🧪 Testing & Validation

### Validation Commands

```bash
# Run integration tests
pytest tests/integration/ -v

# Validate infrastructure
# CDK
cdk synth && cdk diff

# Terraform
terraform validate && terraform plan

# Test S3 access
aws s3 ls s3://prevcarga-data-production/

# Test container deployment
docker run --rm prevcarga:latest prevcarga --version

# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /prevcarga/
```

---

## 📝 Technical Notes

- Deploy to dev environment first
- Validate each stack independently
- Test rollback procedure before production
- Monitor costs during and after deployment
- Keep runbooks updated with actual procedures
- Document any issues encountered

---

## 🔗 Dependencies

**Depends On:**
- PC-100-11A: Production Dockerfile
- PC-102-11A: Container Security and Optimization
- PC-103-11A: AWS Infrastructure as Code

**Blocks:**
- Epic-11B: Production Deployment and Operations

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Infrastructure deployed to dev successfully
- [ ] Integration tests passing
- [ ] Smoke tests passing
- [ ] Runbooks created and validated
- [ ] Rollback tested successfully
- [ ] Monitoring verified
- [ ] Documentation complete
- [ ] Production readiness review approved
- [ ] Committed to repository

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-19 | 1.0.0 | Initial ticket created from Epic-11A | System |

---

**Epic Complete**: All tickets for Epic-11A created
