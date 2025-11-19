# PC-105-11B: AWS Fargate Deployment Setup

**Ticket ID:** PC-105-11B  
**Epic:** [Epic-11B: Production Deployment & Operations](../epics/Epic-11B.md)  
**User Story:** US-1  
**Story Points:** 13  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement automated AWS Fargate deployment with ECS cluster, task definitions, Application Load Balancer, auto-scaling policies, blue-green deployment capability, and rollback procedures for the PrevCarga forecasting system.

**As a** platform engineer  
**I want** automated AWS Fargate deployment  
**So that** I can run the system with managed infrastructure and automatic scaling

---

## ✅ Acceptance Criteria

- [ ] ECS Fargate service with auto-scaling configuration deployed
- [ ] Application Load Balancer with health checks configured
- [ ] Service discovery and networking configuration complete
- [ ] Auto-scaling policies based on CPU and memory implemented
- [ ] Blue-green deployment capability tested and verified
- [ ] Rollback procedures tested and documented
- [ ] Container images published to ECR
- [ ] Health checks passing consistently
- [ ] Service running with minimum 2 tasks
- [ ] Auto-scaling triggers tested under load

---

## 🔧 Implementation Tasks

### 1. Set Up AWS Infrastructure as Code
- [ ] Create AWS CDK project structure
- [ ] Install AWS CDK dependencies
- [ ] Configure AWS credentials and region
- [ ] Initialize CDK stack for PrevCarga
- [ ] Set up VPC and subnet configuration
- [ ] Configure security groups

### 2. Create ECS Cluster and Task Definition
- [ ] Create ECS Fargate cluster `prevcarga-cluster`
- [ ] Enable Container Insights on cluster
- [ ] Create IAM execution role for tasks
- [ ] Create IAM task role with S3/Secrets permissions
- [ ] Define Fargate task with 2 vCPU, 4GB memory
- [ ] Configure container image from ECR

### 3. Configure Container Specifications
- [ ] Set up environment variables (PREVCARGA_ENV, AWS_DEFAULT_REGION)
- [ ] Configure secrets from AWS Secrets Manager
- [ ] Add health check command: `prevcarga status`
- [ ] Configure health check intervals (30s interval, 10s timeout)
- [ ] Set up port mapping (8000)
- [ ] Configure CloudWatch Logs with 1-month retention

### 4. Set Up Application Load Balancer
- [ ] Create Application Load Balancer in public subnets
- [ ] Configure security group for ALB
- [ ] Create target group for port 8000
- [ ] Configure health check path `/health`
- [ ] Set health check parameters (30s interval, 2 healthy threshold)
- [ ] Add listener on port 80

### 5. Deploy Fargate Service
- [ ] Create Fargate service with task definition
- [ ] Configure desired count: 2 tasks
- [ ] Set deployment parameters (min 50%, max 200%)
- [ ] Assign private subnets with NAT egress
- [ ] Attach security groups
- [ ] Set health check grace period (5 minutes)
- [ ] Enable circuit breaker with rollback
- [ ] Register service with target group

### 6. Configure Auto-Scaling
- [ ] Create auto-scaling target (min: 2, max: 10)
- [ ] Configure CPU-based scaling policy (70% target)
- [ ] Set CPU scale-in cooldown (5 minutes)
- [ ] Set CPU scale-out cooldown (2 minutes)
- [ ] Configure memory-based scaling policy (80% target)
- [ ] Set memory cooldown periods
- [ ] Test scaling triggers

### 7. Implement Blue-Green Deployment
- [ ] Create deployment pipeline script
- [ ] Implement image build and push to ECR
- [ ] Create task definition update function
- [ ] Implement ECS service deployment
- [ ] Add deployment health verification
- [ ] Configure deployment timeout (30 minutes)
- [ ] Test blue-green deployment flow

### 8. Create Rollback Procedures
- [ ] Implement automated rollback on failure
- [ ] Create manual rollback script
- [ ] Document rollback decision criteria
- [ ] Test rollback with failed deployment
- [ ] Create rollback runbook
- [ ] Verify service stability post-rollback

### 9. Build and Publish Docker Images
- [ ] Create production Dockerfile
- [ ] Optimize image size and layers
- [ ] Create ECR repository
- [ ] Configure image scanning
- [ ] Build and tag initial image
- [ ] Push image to ECR

### 10. Testing and Validation
- [ ] Test service deployment end-to-end
- [ ] Verify health checks pass consistently
- [ ] Test auto-scaling with load simulation
- [ ] Validate scale-out and scale-in behavior
- [ ] Test blue-green deployment
- [ ] Execute rollback test
- [ ] Verify CloudWatch logs streaming
- [ ] Confirm ALB routing to healthy targets

---

## 📁 Files to Create/Modify

```
infrastructure/
├── cdk/
│   ├── app.py
│   ├── stacks/
│   │   ├── __init__.py
│   │   ├── fargate_stack.py
│   │   ├── networking_stack.py
│   │   └── iam_stack.py
│   └── requirements.txt
├── docker/
│   ├── Dockerfile.production
│   └── .dockerignore
└── scripts/
    ├── deploy.py
    ├── rollback.py
    └── build_and_push.sh

docs/operations/
├── deployment-guide.md
├── rollback-procedures.md
└── fargate-architecture.md
```

---

## 🧪 Testing Requirements

### Unit Tests
- [ ] Test IAM role creation and permissions
- [ ] Test task definition generation
- [ ] Test security group configuration
- [ ] Test auto-scaling policy creation

### Integration Tests
- [ ] Test ECS service deployment
- [ ] Test ALB health check integration
- [ ] Test service discovery
- [ ] Test auto-scaling triggers

### End-to-End Tests
- [ ] Deploy full stack from scratch
- [ ] Verify service accessible via ALB
- [ ] Simulate high load and verify scaling
- [ ] Test deployment pipeline
- [ ] Test rollback procedure

---

## 📚 Documentation Requirements

- [ ] Infrastructure architecture diagram
- [ ] Deployment pipeline documentation
- [ ] Auto-scaling configuration guide
- [ ] Rollback procedures runbook
- [ ] Troubleshooting common deployment issues
- [ ] Cost estimation and optimization guide

---

## 🔗 Dependencies

- **Upstream:** Epic-11A (Infrastructure setup, networking, IAM roles)
- **Upstream:** Docker image creation
- **Upstream:** ECR repository setup
- **Tools:** AWS CDK, Docker, AWS CLI
- **Services:** AWS ECS, ECR, ALB, CloudWatch

---

## 📝 Notes

- Use AWS CDK for infrastructure as code (reproducible deployments)
- Follow least-privilege IAM principles
- Enable Container Insights for enhanced monitoring
- Use secrets manager for sensitive configuration
- Implement circuit breaker to prevent bad deployments
- Plan for multi-AZ deployment for high availability
- Document all manual intervention points
- Keep deployment automation idempotent

---

## ✅ Definition of Done

- [ ] Fargate service deployed and running in production
- [ ] Auto-scaling policies tested and functional
- [ ] Load balancer health checks passing consistently
- [ ] Blue-green deployment capability verified
- [ ] Rollback procedures tested successfully
- [ ] All tests passing
- [ ] Documentation complete and reviewed
- [ ] Code reviewed and approved
- [ ] Infrastructure code committed to repository
- [ ] Deployment verified by operations team
