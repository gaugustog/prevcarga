# PC-106-11B: EventBridge Scheduling and Lambda Orchestration

**Ticket ID:** PC-106-11B  
**Epic:** [Epic-11B: Production Deployment & Operations](../epics/Epic-11B.md)  
**User Story:** US-2  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement automated prediction scheduling using AWS EventBridge with Lambda orchestration to trigger PrevCarga forecasts every 30 minutes, including error handling, retry mechanisms, notification system for failures, and manual trigger capabilities.

**As a** system operator  
**I want** automated prediction scheduling  
**So that** the system generates forecasts every 30 minutes without manual intervention

---

## ✅ Acceptance Criteria

- [ ] EventBridge rule configured for 30-minute intervals
- [ ] Lambda function for prediction orchestration deployed
- [ ] Error handling and retry mechanisms implemented
- [ ] SNS notification system for scheduling failures configured
- [ ] Manual trigger capabilities for ad-hoc predictions working
- [ ] Scheduling monitoring and alerting active
- [ ] Task execution monitoring and logging functional
- [ ] Timeout handling for long-running predictions
- [ ] Graceful degradation on ECS capacity issues
- [ ] All tests passing with mock events

---

## 🔧 Implementation Tasks

### 1. Create Lambda Orchestrator Function
- [ ] Create Lambda function project structure
- [ ] Set up Python 3.11+ runtime
- [ ] Add boto3 for AWS service interaction
- [ ] Implement `PredictionOrchestrator` class
- [ ] Create `lambda_handler` entry point
- [ ] Configure function timeout (15 minutes)

### 2. Implement ECS Task Triggering
- [ ] Create `_trigger_prediction_task()` method
- [ ] Configure ECS run_task with Fargate launch type
- [ ] Set up network configuration (VPC, subnets, security groups)
- [ ] Define container overrides for prediction command
- [ ] Pass prediction parameters (horizons, combination, reconciliation)
- [ ] Handle task launch failures
- [ ] Return task ARN for tracking

### 3. Implement Task Monitoring
- [ ] Create `_monitor_task_execution()` method
- [ ] Use ECS waiter for task completion
- [ ] Configure waiter timeout (30 minutes)
- [ ] Poll task status periodically (30s intervals)
- [ ] Capture exit code and execution status
- [ ] Calculate execution duration
- [ ] Handle task timeout scenarios

### 4. Implement Error Handling
- [ ] Create `_handle_error()` method
- [ ] Categorize error types (network, timeout, task failure)
- [ ] Implement retry logic for transient failures
- [ ] Log detailed error information
- [ ] Publish error metrics to CloudWatch
- [ ] Trigger SNS notifications for critical errors

### 5. Configure EventBridge Scheduling
- [ ] Create EventBridge rule for 30-minute intervals
- [ ] Use cron expression: `rate(30 minutes)`
- [ ] Configure rule target: Lambda function
- [ ] Set up IAM permissions for EventBridge to invoke Lambda
- [ ] Add input transformation for event context
- [ ] Enable rule and verify triggering

### 6. Implement SNS Notification System
- [ ] Create SNS topic for scheduling alerts
- [ ] Configure email subscriptions for operations team
- [ ] Add SMS subscriptions for critical alerts (optional)
- [ ] Create notification templates
- [ ] Implement notification priority levels
- [ ] Test notification delivery

### 7. Add Manual Trigger Capability
- [ ] Create API Gateway endpoint for manual triggers
- [ ] Implement authentication and authorization
- [ ] Accept prediction parameters as input
- [ ] Validate input parameters
- [ ] Trigger Lambda function asynchronously
- [ ] Return task tracking information

### 8. Implement Metrics Publishing
- [ ] Create `_publish_metrics()` method
- [ ] Publish prediction completion metrics
- [ ] Track execution duration
- [ ] Record success/failure rates
- [ ] Count retry attempts
- [ ] Push custom metrics to CloudWatch namespace

### 9. Add CloudWatch Logging
- [ ] Configure structured logging
- [ ] Log all orchestration events
- [ ] Include correlation IDs for tracing
- [ ] Log task ARNs and status transitions
- [ ] Set log retention period (30 days)
- [ ] Create log insights queries

### 10. Testing and Validation
- [ ] Unit test Lambda handler with mock events
- [ ] Test ECS task triggering
- [ ] Test task monitoring and completion detection
- [ ] Test error handling and retry logic
- [ ] Verify EventBridge rule triggering
- [ ] Test SNS notifications
- [ ] Test manual trigger endpoint
- [ ] Validate metrics publishing

---

## 📁 Files to Create/Modify

```
infrastructure/
├── lambda/
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── handler.py
│   │   ├── orchestrator.py
│   │   ├── requirements.txt
│   │   └── tests/
│   │       ├── __init__.py
│   │       └── test_orchestrator.py
│   └── layers/
│       └── aws-sdk/
├── cdk/
│   └── stacks/
│       ├── eventbridge_stack.py
│       ├── lambda_stack.py
│       └── api_gateway_stack.py
└── scripts/
    ├── package_lambda.sh
    └── deploy_lambda.sh

docs/operations/
├── scheduling-guide.md
├── manual-trigger-guide.md
└── troubleshooting-scheduling.md
```

---

## 🧪 Testing Requirements

### Unit Tests
- [ ] Test Lambda handler with various event types
- [ ] Test ECS task parameter construction
- [ ] Test error handling for different failure scenarios
- [ ] Test metrics publishing
- [ ] Test notification formatting

### Integration Tests
- [ ] Test Lambda invocation from EventBridge
- [ ] Test ECS task launch and monitoring
- [ ] Test SNS notification delivery
- [ ] Test manual trigger via API Gateway
- [ ] Test CloudWatch metrics and logs

### End-to-End Tests
- [ ] Verify scheduled predictions run every 30 minutes
- [ ] Test full prediction workflow from trigger to completion
- [ ] Simulate failures and verify error handling
- [ ] Verify notifications reach operations team
- [ ] Test manual trigger and track execution

---

## 📚 Documentation Requirements

- [ ] Lambda orchestrator architecture diagram
- [ ] EventBridge scheduling configuration guide
- [ ] Manual trigger API documentation
- [ ] Error handling and retry logic documentation
- [ ] Notification configuration guide
- [ ] Troubleshooting scheduling issues runbook

---

## 🔗 Dependencies

- **Upstream:** PC-105-11B (Fargate deployment must be complete)
- **Upstream:** SNS topic creation
- **Upstream:** API Gateway setup (for manual triggers)
- **Tools:** AWS Lambda, EventBridge, SNS, API Gateway
- **Services:** AWS ECS, CloudWatch

---

## 📝 Notes

- Lambda function needs appropriate IAM permissions for ECS, SNS, CloudWatch
- Consider Lambda cold start time in scheduling design
- Use exponential backoff for retry logic
- Implement idempotency for prediction tasks
- Monitor Lambda concurrency limits
- Set appropriate timeout for Lambda (15 min to monitor 30 min tasks)
- Use Lambda environment variables for configuration
- Consider using Step Functions for complex orchestration (future enhancement)

---

## ✅ Definition of Done

- [ ] EventBridge rule configured and enabled
- [ ] Lambda orchestrator function deployed and tested
- [ ] Predictions triggering every 30 minutes automatically
- [ ] Error handling and notifications working
- [ ] Manual trigger endpoint functional and documented
- [ ] All tests passing
- [ ] Documentation complete and reviewed
- [ ] Code reviewed and approved
- [ ] Operations team trained on manual triggers
- [ ] Monitoring confirms scheduled executions
