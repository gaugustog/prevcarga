# PC-107-11B: CloudWatch Monitoring and Alerting Infrastructure

**Ticket ID:** PC-107-11B  
**Epic:** [Epic-11B: Production Deployment & Operations](../epics/Epic-11B.md)  
**User Story:** US-3  
**Story Points:** 10  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement comprehensive CloudWatch monitoring and alerting infrastructure including custom dashboards for system health, custom metrics for prediction accuracy and performance, automated alerts for failures and degradation, log aggregation, cost monitoring, and integration with incident management systems.

**As an** operations engineer  
**I want** comprehensive monitoring and alerting  
**So that** I can proactively identify and resolve issues before they impact operations

---

## ✅ Acceptance Criteria

- [ ] CloudWatch dashboards for system health monitoring deployed
- [ ] Custom metrics for prediction accuracy and performance published
- [ ] Automated alerts for system failures and performance degradation configured
- [ ] Log aggregation and searchability working across all services
- [ ] Cost monitoring and optimization alerts active
- [ ] Integration with incident management systems (SNS, PagerDuty) functional
- [ ] All critical alarms tested and verified
- [ ] Dashboard accessible to operations team
- [ ] Alert runbook documentation complete
- [ ] Metrics retention policies configured

---

## 🔧 Implementation Tasks

### 1. Set Up CloudWatch Log Groups
- [ ] Create log group for ECS Fargate tasks: `/ecs/prevcarga`
- [ ] Create log group for Lambda functions: `/aws/lambda/prevcarga-*`
- [ ] Create log group for application logs: `/prevcarga/application`
- [ ] Set retention period: 30 days (configurable)
- [ ] Enable log encryption
- [ ] Configure log streaming

### 2. Create System Health Dashboard
- [ ] Create dashboard: `PrevCarga-SystemHealth`
- [ ] Add ECS CPU utilization widget (average, max)
- [ ] Add ECS memory utilization widget (average, max)
- [ ] Add task count widget (running, pending, stopped)
- [ ] Add ALB metrics widget (response time, request count)
- [ ] Add ALB target health widget
- [ ] Add Lambda invocation metrics widget
- [ ] Add Lambda error rate widget
- [ ] Configure 5-minute granularity

### 3. Create Prediction Performance Dashboard
- [ ] Create dashboard: `PrevCarga-PredictionPerformance`
- [ ] Add MAPE metric widget by model
- [ ] Add MAE metric widget by model
- [ ] Add RMSE metric widget by model
- [ ] Add prediction latency widget (average, p95, p99)
- [ ] Add prediction count widget (successful, failed)
- [ ] Add model selection distribution widget
- [ ] Add data freshness widget
- [ ] Configure 30-minute granularity

### 4. Create Cost Monitoring Dashboard
- [ ] Create dashboard: `PrevCarga-CostMonitoring`
- [ ] Add ECS task runtime cost estimation
- [ ] Add Lambda invocation cost widget
- [ ] Add data transfer cost widget
- [ ] Add S3 storage cost widget
- [ ] Add CloudWatch cost widget
- [ ] Add daily cost trend widget
- [ ] Set cost alert thresholds

### 5. Implement Custom Metrics Collection
- [ ] Create `MetricsCollector` class
- [ ] Implement `publish_prediction_metrics()` method
- [ ] Publish MAPE, MAE, RMSE per model
- [ ] Publish prediction latency metrics
- [ ] Publish data loading time metrics
- [ ] Publish feature generation time metrics
- [ ] Use CloudWatch namespace: `PrevCarga/Predictions`
- [ ] Add model and horizon dimensions

### 6. Configure Critical Alarms
- [ ] Create alarm: `PrevCarga-HighCPU` (threshold: 85%, 2 periods)
- [ ] Create alarm: `PrevCarga-HighMemory` (threshold: 90%, 2 periods)
- [ ] Create alarm: `PrevCarga-HighErrorRate` (threshold: 10 errors/5min)
- [ ] Create alarm: `PrevCarga-TaskFailure` (threshold: 1 failure)
- [ ] Create alarm: `PrevCarga-AccuracyDegradation` (MAPE > 15%, 3 periods)
- [ ] Create alarm: `PrevCarga-HighLatency` (p95 > 5min, 2 periods)
- [ ] Create alarm: `PrevCarga-SchedulingFailure` (missed schedule)
- [ ] Create alarm: `PrevCarga-DataStaleness` (no new data in 2 hours)

### 7. Configure Warning Alarms
- [ ] Create alarm: `PrevCarga-ModerateCPU` (threshold: 70%, 3 periods)
- [ ] Create alarm: `PrevCarga-ModerateMemory` (threshold: 75%, 3 periods)
- [ ] Create alarm: `PrevCarga-IncreasedLatency` (p95 > 3min, 3 periods)
- [ ] Create alarm: `PrevCarga-ReducedAccuracy` (MAPE > 12%, 5 periods)
- [ ] Create alarm: `PrevCarga-HighCost` (daily cost > threshold)
- [ ] Create alarm: `PrevCarga-LowPredictionVolume` (< expected count)

### 8. Set Up SNS Notification Topics
- [ ] Create topic: `prevcarga-critical-alerts`
- [ ] Create topic: `prevcarga-warning-alerts`
- [ ] Create topic: `prevcarga-info-alerts`
- [ ] Configure email subscriptions for operations team
- [ ] Configure SMS for critical alerts (on-call)
- [ ] Test notification delivery
- [ ] Document subscription management

### 9. Integrate with Incident Management
- [ ] Configure PagerDuty integration (if applicable)
- [ ] Set up incident routing rules
- [ ] Configure escalation policies
- [ ] Test incident creation from alarms
- [ ] Document incident response workflow
- [ ] Create incident templates

### 10. Implement Log Insights Queries
- [ ] Create query: "Recent prediction errors"
- [ ] Create query: "Slowest predictions by model"
- [ ] Create query: "Failed task analysis"
- [ ] Create query: "Accuracy degradation timeline"
- [ ] Create query: "Resource utilization spikes"
- [ ] Save queries to dashboard
- [ ] Document query usage

### 11. Set Up Metric Filters
- [ ] Create filter: Count error log entries
- [ ] Create filter: Count warning log entries
- [ ] Create filter: Extract prediction duration
- [ ] Create filter: Extract model selection events
- [ ] Create filter: Track S3 access errors
- [ ] Create metric alarms from filters

### 12. Testing and Validation
- [ ] Verify dashboards display correctly
- [ ] Test custom metrics publishing
- [ ] Trigger each alarm and verify notifications
- [ ] Test SNS delivery to all subscribers
- [ ] Verify log aggregation across services
- [ ] Test log insights queries
- [ ] Validate metric filter creation
- [ ] Simulate failure scenarios

---

## 📁 Files to Create/Modify

```
infrastructure/
├── cdk/
│   └── stacks/
│       ├── monitoring_stack.py
│       ├── alarms_stack.py
│       └── dashboards_stack.py
├── monitoring/
│   ├── __init__.py
│   ├── metrics_collector.py
│   ├── dashboard_definitions.py
│   ├── alarm_definitions.py
│   └── log_insights_queries.py
└── scripts/
    ├── test_alarms.sh
    └── validate_metrics.py

src/prevcarga/
└── monitoring/
    ├── __init__.py
    ├── metrics.py
    └── logging_config.py

docs/operations/
├── monitoring-guide.md
├── dashboard-overview.md
├── alarm-runbook.md
└── log-analysis-guide.md
```

---

## 🧪 Testing Requirements

### Unit Tests
- [ ] Test metrics collector with mock CloudWatch client
- [ ] Test dashboard JSON generation
- [ ] Test alarm configuration validation
- [ ] Test log filter patterns

### Integration Tests
- [ ] Test metrics publishing to CloudWatch
- [ ] Test dashboard creation and rendering
- [ ] Test alarm triggering with test metrics
- [ ] Test SNS notification delivery
- [ ] Test log aggregation across services

### End-to-End Tests
- [ ] Simulate high CPU and verify alarm
- [ ] Simulate prediction failure and verify alert
- [ ] Simulate accuracy degradation and verify notification
- [ ] Verify dashboard updates with live data
- [ ] Test complete incident management workflow

---

## 📚 Documentation Requirements

- [ ] Monitoring architecture diagram
- [ ] Dashboard user guide with screenshots
- [ ] Alarm configuration reference
- [ ] Alert response runbook for each alarm
- [ ] Log insights query cookbook
- [ ] Custom metrics documentation
- [ ] Cost monitoring and optimization guide
- [ ] Troubleshooting monitoring issues

---

## 🔗 Dependencies

- **Upstream:** PC-105-11B (Fargate deployment for ECS metrics)
- **Upstream:** PC-106-11B (Lambda orchestrator for Lambda metrics)
- **Upstream:** SNS topics setup
- **Tools:** AWS CloudWatch, SNS, CloudWatch Logs Insights
- **Optional:** PagerDuty account for incident management

---

## 📝 Notes

- Use CloudWatch Dashboards API for programmatic creation
- Follow AWS best practices for alarm thresholds
- Implement composite alarms for complex conditions
- Use anomaly detection for dynamic thresholds (future enhancement)
- Consider cost implications of custom metrics (charged per metric)
- Implement metric namespaces for organization
- Use dimensions for filtering and aggregation
- Set appropriate retention periods to balance cost and compliance
- Document all custom metrics and their meaning
- Create alert fatigue prevention strategy (adjust thresholds)

---

## ✅ Definition of Done

- [ ] All dashboards deployed and accessible
- [ ] All alarms configured and tested
- [ ] Custom metrics publishing correctly
- [ ] SNS notifications reaching operations team
- [ ] Log aggregation working across all services
- [ ] Log insights queries saved and documented
- [ ] All tests passing
- [ ] Documentation complete and reviewed
- [ ] Operations team trained on dashboards
- [ ] Alarm runbooks created and validated
- [ ] Incident management integration tested
