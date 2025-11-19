# PC-072-08B: Structured Logging System

**Ticket ID:** PC-072-08B  
**Epic:** Epic-08B (Execution Engine & Monitoring)  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 23 (Days 4-5)  

---

## 📋 Description

Implement a comprehensive structured logging system with JSON formatting, correlation ID tracking, performance metrics capture, and integration with log aggregation services (CloudWatch/ELK). This system provides the observability infrastructure required for troubleshooting, monitoring, and operational excellence in production.

### Context
Building on PC-071-08B's parallel execution engine, this ticket provides the logging and monitoring infrastructure that enables rapid issue diagnosis, performance analysis, and system health tracking. Structured JSON logs with correlation IDs enable end-to-end request tracing across async workflows.

### Business Value
- **Rapid Troubleshooting:** Correlation IDs enable issue diagnosis in <10 minutes
- **Performance Monitoring:** Comprehensive metrics for all operations
- **Operational Excellence:** Real-time visibility into system health
- **Compliance:** Audit trail for all system operations
- **Alerting:** Integration with monitoring systems for proactive issue detection

---

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] JSON-structured logs with consistent schema
- [ ] Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- [ ] Context-aware logging with correlation IDs
- [ ] Performance metrics logging (timing, resource usage)
- [ ] Workflow-level logging with context manager
- [ ] Task-level logging with execution details
- [ ] Log aggregation integration (CloudWatch or ELK)
- [ ] Real-time alerting on critical events

### Log Schema Requirements
- [ ] Timestamp in ISO 8601 format
- [ ] Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- [ ] Logger name (module identification)
- [ ] Message (human-readable description)
- [ ] Correlation ID (UUID for request tracing)
- [ ] Custom fields (workflow-specific metadata)
- [ ] Exception traceback (for errors)

### Performance Requirements
- [ ] <5ms overhead per log statement
- [ ] <10% total system overhead for logging
- [ ] Log buffering for high-volume scenarios
- [ ] Async log writing (non-blocking)

### Quality Requirements
- [ ] 80%+ unit test coverage
- [ ] Integration with PC-071-08B (ParallelExecutor)
- [ ] Integration with Epic-08A workflows
- [ ] Log format validation
- [ ] Correlation ID propagation across async tasks

---

## 🔧 Technical Specifications

### Core Components

#### 1. JSONFormatter Class
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: LogRecord from Python logging
            
        Returns:
            JSON string with structured log data
        """
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add custom fields
        if hasattr(record, 'custom_fields'):
            log_data.update(record.custom_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)
```

**Format Details:**
- ISO 8601 timestamp (default formatTime)
- Flat JSON structure (no nesting for searchability)
- Custom fields merged at top level
- Exception includes full traceback string

**Example Output:**
```json
{
  "timestamp": "2023-05-15T14:23:45.123456",
  "level": "INFO",
  "logger": "training_workflow",
  "message": "Starting training workflow",
  "module": "training_workflow",
  "function": "execute_training",
  "line": 125,
  "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "workflow_type": "training",
  "areas": ["area_01", "area_02"],
  "models": ["lgbm", "rf"],
  "event_type": "workflow_start"
}
```

#### 2. StructuredLogger Class
```python
import uuid
from contextlib import contextmanager
from typing import Optional, Dict, Any

class StructuredLogger:
    """Structured logging system with correlation tracking."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for file handler
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
    
    def log(
        self,
        level: str,
        message: str,
        correlation_id: Optional[str] = None,
        **custom_fields
    ):
        """
        Log message with custom fields.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Human-readable log message
            correlation_id: Optional correlation ID for request tracing
            **custom_fields: Additional structured fields
        """
        extra = {}
        if correlation_id:
            extra['correlation_id'] = correlation_id
        if custom_fields:
            extra['custom_fields'] = custom_fields
        
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)
    
    def debug(self, message: str, correlation_id: Optional[str] = None, **fields):
        """Log DEBUG level message."""
        self.log('DEBUG', message, correlation_id, **fields)
    
    def info(self, message: str, correlation_id: Optional[str] = None, **fields):
        """Log INFO level message."""
        self.log('INFO', message, correlation_id, **fields)
    
    def warning(self, message: str, correlation_id: Optional[str] = None, **fields):
        """Log WARNING level message."""
        self.log('WARNING', message, correlation_id, **fields)
    
    def error(self, message: str, correlation_id: Optional[str] = None, **fields):
        """Log ERROR level message."""
        self.log('ERROR', message, correlation_id, **fields)
    
    def critical(self, message: str, correlation_id: Optional[str] = None, **fields):
        """Log CRITICAL level message."""
        self.log('CRITICAL', message, correlation_id, **fields)
```

**Key Features:**
- Convenience methods for each log level
- Automatic correlation ID attachment
- Custom fields as kwargs
- Multiple handler support (console, file, remote)

#### 3. Workflow-Specific Logging Methods
```python
class StructuredLogger:
    # ... (previous methods)
    
    def log_workflow_start(
        self,
        workflow_type: str,
        correlation_id: str,
        parameters: Dict[str, Any]
    ):
        """
        Log workflow start event.
        
        Args:
            workflow_type: 'training', 'prediction', 'backtesting', 'evaluation'
            correlation_id: Unique workflow execution ID
            parameters: Workflow parameters (areas, models, dates, etc.)
        """
        self.log(
            'INFO',
            f"Starting {workflow_type} workflow",
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            parameters=parameters,
            event_type='workflow_start'
        )
    
    def log_workflow_completion(
        self,
        workflow_type: str,
        correlation_id: str,
        duration_seconds: float,
        status: str = 'success',
        results_summary: Optional[Dict[str, Any]] = None
    ):
        """
        Log workflow completion event.
        
        Args:
            workflow_type: 'training', 'prediction', 'backtesting', 'evaluation'
            correlation_id: Unique workflow execution ID
            duration_seconds: Total workflow execution time
            status: 'success' or 'partial_success' or 'failed'
            results_summary: Optional summary of results (counts, metrics, etc.)
        """
        self.log(
            'INFO' if status == 'success' else 'WARNING',
            f"Completed {workflow_type} workflow with status {status}",
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            duration_seconds=duration_seconds,
            status=status,
            results_summary=results_summary or {},
            event_type='workflow_completion'
        )
    
    def log_workflow_error(
        self,
        workflow_type: str,
        correlation_id: str,
        error: Exception,
        duration_seconds: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log workflow error event.
        
        Args:
            workflow_type: 'training', 'prediction', 'backtesting', 'evaluation'
            correlation_id: Unique workflow execution ID
            error: Exception that caused failure
            duration_seconds: Time elapsed before failure
            context: Optional context (current task, step, etc.)
        """
        self.log(
            'ERROR',
            f"Error in {workflow_type} workflow: {str(error)}",
            correlation_id=correlation_id,
            workflow_type=workflow_type,
            duration_seconds=duration_seconds,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            event_type='workflow_error'
        )
    
    def log_task_execution(
        self,
        task_id: str,
        task_type: str,
        correlation_id: str,
        duration_seconds: float,
        status: str,
        resource_usage: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log task execution event (from ParallelExecutor).
        
        Args:
            task_id: Unique task identifier
            task_type: 'training', 'prediction', 'evaluation'
            correlation_id: Parent workflow correlation ID
            duration_seconds: Task execution time
            status: 'success', 'failed', 'timeout'
            resource_usage: CPU, memory, I/O metrics
            metadata: Optional task-specific metadata (area, model, etc.)
        """
        self.log(
            'INFO' if status == 'success' else 'ERROR',
            f"Task {task_id} completed with status {status}",
            correlation_id=correlation_id,
            task_id=task_id,
            task_type=task_type,
            duration_seconds=duration_seconds,
            status=status,
            resource_usage=resource_usage,
            metadata=metadata or {},
            event_type='task_execution'
        )
    
    def log_performance_metrics(
        self,
        correlation_id: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log performance metrics.
        
        Args:
            correlation_id: Parent workflow/task correlation ID
            metric_name: Metric name (e.g., 'prediction_mape', 'training_loss')
            metric_value: Numeric metric value
            metric_unit: Unit (e.g., 'percent', 'seconds', 'count')
            metadata: Optional context (area, model, horizon, etc.)
        """
        self.log(
            'INFO',
            f"Performance metric: {metric_name} = {metric_value} {metric_unit}",
            correlation_id=correlation_id,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            metadata=metadata or {},
            event_type='performance_metric'
        )
    
    def log_drift_detection(
        self,
        correlation_id: str,
        area: str,
        model: str,
        drift_type: str,
        severity: str,
        metric_degradation: float,
        recommended_action: str
    ):
        """
        Log drift detection event.
        
        Args:
            correlation_id: Parent workflow correlation ID
            area: Area identifier
            model: Model identifier
            drift_type: 'GRADUAL', 'ABRUPT', 'CYCLICAL'
            severity: 'INFO', 'WARNING', 'CRITICAL', 'EMERGENCY'
            metric_degradation: Percentage degradation
            recommended_action: Suggested remediation
        """
        level = 'INFO' if severity == 'INFO' else 'WARNING' if severity == 'WARNING' else 'ERROR'
        
        self.log(
            level,
            f"Drift detected: {area}/{model} - {drift_type} ({severity})",
            correlation_id=correlation_id,
            area=area,
            model=model,
            drift_type=drift_type,
            severity=severity,
            metric_degradation=metric_degradation,
            recommended_action=recommended_action,
            event_type='drift_detection'
        )
    
    @contextmanager
    def workflow_context(
        self,
        workflow_type: str,
        parameters: Dict[str, Any]
    ):
        """
        Context manager for workflow-level logging.
        
        Automatically logs workflow start, completion, and errors.
        Yields correlation_id for nested operations.
        
        Usage:
            with logger.workflow_context('training', {'areas': areas}) as correlation_id:
                # Execute workflow
                results = await train_models(areas, correlation_id)
        
        Args:
            workflow_type: 'training', 'prediction', 'backtesting', 'evaluation'
            parameters: Workflow parameters
            
        Yields:
            correlation_id: UUID for this workflow execution
        """
        import time
        
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.log_workflow_start(workflow_type, correlation_id, parameters)
        
        try:
            yield correlation_id
            duration = time.time() - start_time
            self.log_workflow_completion(workflow_type, correlation_id, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_workflow_error(workflow_type, correlation_id, e, duration)
            raise
```

**Workflow Context Manager:**
- Automatic correlation ID generation
- Start/end logging
- Exception handling with logging
- Propagates exceptions after logging

#### 4. Integration with ParallelExecutor
```python
# In PC-071-08B ParallelExecutor
class MonitoredParallelExecutor(ParallelExecutor):
    """ParallelExecutor with integrated structured logging."""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 4,
        resource_thresholds: Optional[Dict[str, float]] = None,
        logger: Optional[StructuredLogger] = None
    ):
        super().__init__(max_concurrent_tasks, resource_thresholds)
        self.logger = logger or StructuredLogger('parallel_executor')
    
    async def execute_parallel_tasks(
        self,
        tasks: List[ExecutionTask],
        strategy: ParallelStrategy = ParallelStrategy.AREA_FIRST,
        adaptive_concurrency: bool = True,
        correlation_id: Optional[str] = None
    ) -> List[TaskResult]:
        """Execute tasks with logging."""
        correlation_id = correlation_id or str(uuid.uuid4())
        
        self.logger.info(
            f"Starting parallel execution with {len(tasks)} tasks",
            correlation_id=correlation_id,
            num_tasks=len(tasks),
            strategy=strategy.value,
            adaptive_concurrency=adaptive_concurrency,
            event_type='parallel_execution_start'
        )
        
        # Execute tasks
        results = await super().execute_parallel_tasks(tasks, strategy, adaptive_concurrency)
        
        # Log results summary
        success_count = sum(1 for r in results if r.status == 'success')
        failed_count = sum(1 for r in results if r.status == 'failed')
        
        self.logger.info(
            f"Parallel execution completed: {success_count} success, {failed_count} failed",
            correlation_id=correlation_id,
            success_count=success_count,
            failed_count=failed_count,
            total_tasks=len(tasks),
            event_type='parallel_execution_completion'
        )
        
        # Log individual task results
        for result in results:
            self.logger.log_task_execution(
                task_id=result.task_id,
                task_type='parallel_task',
                correlation_id=correlation_id,
                duration_seconds=result.duration_seconds,
                status=result.status,
                resource_usage=result.resource_usage
            )
        
        return results
```

### Log Aggregation Integration

#### 1. CloudWatch Integration
```python
import boto3
from datetime import datetime

class CloudWatchHandler(logging.Handler):
    """Custom handler for AWS CloudWatch Logs."""
    
    def __init__(
        self,
        log_group: str,
        log_stream: str,
        region: str = 'us-east-1'
    ):
        """
        Initialize CloudWatch handler.
        
        Args:
            log_group: CloudWatch log group name
            log_stream: CloudWatch log stream name
            region: AWS region
        """
        super().__init__()
        self.client = boto3.client('logs', region_name=region)
        self.log_group = log_group
        self.log_stream = log_stream
        self.sequence_token = None
        
        # Create log group and stream if not exist
        self._ensure_log_stream_exists()
    
    def _ensure_log_stream_exists(self):
        """Create log group and stream if they don't exist."""
        try:
            self.client.create_log_group(logGroupName=self.log_group)
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass
        
        try:
            self.client.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=self.log_stream
            )
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass
    
    def emit(self, record):
        """Send log record to CloudWatch."""
        try:
            log_entry = {
                'timestamp': int(datetime.now().timestamp() * 1000),
                'message': self.format(record)
            }
            
            kwargs = {
                'logGroupName': self.log_group,
                'logStreamName': self.log_stream,
                'logEvents': [log_entry]
            }
            
            if self.sequence_token:
                kwargs['sequenceToken'] = self.sequence_token
            
            response = self.client.put_log_events(**kwargs)
            self.sequence_token = response.get('nextSequenceToken')
        
        except Exception as e:
            self.handleError(record)


# Usage in StructuredLogger
def add_cloudwatch_handler(
    logger: StructuredLogger,
    log_group: str,
    log_stream: str,
    region: str = 'us-east-1'
):
    """Add CloudWatch handler to logger."""
    handler = CloudWatchHandler(log_group, log_stream, region)
    handler.setFormatter(JSONFormatter())
    logger.logger.addHandler(handler)
```

#### 2. ELK (Elasticsearch) Integration
```python
from elasticsearch import Elasticsearch

class ElasticsearchHandler(logging.Handler):
    """Custom handler for Elasticsearch."""
    
    def __init__(
        self,
        hosts: List[str],
        index_prefix: str = 'forecasting-logs'
    ):
        """
        Initialize Elasticsearch handler.
        
        Args:
            hosts: List of Elasticsearch hosts
            index_prefix: Index prefix (date will be appended)
        """
        super().__init__()
        self.client = Elasticsearch(hosts)
        self.index_prefix = index_prefix
    
    def emit(self, record):
        """Send log record to Elasticsearch."""
        try:
            # Parse JSON log message
            log_data = json.loads(self.format(record))
            
            # Add @timestamp for Kibana
            log_data['@timestamp'] = log_data['timestamp']
            
            # Index with date-based index
            index_name = f"{self.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}"
            
            self.client.index(
                index=index_name,
                document=log_data
            )
        
        except Exception as e:
            self.handleError(record)


# Usage in StructuredLogger
def add_elasticsearch_handler(
    logger: StructuredLogger,
    hosts: List[str],
    index_prefix: str = 'forecasting-logs'
):
    """Add Elasticsearch handler to logger."""
    handler = ElasticsearchHandler(hosts, index_prefix)
    handler.setFormatter(JSONFormatter())
    logger.logger.addHandler(handler)
```

### Alerting System

#### Alert Rule Configuration
```python
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]  # Function to evaluate log data
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    notification_channels: List[str]  # ['email', 'slack', 'pagerduty']
    cooldown_seconds: int = 300  # Minimum time between alerts

class AlertManager:
    """Manage alerting based on log events."""
    
    def __init__(self, rules: List[AlertRule]):
        self.rules = rules
        self.last_alert_time = {}  # Track cooldowns
    
    def process_log_event(self, log_data: Dict[str, Any]):
        """
        Process log event and trigger alerts if conditions met.
        
        Args:
            log_data: Parsed JSON log data
        """
        import time
        
        for rule in self.rules:
            # Check if condition met
            if not rule.condition(log_data):
                continue
            
            # Check cooldown
            last_time = self.last_alert_time.get(rule.name, 0)
            if time.time() - last_time < rule.cooldown_seconds:
                continue
            
            # Trigger alert
            self._send_alert(rule, log_data)
            self.last_alert_time[rule.name] = time.time()
    
    def _send_alert(self, rule: AlertRule, log_data: Dict[str, Any]):
        """Send alert through configured channels."""
        alert_message = f"[{rule.severity}] {rule.name}: {log_data.get('message', 'N/A')}"
        
        for channel in rule.notification_channels:
            if channel == 'email':
                self._send_email_alert(alert_message, log_data)
            elif channel == 'slack':
                self._send_slack_alert(alert_message, log_data)
            elif channel == 'pagerduty':
                self._send_pagerduty_alert(alert_message, log_data)
    
    def _send_email_alert(self, message: str, log_data: Dict[str, Any]):
        """Send email alert (placeholder)."""
        # Implement with SMTP or AWS SES
        pass
    
    def _send_slack_alert(self, message: str, log_data: Dict[str, Any]):
        """Send Slack alert (placeholder)."""
        # Implement with Slack webhook
        pass
    
    def _send_pagerduty_alert(self, message: str, log_data: Dict[str, Any]):
        """Send PagerDuty alert (placeholder)."""
        # Implement with PagerDuty API
        pass


# Example alert rules
ALERT_RULES = [
    AlertRule(
        name='workflow_failure',
        condition=lambda log: log.get('event_type') == 'workflow_error',
        severity='CRITICAL',
        notification_channels=['email', 'pagerduty'],
        cooldown_seconds=600
    ),
    AlertRule(
        name='high_task_failure_rate',
        condition=lambda log: (
            log.get('event_type') == 'parallel_execution_completion' and
            log.get('failed_count', 0) / max(log.get('total_tasks', 1), 1) > 0.2
        ),
        severity='WARNING',
        notification_channels=['email', 'slack'],
        cooldown_seconds=300
    ),
    AlertRule(
        name='drift_emergency',
        condition=lambda log: (
            log.get('event_type') == 'drift_detection' and
            log.get('severity') == 'EMERGENCY'
        ),
        severity='CRITICAL',
        notification_channels=['email', 'pagerduty', 'slack'],
        cooldown_seconds=1800
    )
]
```

---

## 📝 Implementation Tasks

### Task 1: Core Logging Infrastructure (Day 4 Morning)
**Estimated Time:** 3 hours

1. **Implement JSONFormatter:**
   - Format LogRecord as JSON
   - Handle correlation_id from extra
   - Handle custom_fields from extra
   - Format exceptions with traceback
   - ISO 8601 timestamp formatting

2. **Implement StructuredLogger:**
   - Initialize with name, level, log_file
   - Add console handler with JSONFormatter
   - Add file handler (optional)
   - Implement log() method
   - Implement convenience methods (debug, info, warning, error, critical)

3. **Write unit tests:**
   - Test JSON format structure
   - Test correlation ID attachment
   - Test custom fields merging
   - Test exception formatting
   - Test multi-handler configuration

**Deliverables:**
- `src/logging/structured_logger.py` with JSONFormatter and StructuredLogger
- `tests/logging/test_structured_logger.py` with unit tests

**Acceptance:**
- JSON logs validate against schema
- All log levels work correctly
- Correlation IDs properly attached

---

### Task 2: Workflow Logging Methods (Day 4 Afternoon)
**Estimated Time:** 3 hours

1. **Implement workflow logging methods:**
   - log_workflow_start()
   - log_workflow_completion()
   - log_workflow_error()
   - log_task_execution()
   - log_performance_metrics()
   - log_drift_detection()

2. **Implement workflow_context manager:**
   - Generate correlation_id
   - Log start event
   - Yield correlation_id
   - Log completion or error
   - Propagate exceptions

3. **Write unit tests:**
   - Test each workflow logging method
   - Test workflow_context happy path
   - Test workflow_context error handling
   - Test correlation ID propagation
   - Mock time for duration calculations

**Deliverables:**
- Workflow logging methods in StructuredLogger
- Comprehensive unit tests

**Acceptance:**
- All workflow events logged correctly
- workflow_context handles success and failure
- Correlation IDs propagate properly

---

### Task 3: ParallelExecutor Integration (Day 4 Evening)
**Estimated Time:** 2 hours

1. **Create MonitoredParallelExecutor:**
   - Extend ParallelExecutor from PC-071-08B
   - Add StructuredLogger integration
   - Log parallel execution start/completion
   - Log individual task results

2. **Update PC-071-08B code:**
   - Add optional logger parameter
   - Add correlation_id parameter to execute_parallel_tasks()
   - Log progress tracking milestones
   - Log concurrency adjustments

3. **Write integration tests:**
   - Test logging during parallel execution
   - Verify correlation ID propagation
   - Test log volume (ensure not excessive)
   - Test task-level logging

**Deliverables:**
- MonitoredParallelExecutor class
- Updated PC-071-08B with logging hooks
- Integration tests

**Acceptance:**
- ParallelExecutor logs all key events
- Correlation IDs link workflow to tasks
- Log volume reasonable (<100 logs per workflow)

---

### Task 4: Log Aggregation Integration (Day 5 Morning)
**Estimated Time:** 3 hours

1. **Implement CloudWatchHandler:**
   - Initialize boto3 client
   - Create log group/stream if needed
   - Implement emit() method
   - Handle sequence tokens
   - Error handling

2. **Implement ElasticsearchHandler:**
   - Initialize Elasticsearch client
   - Implement emit() method
   - Date-based index naming
   - Add @timestamp for Kibana
   - Error handling

3. **Add configuration:**
   - LoggingConfig in SystemConfig (Epic-08A)
   - cloudwatch_enabled, cloudwatch_log_group, cloudwatch_log_stream
   - elasticsearch_enabled, elasticsearch_hosts, elasticsearch_index
   - File logging configuration

4. **Write integration tests:**
   - Mock boto3 for CloudWatch testing
   - Mock Elasticsearch client
   - Test handler initialization
   - Test log emission

**Deliverables:**
- CloudWatchHandler and ElasticsearchHandler
- Configuration additions
- Integration tests with mocks

**Acceptance:**
- CloudWatch handler sends logs successfully
- Elasticsearch handler indexes logs correctly
- Configuration properly wired
- Error handling prevents logging failures from crashing

---

### Task 5: Alerting System (Day 5 Afternoon)
**Estimated Time:** 3 hours

1. **Implement AlertRule and AlertManager:**
   - AlertRule dataclass
   - AlertManager with rule processing
   - Cooldown tracking
   - Notification channel routing

2. **Implement notification channels:**
   - Email alerts (SMTP or SES)
   - Slack alerts (webhook)
   - PagerDuty alerts (API)
   - Configurable per rule

3. **Define default alert rules:**
   - Workflow failure (CRITICAL)
   - High task failure rate (WARNING)
   - Drift emergency (CRITICAL)
   - Resource exhaustion (WARNING)

4. **Write unit tests:**
   - Test alert condition evaluation
   - Test cooldown mechanism
   - Mock notification channels
   - Test rule priority and ordering

**Deliverables:**
- AlertManager and notification channels
- Default alert rules
- Configuration for alerting

**Acceptance:**
- Alerts trigger based on conditions
- Cooldowns prevent alert storms
- Notification channels work
- Configurable and extensible

---

### Task 6: Epic-08A Workflow Integration (Day 5 Evening)
**Estimated Time:** 2 hours

1. **Update Epic-08A workflows:**
   - Add StructuredLogger to TrainingWorkflow
   - Add StructuredLogger to PredictionWorkflow
   - Add StructuredLogger to BacktestingWorkflow
   - Use workflow_context for all executions

2. **Integration examples:**
   - Training workflow with full logging
   - Prediction workflow with performance metrics
   - Backtesting workflow with drift detection logs

3. **Write end-to-end tests:**
   - Full training workflow with log validation
   - Full prediction workflow with log validation
   - Verify log searchability (grep for correlation_id)

**Deliverables:**
- Updated Epic-08A workflows with logging
- End-to-end integration tests
- Usage examples

**Acceptance:**
- All workflows log structured data
- Correlation IDs enable request tracing
- Logs searchable by workflow, task, area, model

---

### Task 7: Documentation and Testing (Day 5 Evening)
**Estimated Time:** 1 hour

1. **Performance testing:**
   - Measure logging overhead (<10% target)
   - Test high-volume logging (1000+ logs/sec)
   - Test async log writing

2. **Documentation:**
   - Log schema reference
   - Correlation ID usage guide
   - CloudWatch setup instructions
   - Elasticsearch/Kibana setup instructions
   - Alert rule configuration guide
   - Troubleshooting guide

3. **Code review preparation:**
   - Run linters
   - Ensure 80%+ test coverage
   - Add type hints
   - Clean up debug code

**Deliverables:**
- Performance benchmarks
- Complete documentation
- Code review ready

**Acceptance:**
- <10% logging overhead
- 80%+ test coverage
- Documentation complete
- Code quality high

---

## 🧪 Testing Requirements

### Unit Tests

#### 1. JSONFormatter Tests
```python
def test_json_formatter_basic():
    """Test basic JSON formatting."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=10,
        msg='Test message',
        args=(),
        exc_info=None
    )
    record.module = 'test'
    record.funcName = 'test_func'
    
    output = formatter.format(record)
    data = json.loads(output)
    
    assert data['level'] == 'INFO'
    assert data['message'] == 'Test message'
    assert data['module'] == 'test'
    assert data['function'] == 'test_func'
    assert data['line'] == 10


def test_json_formatter_with_correlation_id():
    """Test JSON formatting with correlation ID."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=10,
        msg='Test message',
        args=(),
        exc_info=None
    )
    record.correlation_id = 'abc-123'
    record.module = 'test'
    record.funcName = 'test_func'
    
    output = formatter.format(record)
    data = json.loads(output)
    
    assert data['correlation_id'] == 'abc-123'


def test_json_formatter_with_custom_fields():
    """Test JSON formatting with custom fields."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=10,
        msg='Test message',
        args=(),
        exc_info=None
    )
    record.custom_fields = {'area': 'area_01', 'model': 'lgbm'}
    record.module = 'test'
    record.funcName = 'test_func'
    
    output = formatter.format(record)
    data = json.loads(output)
    
    assert data['area'] == 'area_01'
    assert data['model'] == 'lgbm'
```

#### 2. StructuredLogger Tests
```python
def test_structured_logger_basic():
    """Test basic structured logging."""
    logger = StructuredLogger('test_logger')
    
    with patch('logging.Logger.info') as mock_info:
        logger.info('Test message', correlation_id='abc-123', area='area_01')
        
        assert mock_info.called
        call_args = mock_info.call_args
        assert 'correlation_id' in call_args[1]['extra']
        assert call_args[1]['extra']['correlation_id'] == 'abc-123'


def test_workflow_context_success():
    """Test workflow context manager success case."""
    logger = StructuredLogger('test_logger')
    
    with logger.workflow_context('training', {'areas': ['area_01']}) as correlation_id:
        assert correlation_id is not None
        assert len(correlation_id) == 36  # UUID length
    
    # Verify start and completion logged
    # (would need to capture log output)


def test_workflow_context_error():
    """Test workflow context manager error handling."""
    logger = StructuredLogger('test_logger')
    
    with pytest.raises(ValueError):
        with logger.workflow_context('training', {'areas': ['area_01']}) as correlation_id:
            raise ValueError("Test error")
    
    # Verify error logged
    # (would need to capture log output)
```

#### 3. AlertManager Tests
```python
def test_alert_rule_condition():
    """Test alert rule condition evaluation."""
    rule = AlertRule(
        name='test_alert',
        condition=lambda log: log.get('level') == 'ERROR',
        severity='CRITICAL',
        notification_channels=['email']
    )
    
    assert rule.condition({'level': 'ERROR'}) is True
    assert rule.condition({'level': 'INFO'}) is False


def test_alert_manager_cooldown():
    """Test alert cooldown mechanism."""
    rule = AlertRule(
        name='test_alert',
        condition=lambda log: True,
        severity='WARNING',
        notification_channels=[],
        cooldown_seconds=60
    )
    
    manager = AlertManager([rule])
    
    with patch.object(manager, '_send_alert') as mock_send:
        # First alert should trigger
        manager.process_log_event({'message': 'Test 1'})
        assert mock_send.call_count == 1
        
        # Second alert within cooldown should not trigger
        manager.process_log_event({'message': 'Test 2'})
        assert mock_send.call_count == 1
```

### Integration Tests

#### 1. End-to-End Logging
```python
@pytest.mark.asyncio
async def test_e2e_logging_workflow():
    """Test end-to-end logging for complete workflow."""
    logger = StructuredLogger('test_workflow')
    
    with logger.workflow_context('training', {'areas': ['area_01']}) as correlation_id:
        # Simulate workflow execution
        logger.log_task_execution(
            task_id='task_001',
            task_type='training',
            correlation_id=correlation_id,
            duration_seconds=10.5,
            status='success',
            resource_usage={'cpu_percent': 75.0}
        )
        
        logger.log_performance_metrics(
            correlation_id=correlation_id,
            metric_name='mape',
            metric_value=5.2,
            metric_unit='percent',
            metadata={'area': 'area_01', 'model': 'lgbm'}
        )
    
    # Verify logs can be parsed and searched
    # (would need to capture and validate log output)
```

#### 2. CloudWatch Integration
```python
@pytest.mark.integration
def test_cloudwatch_handler_integration(mocker):
    """Test CloudWatch handler integration."""
    mock_client = mocker.Mock()
    mocker.patch('boto3.client', return_value=mock_client)
    
    handler = CloudWatchHandler(
        log_group='test-group',
        log_stream='test-stream'
    )
    
    logger = StructuredLogger('test_logger')
    logger.logger.addHandler(handler)
    
    logger.info('Test message', correlation_id='abc-123')
    
    # Verify put_log_events called
    assert mock_client.put_log_events.called
```

### Performance Tests

#### 1. Logging Overhead
```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_logging_overhead():
    """Measure logging overhead."""
    import time
    
    logger = StructuredLogger('test_logger')
    
    # Measure without logging
    start = time.time()
    for i in range(1000):
        pass
    baseline = time.time() - start
    
    # Measure with logging
    start = time.time()
    for i in range(1000):
        logger.info(f'Message {i}', correlation_id='abc-123')
    with_logging = time.time() - start
    
    overhead = (with_logging - baseline) / with_logging * 100
    assert overhead < 10  # Less than 10% overhead
```

---

## ✅ Success Metrics

### Functional Metrics
- [ ] All logs JSON-formatted and parseable
- [ ] Correlation IDs propagate across async tasks
- [ ] All workflow events logged
- [ ] All task executions logged
- [ ] Performance metrics captured
- [ ] Drift detection logged

### Performance Metrics
- [ ] <5ms per log statement
- [ ] <10% total system overhead
- [ ] Log aggregation latency <1 second
- [ ] Alert delivery <30 seconds

### Quality Metrics
- [ ] 80%+ test coverage
- [ ] Integration with PC-071-08B complete
- [ ] Integration with Epic-08A workflows complete
- [ ] Documentation complete

---

## 📚 Dependencies

### Internal Dependencies
- **PC-071-08B:** ParallelExecutor (for task execution logging)
- **PC-067-08A:** ConfigManager (for logging configuration)
- **PC-068-08A:** TrainingWorkflow (integration)
- **PC-069-08A:** PredictionWorkflow (integration)
- **PC-070-08A:** BacktestingWorkflow (integration)

### External Dependencies
```python
# Standard library
import logging
import json
import uuid
from contextlib import contextmanager
from typing import Dict, Any, Optional, List

# Third-party (optional)
import boto3  # CloudWatch integration (optional)
from elasticsearch import Elasticsearch  # ELK integration (optional)

# Version requirements
# boto3 >= 1.26.0 (optional)
# elasticsearch >= 8.0.0 (optional)
```

---

## 🔄 Related Tickets

### Depends On
- **PC-071-08B:** ParallelExecutor (provides task execution events)
- **PC-067-08A:** ConfigManager (provides logging configuration)

### Blocks
- **Epic-09:** CLI (will use structured logging for all commands)

### Related
- **PC-064-07B:** DriftDetector (drift events logged)
- **PC-060-07A:** MetricsCalculator (performance metrics logged)

---

## 📖 Documentation Requirements

### Code Documentation
- [ ] Docstrings for all classes and methods
- [ ] Type hints for all functions
- [ ] Usage examples in docstrings

### User Documentation
- [ ] **Log Schema Reference:** Complete field documentation
- [ ] **Correlation ID Guide:** How to use for request tracing
- [ ] **CloudWatch Setup:** Step-by-step configuration
- [ ] **Elasticsearch Setup:** Installation and Kibana configuration
- [ ] **Alert Configuration:** How to define custom rules
- [ ] **Troubleshooting Guide:** Common log queries and patterns

### API Documentation
- [ ] StructuredLogger API reference
- [ ] JSONFormatter specification
- [ ] Handler configuration options
- [ ] Alert rule syntax

---

## 🎯 Definition of Done

### Code Complete
- [ ] All classes implemented
- [ ] Type hints and docstrings complete
- [ ] No TODOs or FIXMEs

### Testing Complete
- [ ] 80%+ unit test coverage
- [ ] Integration tests pass
- [ ] Performance overhead <10%
- [ ] CloudWatch/ELK integration tested

### Documentation Complete
- [ ] Log schema documented
- [ ] Setup guides written
- [ ] Troubleshooting guide complete
- [ ] API reference generated

### Integration Complete
- [ ] PC-071-08B integration complete
- [ ] Epic-08A workflows integrated
- [ ] Configuration wired
- [ ] Alerting operational

### Quality Gates
- [ ] Code review approved
- [ ] No critical bugs
- [ ] Linters pass
- [ ] Ready for Epic-09

---

## 📝 Implementation Notes

### Design Decisions

1. **JSON Over Plain Text:**
   - Structured data for easy parsing
   - Searchable in log aggregation systems
   - Standard format for cloud platforms

2. **Correlation IDs:**
   - UUID v4 for uniqueness
   - Propagate through async context
   - Enable end-to-end tracing

3. **Context Manager for Workflows:**
   - Automatic start/end logging
   - Exception handling
   - Cleaner code (Pythonic)

4. **Multiple Handler Support:**
   - Console for development
   - File for local debugging
   - CloudWatch/ELK for production
   - Configurable per environment

### Known Limitations

1. **Log Volume:**
   - High-frequency logging may impact performance
   - Consider sampling for high-volume operations
   - Use log levels appropriately

2. **CloudWatch Costs:**
   - Pay per GB ingested and stored
   - Set retention policies
   - Use log filtering

3. **Elasticsearch Storage:**
   - Disk space grows with log volume
   - Implement index lifecycle management
   - Consider compression

### Future Enhancements

1. **Distributed Tracing:**
   - OpenTelemetry integration
   - Jaeger/Zipkin support
   - Span-based tracing

2. **Log Sampling:**
   - Sample high-frequency logs
   - Preserve critical events
   - Configurable sampling rates

3. **Advanced Analytics:**
   - Log-based metrics
   - Anomaly detection
   - Trend analysis

---

## 🚀 Handoff Checklist

### For Epic-09 (CLI)
- [ ] StructuredLogger available for CLI commands
- [ ] Workflow context manager ready
- [ ] Configuration examples provided

### For Production Deployment
- [ ] CloudWatch or ELK configured
- [ ] Alert rules defined
- [ ] Log retention policies set
- [ ] Monitoring dashboards created
