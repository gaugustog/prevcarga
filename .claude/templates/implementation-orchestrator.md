# PrevCarga Implementation Orchestrator

> Master orchestration logic for autonomous ticket implementation.
> This document defines the decision-making process for implementing tickets.

---

## 1. Orchestration Overview

The orchestrator manages the autonomous implementation of PrevCarga tickets by:
1. Selecting the next ticket(s) based on dependencies and priority
2. Coordinating subagents for implementation
3. Validating quality gates
4. Managing state and rollback

---

## 2. State Management

### State Files Location
```
.claude/state/
├── tickets-status.json       # Status of each ticket
├── dependency-graph.json     # Parsed dependency relationships
├── implementation-log.md     # Chronological log of actions
└── session-state.json        # Current session context
```

### Ticket Status Schema
```json
{
  "ticket_id": "PC-001-00-repository-structure-setup",
  "status": "pending|in_progress|completed|blocked|failed",
  "started_at": "2025-01-17T10:00:00Z",
  "completed_at": null,
  "attempts": 0,
  "last_error": null,
  "validation_results": {
    "tests_passed": null,
    "coverage": null,
    "lint_passed": null,
    "type_check_passed": null
  },
  "git_commits": [],
  "notes": ""
}
```

### Session State Schema
```json
{
  "session_id": "uuid",
  "started_at": "2025-01-17T10:00:00Z",
  "current_ticket": null,
  "tickets_completed_this_session": [],
  "total_implementation_time_minutes": 0,
  "mode": "single|batch|autonomous"
}
```

---

## 3. Ticket Selection Algorithm

### Priority Rules (in order)
1. **Dependency satisfaction**: Only consider tickets whose dependencies are ALL completed
2. **Epic order**: Prefer earlier epics (Epic-00 before Epic-01)
3. **Ticket number**: Within same epic, prefer lower ticket numbers
4. **Critical path**: Prefer tickets that unblock the most other tickets
5. **Story points**: Prefer smaller tickets when multiple options exist

### Selection Pseudocode
```python
def select_next_ticket(tickets: List[Ticket], dependency_graph: Graph) -> Ticket:
    # Filter to ready tickets
    ready = [t for t in tickets
             if t.status == 'pending'
             and all(dep.status == 'completed' for dep in t.dependencies)]

    if not ready:
        return None

    # Sort by priority
    ready.sort(key=lambda t: (
        epic_order(t.epic),           # Epic-00 = 0, Epic-01 = 1, etc.
        t.ticket_number,              # PC-001 < PC-002
        -count_blocked_by(t),         # More blocking = higher priority
        t.story_points or 999         # Smaller = higher priority
    ))

    return ready[0]
```

### Parallel Selection (for batch mode)
```python
def select_parallel_tickets(tickets: List[Ticket], max_parallel: int = 3) -> List[Ticket]:
    ready = get_ready_tickets(tickets)

    # Group by independent subgraphs
    independent_groups = find_independent_groups(ready)

    # Take up to max_parallel tickets from different groups
    selected = []
    for group in independent_groups:
        if len(selected) >= max_parallel:
            break
        selected.append(group[0])

    return selected
```

---

## 4. Implementation Workflow

### Single Ticket Implementation
```
┌──────────────────────────────────────────────────────────────────┐
│                    TICKET IMPLEMENTATION FLOW                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 1. Load Ticket  │
                    │    Spec         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ 2. Verify       │
                    │    Dependencies │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Dependencies met?           │
              └──────────────┬──────────────┘
                   No │             │ Yes
                      ▼             ▼
              ┌──────────┐  ┌─────────────────┐
              │ BLOCKED  │  │ 3. Mark         │
              │ Status   │  │    in_progress  │
              └──────────┘  └────────┬────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────┐
                    │ 4. Implementation Phase     │
                    │    - Read acceptance criteria│
                    │    - Read implementation tasks│
                    │    - Create/modify files    │
                    │    - Write tests            │
                    └────────────┬────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │ 5. Validation Phase         │
                    │    - Run tests              │
                    │    - Check coverage         │
                    │    - Run linting            │
                    │    - Run type checking      │
                    └────────────┬────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │ All validations pass?               │
              └──────────────────┬──────────────────┘
                   No │                 │ Yes
                      ▼                 ▼
              ┌──────────────┐  ┌─────────────────┐
              │ 6a. Fix      │  │ 6b. Mark        │
              │     Issues   │  │     completed   │
              └──────┬───────┘  └────────┬────────┘
                     │                   │
                     │                   ▼
                     │         ┌─────────────────┐
                     └────────►│ 7. Git Commit   │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │ 8. Update State │
                               └─────────────────┘
```

---

## 5. Validation Gates

### Gate 1: Pre-Implementation
Before starting implementation:
- [ ] Ticket spec file exists in `docs/mvp/tickets/`
- [ ] All dependencies are marked `completed`
- [ ] Required source directories exist (create if Epic-00 ticket)

### Gate 2: Post-Implementation
After code is written:
```bash
# Run all validations
pytest tests/ -v --tb=short
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70
ruff check src/ tests/
black --check src/ tests/
mypy src/ --strict
```

### Gate 3: Integration Check
For tickets that depend on others:
- [ ] Existing tests still pass
- [ ] No regressions in previously completed functionality
- [ ] Import structure correct (no circular imports)

### Validation Result Actions
| Result | Action |
|--------|--------|
| All pass | Mark completed, commit, continue |
| Tests fail | Fix tests, retry (max 3 attempts) |
| Coverage low | Add more tests, retry |
| Lint fail | Auto-fix with `ruff --fix`, retry |
| Type error | Fix type hints, retry |
| Max attempts | Mark failed, log error, skip to next |

---

## 6. Rollback Procedures

### File-Level Rollback
If implementation fails after creating/modifying files:
```bash
# Revert all uncommitted changes
git checkout -- .

# Or selective revert
git checkout -- src/path/to/file.py
```

### Ticket-Level Rollback
If a completed ticket needs to be undone:
```bash
# Find the commit
git log --oneline --grep="PC-XXX"

# Revert the commit
git revert <commit-hash> --no-commit

# Update state
# Set ticket status back to 'pending'
```

### Session Rollback
If entire session needs to be rolled back:
1. Identify all commits from session (via session_id in commit messages)
2. Revert in reverse chronological order
3. Reset all affected ticket statuses to previous state

---

## 7. Subagent Coordination

### Agent Responsibilities
```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                │
│  - Ticket selection                                                 │
│  - State management                                                 │
│  - Quality gate enforcement                                         │
│  - Rollback coordination                                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ prevcarga-      │  │ prevcarga-      │  │ prevcarga-      │
│ implementer     │  │ tester          │  │ architect       │
│                 │  │                 │  │                 │
│ - Write code    │  │ - Write tests   │  │ - Review code   │
│ - Follow specs  │  │ - Ensure cover  │  │ - Check patterns│
│ - Use patterns  │  │ - Mock external │  │ - Suggest fixes │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ prevcarga-      │  │ prevcarga-      │  │ prevcarga-      │
│ domain-expert   │  │ quality         │  │ docs            │
│                 │  │                 │  │                 │
│ - Domain logic  │  │ - Run linting   │  │ - Docstrings    │
│ - Forecasting   │  │ - Type checking │  │ - README update │
│ - Validation    │  │ - Security scan │  │ - API docs      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Agent Invocation Order
For a typical ticket:
1. **Implementer**: Write the main code
2. **Tester**: Write tests for the code
3. **Quality**: Run all quality checks
4. **Architect**: Review if structural changes (optional)
5. **Domain Expert**: Review if forecasting logic (optional)
6. **Docs**: Update documentation (if public API)

---

## 8. Autonomous Mode Configuration

### Default Settings
```yaml
autonomous:
  max_tickets_per_session: 10
  max_attempts_per_ticket: 3
  stop_on_failure: false          # Continue to next ticket if one fails
  require_review_for_epics:       # Pause for human review
    - Epic-03                     # Models (critical)
    - Epic-06A                    # Reconciliation (critical)
  parallel_implementation: false  # Sequential by default
  auto_commit: true
  commit_message_template: |
    feat({scope}): {description}

    Implements: {ticket_id}
    Session: {session_id}
```

### Safety Limits
```yaml
safety:
  max_files_per_ticket: 20        # Alert if creating too many files
  max_lines_per_file: 1000        # Split if file too large
  max_test_duration_seconds: 300  # Fail if tests take too long
  require_tests: true             # Cannot mark complete without tests
  min_coverage: 70                # Minimum coverage percentage
```

---

## 9. Error Handling

### Error Categories
| Category | Example | Recovery |
|----------|---------|----------|
| `dependency_not_met` | Required ticket not completed | Block, wait |
| `test_failure` | Unit test fails | Fix, retry |
| `coverage_low` | 65% coverage (needs 70%) | Add tests, retry |
| `lint_error` | Unused import | Auto-fix, retry |
| `type_error` | Missing type hint | Fix, retry |
| `import_error` | Circular import | Refactor, retry |
| `file_conflict` | Merge conflict | Alert user, pause |
| `external_failure` | S3 unavailable | Skip affected tests, continue |

### Error Escalation
```
Attempt 1 → Auto-fix if possible → Retry
Attempt 2 → More aggressive fix → Retry
Attempt 3 → Log detailed error → Mark failed → Continue to next
```

---

## 10. Progress Reporting

### Real-Time Updates
During implementation:
```
[10:00:00] Starting: PC-001-00-repository-structure-setup
[10:00:05] Creating directory structure...
[10:00:10] Writing pyproject.toml...
[10:00:15] Writing conftest.py...
[10:00:20] Running tests... PASSED (5/5)
[10:00:25] Coverage: 78% (meets 70% threshold)
[10:00:30] Linting... PASSED
[10:00:35] Type checking... PASSED
[10:00:40] Committing: feat(setup): implement repository structure
[10:00:45] COMPLETED: PC-001-00-repository-structure-setup
```

### Session Summary
At end of session:
```
═══════════════════════════════════════════════════════════════════
                     SESSION SUMMARY
═══════════════════════════════════════════════════════════════════
Session ID: abc123
Duration: 45 minutes
Mode: autonomous

Tickets Attempted: 5
├── Completed: 4
│   ├── PC-001-00-repository-structure-setup (3m)
│   ├── PC-002-00-uv-environment-setup (5m)
│   ├── PC-003-00-storage-backend-abstraction (8m)
│   └── PC-006-00-logging-configuration (7m)
├── Failed: 1
│   └── PC-004-00-s3-storage-implementation
│       └── Error: boto3 mock configuration issue
└── Blocked: 0

Overall Progress: 4/113 (3.5%)
Epic-00 Progress: 4/8 (50%)

Next Actionable:
1. PC-005-00-local-storage-implementation
2. PC-007-00-configuration-management
═══════════════════════════════════════════════════════════════════
```

---

## 11. Integration Points

### Git Integration
- Each completed ticket = 1 commit (squashed if multiple attempts)
- Commit message includes ticket ID for traceability
- Branch strategy: Direct to `main` for Epic-00/01, feature branches for others (configurable)

### CI/CD Integration
- Push triggers GitHub Actions
- If CI fails, ticket marked as `needs_fix`
- CI must pass before marking `completed` in production mode

### External Tools
- **pytest**: Test execution and coverage
- **ruff**: Linting and auto-fixing
- **black**: Code formatting
- **mypy**: Type checking
- **bandit**: Security scanning
- **pre-commit**: Git hooks (optional)

---

## 12. Command Reference

| Command | Description |
|---------|-------------|
| `/scaffold-ticket-system` | Initialize state files and parse all tickets |
| `/implementation-status` | Show current progress and next actionable tickets |
| `/ticket-implementation PC-XXX` | Implement a specific ticket |
| `/run-autonomous-implementation` | Run autonomous mode |
| `/validate-ticket-system` | Validate all completed tickets |
| `/tickets-accomplishment-review` | Generate accomplishment report |
| `/reset-ticket-state [ticket_id\|all]` | Reset ticket(s) to pending |
