# Run Autonomous Implementation

Execute autonomous ticket implementation for PrevCarga.

## Arguments
- `$ARGUMENTS` - Optional: number of tickets to implement (default: 5), or "until-blocked"

## Task

Run autonomous implementation mode, continuously implementing tickets until:
- Specified number of tickets completed
- No more actionable tickets available
- A critical error occurs
- User interrupts

### 1. Initialize Session

Create new session in `.claude/state/session-state.json`:
```json
{
  "session_id": "uuid-generated",
  "started_at": "2025-01-17T10:00:00Z",
  "current_ticket": null,
  "tickets_completed_this_session": [],
  "total_implementation_time_minutes": 0,
  "mode": "autonomous",
  "config": {
    "max_tickets": 5,
    "stop_on_failure": false,
    "require_review_epics": ["Epic-03", "Epic-06A"]
  }
}
```

### 2. Parse Arguments
- If `$ARGUMENTS` is a number: set `max_tickets` to that number
- If `$ARGUMENTS` is "until-blocked": continue until no actionable tickets
- If `$ARGUMENTS` is empty: default to 5 tickets

### 3. Load State
- Read `.claude/state/tickets-status.json`
- Read `.claude/state/dependency-graph.json`
- Read `.claude/templates/validation-rules.md`
- Read `.claude/templates/implementation-orchestrator.md`

### 4. Main Implementation Loop

```
WHILE tickets_completed < max_tickets AND actionable_tickets_exist:

    1. SELECT next ticket using priority algorithm:
       - Dependencies satisfied
       - Epic order (earlier first)
       - Ticket number (lower first)
       - Blocking count (more blocking = higher priority)

    2. CHECK if ticket's epic requires review:
       - If in require_review_epics list, PAUSE and ask user
       - User can approve, skip, or stop

    3. IMPLEMENT ticket:
       - Follow /ticket-implementation process
       - Handle success/failure

    4. UPDATE state:
       - Mark ticket completed or failed
       - Update session state
       - Log to implementation-log.md

    5. IF failure AND stop_on_failure:
       - BREAK loop
       - Report failure

    6. REPORT progress after each ticket
```

### 5. Ticket Selection Algorithm

```python
def select_next_ticket():
    # Get all pending tickets with satisfied dependencies
    actionable = []
    for ticket in all_tickets:
        if ticket.status != 'pending':
            continue
        if all(dep.status == 'completed' for dep in ticket.dependencies):
            actionable.append(ticket)

    if not actionable:
        return None

    # Sort by priority
    actionable.sort(key=lambda t: (
        epic_order[t.epic],        # Epic-00 = 0, Epic-11B = 21
        t.ticket_number,           # PC-001 = 1
        -len(t.blocks),            # More blocking = negative = higher
    ))

    return actionable[0]
```

### 6. Progress Reporting

After each ticket:
```
───────────────────────────────────────────────────────────────────
 [2/5] ✅ PC-002-00-uv-environment-setup
───────────────────────────────────────────────────────────────────
 Duration: 4m 32s
 Status: COMPLETED

 Session Progress: [████░░░░░░░░░░░░░░░░] 40% (2/5)
 Overall Progress: [██░░░░░░░░░░░░░░░░░░] 10% (12/113)

 Next: PC-003-00-storage-backend-abstraction
───────────────────────────────────────────────────────────────────
```

### 7. Review Checkpoints

For tickets in critical epics (Epic-03, Epic-06A):
```
═══════════════════════════════════════════════════════════════════
                    ⚠️  REVIEW CHECKPOINT
═══════════════════════════════════════════════════════════════════

About to implement: PC-025-03-model-interface
Epic: 03 - End-to-End Models (CRITICAL)

This epic contains core model implementations that require careful
review. Autonomous mode is paused for your approval.

Options:
  1. [APPROVE] Continue with this ticket
  2. [SKIP] Skip this ticket, continue with others
  3. [STOP] End autonomous mode here

Please respond with your choice.
═══════════════════════════════════════════════════════════════════
```

### 8. Session Completion

When loop ends:
```
═══════════════════════════════════════════════════════════════════
              AUTONOMOUS SESSION COMPLETE
═══════════════════════════════════════════════════════════════════

Session ID: abc123-def456
Duration: 47 minutes
Mode: autonomous

Results:
┌─────────────────────────────────────────────────────────────────┐
│ Attempted: 5 │ Completed: 4 │ Failed: 1 │ Skipped: 0           │
└─────────────────────────────────────────────────────────────────┘

Completed Tickets:
  ✅ PC-001-00-repository-structure-setup (3m 12s)
  ✅ PC-002-00-uv-environment-setup (4m 32s)
  ✅ PC-003-00-storage-backend-abstraction (8m 45s)
  ✅ PC-006-00-logging-configuration (5m 18s)

Failed Tickets:
  ❌ PC-004-00-s3-storage-implementation
     Error: moto mock not configured correctly
     Attempts: 3

Progress Update:
  Before Session: 8/113 (7%)
  After Session:  12/113 (10.6%)
  Gain: +4 tickets (+3.6%)

Epic Progress:
  Epic-00: [████████░░] 75% (6/8)

Unblocked by This Session:
  - PC-005-00-local-storage-implementation
  - PC-007-00-configuration-management
  - PC-009-01-data-loader-interface (Epic-01 now actionable!)

Next Actionable Tickets:
  1. PC-005-00-local-storage-implementation
  2. PC-007-00-configuration-management
  3. PC-008-00-pydantic-schemas

Recommendations:
  ⚠️  Fix PC-004 manually before continuing
      Error suggests moto configuration issue
      See: tests/unit/test_storage/test_s3.py

Commands:
  /implementation-status     - Full status dashboard
  /ticket-implementation PC-004 - Retry failed ticket
  /run-autonomous-implementation 3 - Continue with 3 more

═══════════════════════════════════════════════════════════════════
```

### 9. Error Recovery

If session crashes mid-ticket:
```
⚠️  Previous session detected with incomplete ticket

Session: abc123-def456
Incomplete: PC-003-00-storage-backend-abstraction
Status: in_progress (started 15 minutes ago)

Options:
  1. [RESUME] Continue implementing PC-003
  2. [RESET] Reset PC-003 to pending, start fresh
  3. [SKIP] Mark PC-003 as failed, continue with next

Please respond with your choice.
```

### 10. Update Implementation Log

Append to `.claude/state/implementation-log.md`:
```markdown
### Session: abc123-def456 (2025-01-17)

**Duration**: 47 minutes
**Mode**: autonomous
**Tickets**: 4 completed, 1 failed

| Time | Ticket | Status | Duration | Notes |
|------|--------|--------|----------|-------|
| 10:00 | PC-001 | ✅ | 3m 12s | - |
| 10:04 | PC-002 | ✅ | 4m 32s | - |
| 10:08 | PC-003 | ✅ | 8m 45s | - |
| 10:17 | PC-004 | ❌ | 12m 30s | moto mock error |
| 10:30 | PC-006 | ✅ | 5m 18s | - |

**Errors**:
- PC-004: `AttributeError: 'NoneType' object has no attribute 'put_object'`
  at tests/unit/test_storage/test_s3.py:34
```

## Safety Features

1. **Automatic Pause**: Pauses for review on critical epics
2. **Max Attempts**: Each ticket gets max 3 attempts before failing
3. **Rollback**: Can rollback individual tickets on failure
4. **State Persistence**: All progress saved, can resume after crash
5. **Rate Limiting**: Brief pause between tickets to allow interruption

## Notes
- Press Ctrl+C to interrupt at any time (state is saved)
- Failed tickets don't block siblings, only dependents
- Use `stop_on_failure: true` in config for stricter mode
- Review checkpoints can be disabled by removing epics from config
