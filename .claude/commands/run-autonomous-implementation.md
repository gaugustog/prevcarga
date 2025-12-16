# Run Autonomous Implementation

Execute autonomous ticket implementation for PrevCarga R.

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
    "require_review_epics": ["Epic-03", "Epic-04", "Epic-06"]
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

```r
select_next_ticket <- function(all_tickets) {
  # Get all pending tickets with satisfied dependencies
  actionable <- list()

  for (ticket in all_tickets) {
    if (ticket$status != "pending") next

    deps_satisfied <- all(vapply(
      ticket$dependencies,
      function(dep) dep$status == "completed",
      logical(1)
    ))

    if (deps_satisfied) {
      actionable <- c(actionable, list(ticket))
    }
  }

  if (length(actionable) == 0) return(NULL)

  # Sort by priority: epic order, ticket number, blocking count
  # Epic-01 = 1, Epic-12 = 12
  priority_order <- order(
    vapply(actionable, function(t) t$epic, integer(1)),
    vapply(actionable, function(t) t$ticket_number, integer(1)),
    -vapply(actionable, function(t) length(t$blocks), integer(1))
  )

  actionable[[priority_order[1]]]
}
```

### 6. Progress Reporting

After each ticket:
```
───────────────────────────────────────────────────────────────────
 [2/5] ✅ PC-002-01-hive-partitioning-utilities
───────────────────────────────────────────────────────────────────
 Duration: 4m 32s
 Status: COMPLETED

 Session Progress: [████░░░░░░░░░░░░░░░░] 40% (2/5)
 Overall Progress: [██░░░░░░░░░░░░░░░░░░] 10% (11/106)

 Next: PC-003-01-schema-validators
───────────────────────────────────────────────────────────────────
```

### 7. Review Checkpoints

For tickets in critical epics (Epic-03, Epic-04, Epic-06):
```
═══════════════════════════════════════════════════════════════════
                    ⚠️  REVIEW CHECKPOINT
═══════════════════════════════════════════════════════════════════

About to implement: PC-016-03-base-model
Epic: 03 - Model Layer Infrastructure (CRITICAL)

This epic contains core model R6 classes that require careful
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
  ✅ PC-001-01-data-loader-r6-class (8m 12s)
  ✅ PC-002-01-hive-partitioning-utilities (4m 32s)
  ✅ PC-003-01-schema-validators (6m 45s)
  ✅ PC-005-01-resampling (5m 18s)

Failed Tickets:
  ❌ PC-004-01-missing-value-imputation
     Error: testthat test_impute_linear failed
     Attempts: 3

Progress Update:
  Before Session: 8/106 (7.5%)
  After Session:  12/106 (11.3%)
  Gain: +4 tickets (+3.8%)

Epic Progress:
  Epic-01: [████████░░] 75% (6/8)

Unblocked by This Session:
  - PC-006-01-data-catalog
  - PC-007-01-area-codes
  - PC-009-02-base-feature-plugin (Epic-02 now actionable!)

Next Actionable Tickets:
  1. PC-006-01-data-catalog
  2. PC-007-01-area-codes
  3. PC-008-01-data-layer-tests

Recommendations:
  ⚠️  Fix PC-004 manually before continuing
      Error suggests NA handling issue in imputation
      See: tests/testthat/test-imputation.R

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
Incomplete: PC-003-01-schema-validators
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
| 10:00 | PC-001 | ✅ | 8m 12s | DataLoader R6 class |
| 10:09 | PC-002 | ✅ | 4m 32s | Hive partitioning |
| 10:14 | PC-003 | ✅ | 6m 45s | Schema validators |
| 10:21 | PC-004 | ❌ | 12m 30s | Imputation test failure |
| 10:34 | PC-005 | ✅ | 5m 18s | Resampling utilities |

**Errors**:
- PC-004: `Error in test_that(): Test 'impute_linear handles edge cases' failed`
  at tests/testthat/test-imputation.R:45
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
