# Reset Ticket State

Reset one or more tickets to a previous state in PrevCarga R.

## Arguments
- `$ARGUMENTS` - Ticket ID(s) to reset, or "all" for full reset

## Task

Reset ticket status for re-implementation or state recovery.

### 1. Parse Arguments

Handle different input formats:
- Single ticket: `PC-001` or `PC-001-01-data-loader-r6-class`
- Multiple tickets: `PC-001 PC-002 PC-003`
- Epic reset: `epic:01` (reset all Epic-01 tickets)
- Full reset: `all`

### 2. Confirm Reset Operation

**For single/multiple tickets:**
```
═══════════════════════════════════════════════════════════════════
                    TICKET RESET CONFIRMATION
═══════════════════════════════════════════════════════════════════

You are about to reset the following ticket(s):

  PC-004-01-missing-value-imputation
  ├── Current Status: failed
  ├── Attempts: 3
  ├── Last Error: testthat test_impute_linear failed
  └── Files Modified: 4

This will:
  ✓ Set status to 'pending'
  ✓ Clear attempt count
  ✓ Clear error messages
  ✗ NOT revert file changes
  ✗ NOT undo git commits

Confirm reset? [y/N]
═══════════════════════════════════════════════════════════════════
```

**For epic reset:**
```
═══════════════════════════════════════════════════════════════════
                     EPIC RESET CONFIRMATION
═══════════════════════════════════════════════════════════════════

You are about to reset all tickets in Epic-01:

Tickets to Reset (8):
  ✅ PC-001-01-data-loader-r6-class (completed)
  ✅ PC-002-01-hive-partitioning-utilities (completed)
  ✅ PC-003-01-schema-validators (completed)
  ❌ PC-004-01-missing-value-imputation (failed)
  ⬜ PC-005-01-resampling (pending)
  ✅ PC-006-01-data-catalog (completed)
  ⬜ PC-007-01-area-codes (pending)
  ⬜ PC-008-01-data-layer-tests (pending)

⚠️  WARNING: This will also affect dependent tickets!
    Resetting Epic-01 will cascade to:
    - Epic-02: 7 tickets will become BLOCKED
    - Epic-03: 8 tickets will become BLOCKED
    - Total affected: 80+ tickets

Confirm epic reset? [y/N]
═══════════════════════════════════════════════════════════════════
```

**For full reset:**
```
═══════════════════════════════════════════════════════════════════
                ⚠️  FULL SYSTEM RESET WARNING ⚠️
═══════════════════════════════════════════════════════════════════

You are about to reset ALL tickets in the system!

Current Progress: 58/106 (55%)
Completed Tickets: 58
Implementation Time: ~18.5 hours

This will:
  ✓ Reset all 106 tickets to 'pending'
  ✓ Clear all attempt counts and errors
  ✓ Clear implementation log
  ✗ NOT revert any file changes
  ✗ NOT undo any git commits

Type 'RESET ALL' to confirm (or 'n' to cancel):
═══════════════════════════════════════════════════════════════════
```

### 3. Perform Reset

Update `.claude/state/tickets-status.json`:

```json
{
  "PC-004-01-missing-value-imputation": {
    "status": "pending",
    "started_at": null,
    "completed_at": null,
    "attempts": 0,
    "last_error": null,
    "validation_results": null,
    "git_commits": [],
    "notes": "Reset on 2025-01-17 by user"
  }
}
```

### 4. Update Dependent Tickets

If resetting a completed ticket, cascade to dependents:

```r
cascade_reset <- function(ticket_id, tickets_status, dependency_graph) {
  # Reset the ticket itself
  tickets_status <- reset_ticket(ticket_id, tickets_status)

  # Find all tickets that depend on this one
  dependents <- get_dependents(ticket_id, dependency_graph)

  for (dependent in dependents) {
    if (dependent$status == "completed") {
      # Mark as needing re-validation
      dependent$notes <- sprintf(
        "May need re-validation: %s was reset",
        ticket_id
      )
    } else if (dependent$status == "in_progress") {
      # Block it since dependency is no longer met
      dependent$status <- "blocked"
      dependent$notes <- sprintf(
        "Blocked: dependency %s was reset",
        ticket_id
      )
    }
    tickets_status[[dependent$id]] <- dependent
  }

  tickets_status
}
```

### 5. Optional: Git Revert

Offer to revert associated git commits:

```
═══════════════════════════════════════════════════════════════════
                    GIT REVERT OPTION
═══════════════════════════════════════════════════════════════════

The following git commits are associated with PC-004:

  abc123 feat(data): implement missing value imputation
  def456 test(data): add imputation tests

Would you like to revert these commits? [y/N]

Note: This will create revert commits, not delete history.
═══════════════════════════════════════════════════════════════════
```

If yes:
```bash
git revert abc123 --no-commit
git revert def456 --no-commit
git commit -m "revert: undo PC-004 implementation for re-work

Reverts commits abc123, def456
Reason: Reset for re-implementation"
```

### 6. Log the Reset

Append to `.claude/state/implementation-log.md`:

```markdown
### Reset Operation (2025-01-17 15:30:00)

**Tickets Reset**: PC-004-01-missing-value-imputation
**Previous Status**: failed (3 attempts)
**Reason**: User requested reset for re-implementation
**Git Reverted**: No
**Cascaded Effects**: None (no completed dependents)
```

### 7. Output Reset Confirmation

```
═══════════════════════════════════════════════════════════════════
                    ✅ RESET COMPLETE
═══════════════════════════════════════════════════════════════════

Reset Summary:
  Tickets Reset: 1
  └── PC-004-01-missing-value-imputation

New Status:
  PC-004: pending (was: failed)

Cascade Effects:
  No dependent tickets affected.

Updated Files:
  ✓ .claude/state/tickets-status.json
  ✓ .claude/state/implementation-log.md

Next Steps:
  /ticket-implementation PC-004  - Re-implement the ticket
  /implementation-status         - View current progress

═══════════════════════════════════════════════════════════════════
```

### 8. Reset Options

Support additional options via arguments:

| Option | Description |
|--------|-------------|
| `--hard` | Also revert git commits |
| `--cascade` | Also reset all dependent tickets |
| `--dry-run` | Show what would be reset without doing it |
| `--force` | Skip confirmation prompts |

Examples:
```
/reset-ticket-state PC-004                    # Simple reset
/reset-ticket-state PC-004 --hard             # Reset + git revert
/reset-ticket-state epic:01 --cascade         # Reset epic and dependents
/reset-ticket-state all --dry-run             # Preview full reset
```

## Safety Features

1. **Confirmation Required**: Always ask before resetting
2. **Special Confirm for All**: Must type "RESET ALL" for full reset
3. **Cascade Warning**: Warn about dependent ticket effects
4. **Git Separation**: File revert is optional and explicit
5. **Logging**: All resets are logged with timestamp and reason

## Use Cases

1. **Failed Ticket**: Reset to retry implementation
2. **Wrong Implementation**: Reset to start fresh
3. **Dependency Change**: Reset cascade when specs change
4. **Testing**: Full reset to test autonomous implementation
5. **Branching**: Reset before creating feature branch

## Notes
- Resetting does NOT delete code files
- Use `--hard` flag if you also want git revert
- Consider using a new git branch for re-implementation
- Full reset should be rare; prefer single ticket resets
