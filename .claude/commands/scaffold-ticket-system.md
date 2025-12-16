# Scaffold Ticket System

Initialize the PrevCarga ticket implementation tracking system.

## Task

You are initializing the autonomous ticket implementation system for PrevCarga R. Perform the following steps:

### 1. Create State Directory Structure
```
.claude/state/
├── tickets-status.json
├── dependency-graph.json
├── implementation-log.md
└── session-state.json
```

### 2. Parse All Ticket Files
Scan `docs/mvp/tickets/` for all ticket files matching pattern `PC-*.md`:
- Extract ticket ID (e.g., `PC-001-01-data-loader-r6-class`)
- Extract epic reference (e.g., `01`, `02`, `03`)
- Extract dependencies from `Dependencies` section
- Extract effort estimate if available
- Extract acceptance criteria count

### 3. Build Dependency Graph
Create `dependency-graph.json` with:
```json
{
  "nodes": [
    {
      "id": "PC-001",
      "full_id": "PC-001-01-data-loader-r6-class",
      "epic": "01",
      "name": "DataLoader R6 Class",
      "depends_on": [],
      "blocks": ["PC-004", "PC-005", "PC-006"]
    }
  ],
  "edges": [
    {"from": "PC-001", "to": "PC-004", "type": "dependency"}
  ],
  "epics": {
    "01": {"name": "Data Layer", "tickets": ["PC-001", "PC-002", ...]},
    "02": {"name": "Feature Engineering Infrastructure", "tickets": ["PC-009", "PC-010", ...]},
    "03": {"name": "Model Layer Infrastructure", "tickets": ["PC-016", "PC-017", ...]},
    "04": {"name": "Model Layer Hierarchical", "tickets": ["PC-024", "PC-025", ...]},
    "05": {"name": "Combination Infrastructure", "tickets": ["PC-032", "PC-033", ...]},
    "06": {"name": "Reconciliation Infrastructure", "tickets": ["PC-039", "PC-040", ...]},
    "07": {"name": "Evaluation Layer", "tickets": ["PC-047", "PC-048", ...]},
    "08": {"name": "Orchestrator", "tickets": ["PC-057", "PC-058", ...]},
    "09": {"name": "CLI", "tickets": ["PC-064", "PC-065", ...]},
    "10": {"name": "Testing & Validation", "tickets": ["PC-076", "PC-077", ...]},
    "11": {"name": "One-Line Installer", "tickets": ["PC-086", "PC-087", ...]},
    "12": {"name": "Documentation & Deploy", "tickets": ["PC-098", "PC-099", ...]}
  }
}
```

### 4. Initialize Ticket Status
Create `tickets-status.json` with all tickets set to `pending`:
```json
{
  "tickets": {
    "PC-001-01-data-loader-r6-class": {
      "status": "pending",
      "started_at": null,
      "completed_at": null,
      "attempts": 0,
      "last_error": null,
      "validation_results": null,
      "git_commits": [],
      "notes": ""
    }
  },
  "summary": {
    "total": 106,
    "pending": 106,
    "in_progress": 0,
    "completed": 0,
    "blocked": 0,
    "failed": 0
  },
  "last_updated": "2025-01-17T10:00:00Z"
}
```

### 5. Create Implementation Log
Create `implementation-log.md`:
```markdown
# PrevCarga R Implementation Log

## Session History

| Session ID | Date | Tickets Completed | Notes |
|------------|------|-------------------|-------|

## Detailed Log

### [Date]
- [Timestamp] Action: Description
```

### 6. Initialize Session State
Create `session-state.json`:
```json
{
  "session_id": null,
  "started_at": null,
  "current_ticket": null,
  "tickets_completed_this_session": [],
  "total_implementation_time_minutes": 0,
  "mode": null
}
```

### 7. Validate System
After scaffolding:
- [ ] Verify all ticket files were parsed (106 tickets expected)
- [ ] Verify no circular dependencies exist
- [ ] Verify all referenced dependencies exist
- [ ] Report any orphan tickets (no dependencies, not Epic-01)

### 8. Generate Summary Report
Output a summary showing:
- Total tickets found
- Tickets per epic
- Critical path (longest dependency chain)
- Ready-to-start tickets (no dependencies)

## Expected Output

```
═══════════════════════════════════════════════════════════════════
              TICKET SYSTEM SCAFFOLDED SUCCESSFULLY
═══════════════════════════════════════════════════════════════════

Tickets Parsed: 106
├── Epic-01: Data Layer (8 tickets)
├── Epic-02: Feature Engineering (7 tickets)
├── Epic-03: Model Layer Infrastructure (8 tickets)
├── Epic-04: Model Layer Hierarchical (8 tickets)
├── Epic-05: Combination Infrastructure (7 tickets)
├── Epic-06: Reconciliation Infrastructure (8 tickets)
├── Epic-07: Evaluation Layer (10 tickets)
├── Epic-08: Orchestrator (7 tickets)
├── Epic-09: CLI (12 tickets)
├── Epic-10: Testing & Validation (10 tickets)
├── Epic-11: One-Line Installer (12 tickets)
└── Epic-12: Documentation & Deploy (9 tickets)

Dependency Graph:
├── Total edges: 180
├── Circular dependencies: 0 ✓
├── Missing references: 0 ✓

Ready to Start (no dependencies):
1. PC-007-01-area-codes

Critical Path Length: 12 tickets
Critical Path: PC-007 → PC-003 → PC-001 → PC-004 → PC-008 → ...

State files created:
✓ .claude/state/tickets-status.json
✓ .claude/state/dependency-graph.json
✓ .claude/state/implementation-log.md
✓ .claude/state/session-state.json

Run /implementation-status to see current progress.
Run /ticket-implementation PC-007 to start implementing.
═══════════════════════════════════════════════════════════════════
```

## Notes
- This command should be run once at project start
- Re-running will reset all ticket statuses (use with caution)
- If tickets are added/modified, re-run to update the system
