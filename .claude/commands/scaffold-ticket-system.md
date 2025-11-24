# Scaffold Ticket System

Initialize the PrevCarga ticket implementation tracking system.

## Task

You are initializing the autonomous ticket implementation system for PrevCarga. Perform the following steps:

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
- Extract ticket ID (e.g., `PC-001-00-repository-structure-setup`)
- Extract epic reference (e.g., `00`, `01`, `02A`)
- Extract dependencies from `depends_on` or `blocked_by` fields
- Extract story points if available
- Extract acceptance criteria count

### 3. Build Dependency Graph
Create `dependency-graph.json` with:
```json
{
  "nodes": [
    {
      "id": "PC-001",
      "full_id": "PC-001-00-repository-structure-setup",
      "epic": "00",
      "name": "Repository Structure Setup",
      "depends_on": [],
      "blocks": ["PC-002", "PC-003", "PC-006", "PC-007", "PC-008"]
    }
  ],
  "edges": [
    {"from": "PC-001", "to": "PC-002", "type": "dependency"}
  ],
  "epics": {
    "00": {"name": "Project Foundation & Setup", "tickets": ["PC-001", "PC-002", ...]},
    "01": {"name": "Data Infrastructure Layer", "tickets": ["PC-009", "PC-010", ...]}
  }
}
```

### 4. Initialize Ticket Status
Create `tickets-status.json` with all tickets set to `pending`:
```json
{
  "tickets": {
    "PC-001-00-repository-structure-setup": {
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
    "total": 113,
    "pending": 113,
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
# PrevCarga Implementation Log

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
- [ ] Verify all ticket files were parsed
- [ ] Verify no circular dependencies exist
- [ ] Verify all referenced dependencies exist
- [ ] Report any orphan tickets (no dependencies, not Epic-00)

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

Tickets Parsed: 113
├── Epic-00: 8 tickets
├── Epic-01: 12 tickets
├── Epic-02A: 10 tickets
...

Dependency Graph:
├── Total edges: 150
├── Circular dependencies: 0 ✓
├── Missing references: 0 ✓

Ready to Start (no dependencies):
1. PC-001-00-repository-structure-setup

Critical Path Length: 15 tickets
Critical Path: PC-001 → PC-003 → PC-009 → PC-016 → PC-025 → ...

State files created:
✓ .claude/state/tickets-status.json
✓ .claude/state/dependency-graph.json
✓ .claude/state/implementation-log.md
✓ .claude/state/session-state.json

Run /implementation-status to see current progress.
Run /ticket-implementation PC-001 to start implementing.
═══════════════════════════════════════════════════════════════════
```

## Notes
- This command should be run once at project start
- Re-running will reset all ticket statuses (use with caution)
- If tickets are added/modified, re-run to update the system
