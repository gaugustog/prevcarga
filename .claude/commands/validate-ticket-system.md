# Validate Ticket System

Validate the integrity of the PrevCarga ticket implementation system.

## Task

Perform comprehensive validation of:
1. Ticket specifications
2. Implementation state
3. Code quality for completed tickets
4. Dependency graph consistency

### 1. Validate Ticket Specifications

Scan all tickets in `docs/mvp/tickets/`:

#### File Naming
- [ ] Follows pattern: `PC-{number}-{epic}-{name}.md`
- [ ] Number is sequential within epic
- [ ] Epic code matches directory structure

#### Required Sections
For each ticket file, verify presence of:
- [ ] Title with ticket ID
- [ ] Epic reference
- [ ] Story points (optional but recommended)
- [ ] Acceptance criteria (with checkboxes)
- [ ] Implementation tasks (with checkboxes)
- [ ] Dependencies section (if any)
- [ ] Test requirements

#### Content Validation
- [ ] Acceptance criteria are testable
- [ ] Implementation tasks are actionable
- [ ] Code examples follow Python conventions
- [ ] No TODO placeholders in acceptance criteria

### 2. Validate Dependency Graph

Load `.claude/state/dependency-graph.json`:

#### Graph Integrity
- [ ] No circular dependencies
- [ ] All referenced tickets exist
- [ ] No orphan tickets (except Epic-00 starters)
- [ ] Epic ordering respected (earlier epics don't depend on later)

#### Dependency Logic
- [ ] Infrastructure tickets (storage, config) come before users
- [ ] Plugin interfaces before implementations
- [ ] Base classes before derived classes
- [ ] Tests can run for each completed ticket independently

### 3. Validate Completed Tickets

For each ticket marked `completed`:

#### Code Existence
- [ ] All files mentioned in ticket exist
- [ ] Classes/functions mentioned are implemented
- [ ] Imports resolve correctly

#### Test Coverage
```bash
# Run tests for specific ticket's code
pytest tests/ -v --cov=src/{module} --cov-fail-under=70
```

#### Quality Gates
```bash
# Lint check
ruff check src/{module}

# Type check
mypy src/{module} --strict

# Security scan
bandit -r src/{module}
```

#### Git History
- [ ] Commit exists for ticket
- [ ] Commit message references ticket ID
- [ ] No uncommitted changes for ticket files

### 4. Validate State Consistency

#### tickets-status.json
- [ ] All tickets from specs are present
- [ ] Status values are valid (`pending`, `in_progress`, `completed`, `blocked`, `failed`)
- [ ] Timestamps are valid ISO format
- [ ] Completed tickets have `completed_at` set
- [ ] Failed tickets have `last_error` set

#### dependency-graph.json
- [ ] Matches ticket dependencies from spec files
- [ ] Edge count matches expected
- [ ] Epic groupings are correct

#### session-state.json
- [ ] If session active, current_ticket exists
- [ ] Session ID is valid UUID format
- [ ] Completed tickets list matches status file

### 5. Cross-Validation

#### Status vs Dependencies
- [ ] `completed` tickets have all dependencies `completed`
- [ ] `blocked` tickets have at least one dependency not `completed`
- [ ] `in_progress` tickets have all dependencies `completed`

#### Spec vs Implementation
- [ ] Acceptance criteria checkboxes match implementation
- [ ] Implementation tasks checkboxes match actual code
- [ ] Test requirements are covered

### 6. Generate Validation Report

```
═══════════════════════════════════════════════════════════════════
                    TICKET SYSTEM VALIDATION
═══════════════════════════════════════════════════════════════════

Scan Date: 2025-01-17 10:30:00

TICKET SPECIFICATIONS
─────────────────────────────────────────────────────────────────
Total Tickets: 113
├── Valid: 110
├── Warnings: 2
│   ├── PC-045: Missing story points
│   └── PC-078: Vague acceptance criteria
└── Errors: 1
    └── PC-099: Circular dependency detected

DEPENDENCY GRAPH
─────────────────────────────────────────────────────────────────
Nodes: 113
Edges: 156
├── Circular Dependencies: 1 ❌
│   └── PC-041 → PC-043 → PC-041
├── Missing References: 0 ✅
└── Orphan Tickets: 0 ✅

COMPLETED TICKETS VALIDATION
─────────────────────────────────────────────────────────────────
Completed: 25

Quality Check Results:
┌─────────────────┬───────┬───────┬──────────┬───────────┐
│     Ticket      │ Tests │ Cover │   Lint   │   Types   │
├─────────────────┼───────┼───────┼──────────┼───────────┤
│ PC-001          │  ✅   │  85%  │    ✅    │    ✅     │
│ PC-002          │  ✅   │  78%  │    ✅    │    ✅     │
│ PC-003          │  ✅   │  72%  │    ✅    │    ✅     │
│ PC-004          │  ⚠️   │  68%  │    ✅    │    ✅     │
│ ...             │  ...  │  ...  │   ...    │   ...     │
└─────────────────┴───────┴───────┴──────────┴───────────┘

Coverage Issues:
  ⚠️  PC-004: 68% coverage (below 70% threshold)

STATE CONSISTENCY
─────────────────────────────────────────────────────────────────
tickets-status.json:
├── Schema valid: ✅
├── All tickets present: ✅
└── Status consistency: ✅

dependency-graph.json:
├── Schema valid: ✅
├── Matches specs: ✅
└── No stale entries: ✅

session-state.json:
├── Schema valid: ✅
└── No orphan sessions: ✅

CROSS-VALIDATION
─────────────────────────────────────────────────────────────────
Status vs Dependencies: ✅ Consistent
Spec vs Implementation: ⚠️  2 warnings
├── PC-015: Acceptance criterion #3 not verified by tests
└── PC-022: Implementation differs from spec example

═══════════════════════════════════════════════════════════════════
                         SUMMARY
═══════════════════════════════════════════════════════════════════

Overall Status: ⚠️  WARNINGS

Critical Issues: 1
├── Circular dependency: PC-041 ↔ PC-043 (blocks Epic-05B)

Warnings: 4
├── PC-004: Coverage below threshold
├── PC-015: Untested acceptance criterion
├── PC-022: Implementation differs from spec
└── PC-045: Missing story points

Recommendations:
1. Fix circular dependency in PC-041/PC-043 specs
2. Add tests for PC-004 to increase coverage
3. Review PC-015 acceptance criteria #3
4. Update PC-022 spec or implementation to match

Commands:
  /reset-ticket-state PC-004  - Reset for re-implementation
  /ticket-implementation PC-004 - Re-implement with fixes

═══════════════════════════════════════════════════════════════════
```

### 7. Auto-Fix Options

For certain issues, offer auto-fix:

```
Auto-fixable issues found:

1. Lint errors in 3 files
   Run: ruff check --fix src/

2. Format issues in 5 files
   Run: black src/

3. Missing __init__.py in tests/unit/test_models/
   Create empty file

Apply auto-fixes? [y/N]
```

## Notes
- Run this command periodically to ensure system integrity
- Critical issues should be fixed before continuing implementation
- Warnings can be addressed during normal development
- Use `--strict` flag to fail on warnings too
