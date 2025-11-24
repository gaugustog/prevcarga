# Ticket Implementation

Implement a specific ticket from the PrevCarga backlog.

## Arguments
- `$ARGUMENTS` - Ticket ID to implement (e.g., `PC-001`, `PC-001-00-repository-structure-setup`)

## Task

Implement the specified ticket following the PrevCarga development standards.

### 1. Load Ticket Information
- Parse ticket ID from arguments: `$ARGUMENTS`
- Load ticket spec from `docs/mvp/tickets/PC-{number}-{epic}-{name}.md`
- Load validation rules from `.claude/templates/validation-rules.md`
- Load current status from `.claude/state/tickets-status.json`

### 2. Verify Prerequisites

#### Check Status
- If ticket is `completed`: Ask if user wants to re-implement
- If ticket is `in_progress`: Resume implementation
- If ticket is `blocked`: Show what's blocking and stop
- If ticket is `pending`: Proceed with implementation

#### Check Dependencies
For each dependency listed in the ticket:
- Verify it exists in the dependency graph
- Verify its status is `completed`
- If any dependency not met, show which ones and stop

### 3. Update State
Mark ticket as `in_progress`:
```json
{
  "status": "in_progress",
  "started_at": "2025-01-17T10:00:00Z",
  "attempts": 1
}
```

### 4. Implementation Phase

#### 4.1 Read Ticket Spec
Extract from ticket file:
- **Acceptance Criteria**: List of checkboxes to satisfy
- **Implementation Tasks**: Step-by-step implementation guide
- **Code Examples**: Reference implementations
- **Test Requirements**: What tests to write

#### 4.2 Create/Modify Files
Following the implementation tasks:
- Create new files as specified
- Modify existing files if needed
- Follow patterns from `.claude/templates/validation-rules.md`:
  - Plugin pattern for features/models
  - Registry pattern for discovery
  - Factory pattern for storage
  - Pydantic for configuration

#### 4.3 Write Tests
For each new component:
- Create corresponding test file in `tests/`
- Write unit tests covering:
  - Happy path
  - Edge cases
  - Error conditions
- Aim for 70%+ coverage on new code

### 5. Validation Phase

Run all quality checks:

```bash
# Run tests
pytest tests/ -v --tb=short

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint check
ruff check src/ tests/

# Format check
black --check src/ tests/

# Type check
mypy src/ --strict
```

### 6. Handle Validation Results

#### All Pass
- Mark ticket as `completed`
- Create git commit
- Update dependency graph (unblock dependent tickets)
- Log success

#### Some Fail
- Attempt to fix issues (up to 3 attempts total)
- If lint/format issues: auto-fix with `ruff --fix` and `black`
- If test failures: analyze and fix
- If coverage low: add more tests
- If max attempts reached: mark as `failed`, log error

### 7. Git Commit

Create a conventional commit:
```
feat({scope}): {ticket_description}

Implements acceptance criteria:
- [x] Criterion 1
- [x] Criterion 2

Ticket: {ticket_id}
```

### 8. Update State

On success:
```json
{
  "status": "completed",
  "completed_at": "2025-01-17T10:30:00Z",
  "validation_results": {
    "tests_passed": true,
    "coverage": 78,
    "lint_passed": true,
    "type_check_passed": true
  },
  "git_commits": ["abc123"]
}
```

On failure:
```json
{
  "status": "failed",
  "attempts": 3,
  "last_error": "Test test_xyz failed: AssertionError at line 45"
}
```

### 9. Output Progress

During implementation:
```
═══════════════════════════════════════════════════════════════════
         IMPLEMENTING: PC-025-03-model-interface
═══════════════════════════════════════════════════════════════════

📋 Ticket: Model Interface
📦 Epic: 03 - End-to-End Models
📊 Points: 5
⏱️  Started: 10:00:00

Dependencies:
  ✅ PC-016-02A-feature-plugin-interface

Progress:
  [████████████░░░░░░░░] 60%

Current Task:
  ▶ Writing BaseModel abstract class...

Files Modified:
  ✓ src/models/__init__.py (created)
  ✓ src/models/base.py (created)
  ▶ tests/unit/test_models/test_base.py (in progress)

═══════════════════════════════════════════════════════════════════
```

On completion:
```
═══════════════════════════════════════════════════════════════════
         ✅ COMPLETED: PC-025-03-model-interface
═══════════════════════════════════════════════════════════════════

Duration: 12 minutes
Attempts: 1

Validation Results:
  ✅ Tests: 8 passed, 0 failed
  ✅ Coverage: 82% (threshold: 70%)
  ✅ Linting: No issues
  ✅ Type Check: No errors

Files Created/Modified:
  + src/models/__init__.py
  + src/models/base.py
  + tests/unit/test_models/__init__.py
  + tests/unit/test_models/test_base.py

Git Commit: abc123def
  feat(models): implement BaseModel abstract interface

Now Unblocked:
  - PC-026-03-model-registry
  - PC-027-03-lgbm-model
  - PC-030-03-random-forest-model

Next suggested: /ticket-implementation PC-026
═══════════════════════════════════════════════════════════════════
```

## Error Handling

If ticket file not found:
```
❌ Error: Ticket not found

Could not find ticket spec for: PC-999

Expected location: docs/mvp/tickets/PC-999-*.md

Available tickets matching 'PC-99*':
  - PC-099-...

Run /implementation-status to see all available tickets.
```

If dependencies not met:
```
🚫 Cannot implement: PC-025-03-model-interface

Missing dependencies:
  ⬜ PC-016-02A-feature-plugin-interface (pending)

Implement dependencies first:
  /ticket-implementation PC-016
```

## Notes
- Always read the full ticket spec before implementing
- Follow existing code patterns in the codebase
- Ask for clarification if ticket spec is ambiguous
- Create atomic commits (one commit per ticket)
