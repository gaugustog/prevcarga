# Ticket Implementation

Implement a specific ticket from the PrevCarga backlog.

## Arguments
- `$ARGUMENTS` - Ticket ID to implement (e.g., `PC-001`, `PC-001-01-data-loader-r6-class`)

## Task

Implement the specified ticket following the PrevCarga R development standards.

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
- Create new files as specified in `R/` directory
- Modify existing files if needed
- Follow patterns from `.claude/templates/validation-rules.md`:
  - R6 class pattern for all components
  - Registry pattern for plugin discovery
  - Factory pattern for storage backends
  - checkmate for input validation

#### 4.3 Write Tests
For each new R6 class or function:
- Create corresponding test file in `tests/testthat/`
- Write unit tests using testthat covering:
  - Happy path
  - Edge cases
  - Error conditions
- Aim for 70%+ coverage on new code

### 5. Validation Phase

Run all quality checks:

```bash
# Run R CMD check
R CMD check . --no-manual --as-cran

# Run tests with testthat
Rscript -e "testthat::test_local(reporter = 'summary')"

# Check coverage
Rscript -e "covr::package_coverage(type = 'tests')"

# Lint check with lintr
Rscript -e "lintr::lint_package()"

# Check documentation
Rscript -e "devtools::document(); devtools::check_man()"
```

### 6. Handle Validation Results

#### All Pass
- Mark ticket as `completed`
- Create git commit
- Update dependency graph (unblock dependent tickets)
- Log success

#### Some Fail
- Attempt to fix issues (up to 3 attempts total)
- If lint issues: review and fix manually (lintr has no auto-fix)
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
    "r_cmd_check_passed": true,
    "tests_passed": true,
    "coverage": 78,
    "lint_passed": true,
    "documentation_complete": true
  },
  "git_commits": ["abc123"]
}
```

On failure:
```json
{
  "status": "failed",
  "attempts": 3,
  "last_error": "Test test_data_loader failed: Error in test_that() at line 45"
}
```

### 9. Output Progress

During implementation:
```
═══════════════════════════════════════════════════════════════════
         IMPLEMENTING: PC-016-03-base-model
═══════════════════════════════════════════════════════════════════

📋 Ticket: BaseModel Abstract Class
📦 Epic: 03 - Model Layer Infrastructure
📊 Effort: 1.5 days
⏱️  Started: 10:00:00

Dependencies:
  ✅ PC-009-02-base-feature-plugin

Progress:
  [████████████░░░░░░░░] 60%

Current Task:
  ▶ Writing BaseModel R6 class...

Files Modified:
  ✓ R/models/base.R (created)
  ✓ R/models/registry.R (created)
  ▶ tests/testthat/test-models-base.R (in progress)

═══════════════════════════════════════════════════════════════════
```

On completion:
```
═══════════════════════════════════════════════════════════════════
         ✅ COMPLETED: PC-016-03-base-model
═══════════════════════════════════════════════════════════════════

Duration: 12 minutes
Attempts: 1

Validation Results:
  ✅ R CMD check: 0 errors, 0 warnings, 0 notes
  ✅ Tests: 8 passed, 0 failed
  ✅ Coverage: 82% (threshold: 70%)
  ✅ Linting: No issues
  ✅ Documentation: Complete

Files Created/Modified:
  + R/models/base.R
  + R/models/registry.R
  + tests/testthat/test-models-base.R
  ~ NAMESPACE (updated)

Git Commit: abc123def
  feat(models): implement BaseModel R6 abstract class

Now Unblocked:
  - PC-017-03-model-registry
  - PC-019-03-model-artifact
  - PC-020-03-universal-trainer

Next suggested: /ticket-implementation PC-017
═══════════════════════════════════════════════════════════════════
```

## Error Handling

If ticket file not found:
```
❌ Error: Ticket not found

Could not find ticket spec for: PC-999

Expected location: docs/mvp/tickets/PC-999-*.md

Available tickets matching 'PC-99*':
  - PC-099-12-sphinx-documentation-site

Run /implementation-status to see all available tickets.
```

If dependencies not met:
```
🚫 Cannot implement: PC-016-03-base-model

Missing dependencies:
  ⬜ PC-009-02-base-feature-plugin (pending)

Implement dependencies first:
  /ticket-implementation PC-009
```

## Notes
- Always read the full ticket spec before implementing
- Follow existing code patterns in the R/ directory
- Use R6 classes for all major components
- Use roxygen2 comments for documentation
- Ask for clarification if ticket spec is ambiguous
- Create atomic commits (one commit per ticket)
