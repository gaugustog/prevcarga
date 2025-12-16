# Implementation Tickets

This directory contains detailed implementation tickets for the PrevCargaONS MVP.

## Ticket Naming Convention

```
PC-{NNN}-{EE}-{description}.md
```

- `PC` - PrevCarga prefix
- `NNN` - Sequential ticket number (001-999)
- `EE` - Epic number (01-12)
- `description` - Kebab-case description

## Tickets by Epic

### EPIC-01: Data Layer

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-001-01](PC-001-01-data-loader-r6-class.md) | DataLoader R6 Class | High | 3 days | Pending |
| [PC-002-01](PC-002-01-hive-partitioning-utilities.md) | Hive Partitioning Utilities | High | 1 day | Pending |
| [PC-003-01](PC-003-01-schema-validators.md) | Schema Validators | High | 2 days | Pending |
| [PC-004-01](PC-004-01-missing-value-imputation.md) | Missing Value Imputation | High | 2 days | Pending |
| [PC-005-01](PC-005-01-resampling.md) | Resampling | Medium | 1 day | Pending |
| [PC-006-01](PC-006-01-data-catalog.md) | DataCatalog | Medium | 1 day | Pending |
| [PC-007-01](PC-007-01-area-codes.md) | Area Codes | High | 0.5 days | Pending |
| [PC-008-01](PC-008-01-data-layer-tests.md) | Data Layer Tests | High | 2 days | Pending |

**Total Effort:** ~12.5 days (2 weeks)

---

### EPIC-02: Feature Engineering Infrastructure

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-009-02](PC-009-02-base-feature-plugin.md) | BaseFeaturePlugin Abstract Class | High | 1 day | Pending |
| [PC-010-02](PC-010-02-feature-plugin-registry.md) | FeaturePluginRegistry | High | 1 day | Pending |
| [PC-011-02](PC-011-02-feature-pipeline.md) | FeaturePipeline | High | 1.5 days | Pending |
| [PC-012-02](PC-012-02-feature-configuration.md) | Feature Configuration Schema | Medium | 1 day | Pending |
| [PC-013-02](PC-013-02-feature-evaluator.md) | FeatureEvaluator Structure | Medium | 1 day | Pending |
| [PC-014-02](PC-014-02-feature-utilities.md) | Common Feature Utilities | Medium | 1 day | Pending |
| [PC-015-02](PC-015-02-feature-tests.md) | Feature Infrastructure Tests | High | 1.5 days | Pending |

**Total Effort:** ~8 days (~1.5 weeks)

---

### EPIC-03: Model Layer - Infrastructure

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-016-03](PC-016-03-base-model.md) | BaseModel Abstract Class | High | 1.5 days | Pending |
| [PC-017-03](PC-017-03-model-registry.md) | ModelRegistry | High | 1 day | Pending |
| [PC-018-03](PC-018-03-semantic-versioning.md) | Semantic Versioning | Medium | 0.5 days | Pending |
| [PC-019-03](PC-019-03-model-artifact.md) | ModelArtifact | High | 1 day | Pending |
| [PC-020-03](PC-020-03-universal-trainer.md) | UniversalTrainer | High | 2 days | Pending |
| [PC-021-03](PC-021-03-model-configuration.md) | Model Configuration Schema | Medium | 1 day | Pending |
| [PC-022-03](PC-022-03-model-storage.md) | Model Storage Utilities | Medium | 0.5 days | Pending |
| [PC-023-03](PC-023-03-model-tests.md) | Model Infrastructure Tests | High | 2 days | Pending |

**Total Effort:** ~9.5 days (~2 weeks)

---

### EPIC-04: Model Layer - Hierarchical Support

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-024-04](PC-024-04-hierarchical-model.md) | HierarchicalModel Base Class | High | 1.5 days | Pending |
| [PC-025-04](PC-025-04-multi-model-manager.md) | MultiModelManager Pattern | High | 1.5 days | Pending |
| [PC-026-04](PC-026-04-parallel-training.md) | Parallel Training Infrastructure | High | 1.5 days | Pending |
| [PC-027-04](PC-027-04-inference-workflow.md) | Inference Workflow Infrastructure | High | 1.5 days | Pending |
| [PC-028-04](PC-028-04-two-stage-inference.md) | Two-Stage Inference Support | High | 1.5 days | Pending |
| [PC-029-04](PC-029-04-batch-inference.md) | Batch Inference Runner | Medium | 1 day | Pending |
| [PC-030-04](PC-030-04-feature-selection.md) | Dynamic Feature Selection | Medium | 1 day | Pending |
| [PC-031-04](PC-031-04-hierarchical-tests.md) | Hierarchical Model Tests | High | 2 days | Pending |

**Total Effort:** ~11.5 days (~2.5 weeks)

---

### EPIC-05: Combination Infrastructure

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-032-05](PC-032-05-base-combiner.md) | BaseCombiner Abstract Class | High | 1 day | Pending |
| [PC-033-05](PC-033-05-combiner-registry.md) | CombinerRegistry | High | 1 day | Pending |
| [PC-034-05](PC-034-05-combination-workflow.md) | CombinationWorkflow | High | 1.5 days | Pending |
| [PC-035-05](PC-035-05-bias-correction.md) | Bias Correction Infrastructure | Medium | 1 day | Pending |
| [PC-036-05](PC-036-05-weight-optimizer.md) | Weight Optimizer Framework | Medium | 1 day | Pending |
| [PC-037-05](PC-037-05-combination-config.md) | Combination Configuration Schema | Medium | 0.5 days | Pending |
| [PC-038-05](PC-038-05-combination-tests.md) | Combination Infrastructure Tests | High | 1.5 days | Pending |

**Total Effort:** ~7.5 days (~1.5 weeks)

---

### EPIC-06: Reconciliation Infrastructure

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-039-06](PC-039-06-base-reconciler.md) | BaseReconciler Abstract Class | High | 1 day | Pending |
| [PC-040-06](PC-040-06-reconciler-registry.md) | ReconcilerRegistry | High | 0.5 days | Pending |
| [PC-041-06](PC-041-06-hierarchy-builder.md) | HierarchyBuilder | High | 1.5 days | Pending |
| [PC-042-06](PC-042-06-reconciliation-workflow.md) | ReconciliationWorkflow | High | 1.5 days | Pending |
| [PC-043-06](PC-043-06-loss-calculator.md) | LossCalculator | High | 1 day | Pending |
| [PC-044-06](PC-044-06-hierarchy-config.md) | Hierarchy Configuration Schema | Medium | 0.5 days | Pending |
| [PC-045-06](PC-045-06-coherence-validator.md) | Coherence Validator | High | 0.5 days | Pending |
| [PC-046-06](PC-046-06-reconciliation-tests.md) | Reconciliation Infrastructure Tests | High | 2 days | Pending |

**Total Effort:** ~8.5 days (~2 weeks)

---

### EPIC-07: Evaluation Layer

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-047-07](PC-047-07-metrics-calculator.md) | Metrics Calculator | High | 1 day | Pending |
| [PC-048-07](PC-048-07-metrics-by-period.md) | Metrics by Period (Patamar) | High | 1 day | Pending |
| [PC-049-07](PC-049-07-metrics-by-horizon.md) | Metrics by Horizon | High | 0.5 days | Pending |
| [PC-050-07](PC-050-07-drift-detector.md) | DriftDetector | High | 1.5 days | Pending |
| [PC-051-07](PC-051-07-model-comparator.md) | ModelComparator | High | 1.5 days | Pending |
| [PC-052-07](PC-052-07-html-reporter.md) | HTMLReporter | Medium | 1 day | Pending |
| [PC-053-07](PC-053-07-csv-reporter.md) | CSVReporter | Medium | 0.5 days | Pending |
| [PC-054-07](PC-054-07-highcharter-reporter.md) | HighcharterReporter | Medium | 1.5 days | Pending |
| [PC-055-07](PC-055-07-highcharter-templates.md) | Highcharter Report Templates | Medium | 2 days | Pending |
| [PC-056-07](PC-056-07-evaluation-tests.md) | Evaluation Infrastructure Tests | High | 2 days | Pending |

**Total Effort:** ~12.5 days (~2.5 weeks)

---

### EPIC-08: Orchestrator

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-057-08](PC-057-08-config-manager.md) | ConfigManager | High | 1.5 days | Pending |
| [PC-058-08](PC-058-08-train-workflow.md) | TrainWorkflow | High | 2.5 days | Pending |
| [PC-059-08](PC-059-08-predict-workflow.md) | PredictWorkflow | High | 2 days | Pending |
| [PC-060-08](PC-060-08-backtest-workflow.md) | BacktestWorkflow | High | 2.5 days | Pending |
| [PC-061-08](PC-061-08-parallel-executor.md) | ParallelExecutor | High | 1.5 days | Pending |
| [PC-062-08](PC-062-08-structured-logger.md) | StructuredLogger | Medium | 1.5 days | Pending |
| [PC-063-08](PC-063-08-orchestrator-tests.md) | Orchestrator Tests | High | 2 days | Pending |

**Total Effort:** ~13.5 days (~3 weeks)

---

### EPIC-09: CLI

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-064-09](PC-064-09-cli-entry-point.md) | CLI Entry Point | High | 1 day | Pending |
| [PC-065-09](PC-065-09-train-command.md) | Train Command | High | 1.5 days | Pending |
| [PC-066-09](PC-066-09-predict-command.md) | Predict Command | High | 1.5 days | Pending |
| [PC-067-09](PC-067-09-backtest-command.md) | Backtest Command | High | 1.5 days | Pending |
| [PC-068-09](PC-068-09-feature-commands.md) | Feature Commands | Medium | 1 day | Pending |
| [PC-069-09](PC-069-09-evaluation-commands.md) | Evaluation Commands | Medium | 1.5 days | Pending |
| [PC-070-09](PC-070-09-interactive-shell.md) | Interactive Shell | Medium | 2 days | Pending |
| [PC-071-09](PC-071-09-display-utilities.md) | Display Utilities | Medium | 1 day | Pending |
| [PC-072-09](PC-072-09-input-validators.md) | Input Validators | Medium | 0.5 days | Pending |
| [PC-073-09](PC-073-09-shell-entry-script.md) | Shell Entry Script | High | 0.5 days | Pending |
| [PC-074-09](PC-074-09-report-command.md) | Report Command | Medium | 1.5 days | Pending |
| [PC-075-09](PC-075-09-cli-tests.md) | CLI Tests | High | 2 days | Pending |

**Total Effort:** ~15.5 days (~2 weeks)

---

### EPIC-10: Testing & Validation

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-076-10](PC-076-10-baseline-validation-framework.md) | Baseline Validation Framework | High | 2 days | Pending |
| [PC-077-10](PC-077-10-model-results-reproduction.md) | Model Results Reproduction | High | 3 days | Pending |
| [PC-078-10](PC-078-10-1year-backtest.md) | 1-Year Backtest Execution | High | 3 days | Pending |
| [PC-079-10](PC-079-10-combination-validation.md) | Combination Validation | High | 2 days | Pending |
| [PC-080-10](PC-080-10-reconciliation-validation.md) | Reconciliation Validation | High | 2 days | Pending |
| [PC-081-10](PC-081-10-performance-benchmarking.md) | Performance Benchmarking | High | 2 days | Pending |
| [PC-082-10](PC-082-10-test-coverage-verification.md) | Test Coverage Verification | High | 1 day | Pending |
| [PC-083-10](PC-083-10-integration-test-suite.md) | Integration Test Suite | High | 2 days | Pending |
| [PC-084-10](PC-084-10-edge-case-testing.md) | Edge Case Testing | High | 2 days | Pending |
| [PC-085-10](PC-085-10-validation-report.md) | Validation Report | High | 2 days | Pending |

**Total Effort:** ~21 days (~3 weeks)

---

### EPIC-11: One-Line Installer

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-086-11](PC-086-11-installer-script-structure.md) | Installer Script Structure | High | 0.5 days | Pending |
| [PC-087-11](PC-087-11-os-detection.md) | OS Detection | High | 0.5 days | Pending |
| [PC-088-11](PC-088-11-r-installation.md) | R Installation | High | 1 day | Pending |
| [PC-089-11](PC-089-11-directory-structure.md) | Directory Structure | High | 0.5 days | Pending |
| [PC-090-11](PC-090-11-shell-extraction.md) | Shell Extraction | High | 0.5 days | Pending |
| [PC-091-11](PC-091-11-launcher-script.md) | Launcher Script | High | 0.5 days | Pending |
| [PC-092-11](PC-092-11-renv-configuration.md) | renv Configuration | High | 1 day | Pending |
| [PC-093-11](PC-093-11-path-configuration.md) | PATH Configuration | High | 0.5 days | Pending |
| [PC-094-11](PC-094-11-default-configuration.md) | Default Configuration | Medium | 0.5 days | Pending |
| [PC-095-11](PC-095-11-success-message.md) | Success Message | Medium | 0.25 days | Pending |
| [PC-096-11](PC-096-11-build-script.md) | Build Script | High | 0.5 days | Pending |
| [PC-097-11](PC-097-11-installer-tests.md) | Installer Tests | High | 1 day | Pending |

**Total Effort:** ~7.25 days (~1 week)

---

### EPIC-12: Documentation & Deploy

| Ticket | Title | Priority | Effort | Status |
|--------|-------|----------|--------|--------|
| [PC-098-12](PC-098-12-complete-readme.md) | Complete README | High | 1 day | Pending |
| [PC-099-12](PC-099-12-sphinx-documentation-site.md) | Sphinx Documentation Site | High | 1 day | Pending |
| [PC-100-12](PC-100-12-documentation-pages.md) | Documentation Pages | High | 3 days | Pending |
| [PC-101-12](PC-101-12-docker-installation-documentation.md) | Docker Installation Documentation | Medium | 0.5 days | Pending |
| [PC-102-12](PC-102-12-wsl-installation-documentation.md) | WSL Installation Documentation | Medium | 0.5 days | Pending |
| [PC-103-12](PC-103-12-dockerfile.md) | Dockerfile | High | 1 day | Pending |
| [PC-104-12](PC-104-12-docker-compose.md) | Docker Compose | High | 0.5 days | Pending |
| [PC-105-12](PC-105-12-github-actions-ci.md) | GitHub Actions CI | High | 1 day | Pending |
| [PC-106-12](PC-106-12-github-actions-release.md) | GitHub Actions Release | High | 1 day | Pending |

**Total Effort:** ~9.5 days (~2 weeks)

---

## Epic Coverage

| Epic | Tickets | Status |
|------|---------|--------|
| EPIC-00: Infrastructure | - | Not Started |
| EPIC-01: Data Layer | 8 | Tickets Created |
| EPIC-02: Feature Engineering | 7 | Tickets Created |
| EPIC-03: Model Layer Infrastructure | 8 | Tickets Created |
| EPIC-04: Model Layer Hierarchical | 8 | Tickets Created |
| EPIC-05: Combination Infrastructure | 7 | Tickets Created |
| EPIC-06: Hierarchical Reconciliation | 8 | Tickets Created |
| EPIC-07: Evaluation Layer | 10 | Tickets Created |
| EPIC-08: Orchestrator | 7 | Tickets Created |
| EPIC-09: CLI | 12 | Tickets Created |
| EPIC-10: Testing & Validation | 10 | Tickets Created |
| EPIC-11: One-Line Installer | 12 | Tickets Created |
| EPIC-12: Documentation & Deploy | 9 | Tickets Created |

---

## Ticket Status Legend

| Status | Description |
|--------|-------------|
| Pending | Not yet started |
| In Progress | Currently being worked on |
| Review | Code complete, awaiting review |
| Done | Merged to develop branch |

---

## Dependencies

### EPIC-01: Data Layer
```
PC-007-01 (Area Codes)
    ↓
PC-003-01 (Schema Validators)
    ↓
PC-002-01 (Hive Utilities)
    ↓
PC-001-01 (DataLoader)
    ↓
PC-006-01 (DataCatalog)
    ↓
PC-004-01 (Imputation) ─── PC-005-01 (Resampling)
    ↓
PC-008-01 (Tests)
```

### EPIC-02: Feature Engineering
```
PC-009-02 (BaseFeaturePlugin)
    ↓
PC-010-02 (Registry) ─── PC-014-02 (Utilities)
    ↓
PC-011-02 (Pipeline)
    ↓
PC-012-02 (Configuration) ─── PC-013-02 (Evaluator)
    ↓
PC-015-02 (Tests)
```

### EPIC-03: Model Layer Infrastructure
```
PC-016-03 (BaseModel)
    ↓
PC-017-03 (Registry) ─── PC-018-03 (Versioning)
    ↓
PC-019-03 (Artifact) ─── PC-022-03 (Storage)
    ↓
PC-021-03 (Configuration)
    ↓
PC-020-03 (UniversalTrainer)
    ↓
PC-023-03 (Tests)
```

### EPIC-04: Model Layer Hierarchical
```
PC-024-04 (HierarchicalModel) ─── PC-025-04 (MultiModelManager)
    ↓                                   ↓
PC-026-04 (ParallelTrainer) ────────────┘
    ↓
PC-027-04 (InferenceWorkflow)
    ↓
PC-028-04 (TwoStageInference) ─── PC-029-04 (BatchInference)
    ↓
PC-030-04 (FeatureSelection)
    ↓
PC-031-04 (Tests)
```

### EPIC-05: Combination Infrastructure
```
PC-032-05 (BaseCombiner)
    ↓
PC-033-05 (CombinerRegistry)
    ↓
PC-034-05 (CombinationWorkflow)
    ↓
PC-035-05 (BiasCorrection) ─── PC-036-05 (WeightOptimizer)
    ↓
PC-037-05 (Configuration)
    ↓
PC-038-05 (Tests)
```

### EPIC-06: Reconciliation Infrastructure
```
PC-039-06 (BaseReconciler)
    ↓
PC-040-06 (ReconcilerRegistry)
    ↓
PC-041-06 (HierarchyBuilder) ─── PC-044-06 (Configuration)
    ↓
PC-043-06 (LossCalculator) ─── PC-045-06 (CoherenceValidator)
    ↓
PC-042-06 (ReconciliationWorkflow)
    ↓
PC-046-06 (Tests)
```

### EPIC-07: Evaluation Layer
```
PC-047-07 (MetricsCalculator)
    ↓
PC-048-07 (MetricsByPeriod) ─── PC-049-07 (MetricsByHorizon)
    ↓
PC-050-07 (DriftDetector) ─── PC-051-07 (ModelComparator)
    ↓
PC-052-07 (HTMLReporter) ─── PC-053-07 (CSVReporter)
    ↓
PC-054-07 (HighcharterReporter)
    ↓
PC-055-07 (ReportTemplates)
    ↓
PC-056-07 (Tests)
```

### EPIC-08: Orchestrator
```
PC-057-08 (ConfigManager) ─── PC-062-08 (StructuredLogger)
    ↓                              ↓
PC-061-08 (ParallelExecutor) ──────┘
    ↓
PC-058-08 (TrainWorkflow)
    ↓
PC-059-08 (PredictWorkflow) ─── PC-060-08 (BacktestWorkflow)
    ↓                                   ↓
    └───────────────────────────────────┘
                    ↓
           PC-063-08 (Tests)
```

### EPIC-09: CLI
```
PC-064-09 (Entry Point) ─── PC-072-09 (Validators)
    ↓                              ↓
PC-071-09 (Display) ───────────────┘
    ↓
PC-065-09 (Train) ─── PC-066-09 (Predict) ─── PC-067-09 (Backtest)
    ↓                      ↓                        ↓
PC-068-09 (Features) ── PC-069-09 (Evaluation) ────┘
    ↓
PC-074-09 (Report) ─── PC-070-09 (Shell)
    ↓                      ↓
PC-073-09 (Entry Script) ──┘
    ↓
PC-075-09 (Tests)
```

### EPIC-10: Testing & Validation
```
PC-076-10 (BaselineValidator)
    ↓
PC-077-10 (ModelReproduction)
    ↓
PC-078-10 (YearBacktest) ─── PC-079-10 (CombinationValidation)
    ↓                              ↓
PC-080-10 (ReconciliationValidation) ──┘
    ↓
PC-081-10 (PerformanceBenchmark) ─── PC-082-10 (CoverageVerification)
    ↓                                       ↓
PC-083-10 (IntegrationTests) ── PC-084-10 (EdgeCases)
    ↓                                  ↓
    └──────────────────────────────────┘
                    ↓
           PC-085-10 (ValidationReport)
```

### EPIC-11: One-Line Installer
```
PC-086-11 (ScriptStructure)
    ↓
PC-087-11 (OSDetection)
    ↓
PC-088-11 (RInstallation)
    ↓
PC-089-11 (DirectoryStructure)
    ↓
PC-090-11 (ShellExtraction) ─── PC-091-11 (LauncherScript)
    ↓                                  ↓
PC-092-11 (renvConfiguration) ─────────┘
    ↓
PC-093-11 (PATHConfiguration) ─── PC-094-11 (DefaultConfiguration)
    ↓                                      ↓
PC-095-11 (SuccessMessage) ────────────────┘
    ↓
PC-096-11 (BuildScript)
    ↓
PC-097-11 (InstallerTests)
```

### EPIC-12: Documentation & Deploy
```
PC-098-12 (README)
    ↓
PC-099-12 (SphinxSetup)
    ↓
PC-100-12 (DocPages) ─── PC-101-12 (DockerDocs) ─── PC-102-12 (WSLDocs)
    ↓
PC-103-12 (Dockerfile)
    ↓
PC-104-12 (DockerCompose)
    ↓
PC-105-12 (GitHubCI)
    ↓
PC-106-12 (GitHubRelease)
```

---

## Quick Start

### EPIC-01: Data Layer

1. Start with foundational tickets:
   - PC-007-01: Area Codes (defines constants used everywhere)
   - PC-002-01: Hive Utilities (used by DataLoader)

2. Build core components:
   - PC-003-01: Schema Validators
   - PC-001-01: DataLoader

3. Add preprocessing:
   - PC-004-01: Imputation
   - PC-005-01: Resampling

4. Integrate:
   - PC-006-01: DataCatalog
   - PC-008-01: Tests

### EPIC-02: Feature Engineering

1. Start with base infrastructure:
   - PC-009-02: BaseFeaturePlugin (foundation class)
   - PC-014-02: Common Utilities (helpers for plugins)

2. Build registry and pipeline:
   - PC-010-02: FeaturePluginRegistry
   - PC-011-02: FeaturePipeline

3. Add configuration and evaluation:
   - PC-012-02: Feature Configuration
   - PC-013-02: FeatureEvaluator

4. Complete with tests:
   - PC-015-02: Feature Tests

### EPIC-03: Model Layer

1. Start with base infrastructure:
   - PC-016-03: BaseModel (foundation class)
   - PC-018-03: Semantic Versioning

2. Build registry and storage:
   - PC-017-03: ModelRegistry
   - PC-022-03: Model Storage Utilities

3. Add artifacts and configuration:
   - PC-019-03: ModelArtifact
   - PC-021-03: Model Configuration

4. Implement training workflow:
   - PC-020-03: UniversalTrainer

5. Complete with tests:
   - PC-023-03: Model Tests

### EPIC-04: Model Layer - Hierarchical Support

1. Start with base hierarchical patterns:
   - PC-024-04: HierarchicalModel (DM→Profile pattern)
   - PC-025-04: MultiModelManager (216-model pattern)

2. Add parallel training:
   - PC-026-04: ParallelTrainer (future-based parallelism)

3. Build inference infrastructure:
   - PC-027-04: InferenceWorkflow (base workflow)
   - PC-028-04: TwoStageInference (D+0/D+1 pattern)
   - PC-029-04: BatchInferenceRunner (multi-area inference)

4. Add feature selection:
   - PC-030-04: Dynamic Feature Selection (horizon-aware)

5. Complete with tests:
   - PC-031-04: Hierarchical Model Tests

### EPIC-05: Combination Infrastructure

1. Start with base combiner:
   - PC-032-05: BaseCombiner (abstract class)
   - PC-033-05: CombinerRegistry (plugin management)

2. Build combination workflow:
   - PC-034-05: CombinationWorkflow (orchestration)

3. Add optimization components:
   - PC-035-05: BiasCorrection (post-combination correction)
   - PC-036-05: WeightOptimizer (weight computation)

4. Add configuration:
   - PC-037-05: Combination Configuration Schema

5. Complete with tests:
   - PC-038-05: Combination Infrastructure Tests

### EPIC-06: Reconciliation Infrastructure

1. Start with base reconciler:
   - PC-039-06: BaseReconciler (abstract class)
   - PC-040-06: ReconcilerRegistry (plugin management)

2. Build hierarchy structure:
   - PC-041-06: HierarchyBuilder (summing matrix S)
   - PC-044-06: Hierarchy Configuration Schema

3. Add calculation and validation:
   - PC-043-06: LossCalculator (by difference)
   - PC-045-06: CoherenceValidator (constraint checking)

4. Build reconciliation workflow:
   - PC-042-06: ReconciliationWorkflow (orchestration)

5. Complete with tests:
   - PC-046-06: Reconciliation Infrastructure Tests

### EPIC-07: Evaluation Layer

1. Start with core metrics:
   - PC-047-07: MetricsCalculator (MAPE, MAE, RMSE)

2. Add period and horizon breakdowns:
   - PC-048-07: Metrics by Period (patamar classification)
   - PC-049-07: Metrics by Horizon (D+0 to D+8)

3. Build analysis tools:
   - PC-050-07: DriftDetector (performance monitoring)
   - PC-051-07: ModelComparator (statistical tests)

4. Add reporting:
   - PC-052-07: HTMLReporter (static reports)
   - PC-053-07: CSVReporter (data export)
   - PC-054-07: HighcharterReporter (interactive charts)

5. Create report templates:
   - PC-055-07: Highcharter Report Templates (5 templates)

6. Complete with tests:
   - PC-056-07: Evaluation Infrastructure Tests

### EPIC-08: Orchestrator

1. Start with foundational components:
   - PC-057-08: ConfigManager (YAML loading, validation)
   - PC-062-08: StructuredLogger (JSON logging, context tracking)

2. Add parallel execution:
   - PC-061-08: ParallelExecutor (future-based parallelism)

3. Build workflow orchestrators:
   - PC-058-08: TrainWorkflow (multi-area, multi-model training)
   - PC-059-08: PredictWorkflow (batch and intraday modes)
   - PC-060-08: BacktestWorkflow (walk-forward validation)

4. Complete with tests:
   - PC-063-08: Orchestrator Tests

### EPIC-09: CLI

1. Start with foundation:
   - PC-064-09: CLI Entry Point (command routing, help)
   - PC-072-09: Input Validators (dates, areas, models)
   - PC-071-09: Display Utilities (progress bars, colors)

2. Build core commands:
   - PC-065-09: Train Command (model training)
   - PC-066-09: Predict Command (batch/intraday)
   - PC-067-09: Backtest Command (walk-forward)

3. Add utility commands:
   - PC-068-09: Feature Commands (gen-features, eval-features)
   - PC-069-09: Evaluation Commands (eval-model, combine)
   - PC-074-09: Report Command (highcharter reports)

4. Build interactive experience:
   - PC-070-09: Interactive Shell (REPL mode)
   - PC-073-09: Shell Entry Script (installation, completion)

5. Complete with tests:
   - PC-075-09: CLI Tests

### EPIC-10: Testing & Validation

1. Start with baseline validation framework:
   - PC-076-10: BaselineValidator (compare against legacy results)
   - PC-077-10: ModelReproduction (validate contributed plugins)

2. Execute comprehensive backtest:
   - PC-078-10: YearBacktest (1-year 2024 backtest)
   - PC-079-10: CombinationValidation (verify combiners)
   - PC-080-10: ReconciliationValidation (verify hierarchical consistency)

3. Run performance and coverage validation:
   - PC-081-10: PerformanceBenchmark (training <30min, prediction <5min)
   - PC-082-10: CoverageVerification (≥70% overall, ≥80% critical)

4. Execute test suites:
   - PC-083-10: IntegrationTests (end-to-end workflows)
   - PC-084-10: EdgeCaseTesting (missing data, DST, holidays)

5. Generate validation report:
   - PC-085-10: ValidationReport (comprehensive HTML report)

### EPIC-11: One-Line Installer

1. Start with script foundation:
   - PC-086-11: Installer Script Structure (bash setup, color output)
   - PC-087-11: OS Detection (Linux distros, macOS)

2. Implement R installation:
   - PC-088-11: R Installation (apt, dnf, yum, brew)

3. Create directory and file structure:
   - PC-089-11: Directory Structure (~/.prevcarga hierarchy)
   - PC-090-11: Shell Extraction (base64 decode R shell)
   - PC-091-11: Launcher Script (prevcarga wrapper)

4. Configure environment:
   - PC-092-11: renv Configuration (reproducible packages)
   - PC-093-11: PATH Configuration (shell RC files, completions)
   - PC-094-11: Default Configuration (config.yaml generation)

5. Finalize and test:
   - PC-095-11: Success Message (colored banner, next steps)
   - PC-096-11: Build Script (embed shell, calculate checksums)
   - PC-097-11: Installer Tests (bats tests, Docker integration)

### EPIC-12: Documentation & Deploy

1. Start with core documentation:
   - PC-098-12: Complete README (badges, installation, usage)
   - PC-099-12: Sphinx Documentation Site (conf.py, RTD theme)

2. Create documentation pages:
   - PC-100-12: Documentation Pages (14 pages across 4 sections)
   - PC-101-12: Docker Installation Documentation
   - PC-102-12: WSL Installation Documentation

3. Build Docker infrastructure:
   - PC-103-12: Dockerfile (multi-stage, non-root, health checks)
   - PC-104-12: Docker Compose (dev and prod configurations)

4. Configure CI/CD:
   - PC-105-12: GitHub Actions CI (tests, lint, coverage, docs)
   - PC-106-12: GitHub Actions Release (Docker push, GitHub releases)
