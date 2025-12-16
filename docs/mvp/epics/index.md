# PrevCargaONS Epic Index

**Project:** PrevCargaONS (R CLI Implementation)
**Version:** 1.0.0
**Last Updated:** 2025-12-07
**Reference:** [MVP Plan R](../mvp-plan-r.md)

---

## Overview

Development epics for the PrevCargaONS electric load forecasting system - an R-native CLI implementation with pluggable architecture.

**Key Metrics:**
- 26 time series (21 areas + 4 subsystems + 1 national)
- Pluggable model architecture (R6 classes)
- 30-week timeline (~7.5 months)
- 8 major milestones

**Important:** These epics define the **infrastructure and plugin structures**, not plugin implementations. Model, feature, combiner, and reconciler implementations are contributed separately via the plugin system.

---

## Milestones

| Milestone | Week | Deliverable | Epics |
|-----------|------|-------------|-------|
| **M1: MVP Data + Features** | 6 | Data pipeline + feature plugin infrastructure | EPIC-00, EPIC-01, EPIC-02 |
| **M2: First Model** | 8 | Model plugin infrastructure working | EPIC-03 |
| **M3: All Core Models** | 15 | Multiple model plugins can be registered | EPIC-04 |
| **M4: Combination & Reconciliation** | 20 | Complete system (without CLI) | EPIC-05, EPIC-06, EPIC-07 |
| **M5: Functional CLI** | 23 | CLI with all commands | EPIC-08, EPIC-09 |
| **M6: Complete Validation** | 26 | Results validated vs baseline | EPIC-10 |
| **M7: One-Line Installer** | 28 | curl installer for easy deployment | EPIC-11 |
| **M8: Production** | 30 | Docker deploy + Sphinx documentation | EPIC-12 |

---

## Epic Dependency Graph

```
EPIC-00 (Initial Setup)
    │
    └──► EPIC-01 (Data Layer)
             │
             └──► EPIC-02 (Feature Engineering Infrastructure)
                      │
                      └──► EPIC-03 (Model Layer - Infrastructure)
                               │
                               └──► EPIC-04 (Model Layer - Hierarchical Support)
                                        │
                                        ├──► EPIC-05 (Combination Infrastructure)
                                        │
                                        └──► EPIC-06 (Reconciliation Infrastructure)
                                                 │
                                                 └──► EPIC-07 (Evaluation Layer)
                                                          │
                                                          └──► EPIC-08 (Orchestrator)
                                                                   │
                                                                   └──► EPIC-09 (CLI)
                                                                            │
                                                                            └──► EPIC-10 (Testing & Validation)
                                                                                     │
                                                                                     └──► EPIC-11 (One-Line Installer)
                                                                                              │
                                                                                              └──► EPIC-12 (Documentation & Deploy)
```

---

## Epics Summary

| Epic ID | Title | Duration | Status | Dependencies |
|---------|-------|----------|--------|--------------|
| [EPIC-00](./EPIC-00-initial-setup.md) | Initial Setup | 1 week | Not Started | None |
| [EPIC-01](./EPIC-01-data-layer.md) | Data Layer | 2 weeks | Not Started | EPIC-00 |
| [EPIC-02](./EPIC-02-feature-engineering.md) | Feature Engineering Infrastructure | 3 weeks | Not Started | EPIC-01 |
| [EPIC-03](./EPIC-03-model-layer-part1.md) | Model Layer - Infrastructure | 4 weeks | Not Started | EPIC-02 |
| [EPIC-04](./EPIC-04-model-layer-part2.md) | Model Layer - Hierarchical Support | 3 weeks | Not Started | EPIC-03 |
| [EPIC-05](./EPIC-05-combination-layer.md) | Combination Infrastructure | 3 weeks | Not Started | EPIC-04 |
| [EPIC-06](./EPIC-06-reconciliation-layer.md) | Reconciliation Infrastructure | 2 weeks | Not Started | EPIC-04 |
| [EPIC-07](./EPIC-07-evaluation-layer.md) | Evaluation Layer | 2 weeks | Not Started | EPIC-05, EPIC-06 |
| [EPIC-08](./EPIC-08-orchestrator.md) | Orchestrator | 3 weeks | Not Started | EPIC-07 |
| [EPIC-09](./EPIC-09-cli.md) | CLI | 2 weeks | Not Started | EPIC-08 |
| [EPIC-10](./EPIC-10-testing-validation.md) | Testing & Validation | 3 weeks | Not Started | EPIC-09 |
| [EPIC-11](./EPIC-11-installer.md) | One-Line Installer | 1 week | Not Started | EPIC-10 |
| [EPIC-12](./EPIC-12-documentation-deploy.md) | Documentation & Deploy | 2 weeks | Not Started | EPIC-11 |

---

## Success Metrics

### Technical Metrics
- **Accuracy**: MAPE within ±5% of baseline (with contributed plugins)
- **Performance**:
  - Training 1 area + 1 model: <30min
  - Intraday prediction: <5min
  - 1-year backtest: <8h
- **Quality**: Test coverage >70%

### Operational Metrics
- **Availability**: 99.5% uptime for prediction service
- **Latency**: P95 prediction latency <3min
- **Extensibility**: New model plugin registration in <1 day

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [MVP Plan R](../mvp-plan-r.md) | Complete system specification |
| [Plugin Guide: Overview](../plugin-guide/00-overview.md) | Plugin architecture |
| [Plugin Guide: Data Importing](../plugin-guide/01-data-importing.md) | Storage backends |
| [Plugin Guide: Feature Engineering](../plugin-guide/02-feature-engineering.md) | Feature plugins |
| [Plugin Guide: Model Training](../plugin-guide/03-model-training.md) | Model plugins |
| [Plugin Guide: Model Inference](../plugin-guide/04-model-inference.md) | Inference patterns |
| [Plugin Guide: Combination](../plugin-guide/05-combination.md) | Combiner plugins |
| [Plugin Guide: Reconciliation](../plugin-guide/06-reconciliation.md) | Reconciler plugins |

---

## Status Legend

| Status | Description |
|--------|-------------|
| Not Started | Epic not yet begun |
| In Progress | Active development |
| In Review | Code complete, under review |
| Done | Accepted and merged |
| Blocked | Waiting on dependency or decision |

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-07 | 1.0.0 | Initial R CLI epic index created |
| 2025-12-07 | 1.1.0 | Added One-Line Installer epic |
| 2025-12-07 | 1.2.0 | Swapped EPIC-11/EPIC-12: Installer before Documentation |
