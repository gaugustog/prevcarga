# PC-015-02: Feature Infrastructure Tests

**Epic:** [EPIC-02: Feature Engineering Infrastructure](../epics/EPIC-02-feature-engineering.md)
**Task Reference:** T-02.7
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Create comprehensive test suite for the Feature Engineering Infrastructure (EPIC-02), covering the base class, registry, pipeline, configuration, evaluator, and utilities with ≥80% coverage target.

---

## Acceptance Criteria

- [ ] Test file created: `tests/testthat/test-features-base.R`
- [ ] Test file created: `tests/testthat/test-features-registry.R`
- [ ] Test file created: `tests/testthat/test-features-pipeline.R`
- [ ] Test file created: `tests/testthat/test-features-config.R`
- [ ] Test file created: `tests/testthat/test-features-evaluation.R`
- [ ] Test file created: `tests/testthat/test-features-common.R`
- [ ] Mock plugins for isolated testing
- [ ] Integration tests for full pipeline
- [ ] Overall coverage ≥80% for Feature module

---

## Technical Specification

### Test Directory Structure
```
tests/
└── testthat/
   ├── test-features-base.R
   ├── test-features-registry.R
   ├── test-features-pipeline.R
   ├── test-features-config.R
   ├── test-features-evaluation.R
   ├── test-features-common.R
   └── helper-feature-mocks.R
```

### Mock Plugins

```r
# tests/testthat/helper-feature-mocks.R

#' Mock stateless plugin for testing
MockStatelessPlugin <- R6::R6Class(
 "MockStatelessPlugin",
 inherit = BaseFeaturePlugin,
 public = list(
   initialize = function(name = "mock_stateless", config = list()) {
     super$initialize(name, config)
     self$is_fitted <- TRUE
   },

   transform = function(dt, ...) {
     result <- data.table::copy(dt)
     prefix <- self$get_config("prefix", "mock")
     result[, paste0(prefix, "_feature") := 1.0]
     result
   },

   get_feature_names = function() {
     prefix <- self$get_config("prefix", "mock")
     paste0(prefix, "_feature")
   }
 )
)


#' Mock stateful plugin for testing
MockStatefulPlugin <- R6::R6Class(
 "MockStatefulPlugin",
 inherit = BaseFeaturePlugin,
 private = list(
   learned_mean = NULL
 ),
 public = list(
   initialize = function(name = "mock_stateful", config = list()) {
     super$initialize(name, config)
   },

   fit = function(dt, ...) {
     col <- self$get_config("column", "value")
     private$learned_mean <- mean(dt[[col]], na.rm = TRUE)
     self$is_fitted <- TRUE
     invisible(self)
   },

   transform = function(dt, ...) {
     if (!self$is_fitted) {
       stop("Plugin must be fitted first")
     }

     col <- self$get_config("column", "value")
     result <- data.table::copy(dt)
     result[, centered := get(col) - private$learned_mean]
     result
   },

   get_feature_names = function() {
     "centered"
   }
 )
)


#' Mock plugin that fails
MockFailingPlugin <- R6::R6Class(
 "MockFailingPlugin",
 inherit = BaseFeaturePlugin,
 public = list(
   transform = function(dt, ...) {
     stop("Intentional failure for testing")
   },
   get_feature_names = function() {
     "will_fail"
   }
 )
)


#' Create sample feature data for testing
create_feature_test_data <- function(n = 100) {
 data.table::data.table(
   DataHora = seq(
     as.POSIXct("2024-01-01 00:00:00"),
     by = "hour",
     length.out = n
   ),
   value = rnorm(n, 100, 10),
   category = sample(c("A", "B", "C"), n, replace = TRUE),
   area_code = sample(c("RJ", "SP"), n, replace = TRUE)
 )
}
```

### Base Class Tests

```r
# tests/testthat/test-features-base.R

describe("BaseFeaturePlugin", {
 describe("initialization", {
   it("can be instantiated", {
     plugin <- BaseFeaturePlugin$new()
     expect_s3_class(plugin, "BaseFeaturePlugin")
   })

   it("uses class name as default name", {
     plugin <- BaseFeaturePlugin$new()
     expect_equal(plugin$name, "BaseFeaturePlugin")
   })

   it("accepts custom name", {
     plugin <- BaseFeaturePlugin$new(name = "custom")
     expect_equal(plugin$name, "custom")
   })

   it("accepts config", {
     plugin <- BaseFeaturePlugin$new(config = list(a = 1, b = 2))
     expect_equal(plugin$config$a, 1)
     expect_equal(plugin$config$b, 2)
   })
 })

 describe("abstract methods", {
   it("transform() throws error", {
     plugin <- BaseFeaturePlugin$new()
     expect_error(
       plugin$transform(data.table::data.table()),
       "must be implemented"
     )
   })

   it("get_feature_names() throws error", {
     plugin <- BaseFeaturePlugin$new()
     expect_error(
       plugin$get_feature_names(),
       "must be implemented"
     )
   })
 })

 describe("validate_input", {
   it("accepts valid data.table", {
     plugin <- BaseFeaturePlugin$new()
     dt <- data.table::data.table(a = 1:10)
     expect_true(plugin$validate_input(dt))
   })

   it("rejects empty data.table", {
     plugin <- BaseFeaturePlugin$new()
     dt <- data.table::data.table()
     expect_error(plugin$validate_input(dt))
   })

   it("rejects non-data.table", {
     plugin <- BaseFeaturePlugin$new()
     expect_error(plugin$validate_input(data.frame(a = 1)))
   })
 })

 describe("get_config", {
   it("returns config value", {
     plugin <- BaseFeaturePlugin$new(config = list(key = "value"))
     expect_equal(plugin$get_config("key"), "value")
   })

   it("returns default for missing key", {
     plugin <- BaseFeaturePlugin$new()
     expect_equal(plugin$get_config("missing", "default"), "default")
   })
 })

 describe("subclass implementation", {
   it("MockStatelessPlugin works", {
     plugin <- MockStatelessPlugin$new()
     dt <- create_feature_test_data(10)

     result <- plugin$transform(dt)

     expect_true("mock_feature" %in% names(result))
     expect_equal(plugin$get_feature_names(), "mock_feature")
   })

   it("MockStatefulPlugin requires fit", {
     plugin <- MockStatefulPlugin$new(config = list(column = "value"))
     dt <- create_feature_test_data(10)

     expect_error(plugin$transform(dt), "fitted")

     plugin$fit(dt)
     result <- plugin$transform(dt)

     expect_true("centered" %in% names(result))
   })
 })
})
```

### Registry Tests

```r
# tests/testthat/test-features-registry.R

describe("FeaturePluginRegistry", {
 describe("initialization", {
   it("starts empty", {
     registry <- FeaturePluginRegistry$new()
     expect_equal(registry$count(), 0)
   })
 })

 describe("registration", {
   it("registers valid plugin", {
     registry <- FeaturePluginRegistry$new()

     expect_message(
       registry$register("mock", MockStatelessPlugin),
       "Registered"
     )
     expect_equal(registry$count(), 1)
   })

   it("rejects non-R6 class", {
     registry <- FeaturePluginRegistry$new()
     expect_error(
       registry$register("invalid", function() {}),
       "R6 class generator"
     )
   })

   it("rejects non-BaseFeaturePlugin", {
     NonPlugin <- R6::R6Class("NonPlugin")
     registry <- FeaturePluginRegistry$new()

     expect_error(
       registry$register("non_plugin", NonPlugin),
       "BaseFeaturePlugin"
     )
   })

   it("warns on duplicate registration", {
     registry <- FeaturePluginRegistry$new()
     registry$register("mock", MockStatelessPlugin)

     expect_warning(
       registry$register("mock", MockStatelessPlugin),
       "Overwriting"
     )
   })
 })

 describe("retrieval", {
   it("get() returns plugin instance", {
     registry <- FeaturePluginRegistry$new()
     registry$register("mock", MockStatelessPlugin)

     plugin <- registry$get("mock")

     expect_s3_class(plugin, "MockStatelessPlugin")
     expect_s3_class(plugin, "BaseFeaturePlugin")
   })

   it("get() passes config", {
     registry <- FeaturePluginRegistry$new()
     registry$register("mock", MockStatelessPlugin)

     plugin <- registry$get("mock", config = list(prefix = "test"))

     expect_equal(plugin$get_config("prefix"), "test")
   })

   it("get() errors on missing plugin", {
     registry <- FeaturePluginRegistry$new()

     expect_error(
       registry$get("missing"),
       "not registered"
     )
   })

   it("has() returns correct status", {
     registry <- FeaturePluginRegistry$new()
     registry$register("mock", MockStatelessPlugin)

     expect_true(registry$has("mock"))
     expect_false(registry$has("missing"))
   })

   it("list_plugins() returns names", {
     registry <- FeaturePluginRegistry$new()
     registry$register("plugin1", MockStatelessPlugin)
     registry$register("plugin2", MockStatefulPlugin)

     plugins <- registry$list_plugins()

     expect_setequal(plugins, c("plugin1", "plugin2"))
   })
 })

 describe("unregistration", {
   it("unregister() removes plugin", {
     registry <- FeaturePluginRegistry$new()
     registry$register("mock", MockStatelessPlugin)

     registry$unregister("mock")

     expect_false(registry$has("mock"))
   })

   it("unregister() warns on missing", {
     registry <- FeaturePluginRegistry$new()

     expect_warning(
       registry$unregister("missing"),
       "not registered"
     )
   })

   it("clear() removes all", {
     registry <- FeaturePluginRegistry$new()
     registry$register("p1", MockStatelessPlugin)
     registry$register("p2", MockStatelessPlugin)

     registry$clear()

     expect_equal(registry$count(), 0)
   })
 })
})
```

### Pipeline Tests

```r
# tests/testthat/test-features-pipeline.R

describe("FeaturePipeline", {
 describe("initialization", {
   it("starts empty", {
     pipeline <- FeaturePipeline$new()
     expect_equal(pipeline$length(), 0)
   })
 })

 describe("adding plugins", {
   it("add() appends plugin", {
     pipeline <- FeaturePipeline$new()
     plugin <- MockStatelessPlugin$new()

     pipeline$add(plugin)

     expect_equal(pipeline$length(), 1)
   })

   it("add() with position inserts", {
     pipeline <- FeaturePipeline$new()
     pipeline$add(MockStatelessPlugin$new(name = "first"))
     pipeline$add(MockStatelessPlugin$new(name = "third"))
     pipeline$add(MockStatelessPlugin$new(name = "second"), position = 2)

     plugins <- pipeline$list_plugins()

     expect_equal(plugins, c("first", "second", "third"))
   })

   it("warns on duplicate name", {
     pipeline <- FeaturePipeline$new()
     pipeline$add(MockStatelessPlugin$new(name = "dup"))

     expect_warning(
       pipeline$add(MockStatelessPlugin$new(name = "dup")),
       "already in pipeline"
     )
   })
 })

 describe("transformation", {
   it("transform() applies all plugins", {
     pipeline <- FeaturePipeline$new()
     pipeline$add(MockStatelessPlugin$new(config = list(prefix = "a")))
     pipeline$add(MockStatelessPlugin$new(name = "p2", config = list(prefix = "b")))

     dt <- create_feature_test_data(10)
     result <- pipeline$transform(dt)

     expect_true("a_feature" %in% names(result))
     expect_true("b_feature" %in% names(result))
   })

   it("transform() on empty pipeline warns", {
     pipeline <- FeaturePipeline$new()
     dt <- create_feature_test_data(10)

     expect_warning(
       result <- pipeline$transform(dt),
       "no plugins"
     )
   })

   it("fit() fits stateful plugins", {
     pipeline <- FeaturePipeline$new()
     plugin <- MockStatefulPlugin$new(config = list(column = "value"))
     pipeline$add(plugin)

     dt <- create_feature_test_data(100)
     pipeline$fit(dt)

     expect_true(plugin$is_fitted)
   })

   it("fit_transform() combines both", {
     pipeline <- FeaturePipeline$new()
     pipeline$add(MockStatefulPlugin$new(config = list(column = "value")))

     dt <- create_feature_test_data(100)
     result <- pipeline$fit_transform(dt)

     expect_true("centered" %in% names(result))
   })
 })

 describe("feature names", {
   it("get_all_feature_names() aggregates", {
     pipeline <- FeaturePipeline$new()
     pipeline$add(MockStatelessPlugin$new(config = list(prefix = "a")))
     pipeline$add(MockStatelessPlugin$new(name = "p2", config = list(prefix = "b")))

     features <- pipeline$get_all_feature_names()

     expect_setequal(features, c("a_feature", "b_feature"))
   })
 })

 describe("from_config", {
   it("builds pipeline from config", {
     registry <- FeaturePluginRegistry$new()
     registry$register("mock", MockStatelessPlugin)

     config <- list(
       plugins = list(
         list(name = "mock", config = list(prefix = "test"))
       )
     )

     pipeline <- FeaturePipeline$new(registry = registry)
     pipeline$from_config(config)

     expect_equal(pipeline$length(), 1)
   })

   it("skips disabled plugins", {
     registry <- FeaturePluginRegistry$new()
     registry$register("mock", MockStatelessPlugin)

     config <- list(
       plugins = list(
         list(name = "mock", enabled = FALSE)
       )
     )

     pipeline <- FeaturePipeline$new(registry = registry)
     pipeline$from_config(config)

     expect_equal(pipeline$length(), 0)
   })
 })
})
```

### Utilities Tests

```r
# tests/testthat/test-features-common.R

describe("cyclical_encode", {
 it("returns sin and cos columns", {
   result <- cyclical_encode(0:23, period = 24, prefix = "hour")

   expect_equal(names(result), c("hour_sin", "hour_cos"))
   expect_equal(nrow(result), 24)
 })

 it("hour 0 equals hour 24", {
   hour_0 <- cyclical_encode(0, period = 24)
   hour_24 <- cyclical_encode(24, period = 24)

   expect_equal(hour_0$cyclical_sin, hour_24$cyclical_sin, tolerance = 1e-10)
   expect_equal(hour_0$cyclical_cos, hour_24$cyclical_cos, tolerance = 1e-10)
 })
})

describe("create_lags", {
 it("creates correct lag values", {
   dt <- data.table::data.table(
     DataHora = as.POSIXct("2024-01-01") + (0:9) * 3600,
     value = 1:10
   )

   result <- create_lags(dt, "value", lags = c(1, 2))

   expect_true("value_lag1" %in% names(result))
   expect_true("value_lag2" %in% names(result))
   expect_true(is.na(result$value_lag1[1]))
   expect_equal(result$value_lag1[2], 1)
 })

 it("handles groups correctly", {
   dt <- data.table::data.table(
     DataHora = rep(as.POSIXct("2024-01-01") + (0:4) * 3600, 2),
     value = c(1:5, 11:15),
     area_code = rep(c("A", "B"), each = 5)
   )

   result <- create_lags(dt, "value", lags = 1, group_cols = "area_code")

   # First obs of each group should be NA
   expect_true(is.na(result[area_code == "A"]$value_lag1[1]))
   expect_true(is.na(result[area_code == "B"]$value_lag1[1]))

   # Second obs should have correct lag
   expect_equal(result[area_code == "A"]$value_lag1[2], 1)
   expect_equal(result[area_code == "B"]$value_lag1[2], 11)
 })
})

describe("create_rolling_stats", {
 it("computes rolling mean", {
   dt <- data.table::data.table(
     DataHora = as.POSIXct("2024-01-01") + (0:9) * 3600,
     value = rep(10, 10)
   )

   result <- create_rolling_stats(dt, "value", windows = 3, stats = "mean")

   expect_true("value_roll3_mean" %in% names(result))
   expect_equal(result$value_roll3_mean[3], 10)
 })
})

describe("one_hot_encode", {
 it("creates dummy columns", {
   dt <- data.table::data.table(
     category = c("A", "B", "C", "A")
   )

   result <- one_hot_encode(dt, "category")

   expect_true("category_A" %in% names(result))
   expect_true("category_B" %in% names(result))
   expect_true("category_C" %in% names(result))
 })

 it("drop_first removes one category", {
   dt <- data.table::data.table(
     category = c("A", "B", "C")
   )

   result <- one_hot_encode(dt, "category", drop_first = TRUE)

   # A is dropped (first alphabetically)
   expect_false("category_A" %in% names(result))
   expect_true("category_B" %in% names(result))
   expect_true("category_C" %in% names(result))
 })
})

describe("create_interactions", {
 it("creates pairwise interactions", {
   dt <- data.table::data.table(
     a = c(1, 2, 3),
     b = c(4, 5, 6)
   )

   result <- create_interactions(dt, c("a", "b"), degree = 2)

   expect_true("a_x_b" %in% names(result))
   expect_equal(result$a_x_b, c(4, 10, 18))
 })
})

describe("create_differences", {
 it("computes absolute differences", {
   dt <- data.table::data.table(
     DataHora = as.POSIXct("2024-01-01") + (0:4) * 3600,
     value = c(10, 12, 15, 14, 20)
   )

   result <- create_differences(dt, "value", lags = 1)

   expect_true("value_diff1" %in% names(result))
   expect_true(is.na(result$value_diff1[1]))
   expect_equal(result$value_diff1[2], 2)  # 12 - 10
 })

 it("computes percent differences", {
   dt <- data.table::data.table(
     DataHora = as.POSIXct("2024-01-01") + (0:2) * 3600,
     value = c(100, 110, 121)
   )

   result <- create_differences(dt, "value", lags = 1, pct = TRUE)

   expect_equal(result$value_pct1[2], 0.1, tolerance = 1e-10)
   expect_equal(result$value_pct1[3], 0.1, tolerance = 1e-10)
 })
})
```

---

## Test Coverage Targets

| Module | Target Coverage |
|--------|-----------------|
| `R/features/base.R` | ≥90% |
| `R/features/registry.R` | ≥90% |
| `R/features/pipeline.R` | ≥85% |
| `R/features/config.R` | ≥85% |
| `R/features/evaluation.R` | ≥80% |
| `R/features/common.R` | ≥90% |
| **Overall Feature Module** | **≥80%** |

---

## Definition of Done

- [ ] All test files created
- [ ] Mock plugins implemented
- [ ] All tests passing
- [ ] Coverage ≥80% verified with covr
- [ ] Integration tests included
- [ ] Edge cases covered
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use `testthat::describe()` and `it()` for BDD-style tests
- Mock plugins isolate tests from actual implementations
- Test both happy path and error cases
- Include boundary condition tests
- Run coverage with: `covr::package_coverage()`
