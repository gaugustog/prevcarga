# PC-041-06: HierarchyBuilder

**Epic:** [EPIC-06: Reconciliation Infrastructure](../epics/EPIC-06-reconciliation-layer.md)
**Task Reference:** T-06.3
**Priority:** High
**Estimated Effort:** 1.5 days

---

## Summary

Implement the `HierarchyBuilder` R6 class for constructing the summing matrix S that encodes the SIN hierarchy structure, including subsystems, areas, and loss areas.

---

## Acceptance Criteria

- [ ] Hierarchy module created in `R/reconciliation/hierarchy.R`
- [ ] `HierarchyBuilder` R6 class for hierarchy management
- [ ] Summing matrix S construction
- [ ] Support for SIN hierarchy (national → subsystems → areas)
- [ ] Loss area handling (PESE, PES, PENE, PEN)
- [ ] Hierarchy validation

---

## Technical Specification

### File Location
```
R/reconciliation/hierarchy.R
```

### HierarchyBuilder R6 Class

```r
#' @title HierarchyBuilder
#' @description Builds hierarchy structures and summing matrices for reconciliation
#'
#' The SIN hierarchy structure:
#' - Level 0: SIN (national total)
#' - Level 1: Subsystems (SECO, S, NE, N)
#' - Level 2: Areas (RJ, SP, ...) + Loss Areas (PESE, PES, PENE, PEN)
#'
#' The summing matrix S encodes: all_series = S × bottom_series
#'
#' @export
HierarchyBuilder <- R6::R6Class(
  "HierarchyBuilder",
  private = list(
    config = NULL,
    summing_matrix = NULL,
    series_info = NULL
  ),
  public = list(
    #' @description Initialize hierarchy builder
    #' @param config Hierarchy configuration
    initialize = function(config = NULL) {
      if (is.null(config)) {
        config <- private$default_sin_config()
      }
      private$config <- config
      private$build_series_info()
    },

    #' @description Build the summing matrix S
    #' @return Matrix S where all_series = S × bottom_series
    build_summing_matrix = function() {
      if (!is.null(private$summing_matrix)) {
        return(private$summing_matrix)
      }

      bottom_series <- self$get_bottom_series()
      all_series <- self$get_all_series()

      n_bottom <- length(bottom_series)
      n_all <- length(all_series)

      # Initialize S matrix
      S <- matrix(0, nrow = n_all, ncol = n_bottom)
      rownames(S) <- all_series
      colnames(S) <- bottom_series

      # Build aggregation relationships
      for (i in seq_along(all_series)) {
        series <- all_series[i]
        children <- self$get_children(series)

        if (length(children) == 0) {
          # Bottom-level series: identity
          S[series, series] <- 1
        } else {
          # Aggregate series: sum of children
          for (child in children) {
            if (child %in% bottom_series) {
              S[series, child] <- 1
            } else {
              # Child is also an aggregate - get its bottom series
              child_bottom <- self$get_descendant_bottom_series(child)
              S[series, child_bottom] <- 1
            }
          }
        }
      }

      private$summing_matrix <- S
      S
    },

    #' @description Get all series in the hierarchy
    #' @return Character vector of series names (ordered top to bottom)
    get_all_series = function() {
      c(
        private$config$national,
        names(private$config$subsystems),
        self$get_bottom_series()
      )
    },

    #' @description Get bottom-level series (areas + loss areas)
    #' @return Character vector of bottom series names
    get_bottom_series = function() {
      areas <- unlist(private$config$subsystems, use.names = FALSE)
      loss_areas <- names(private$config$loss_areas)

      c(areas, loss_areas)
    },

    #' @description Get direct children of a series
    #' @param series Series name
    #' @return Character vector of child series names
    get_children = function(series) {
      if (series == private$config$national) {
        # National: children are subsystems
        return(names(private$config$subsystems))
      }

      if (series %in% names(private$config$subsystems)) {
        # Subsystem: children are its areas + loss area
        subsystem <- series
        areas <- private$config$subsystems[[subsystem]]

        # Find loss area for this subsystem
        loss_area <- NULL
        for (la in names(private$config$loss_areas)) {
          if (private$config$loss_areas[[la]]$subsystem == subsystem) {
            loss_area <- la
            break
          }
        }

        return(c(areas, loss_area))
      }

      # Bottom-level series: no children
      character(0)
    },

    #' @description Get all descendant bottom series
    #' @param series Series name
    #' @return Character vector of bottom series names
    get_descendant_bottom_series = function(series) {
      bottom <- self$get_bottom_series()

      if (series %in% bottom) {
        return(series)
      }

      children <- self$get_children(series)
      if (length(children) == 0) {
        return(character(0))
      }

      descendants <- character(0)
      for (child in children) {
        child_descendants <- self$get_descendant_bottom_series(child)
        descendants <- c(descendants, child_descendants)
      }

      unique(descendants)
    },

    #' @description Get parent of a series
    #' @param series Series name
    #' @return Parent series name or NULL
    get_parent = function(series) {
      if (series == private$config$national) {
        return(NULL)
      }

      # Check if it's a subsystem
      if (series %in% names(private$config$subsystems)) {
        return(private$config$national)
      }

      # Find which subsystem contains this area
      for (subsystem in names(private$config$subsystems)) {
        areas <- private$config$subsystems[[subsystem]]
        if (series %in% areas) {
          return(subsystem)
        }
      }

      # Check loss areas
      for (la in names(private$config$loss_areas)) {
        if (series == la) {
          return(private$config$loss_areas[[la]]$subsystem)
        }
      }

      NULL
    },

    #' @description Get aggregation levels
    #' @return List with level names and series
    get_aggregation_levels = function() {
      list(
        national = private$config$national,
        subsystems = names(private$config$subsystems),
        areas = unlist(private$config$subsystems, use.names = FALSE),
        loss_areas = names(private$config$loss_areas)
      )
    },

    #' @description Get series info
    #' @return data.table with series information
    get_series_info = function() {
      data.table::copy(private$series_info)
    },

    #' @description Get loss area configuration
    #' @param loss_area Loss area code (optional, returns all if NULL)
    #' @return List with loss area config
    get_loss_area_config = function(loss_area = NULL) {
      if (is.null(loss_area)) {
        return(private$config$loss_areas)
      }
      private$config$loss_areas[[loss_area]]
    },

    #' @description Validate hierarchy configuration
    #' @return TRUE if valid, error otherwise
    validate = function() {
      # Check national exists
      if (is.null(private$config$national)) {
        stop("Hierarchy config missing 'national' field")
      }

      # Check subsystems exist
      if (is.null(private$config$subsystems) ||
          length(private$config$subsystems) == 0) {
        stop("Hierarchy config missing 'subsystems' field")
      }

      # Check each subsystem has areas
      for (subsystem in names(private$config$subsystems)) {
        if (length(private$config$subsystems[[subsystem]]) == 0) {
          stop(sprintf("Subsystem '%s' has no areas", subsystem))
        }
      }

      # Check loss areas reference valid subsystems
      for (la in names(private$config$loss_areas)) {
        la_config <- private$config$loss_areas[[la]]
        if (!la_config$subsystem %in% names(private$config$subsystems)) {
          stop(sprintf(
            "Loss area '%s' references unknown subsystem '%s'",
            la, la_config$subsystem
          ))
        }
      }

      # Check for duplicate area codes
      all_areas <- c(
        unlist(private$config$subsystems, use.names = FALSE),
        names(private$config$loss_areas)
      )
      dupes <- all_areas[duplicated(all_areas)]
      if (length(dupes) > 0) {
        stop(sprintf("Duplicate area codes: %s", paste(dupes, collapse = ", ")))
      }

      invisible(TRUE)
    },

    #' @description Get hierarchy configuration
    #' @return Configuration list
    get_config = function() {
      private$config
    },

    #' @description Print hierarchy summary
    print = function() {
      cat("HierarchyBuilder\n")
      cat(sprintf("  National: %s\n", private$config$national))
      cat(sprintf("  Subsystems: %d\n", length(private$config$subsystems)))

      for (subsystem in names(private$config$subsystems)) {
        n_areas <- length(private$config$subsystems[[subsystem]])
        cat(sprintf("    %s: %d areas\n", subsystem, n_areas))
      }

      cat(sprintf("  Loss areas: %d\n", length(private$config$loss_areas)))
      cat(sprintf("  Total series: %d\n", length(self$get_all_series())))
      cat(sprintf("  Bottom series: %d\n", length(self$get_bottom_series())))

      invisible(self)
    }
  ),

  private = list(
    default_sin_config = function() {
      list(
        national = "SIN",
        subsystems = list(
          SECO = c("RJ", "SP", "MG", "ES", "MT", "MS", "AC", "RO", "DF", "GO"),
          S = c("PR", "SC", "RS"),
          NE = c("ALPE", "PBRN", "BASE", "CE", "PI", "BAOE"),
          N = c("AM", "PA", "MA", "TO", "RR", "AP")
        ),
        loss_areas = list(
          PESE = list(subsystem = "SECO", method = "by_difference"),
          PES = list(subsystem = "S", method = "by_difference"),
          PENE = list(subsystem = "NE", method = "model"),
          PEN = list(subsystem = "N", method = "by_difference")
        )
      )
    },

    build_series_info = function() {
      info <- list()

      # National
      info[[private$config$national]] <- list(
        series = private$config$national,
        level = 0,
        type = "national",
        parent = NA_character_
      )

      # Subsystems
      for (subsystem in names(private$config$subsystems)) {
        info[[subsystem]] <- list(
          series = subsystem,
          level = 1,
          type = "subsystem",
          parent = private$config$national
        )
      }

      # Areas
      for (subsystem in names(private$config$subsystems)) {
        for (area in private$config$subsystems[[subsystem]]) {
          info[[area]] <- list(
            series = area,
            level = 2,
            type = "area",
            parent = subsystem
          )
        }
      }

      # Loss areas
      for (la in names(private$config$loss_areas)) {
        la_config <- private$config$loss_areas[[la]]
        info[[la]] <- list(
          series = la,
          level = 2,
          type = "loss_area",
          parent = la_config$subsystem,
          method = la_config$method
        )
      }

      private$series_info <- data.table::rbindlist(info, fill = TRUE)
    }
  )
)
```

### Factory Function

```r
#' Create a hierarchy builder for SIN
#'
#' @param config Optional custom configuration
#' @return HierarchyBuilder instance
#' @export
create_sin_hierarchy <- function(config = NULL) {
  HierarchyBuilder$new(config = config)
}
```

### Usage Example

```r
# Create default SIN hierarchy
hierarchy <- create_sin_hierarchy()

# Get summing matrix
S <- hierarchy$build_summing_matrix()
# Matrix: 31 rows (all series) × 27 columns (bottom series)

# Get series info
hierarchy$get_series_info()
#    series level      type parent        method
# 1:    SIN     0  national   <NA>          <NA>
# 2:   SECO     1 subsystem    SIN          <NA>
# 3:      S     1 subsystem    SIN          <NA>
# ...
# 27:  PESE     2 loss_area   SECO by_difference

# Get aggregation levels
levels <- hierarchy$get_aggregation_levels()
# $national: "SIN"
# $subsystems: c("SECO", "S", "NE", "N")
# $areas: c("RJ", "SP", ...)
# $loss_areas: c("PESE", "PES", "PENE", "PEN")

# Get children of SECO
hierarchy$get_children("SECO")
# [1] "RJ" "SP" "MG" "ES" "MT" "MS" "AC" "RO" "DF" "GO" "PESE"

# Validate
hierarchy$validate()

# Custom hierarchy
custom_config <- list(
  national = "TOTAL",
  subsystems = list(
    REGION_A = c("A1", "A2", "A3"),
    REGION_B = c("B1", "B2")
  ),
  loss_areas = list(
    LOSS_A = list(subsystem = "REGION_A", method = "by_difference")
  )
)
custom_hierarchy <- create_sin_hierarchy(config = custom_config)
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Initialize with default config | SIN hierarchy |
| TC-002 | Initialize with custom config | Custom hierarchy |
| TC-003 | build_summing_matrix() dimensions | Correct shape |
| TC-004 | build_summing_matrix() values | Correct aggregations |
| TC-005 | get_all_series() | All 31 series |
| TC-006 | get_bottom_series() | 27 bottom series |
| TC-007 | get_children("SIN") | 4 subsystems |
| TC-008 | get_children("SECO") | 11 (10 areas + 1 loss) |
| TC-009 | get_children("RJ") | Empty |
| TC-010 | get_parent("SECO") | "SIN" |
| TC-011 | get_parent("RJ") | "SECO" |
| TC-012 | get_descendant_bottom_series("SIN") | All 27 |
| TC-013 | validate() valid config | TRUE |
| TC-014 | validate() missing national | Error |
| TC-015 | validate() duplicate areas | Error |

---

## Dependencies

None (can be used independently)

---

## Definition of Done

- [ ] HierarchyBuilder R6 class implemented
- [ ] Summing matrix construction working
- [ ] All hierarchy navigation methods
- [ ] SIN default configuration
- [ ] Custom configuration support
- [ ] Validation working
- [ ] roxygen2 documentation complete
- [ ] Unit tests (≥90% coverage)
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- The summing matrix S is fundamental to all reconciliation methods
- Default config matches Brazilian SIN hierarchy structure
- Loss areas are bottom-level but computed by difference
- Consider adding graph visualization of hierarchy in future
- May need to support temporal hierarchy in future (daily → hourly)
