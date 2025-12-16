# Plugin Guide: Data Importing

> **Note:** This documentation will evolve into the project's CONTRIBUTING guide.

## Storage Backend Architecture

PrevCargaONS uses an abstract storage backend that supports both local filesystem (development) and S3 (production).

## Data Structure (Hive-style Partitioning)

All data uses Hive-style partitioning for efficient querying:

```
data/
├── load/                                  # Load data (primary dataset)
│   └── area_code={CODE}/                  # Ex: area_code=RJ, area_code=SP
│       └── year={YEAR}/                   # Ex: year=2024
│           └── data.parquet               # Hourly load values

├── auxiliary/                             # Supporting metadata
│   ├── dst_periods/
│   │   └── data.parquet                   # DST transition periods
│   ├── holidays/
│   │   └── area_code={CODE}/year={YEAR}/data.parquet
│   └── load_tiers/
│       └── area_code={CODE}/data.parquet  # Static config

├── weather/
│   ├── forecast/
│   │   └── area_code={CODE}/year={YEAR}/data.parquet
│   └── observed/
│       └── area_code={CODE}/year={YEAR}/data.parquet

├── models/                                # Trained models
└── results/                               # Forecasts and evaluations
```

## Area Codes Reference

### Areas (21 total - Models Can Run Here)
```
SECO: RJ, SP, MG, ES, MT, MS, AC, RO, DF, GO, PESE
S:    PR, SC, RS, PES
NE:   ALPE, PBRN, BASE, CE, PI, BAOE, PENE
N:    AM, PA, MA, TO, RR, AP, PEN
```

### Loss Areas (Can Have Models OR Be Calculated by Difference)
```
PESE (SECO), PES (S), PENE (NE), PEN (N)
```

These are regular areas that can either:
- Have forecasting models run for them, OR
- Be calculated as: `Loss = Subsystem - Σ(other_areas)`

### Subsystems (Reconciliation Only - No Models)
```
SECO, S, NE, N
```

### National (Reconciliation Only - No Models)
```
SIN (area_code=SIN)
```

## Storage Backend Interface

```r
#' @title StorageBackend
#' @description Abstract base class for storage backends
StorageBackend <- R6::R6Class(
  "StorageBackend",
  public = list(
    backend_type = NULL,

    read_parquet = function(path) {
      stop("read_parquet() must be implemented by subclass")
    },

    write_parquet = function(dt, path) {
      stop("write_parquet() must be implemented by subclass")
    },

    read_rds = function(path) {
      stop("read_rds() must be implemented by subclass")
    },

    write_rds = function(obj, path) {
      stop("write_rds() must be implemented by subclass")
    },

    exists = function(path) {
      stop("exists() must be implemented by subclass")
    },

    list_files = function(path, pattern = NULL) {
      stop("list_files() must be implemented by subclass")
    }
  )
)
```

## Local Backend Implementation

```r
LocalStorageBackend <- R6::R6Class(
  "LocalStorageBackend",
  inherit = StorageBackend,
  private = list(
    base_path = NULL
  ),
  public = list(
    initialize = function(base_path = "./data") {
      self$backend_type <- "local"
      private$base_path <- normalizePath(base_path, mustWork = FALSE)
    },

    read_parquet = function(path) {
      full_path <- file.path(private$base_path, path)
      arrow::read_parquet(full_path) |> data.table::setDT()
    },

    write_parquet = function(dt, path) {
      full_path <- file.path(private$base_path, path)
      dir.create(dirname(full_path), recursive = TRUE, showWarnings = FALSE)
      arrow::write_parquet(dt, full_path)
    }
  )
)
```

## S3 Backend Implementation

```r
S3StorageBackend <- R6::R6Class(
  "S3StorageBackend",
  inherit = StorageBackend,
  private = list(
    bucket = NULL,
    s3_client = NULL
  ),
  public = list(
    initialize = function(bucket, region = "us-east-1") {
      self$backend_type <- "s3"
      private$bucket <- bucket
      private$s3_client <- paws::s3(config = list(region = region))
    },

    read_parquet = function(path) {
      tmp_file <- tempfile(fileext = ".parquet")
      on.exit(unlink(tmp_file))

      private$s3_client$download_file(
        Bucket = private$bucket,
        Key = path,
        Filename = tmp_file
      )
      arrow::read_parquet(tmp_file) |> data.table::setDT()
    }
  )
)
```

## Storage Factory

```r
StorageFactory <- R6::R6Class(
  "StorageFactory",
  public = list(
    from_config = function(config) {
      backend_type <- config$storage$backend %||% "local"

      if (backend_type == "s3") {
        S3StorageBackend$new(
          bucket = config$storage$s3$bucket,
          region = config$storage$s3$region
        )
      } else {
        LocalStorageBackend$new(
          base_path = config$storage$local$base_path
        )
      }
    }
  )
)
```

## DataLoader Usage

```r
# Create storage backend
backend <- StorageFactory$new()$from_config(config)

# Create data loader
loader <- DataLoader$new(storage = backend)

# Load data for specific areas and period
load_data <- loader$load_carga(
  areas = c("RJ", "SP", "MG"),
  start_date = "2023-01-01",
  end_date = "2024-12-31"
)

# Load weather data
weather <- loader$load_weather(
  areas = c("RJ", "SP"),
  type = "forecast",
  start_date = "2024-01-01",
  end_date = "2024-12-31"
)

# Load holidays
holidays <- loader$load_holidays(
  areas = c("RJ", "SP"),
  years = 2023:2024
)
```

## Data Schemas

### Load Data Schema
| Column | Type | Description |
|--------|------|-------------|
| `DataHora` | datetime | Timestamp (hourly) |
| `CargaGlobal` | numeric | Load in MW |
| `area_code` | character | Area identifier |

### Weather Data Schema
| Column | Type | Description |
|--------|------|-------------|
| `DataHora` | datetime | Timestamp |
| `temperatura` | numeric | Temperature (C) |
| `umidade` | numeric | Humidity (%) |
| `area_code` | character | Area identifier |

### Holiday Data Schema
| Column | Type | Description |
|--------|------|-------------|
| `Data` | date | Holiday date |
| `nome` | character | Holiday name |
| `tipo` | character | Holiday type |
| `area_code` | character | Area identifier |

## Next Steps

- [Feature Engineering](02-feature-engineering.md) - Creating feature plugins
- [Model Training](03-model-training.md) - Creating model plugins
