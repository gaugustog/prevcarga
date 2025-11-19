# PC-066-07B: Multi-Format Report Generator

**Ticket ID:** PC-066-07B  
**Epic:** Epic-07B - Monitoring & Reporting  
**Story Points:** 8  
**Priority:** High  
**Assignee:** TBD  
**Sprint:** Week 20 (Days 4-5)

---

## 📋 Description

Implement comprehensive multi-format reporting system that generates automated performance reports in HTML (with interactive visualizations), CSV (for detailed analysis), and PDF (for executive summaries). Includes Jinja2 templating, customizable report templates for different stakeholder audiences, and automated scheduling capabilities.

---

## 🎯 Acceptance Criteria

- [ ] `ReportGenerator` creates HTML reports with interactive Plotly charts
- [ ] CSV export functionality with proper formatting and metadata
- [ ] PDF summary generation for executive reporting
- [ ] Jinja2 template system with modular components
- [ ] Customizable report templates for different audiences
- [ ] ChartGenerator for performance visualizations
- [ ] Report generation completes in <30 seconds
- [ ] Automated scheduling capability (cron-compatible)
- [ ] Unit tests with 75%+ coverage

---

## 📐 Technical Specifications

### Core Components

#### 1. Configuration
```python
@dataclass
class ReportConfig:
    template_dir: str = "templates"
    output_dir: str = "reports"
    include_charts: bool = True
    chart_format: str = "png"  # or "svg"
    report_title: str = "Forecast Performance Report"
    author: str = "PrevCarga System"
```

#### 2. ReportGenerator
```python
class ReportGenerator:
    def __init__(self, template_config: ReportConfig)
    
    def generate_html_report(
        self,
        evaluation_results: Dict[str, any],
        template_name: str = 'standard'
    ) -> str  # Returns path to generated HTML
    
    def export_csv_data(
        self,
        evaluation_results: Dict[str, any],
        output_path: str
    ) -> None
    
    def generate_pdf_summary(
        self,
        evaluation_results: Dict[str, any],
        output_path: str
    ) -> None
    
    def _setup_jinja2(self) -> Environment
    def _generate_report_charts(self, evaluation_results: Dict[str, any]) -> Dict[str, str]
```

#### 3. ChartGenerator
```python
class ChartGenerator:
    def create_performance_chart(
        self,
        metrics: Dict[str, MetricsResult]
    ) -> str  # Returns HTML or file path
    
    def create_horizon_chart(
        self,
        horizon_analysis: Dict
    ) -> str
    
    def create_period_chart(
        self,
        period_analysis: Dict
    ) -> str
    
    def create_comparison_chart(
        self,
        comparison_result: ComparisonResult
    ) -> str
    
    def create_drift_chart(
        self,
        drift_history: List[DriftDetectionResult]
    ) -> str
```

### Report Structure

#### HTML Report Sections
1. **Executive Summary**
   - Overall performance metrics
   - Best/worst performing areas
   - Key findings and recommendations

2. **Detailed Metrics**
   - Performance by area (26 series)
   - Performance by horizon (D+0 to D+8)
   - Performance by time period (peak/off-peak, seasonal)

3. **Model Comparison**
   - Pairwise comparison tables
   - Statistical significance results
   - Model rankings

4. **Drift Detection**
   - Recent drift alerts
   - Performance trends
   - Severity distribution

5. **Visualizations**
   - Interactive Plotly charts
   - Performance trends
   - Horizon decay curves
   - Period comparisons

#### CSV Export Structure
```csv
# Forecast Performance Report
# Generated: 2025-11-19T10:30:00
# Author: PrevCarga System

series_id,mape,mae,rmse,r2,sample_size,best_model,drift_detected
area_01,2.5,150.2,180.3,0.92,1000,lgbm,False
area_02,2.8,165.4,195.2,0.90,1000,lgbm,False
...
```

#### PDF Summary (Executive)
- 2-4 page summary
- Key metrics and KPIs
- Top recommendations
- Critical alerts
- High-level visualizations

---

## 🔧 Implementation Tasks

### Task 1: Core Infrastructure (2 hours)
- [ ] Implement `ReportConfig` dataclass
- [ ] Set up Jinja2 template environment
- [ ] Create output directory management
- [ ] Add timestamp and file naming utilities

### Task 2: HTML Report Generation (4 hours)
- [ ] Create standard HTML template
- [ ] Create executive summary template
- [ ] Implement Jinja2 template rendering
- [ ] Add CSS styling for reports
- [ ] Integrate Plotly.js for interactive charts
- [ ] Add navigation and table of contents

### Task 3: CSV Export (2 hours)
- [ ] Implement metrics-to-DataFrame conversion
- [ ] Add metadata header generation
- [ ] Implement CSV export with pandas
- [ ] Add multi-sheet support (optional, using Excel)
- [ ] Handle missing values properly

### Task 4: PDF Generation (3 hours)
- [ ] Create executive summary template
- [ ] Implement HTML-to-PDF conversion (WeasyPrint)
- [ ] Add PDF styling and layout
- [ ] Handle page breaks and formatting
- [ ] Add headers/footers with metadata

### Task 5: Chart Generation (4 hours)
- [ ] Implement performance bar/line charts
- [ ] Implement horizon decay curves
- [ ] Implement period comparison charts
- [ ] Implement model comparison charts
- [ ] Implement drift history time series
- [ ] Add chart customization options

### Task 6: Template System (3 hours)
- [ ] Create modular template components
- [ ] Implement template inheritance
- [ ] Add customization parameters
- [ ] Create multiple audience templates:
  - Technical (detailed)
  - Executive (summary)
  - Operational (actionable)

### Task 7: Automation and Scheduling (2 hours)
- [ ] Add report generation scheduling
- [ ] Implement command-line interface
- [ ] Add email distribution capability (optional)
- [ ] Create cron job examples

### Task 8: Integration and Testing (4 hours)
- [ ] Unit tests for each report format
- [ ] Integration tests with all Epic-07A/07B components
- [ ] Performance testing (30s target)
- [ ] Output validation and quality checks
- [ ] End-to-end report generation pipeline

---

## 🧪 Testing Requirements

### Unit Tests
```python
def test_report_config_defaults()
def test_jinja2_setup()
def test_html_report_generation()
def test_csv_export_structure()
def test_csv_metadata_header()
def test_pdf_generation_basic()
def test_chart_generation_performance()
def test_chart_generation_horizon()
def test_chart_generation_period()
def test_template_rendering()
```

### Integration Tests
```python
def test_full_html_report_pipeline()
def test_csv_export_pipeline()
def test_pdf_generation_pipeline()
def test_all_formats_generation()
def test_integration_with_metrics_calculator()
def test_integration_with_drift_detector()
def test_integration_with_model_comparator()
```

### Output Validation Tests
```python
def test_html_valid_structure()
def test_csv_valid_format()
def test_pdf_valid_document()
def test_charts_render_correctly()
def test_report_completeness()
```

### Performance Tests
```python
def test_html_generation_speed()
def test_csv_export_speed()
def test_pdf_generation_speed()
def test_full_report_under_30_seconds()
```

---

## 📊 Success Metrics

### Performance Targets
- HTML report generation: **<15 seconds**
- CSV export: **<5 seconds**
- PDF generation: **<20 seconds**
- Complete report (all formats): **<30 seconds**

### Quality Targets
- Unit test coverage: **≥75%**
- Report completeness: **100%** of requested sections
- Chart rendering success: **>95%**
- Valid HTML/CSS: **100%** (W3C validation)

---

## 📚 Dependencies

### Required Packages
```python
jinja2 >= 3.1.0              # Template engine
pandas >= 2.0.0              # CSV export
plotly >= 5.14.0             # Interactive charts
weasyprint >= 59.0           # PDF generation (optional)
# OR
reportlab >= 4.0.0           # Alternative PDF generation
```

### Code Dependencies
- **Requires:** PC-060-07A (MetricsCalculator)
- **Requires:** PC-061-07A (PercentileAnalyzer)
- **Requires:** PC-062-07A (TimePeriodAnalyzer)
- **Requires:** PC-063-07A (HorizonAnalyzer)
- **Requires:** PC-064-07B (DriftDetector)
- **Requires:** PC-065-07B (ModelComparator)

---

## 🔗 Related Tickets
- **Depends on:** PC-060-63-07A (all Epic-07A tickets)
- **Depends on:** PC-064-065-07B (Drift Detector, Model Comparator)
- **Completes:** Epic-07B

---

## 📖 Documentation Requirements

- [ ] API documentation for ReportGenerator
- [ ] Template customization guide
- [ ] Chart configuration options
- [ ] Scheduling setup instructions
- [ ] Example reports for each template
- [ ] Troubleshooting guide for PDF generation

---

## 💡 Implementation Notes

### Jinja2 Template Structure
```
templates/
├── base.html              # Base template with common structure
├── standard.html          # Standard detailed report
├── executive.html         # Executive summary
├── operational.html       # Operational focus
└── components/
    ├── header.html
    ├── metrics_table.html
    ├── charts.html
    └── footer.html
```

### HTML Report Example Structure
```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Report styling */
    </style>
</head>
<body>
    <header>
        <h1>{{ title }}</h1>
        <p>Generated: {{ generation_time }}</p>
    </header>
    
    <section id="executive-summary">
        <h2>Executive Summary</h2>
        <p>Overall MAPE: {{ results.mean_mape }}%</p>
        <!-- Summary content -->
    </section>
    
    <section id="detailed-metrics">
        <h2>Detailed Metrics</h2>
        <table>
            <!-- Metrics table -->
        </table>
    </section>
    
    <section id="visualizations">
        <h2>Performance Visualizations</h2>
        <div id="chart-performance">{{ charts.performance_by_series }}</div>
        <div id="chart-horizon">{{ charts.horizon_decay }}</div>
    </section>
</body>
</html>
```

### Plotly Chart Example
```python
import plotly.graph_objects as go

def create_performance_chart(metrics: Dict[str, MetricsResult]) -> str:
    series_ids = list(metrics.keys())
    mapes = [metrics[sid].mape for sid in series_ids]
    
    fig = go.Figure(data=[
        go.Bar(x=series_ids, y=mapes, name='MAPE')
    ])
    
    fig.update_layout(
        title='Performance by Area',
        xaxis_title='Area',
        yaxis_title='MAPE (%)',
        template='plotly_white'
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id='performance-chart')
```

### PDF Generation Notes
**WeasyPrint Approach:**
- Converts HTML/CSS to PDF
- Good for well-styled HTML reports
- Requires system dependencies (cairo, pango)

**ReportLab Approach:**
- Python-native PDF generation
- More control over layout
- Steeper learning curve

**Recommendation:** Start with HTML → WeasyPrint for simplicity

### CSV Metadata Header
```python
def _add_csv_metadata(self, file_handle):
    file_handle.write(f"# {self.config.report_title}\n")
    file_handle.write(f"# Generated: {pd.Timestamp.now().isoformat()}\n")
    file_handle.write(f"# Author: {self.config.author}\n")
    file_handle.write(f"# Sections: Metrics, Horizon Analysis, Period Analysis\n")
    file_handle.write("\n")
```

### Scheduling Example
```bash
# Cron job for daily report generation
0 6 * * * /path/to/venv/bin/python /path/to/generate_report.py --output-dir=/reports/daily

# Weekly executive summary
0 8 * * 1 /path/to/venv/bin/python /path/to/generate_report.py --template=executive --email=executives@example.com
```

---

## ✅ Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing with ≥75% coverage
- [ ] HTML reports with interactive charts working
- [ ] CSV export functional with metadata
- [ ] PDF generation implemented (basic)
- [ ] Multiple templates created and tested
- [ ] Chart generation for all visualization types
- [ ] Performance benchmarks met (<30s total)
- [ ] Code reviewed and approved
- [ ] Documentation complete with examples
- [ ] Integration tested with all Epic-07A/07B components
- [ ] Epic-07B complete and ready for Epic-08
