# PC-090-10B: Code Quality and Coverage Validator Implementation

**Ticket ID:** PC-090-10B  
**Epic:** Epic-10B (Quality Assurance & Acceptance Validation)  
**Story:** User Story 2 - Code Quality and Test Coverage Validation  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** Critical  
**Estimated Effort:** 5 days  
**Sprint:** Week 29

---

## 📋 Description

Implement comprehensive code quality validation framework to ensure code meets production standards including test coverage analysis (>70% target), code quality metrics validation, and performance regression testing. The validator must integrate with CI/CD pipeline and enforce quality gates.

---

## 🎯 Acceptance Criteria

- [ ] `CodeQualityValidator` class with comprehensive quality analysis
- [ ] Test coverage analysis with >70% threshold enforcement
- [ ] Coverage reporting by component with detailed breakdowns
- [ ] Code quality metrics (cyclomatic complexity, maintainability index, duplication)
- [ ] Quality threshold validation and gate enforcement
- [ ] Performance regression testing framework
- [ ] Coverage trend tracking and historical comparison
- [ ] Quality dashboard generation with visualizations
- [ ] CI/CD pipeline integration for automated validation
- [ ] Detailed quality reports with actionable recommendations
- [ ] Unit tests achieve >80% coverage
- [ ] Integration tests validate end-to-end quality workflows

---

## 🏗️ Technical Implementation

### **1. CodeQualityValidator Class**

**Location:** `src/validation/code_quality_validator.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
from pathlib import Path
import subprocess
import json

@dataclass
class CoverageValidationResult:
    """Results from test coverage validation."""
    overall_coverage: float
    component_results: Dict[str, Dict]
    meets_requirements: bool
    coverage_report_path: str
    low_coverage_files: List[str]
    
    def get_components_below_threshold(self, threshold: float = 0.7) -> List[str]:
        """Get list of components below coverage threshold."""
        return [
            comp for comp, result in self.component_results.items()
            if result['line_coverage'] < threshold
        ]

@dataclass
class CodeQualityValidationResult:
    """Results from code quality metrics validation."""
    validations: Dict[str, Dict]
    meets_standards: bool
    quality_issues: List[str]
    
    def get_failing_metrics(self) -> List[str]:
        """Get list of metrics that fail standards."""
        return [
            metric for metric, result in self.validations.items()
            if not result.get('meets_standards', False)
        ]

@dataclass
class QualityConfig:
    """Configuration for quality validation."""
    min_coverage: float = 0.7  # 70% minimum coverage
    max_complexity: int = 10  # Maximum cyclomatic complexity
    min_maintainability: float = 60.0  # Minimum maintainability index
    max_duplication: float = 0.05  # Maximum 5% duplication
    components: List[str] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = [
                'data',
                'features',
                'models',
                'combination',
                'reconciliation',
                'evaluation',
                'workflows',
                'cli',
                'validation'
            ]

class CodeQualityValidator:
    """Validates code quality and test coverage."""
    
    def __init__(self, config: QualityConfig):
        """Initialize code quality validator.
        
        Args:
            config: Configuration for quality validation
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.coverage_analyzer = CoverageAnalyzer()
        self.quality_analyzer = CodeQualityAnalyzer()
    
    def validate_test_coverage(self) -> CoverageValidationResult:
        """Validate that test coverage meets requirements.
        
        Returns:
            CoverageValidationResult with detailed coverage analysis
        """
        self.logger.info("Starting test coverage validation")
        
        # Generate coverage report
        coverage_report = self.coverage_analyzer.generate_coverage_report()
        
        validation_results = {}
        low_coverage_files = []
        
        for component in self.config.components:
            self.logger.info(f"Analyzing coverage for component: {component}")
            component_coverage = coverage_report.get_component_coverage(component)
            
            meets_requirement = component_coverage.line_coverage >= self.config.min_coverage
            
            if not meets_requirement:
                low_coverage_files.extend(component_coverage.get_low_coverage_files())
            
            validation_results[component] = {
                'line_coverage': component_coverage.line_coverage,
                'branch_coverage': component_coverage.branch_coverage,
                'function_coverage': component_coverage.function_coverage,
                'meets_requirements': meets_requirement,
                'missing_coverage_files': component_coverage.get_low_coverage_files(),
                'total_lines': component_coverage.total_lines,
                'covered_lines': component_coverage.covered_lines
            }
        
        overall_coverage = coverage_report.overall_coverage
        meets_requirements = overall_coverage >= self.config.min_coverage
        
        return CoverageValidationResult(
            overall_coverage=overall_coverage,
            component_results=validation_results,
            meets_requirements=meets_requirements,
            coverage_report_path=coverage_report.generate_html_report(),
            low_coverage_files=low_coverage_files
        )
    
    def validate_code_quality(self) -> CodeQualityValidationResult:
        """Validate code quality metrics.
        
        Returns:
            CodeQualityValidationResult with quality metrics analysis
        """
        self.logger.info("Starting code quality validation")
        
        quality_metrics = self.quality_analyzer.analyze_codebase()
        
        quality_validation = {}
        quality_issues = []
        
        # Validate cyclomatic complexity
        complexity_validation = {
            'average': quality_metrics.average_complexity,
            'max': quality_metrics.max_complexity,
            'violations': quality_metrics.complexity_violations,
            'meets_standards': quality_metrics.max_complexity <= self.config.max_complexity
        }
        if not complexity_validation['meets_standards']:
            quality_issues.append(
                f"Cyclomatic complexity exceeds threshold: "
                f"{quality_metrics.max_complexity} > {self.config.max_complexity}"
            )
        quality_validation['cyclomatic_complexity'] = complexity_validation
        
        # Validate maintainability index
        maintainability_validation = {
            'average': quality_metrics.average_maintainability,
            'low_maintainability_files': quality_metrics.low_maintainability_files,
            'meets_standards': quality_metrics.average_maintainability >= self.config.min_maintainability
        }
        if not maintainability_validation['meets_standards']:
            quality_issues.append(
                f"Maintainability index below threshold: "
                f"{quality_metrics.average_maintainability} < {self.config.min_maintainability}"
            )
        quality_validation['maintainability_index'] = maintainability_validation
        
        # Validate code duplication
        duplication_validation = {
            'percentage': quality_metrics.duplication_percentage,
            'duplicate_blocks': quality_metrics.duplicate_blocks,
            'meets_standards': quality_metrics.duplication_percentage <= self.config.max_duplication
        }
        if not duplication_validation['meets_standards']:
            quality_issues.append(
                f"Code duplication exceeds threshold: "
                f"{quality_metrics.duplication_percentage*100:.1f}% > {self.config.max_duplication*100:.1f}%"
            )
        quality_validation['code_duplication'] = duplication_validation
        
        meets_standards = len(quality_issues) == 0
        
        return CodeQualityValidationResult(
            validations=quality_validation,
            meets_standards=meets_standards,
            quality_issues=quality_issues
        )
    
    def generate_quality_report(
        self,
        coverage_result: CoverageValidationResult,
        quality_result: CodeQualityValidationResult
    ) -> str:
        """Generate comprehensive quality report.
        
        Args:
            coverage_result: Coverage validation results
            quality_result: Code quality validation results
            
        Returns:
            Path to generated HTML report
        """
        from jinja2 import Template
        
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Quality Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .pass { color: green; font-weight: bold; }
                .fail { color: red; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>Code Quality Report</h1>
            
            <h2>Test Coverage Summary</h2>
            <p>Overall Coverage: <span class="{{ 'pass' if coverage_result.meets_requirements else 'fail' }}">
                {{ "%.1f"|format(coverage_result.overall_coverage * 100) }}%
            </span></p>
            
            <table>
                <tr>
                    <th>Component</th>
                    <th>Line Coverage</th>
                    <th>Branch Coverage</th>
                    <th>Status</th>
                </tr>
                {% for component, result in coverage_result.component_results.items() %}
                <tr>
                    <td>{{ component }}</td>
                    <td>{{ "%.1f"|format(result.line_coverage * 100) }}%</td>
                    <td>{{ "%.1f"|format(result.branch_coverage * 100) }}%</td>
                    <td class="{{ 'pass' if result.meets_requirements else 'fail' }}">
                        {{ 'PASS' if result.meets_requirements else 'FAIL' }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Code Quality Summary</h2>
            <p>Overall Status: <span class="{{ 'pass' if quality_result.meets_standards else 'fail' }}">
                {{ 'PASS' if quality_result.meets_standards else 'FAIL' }}
            </span></p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Status</th>
                </tr>
                {% for metric, result in quality_result.validations.items() %}
                <tr>
                    <td>{{ metric }}</td>
                    <td>{{ result.average if 'average' in result else result.percentage }}</td>
                    <td>{{ result.threshold if 'threshold' in result else 'N/A' }}</td>
                    <td class="{{ 'pass' if result.meets_standards else 'fail' }}">
                        {{ 'PASS' if result.meets_standards else 'FAIL' }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            {% if quality_result.quality_issues %}
            <h2>Quality Issues</h2>
            <ul>
                {% for issue in quality_result.quality_issues %}
                <li>{{ issue }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </body>
        </html>
        """)
        
        html_content = template.render(
            coverage_result=coverage_result,
            quality_result=quality_result
        )
        
        report_path = Path('reports/quality_report.html')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html_content)
        
        return str(report_path)

class CoverageAnalyzer:
    """Analyzes test coverage."""
    
    def generate_coverage_report(self) -> 'CoverageReport':
        """Generate coverage report using pytest-cov.
        
        Returns:
            CoverageReport with detailed coverage data
        """
        # Run pytest with coverage
        subprocess.run([
            'pytest',
            '--cov=prevcarga',
            '--cov-report=json',
            '--cov-report=html',
            'tests/'
        ], check=True)
        
        # Load coverage data
        with open('.coverage.json', 'r') as f:
            coverage_data = json.load(f)
        
        return CoverageReport(coverage_data)

class CoverageReport:
    """Represents test coverage report."""
    
    def __init__(self, coverage_data: Dict):
        """Initialize coverage report.
        
        Args:
            coverage_data: Raw coverage data from pytest-cov
        """
        self.coverage_data = coverage_data
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate coverage metrics."""
        totals = self.coverage_data.get('totals', {})
        self.overall_coverage = totals.get('percent_covered', 0) / 100.0
    
    def get_component_coverage(self, component: str) -> 'ComponentCoverage':
        """Get coverage for specific component.
        
        Args:
            component: Component name
            
        Returns:
            ComponentCoverage object
        """
        files = self.coverage_data.get('files', {})
        
        component_files = {
            path: data for path, data in files.items()
            if f'prevcarga/{component}/' in path
        }
        
        if not component_files:
            return ComponentCoverage(component, [], 0.0, 0.0, 0.0)
        
        total_lines = sum(f['summary']['num_statements'] for f in component_files.values())
        covered_lines = sum(f['summary']['covered_lines'] for f in component_files.values())
        total_branches = sum(f['summary'].get('num_branches', 0) for f in component_files.values())
        covered_branches = sum(f['summary'].get('covered_branches', 0) for f in component_files.values())
        
        line_coverage = covered_lines / total_lines if total_lines > 0 else 0.0
        branch_coverage = covered_branches / total_branches if total_branches > 0 else 0.0
        
        return ComponentCoverage(
            component=component,
            files=list(component_files.keys()),
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=line_coverage  # Simplified
        )
    
    def generate_html_report(self) -> str:
        """Generate HTML coverage report.
        
        Returns:
            Path to HTML report
        """
        return 'htmlcov/index.html'

@dataclass
class ComponentCoverage:
    """Coverage data for a component."""
    component: str
    files: List[str]
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    total_lines: int = 0
    covered_lines: int = 0
    
    def get_low_coverage_files(self, threshold: float = 0.7) -> List[str]:
        """Get files with coverage below threshold."""
        # Simplified - would need per-file analysis
        return [] if self.line_coverage >= threshold else self.files

class CodeQualityAnalyzer:
    """Analyzes code quality metrics."""
    
    def analyze_codebase(self) -> 'QualityMetrics':
        """Analyze codebase for quality metrics.
        
        Returns:
            QualityMetrics object
        """
        # Run radon for complexity and maintainability
        complexity_result = subprocess.run(
            ['radon', 'cc', 'prevcarga/', '-j'],
            capture_output=True,
            text=True
        )
        
        maintainability_result = subprocess.run(
            ['radon', 'mi', 'prevcarga/', '-j'],
            capture_output=True,
            text=True
        )
        
        complexity_data = json.loads(complexity_result.stdout) if complexity_result.stdout else {}
        maintainability_data = json.loads(maintainability_result.stdout) if maintainability_result.stdout else {}
        
        return QualityMetrics(complexity_data, maintainability_data)

@dataclass
class QualityMetrics:
    """Code quality metrics."""
    complexity_data: Dict
    maintainability_data: Dict
    
    def __post_init__(self):
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate aggregate metrics."""
        # Calculate complexity metrics
        all_complexities = []
        for file_data in self.complexity_data.values():
            for func_data in file_data:
                all_complexities.append(func_data.get('complexity', 0))
        
        self.average_complexity = sum(all_complexities) / len(all_complexities) if all_complexities else 0
        self.max_complexity = max(all_complexities) if all_complexities else 0
        self.complexity_violations = [c for c in all_complexities if c > 10]
        
        # Calculate maintainability metrics
        all_mi = []
        self.low_maintainability_files = []
        for file_path, mi_data in self.maintainability_data.items():
            mi = mi_data.get('mi', 0)
            all_mi.append(mi)
            if mi < 60:
                self.low_maintainability_files.append(file_path)
        
        self.average_maintainability = sum(all_mi) / len(all_mi) if all_mi else 0
        
        # Simplified duplication analysis
        self.duplication_percentage = 0.02  # Placeholder
        self.duplicate_blocks = []
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/validation/test_code_quality_validator.py`

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from prevcarga.validation.code_quality_validator import (
    CodeQualityValidator,
    QualityConfig,
    CoverageAnalyzer,
    CodeQualityAnalyzer
)

class TestCodeQualityValidator:
    """Test suite for CodeQualityValidator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return QualityConfig(
            min_coverage=0.7,
            max_complexity=10,
            components=['data', 'models']
        )
    
    @pytest.fixture
    def validator(self, config):
        """Create validator instance."""
        return CodeQualityValidator(config)
    
    def test_validate_test_coverage(self, validator):
        """Test coverage validation."""
        with patch.object(validator.coverage_analyzer, 'generate_coverage_report') as mock_report:
            mock_coverage = Mock()
            mock_coverage.overall_coverage = 0.75
            mock_coverage.get_component_coverage.return_value = Mock(
                line_coverage=0.75,
                branch_coverage=0.70,
                function_coverage=0.72,
                get_low_coverage_files=lambda: []
            )
            mock_coverage.generate_html_report.return_value = 'report.html'
            mock_report.return_value = mock_coverage
            
            result = validator.validate_test_coverage()
            
            assert result.overall_coverage == 0.75
            assert result.meets_requirements is True
    
    def test_validate_code_quality(self, validator):
        """Test code quality validation."""
        with patch.object(validator.quality_analyzer, 'analyze_codebase') as mock_analyze:
            mock_metrics = Mock()
            mock_metrics.average_complexity = 5
            mock_metrics.max_complexity = 8
            mock_metrics.complexity_violations = []
            mock_metrics.average_maintainability = 70
            mock_metrics.low_maintainability_files = []
            mock_metrics.duplication_percentage = 0.03
            mock_metrics.duplicate_blocks = []
            mock_analyze.return_value = mock_metrics
            
            result = validator.validate_code_quality()
            
            assert result.meets_standards is True
            assert len(result.quality_issues) == 0
```

---

## 📊 Success Metrics

- [ ] Overall test coverage >70%
- [ ] All components meet coverage threshold
- [ ] Code quality metrics meet all standards
- [ ] Cyclomatic complexity <10
- [ ] Maintainability index >60
- [ ] Code duplication <5%

---

## 🔗 Dependencies

**Requires:**
- All codebase components (for analysis)
- pytest, pytest-cov (test coverage)
- radon (code metrics)

**Blocks:**
- PC-091-10B (Acceptance Validator - needs quality validation)

---

## 📚 Documentation

- [ ] Code quality framework documented
- [ ] Quality metrics and thresholds documented
- [ ] Coverage analysis usage guide
- [ ] CI/CD integration guide

---

## 🔄 Implementation Steps

1. **Day 1-2: Coverage Analysis**
   - Implement `CoverageAnalyzer` class
   - Implement coverage report generation
   - Create component-level analysis

2. **Day 3-4: Quality Analysis**
   - Implement `CodeQualityAnalyzer` class
   - Integrate radon for metrics
   - Create quality validation logic

3. **Day 5: Testing and Reporting**
   - Complete unit tests
   - Implement quality report generation
   - Write documentation

---

**Estimated Completion:** Week 29, Day 5  
**Review Required:** QA Team, Tech Lead
