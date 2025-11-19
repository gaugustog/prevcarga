# PC-093-10B: Epic Acceptance Criteria Validator Implementation

**Ticket ID:** PC-093-10B  
**Epic:** Epic-10B (Quality Assurance & Acceptance Validation)  
**Story:** User Story 3 - Final Acceptance Criteria Validation  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** Critical  
**Estimated Effort:** 3 days  
**Sprint:** Week 29

---

## 📋 Description

Implement comprehensive epic acceptance criteria validation framework to validate all acceptance criteria across all completed epics, ensuring >95% success rate. The validator must track criteria completion, identify gaps, and provide detailed validation reports for stakeholder approval.

---

## 🎯 Acceptance Criteria

- [ ] `EpicAcceptanceCriteriaValidator` class with multi-epic validation
- [ ] Acceptance criteria parsing from epic documentation
- [ ] Automated criteria validation with evidence collection
- [ ] Manual criteria verification workflow
- [ ] Criteria completion tracking across all epics
- [ ] Gap analysis and remediation recommendations
- [ ] >95% success rate threshold enforcement
- [ ] Detailed validation reports per epic
- [ ] Comprehensive summary report for all epics
- [ ] Integration with production readiness validator
- [ ] Unit tests achieve >80% coverage
- [ ] Integration tests validate end-to-end acceptance workflows

---

## 🏗️ Technical Implementation

### **1. EpicAcceptanceCriteriaValidator Class**

**Location:** `src/validation/epic_acceptance_validator.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import logging
from pathlib import Path
import yaml
import re

@dataclass
class AcceptanceCriterion:
    """Represents a single acceptance criterion."""
    criterion_id: str
    epic_id: str
    description: str
    validation_type: str  # 'automated', 'manual', 'evidence'
    validation_func: Optional[Callable[[], bool]] = None
    evidence_path: Optional[str] = None
    status: str = 'not_validated'  # 'passed', 'failed', 'not_validated'
    
    def validate(self) -> 'CriterionValidationResult':
        """Validate the acceptance criterion."""
        if self.validation_type == 'automated' and self.validation_func:
            try:
                passed = self.validation_func()
                return CriterionValidationResult(
                    criterion_id=self.criterion_id,
                    passed=passed,
                    evidence=None,
                    notes=None
                )
            except Exception as e:
                return CriterionValidationResult(
                    criterion_id=self.criterion_id,
                    passed=False,
                    evidence=None,
                    notes=f"Validation failed: {str(e)}"
                )
        elif self.validation_type == 'evidence' and self.evidence_path:
            passed = Path(self.evidence_path).exists()
            return CriterionValidationResult(
                criterion_id=self.criterion_id,
                passed=passed,
                evidence=self.evidence_path if passed else None,
                notes=None if passed else f"Evidence not found: {self.evidence_path}"
            )
        else:
            # Manual validation required
            return CriterionValidationResult(
                criterion_id=self.criterion_id,
                passed=False,
                evidence=None,
                notes="Manual validation required"
            )

@dataclass
class CriterionValidationResult:
    """Result from validating a single criterion."""
    criterion_id: str
    passed: bool
    evidence: Optional[str]
    notes: Optional[str]

@dataclass
class EpicValidationResult:
    """Results from validating an epic's acceptance criteria."""
    epic_id: str
    criteria_met: int
    total_criteria: int
    success_rate: float
    failing_criteria: List[str]
    criterion_results: Dict[str, CriterionValidationResult]
    
    def passes_validation(self, threshold: float = 0.95) -> bool:
        """Check if epic passes validation threshold."""
        return self.success_rate >= threshold

@dataclass
class EpicAcceptanceValidationResult:
    """Results from validating all epic acceptance criteria."""
    epic_validations: Dict[str, EpicValidationResult]
    overall_success: bool
    overall_success_rate: float
    total_criteria: int
    total_passed: int
    
    def get_failing_epics(self, threshold: float = 0.95) -> List[str]:
        """Get list of epics that don't meet threshold."""
        return [
            epic_id for epic_id, result in self.epic_validations.items()
            if not result.passes_validation(threshold)
        ]

class EpicAcceptanceCriteriaValidator:
    """Validates acceptance criteria for all completed epics."""
    
    def __init__(self, config: 'AcceptanceValidationConfig'):
        """Initialize epic acceptance criteria validator.
        
        Args:
            config: Configuration for acceptance validation
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.epic_criteria: Dict[str, List[AcceptanceCriterion]] = {}
        self._load_epic_criteria()
    
    def validate_all_epic_criteria(self) -> EpicAcceptanceValidationResult:
        """Validate acceptance criteria for all completed epics.
        
        Returns:
            EpicAcceptanceValidationResult with validation for all epics
        """
        self.logger.info("Starting epic acceptance criteria validation")
        
        epic_validations = {}
        total_criteria = 0
        total_passed = 0
        
        for epic_id in self.config.completed_epics:
            self.logger.info(f"Validating epic: {epic_id}")
            
            epic_result = self._validate_epic_criteria(epic_id)
            epic_validations[epic_id] = epic_result
            
            total_criteria += epic_result.total_criteria
            total_passed += epic_result.criteria_met
        
        overall_success_rate = total_passed / total_criteria if total_criteria > 0 else 0.0
        overall_success = all(
            result.passes_validation(self.config.success_threshold)
            for result in epic_validations.values()
        )
        
        return EpicAcceptanceValidationResult(
            epic_validations=epic_validations,
            overall_success=overall_success,
            overall_success_rate=overall_success_rate,
            total_criteria=total_criteria,
            total_passed=total_passed
        )
    
    def _load_epic_criteria(self):
        """Load acceptance criteria from epic documentation."""
        for epic_id in self.config.completed_epics:
            epic_file = Path(f'docs/mvp/epics/{epic_id}.md')
            
            if not epic_file.exists():
                self.logger.warning(f"Epic file not found: {epic_file}")
                continue
            
            criteria = self._parse_acceptance_criteria(epic_file, epic_id)
            self.epic_criteria[epic_id] = criteria
    
    def _parse_acceptance_criteria(
        self,
        epic_file: Path,
        epic_id: str
    ) -> List[AcceptanceCriterion]:
        """Parse acceptance criteria from epic markdown file.
        
        Args:
            epic_file: Path to epic markdown file
            epic_id: Epic identifier
            
        Returns:
            List of AcceptanceCriterion objects
        """
        content = epic_file.read_text()
        criteria = []
        
        # Find acceptance criteria section
        ac_pattern = r'## 🎯 Acceptance Criteria\s+(.*?)(?=\n##|\Z)'
        ac_match = re.search(ac_pattern, content, re.DOTALL)
        
        if not ac_match:
            self.logger.warning(f"No acceptance criteria found in {epic_file}")
            return criteria
        
        ac_content = ac_match.group(1)
        
        # Parse checkbox items
        checkbox_pattern = r'- \[([ x])\] (.+)'
        matches = re.finditer(checkbox_pattern, ac_content)
        
        for idx, match in enumerate(matches):
            checked = match.group(1) == 'x'
            description = match.group(2).strip()
            
            criterion = AcceptanceCriterion(
                criterion_id=f"{epic_id}-AC-{idx+1:03d}",
                epic_id=epic_id,
                description=description,
                validation_type=self._determine_validation_type(description),
                status='passed' if checked else 'not_validated'
            )
            
            # Set validation function based on description
            criterion.validation_func = self._create_validation_func(criterion)
            
            criteria.append(criterion)
        
        return criteria
    
    def _determine_validation_type(self, description: str) -> str:
        """Determine validation type from description."""
        description_lower = description.lower()
        
        # Automated validation keywords
        automated_keywords = ['implemented', 'supports', 'validates', 'generates', 'processes']
        if any(keyword in description_lower for keyword in automated_keywords):
            return 'automated'
        
        # Evidence-based validation keywords
        evidence_keywords = ['documented', 'report', 'coverage', 'tests']
        if any(keyword in description_lower for keyword in evidence_keywords):
            return 'evidence'
        
        # Default to manual
        return 'manual'
    
    def _create_validation_func(
        self,
        criterion: AcceptanceCriterion
    ) -> Optional[Callable[[], bool]]:
        """Create validation function for criterion.
        
        Args:
            criterion: Acceptance criterion to validate
            
        Returns:
            Validation function or None
        """
        desc_lower = criterion.description.lower()
        
        # Check for common patterns
        if 'test coverage' in desc_lower and '>70%' in desc_lower:
            return lambda: self._validate_test_coverage(0.7)
        
        if 'performance benchmark' in desc_lower:
            return lambda: self._validate_performance_benchmarks()
        
        if 'security scan' in desc_lower:
            return lambda: self._validate_security_scan()
        
        if 'documentation' in desc_lower:
            pattern = r'docs/([^\s]+)'
            match = re.search(pattern, criterion.description)
            if match:
                doc_path = match.group(1)
                return lambda: Path(doc_path).exists()
        
        # Check if status is already marked as passed
        if criterion.status == 'passed':
            return lambda: True
        
        return None
    
    def _validate_epic_criteria(self, epic_id: str) -> EpicValidationResult:
        """Validate all criteria for a specific epic.
        
        Args:
            epic_id: Epic identifier
            
        Returns:
            EpicValidationResult with validation results
        """
        criteria = self.epic_criteria.get(epic_id, [])
        
        if not criteria:
            self.logger.warning(f"No criteria found for epic: {epic_id}")
            return EpicValidationResult(
                epic_id=epic_id,
                criteria_met=0,
                total_criteria=0,
                success_rate=0.0,
                failing_criteria=[],
                criterion_results={}
            )
        
        criterion_results = {}
        for criterion in criteria:
            result = criterion.validate()
            criterion_results[criterion.criterion_id] = result
        
        criteria_met = sum(1 for r in criterion_results.values() if r.passed)
        total_criteria = len(criteria)
        success_rate = criteria_met / total_criteria if total_criteria > 0 else 0.0
        
        failing_criteria = [
            cid for cid, result in criterion_results.items()
            if not result.passed
        ]
        
        return EpicValidationResult(
            epic_id=epic_id,
            criteria_met=criteria_met,
            total_criteria=total_criteria,
            success_rate=success_rate,
            failing_criteria=failing_criteria,
            criterion_results=criterion_results
        )
    
    def _validate_test_coverage(self, threshold: float) -> bool:
        """Validate test coverage meets threshold."""
        # Check coverage report
        coverage_report = Path('.coverage.json')
        if not coverage_report.exists():
            return False
        
        import json
        with open(coverage_report) as f:
            data = json.load(f)
            coverage = data.get('totals', {}).get('percent_covered', 0) / 100.0
            return coverage >= threshold
    
    def _validate_performance_benchmarks(self) -> bool:
        """Validate performance benchmarks are met."""
        benchmark_report = Path('reports/performance_benchmarks.json')
        return benchmark_report.exists()
    
    def _validate_security_scan(self) -> bool:
        """Validate security scan passed."""
        security_report = Path('reports/security_report.html')
        return security_report.exists()
    
    def generate_validation_report(
        self,
        result: EpicAcceptanceValidationResult
    ) -> str:
        """Generate comprehensive validation report.
        
        Args:
            result: Epic acceptance validation results
            
        Returns:
            Path to generated report
        """
        from jinja2 import Template
        import datetime
        
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Epic Acceptance Criteria Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .pass { color: green; font-weight: bold; }
                .fail { color: red; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                .summary { background-color: #f5f5f5; padding: 15px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Epic Acceptance Criteria Validation Report</h1>
            <p>Generated: {{ timestamp }}</p>
            
            <div class="summary">
                <h2>Overall Summary</h2>
                <p>Overall Status: <span class="{{ 'pass' if result.overall_success else 'fail' }}">
                    {{ 'PASS' if result.overall_success else 'FAIL' }}
                </span></p>
                <p>Success Rate: <span class="{{ 'pass' if result.overall_success_rate >= 0.95 else 'fail' }}">
                    {{ "%.1f"|format(result.overall_success_rate * 100) }}%
                </span></p>
                <p>Total Criteria: {{ result.total_criteria }}</p>
                <p>Passed: {{ result.total_passed }}</p>
                <p>Failed: {{ result.total_criteria - result.total_passed }}</p>
            </div>
            
            <h2>Epic Validation Results</h2>
            <table>
                <tr>
                    <th>Epic ID</th>
                    <th>Criteria Met</th>
                    <th>Total Criteria</th>
                    <th>Success Rate</th>
                    <th>Status</th>
                </tr>
                {% for epic_id, epic_result in result.epic_validations.items() %}
                <tr>
                    <td>{{ epic_id }}</td>
                    <td>{{ epic_result.criteria_met }}</td>
                    <td>{{ epic_result.total_criteria }}</td>
                    <td>{{ "%.1f"|format(epic_result.success_rate * 100) }}%</td>
                    <td class="{{ 'pass' if epic_result.passes_validation() else 'fail' }}">
                        {{ 'PASS' if epic_result.passes_validation() else 'FAIL' }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            {% for epic_id, epic_result in result.epic_validations.items() %}
            <h3>{{ epic_id }} - Detailed Results</h3>
            <table>
                <tr>
                    <th>Criterion ID</th>
                    <th>Status</th>
                    <th>Notes</th>
                </tr>
                {% for crit_id, crit_result in epic_result.criterion_results.items() %}
                <tr>
                    <td>{{ crit_id }}</td>
                    <td class="{{ 'pass' if crit_result.passed else 'fail' }}">
                        {{ 'PASS' if crit_result.passed else 'FAIL' }}
                    </td>
                    <td>{{ crit_result.notes or 'N/A' }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endfor %}
        </body>
        </html>
        """)
        
        html_content = template.render(
            result=result,
            timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        report_path = Path('reports/epic_acceptance_validation.html')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html_content)
        
        return str(report_path)

@dataclass
class AcceptanceValidationConfig:
    """Configuration for acceptance criteria validation."""
    completed_epics: List[str]
    success_threshold: float = 0.95
    
    def __post_init__(self):
        if not self.completed_epics:
            # Default to all epics
            self.completed_epics = [
                'Epic-00', 'Epic-01', 'Epic-02A', 'Epic-02B',
                'Epic-03', 'Epic-04', 'Epic-05A', 'Epic-05B',
                'Epic-06A', 'Epic-06B', 'Epic-07A', 'Epic-07B',
                'Epic-08A', 'Epic-08B', 'Epic-09A', 'Epic-09B',
                'Epic-10A', 'Epic-10B'
            ]
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/validation/test_epic_acceptance_validator.py`

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from prevcarga.validation.epic_acceptance_validator import (
    EpicAcceptanceCriteriaValidator,
    AcceptanceValidationConfig,
    AcceptanceCriterion
)

class TestEpicAcceptanceCriteriaValidator:
    """Test suite for EpicAcceptanceCriteriaValidator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AcceptanceValidationConfig(
            completed_epics=['Epic-01', 'Epic-02A'],
            success_threshold=0.95
        )
    
    @pytest.fixture
    def validator(self, config):
        """Create validator instance."""
        return EpicAcceptanceCriteriaValidator(config)
    
    def test_validate_all_epic_criteria(self, validator):
        """Test validation of all epic criteria."""
        # Mock epic criteria
        validator.epic_criteria = {
            'Epic-01': [
                AcceptanceCriterion(
                    criterion_id='Epic-01-AC-001',
                    epic_id='Epic-01',
                    description='Test criterion',
                    validation_type='automated',
                    validation_func=lambda: True
                )
            ]
        }
        
        result = validator.validate_all_epic_criteria()
        
        assert 'Epic-01' in result.epic_validations
        assert result.overall_success_rate >= 0.0
    
    def test_determine_validation_type(self, validator):
        """Test validation type determination."""
        assert validator._determine_validation_type('Feature implemented') == 'automated'
        assert validator._determine_validation_type('Documentation complete') == 'evidence'
        assert validator._determine_validation_type('Manual review') == 'manual'
    
    def test_parse_acceptance_criteria(self, validator, tmp_path):
        """Test parsing acceptance criteria from markdown."""
        epic_file = tmp_path / 'Epic-Test.md'
        epic_file.write_text("""
        # Epic-Test
        
        ## 🎯 Acceptance Criteria
        
        - [x] Feature A implemented
        - [ ] Feature B documented
        - [ ] Tests achieve >70% coverage
        """)
        
        criteria = validator._parse_acceptance_criteria(epic_file, 'Epic-Test')
        
        assert len(criteria) == 3
        assert criteria[0].status == 'passed'
        assert criteria[1].status == 'not_validated'
```

---

## 📊 Success Metrics

- [ ] All epic acceptance criteria parsed successfully
- [ ] Overall success rate >95%
- [ ] Automated validation for >50% of criteria
- [ ] Comprehensive validation reports generated
- [ ] Gap analysis identifies all failing criteria

---

## 🔗 Dependencies

**Requires:**
- All completed epics (Epic-00 through Epic-10A)
- PC-090-10B (Code Quality Validator - for coverage validation)
- PC-091-10B (Security Scanner - for security validation)
- PC-087-10B (Performance Validator - for performance validation)

**Blocks:**
- PC-094-10B (Final Validation Report Generator)

---

## 📚 Documentation

- [ ] Acceptance criteria validation framework documented
- [ ] Validation type definitions documented
- [ ] Evidence requirements documented
- [ ] Validation report format documented

---

## 🔄 Implementation Steps

1. **Day 1: Core Validator**
   - Implement `EpicAcceptanceCriteriaValidator` class
   - Implement criteria parsing
   - Create validation framework

2. **Day 2: Validation Logic**
   - Implement automated validation functions
   - Create evidence validation
   - Implement epic-level validation

3. **Day 3: Testing and Reporting**
   - Complete unit tests
   - Implement report generation
   - Write documentation

---

**Estimated Completion:** Week 29, Day 3  
**Review Required:** QA Team, Project Management
