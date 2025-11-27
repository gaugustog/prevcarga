"""Tests for the epic acceptance criteria validator.

Tests cover:
- ValidationType and CriterionStatus enums
- AcceptanceValidationConfig dataclass
- CriterionValidationResult dataclass
- AcceptanceCriterion dataclass and validation
- EpicValidationResult dataclass
- EpicAcceptanceValidationResult dataclass
- GapAnalysisResult dataclass
- EpicAcceptanceCriteriaValidator class methods
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.validation.epic_acceptance_validator import (
    AcceptanceCriterion,
    AcceptanceValidationConfig,
    CriterionStatus,
    CriterionValidationResult,
    EpicAcceptanceCriteriaValidator,
    EpicAcceptanceValidationResult,
    EpicValidationResult,
    GapAnalysisResult,
    ValidationType,
)


class TestValidationType:
    """Tests for ValidationType enum."""

    def test_validation_type_values(self) -> None:
        """Test ValidationType enum values."""
        assert ValidationType.AUTOMATED.value == "automated"
        assert ValidationType.EVIDENCE.value == "evidence"
        assert ValidationType.MANUAL.value == "manual"

    def test_validation_type_members(self) -> None:
        """Test that all expected members exist."""
        members = [m.value for m in ValidationType]
        assert "automated" in members
        assert "evidence" in members
        assert "manual" in members
        assert len(members) == 3


class TestCriterionStatus:
    """Tests for CriterionStatus enum."""

    def test_criterion_status_values(self) -> None:
        """Test CriterionStatus enum values."""
        assert CriterionStatus.NOT_VALIDATED.value == "not_validated"
        assert CriterionStatus.PASSED.value == "passed"
        assert CriterionStatus.FAILED.value == "failed"
        assert CriterionStatus.SKIPPED.value == "skipped"

    def test_criterion_status_members(self) -> None:
        """Test that all expected members exist."""
        members = [m.value for m in CriterionStatus]
        assert len(members) == 4


class TestAcceptanceValidationConfig:
    """Tests for AcceptanceValidationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AcceptanceValidationConfig()

        assert len(config.completed_epics) > 0
        assert "Epic-00" in config.completed_epics
        assert "Epic-10B" in config.completed_epics
        assert config.success_threshold == 0.95
        assert config.epic_docs_path == "docs/mvp/epics"
        assert "reports" in config.evidence_paths
        assert config.skip_manual_criteria is False

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AcceptanceValidationConfig(
            completed_epics=["Epic-00", "Epic-01"],
            success_threshold=0.90,
            epic_docs_path="/custom/path",
            evidence_paths=["custom/evidence"],
            skip_manual_criteria=True,
        )

        assert config.completed_epics == ["Epic-00", "Epic-01"]
        assert config.success_threshold == 0.90
        assert config.epic_docs_path == "/custom/path"
        assert config.evidence_paths == ["custom/evidence"]
        assert config.skip_manual_criteria is True

    def test_config_to_dict(self) -> None:
        """Test configuration to_dict method."""
        config = AcceptanceValidationConfig(
            completed_epics=["Epic-00"],
            success_threshold=0.85,
        )

        result = config.to_dict()

        assert result["completed_epics"] == ["Epic-00"]
        assert result["success_threshold"] == 0.85
        assert "epic_docs_path" in result
        assert "evidence_paths" in result
        assert "skip_manual_criteria" in result


class TestCriterionValidationResult:
    """Tests for CriterionValidationResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic result creation."""
        result = CriterionValidationResult(
            criterion_id="Epic-00-AC-001",
            passed=True,
        )

        assert result.criterion_id == "Epic-00-AC-001"
        assert result.passed is True
        assert result.evidence is None
        assert result.notes is None
        assert result.validation_type == ValidationType.AUTOMATED
        assert isinstance(result.timestamp, datetime)

    def test_result_with_evidence(self) -> None:
        """Test result with evidence."""
        result = CriterionValidationResult(
            criterion_id="Epic-00-AC-001",
            passed=True,
            evidence="/path/to/report.html",
            notes="Report generated successfully",
            validation_type=ValidationType.EVIDENCE,
        )

        assert result.evidence == "/path/to/report.html"
        assert result.notes == "Report generated successfully"
        assert result.validation_type == ValidationType.EVIDENCE

    def test_result_to_dict(self) -> None:
        """Test result to_dict method."""
        result = CriterionValidationResult(
            criterion_id="Epic-00-AC-001",
            passed=False,
            notes="Failed validation",
            validation_type=ValidationType.MANUAL,
        )

        data = result.to_dict()

        assert data["criterion_id"] == "Epic-00-AC-001"
        assert data["passed"] is False
        assert data["notes"] == "Failed validation"
        assert data["validation_type"] == "manual"
        assert "timestamp" in data


class TestAcceptanceCriterion:
    """Tests for AcceptanceCriterion dataclass."""

    def test_basic_criterion(self) -> None:
        """Test basic criterion creation."""
        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="All tickets completed",
        )

        assert criterion.criterion_id == "Epic-00-AC-001"
        assert criterion.epic_id == "Epic-00"
        assert criterion.description == "All tickets completed"
        assert criterion.validation_type == ValidationType.MANUAL
        assert criterion.validation_func is None
        assert criterion.status == CriterionStatus.NOT_VALIDATED
        assert criterion.is_checked is False

    def test_criterion_with_validation_func(self) -> None:
        """Test criterion with validation function."""
        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="Test coverage > 80%",
            validation_type=ValidationType.AUTOMATED,
            validation_func=lambda: True,
        )

        result = criterion.validate()

        assert result.passed is True
        assert result.validation_type == ValidationType.AUTOMATED

    def test_criterion_validate_when_checked(self) -> None:
        """Test validation when criterion is already checked."""
        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="Implementation complete",
            is_checked=True,
        )

        result = criterion.validate()

        assert result.passed is True
        assert "Marked as completed" in result.notes

    def test_criterion_validate_evidence_exists(self) -> None:
        """Test validation with evidence file that exists."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            evidence_path = f.name

        try:
            criterion = AcceptanceCriterion(
                criterion_id="Epic-00-AC-001",
                epic_id="Epic-00",
                description="Report generated",
                validation_type=ValidationType.EVIDENCE,
                evidence_path=evidence_path,
            )

            result = criterion.validate()

            assert result.passed is True
            assert result.evidence == evidence_path
        finally:
            Path(evidence_path).unlink(missing_ok=True)

    def test_criterion_validate_evidence_missing(self) -> None:
        """Test validation with missing evidence file."""
        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="Report generated",
            validation_type=ValidationType.EVIDENCE,
            evidence_path="/nonexistent/report.html",
        )

        result = criterion.validate()

        assert result.passed is False
        assert "Evidence not found" in result.notes

    def test_criterion_validate_automated_fails(self) -> None:
        """Test automated validation that fails."""
        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="Tests pass",
            validation_type=ValidationType.AUTOMATED,
            validation_func=lambda: False,
        )

        result = criterion.validate()

        assert result.passed is False
        assert "Automated validation failed" in result.notes

    def test_criterion_validate_automated_exception(self) -> None:
        """Test automated validation that raises exception."""
        def failing_func() -> bool:
            raise ValueError("Test error")

        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="Tests pass",
            validation_type=ValidationType.AUTOMATED,
            validation_func=failing_func,
        )

        result = criterion.validate()

        assert result.passed is False
        assert "Validation error" in result.notes
        assert "Test error" in result.notes

    def test_criterion_validate_manual_required(self) -> None:
        """Test manual validation required."""
        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="Review by team lead",
            validation_type=ValidationType.MANUAL,
        )

        result = criterion.validate()

        assert result.passed is False
        assert "Manual validation required" in result.notes

    def test_criterion_to_dict(self) -> None:
        """Test criterion to_dict method."""
        criterion = AcceptanceCriterion(
            criterion_id="Epic-00-AC-001",
            epic_id="Epic-00",
            description="Test description",
            validation_type=ValidationType.EVIDENCE,
            evidence_path="/path/to/evidence.html",
            status=CriterionStatus.PASSED,
            is_checked=True,
        )

        data = criterion.to_dict()

        assert data["criterion_id"] == "Epic-00-AC-001"
        assert data["epic_id"] == "Epic-00"
        assert data["description"] == "Test description"
        assert data["validation_type"] == "evidence"
        assert data["evidence_path"] == "/path/to/evidence.html"
        assert data["status"] == "passed"
        assert data["is_checked"] is True


class TestEpicValidationResult:
    """Tests for EpicValidationResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic result creation."""
        result = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=9,
            total_criteria=10,
            success_rate=0.90,
            failing_criteria=["Epic-00-AC-010"],
            criterion_results={},
        )

        assert result.epic_id == "Epic-00"
        assert result.criteria_met == 9
        assert result.total_criteria == 10
        assert result.success_rate == 0.90
        assert len(result.failing_criteria) == 1

    def test_passes_validation_above_threshold(self) -> None:
        """Test passes_validation above threshold."""
        result = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=96,
            total_criteria=100,
            success_rate=0.96,
            failing_criteria=[],
            criterion_results={},
        )

        assert result.passes_validation() is True
        assert result.passes_validation(0.95) is True
        assert result.passes_validation(0.96) is True
        assert result.passes_validation(0.97) is False

    def test_passes_validation_below_threshold(self) -> None:
        """Test passes_validation below threshold."""
        result = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=90,
            total_criteria=100,
            success_rate=0.90,
            failing_criteria=[],
            criterion_results={},
        )

        assert result.passes_validation() is False
        assert result.passes_validation(0.90) is True
        assert result.passes_validation(0.85) is True

    def test_get_failing_criteria_details(self) -> None:
        """Test get_failing_criteria_details method."""
        crit_result = CriterionValidationResult(
            criterion_id="Epic-00-AC-001",
            passed=False,
            notes="Failed test",
            validation_type=ValidationType.AUTOMATED,
        )

        result = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=0,
            total_criteria=1,
            success_rate=0.0,
            failing_criteria=["Epic-00-AC-001"],
            criterion_results={"Epic-00-AC-001": crit_result},
        )

        details = result.get_failing_criteria_details()

        assert len(details) == 1
        assert details[0]["criterion_id"] == "Epic-00-AC-001"
        assert details[0]["notes"] == "Failed test"
        assert details[0]["validation_type"] == "automated"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        result = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=5,
            total_criteria=5,
            success_rate=1.0,
            failing_criteria=[],
            criterion_results={},
        )

        data = result.to_dict()

        assert data["epic_id"] == "Epic-00"
        assert data["criteria_met"] == 5
        assert data["total_criteria"] == 5
        assert data["success_rate"] == 1.0
        assert "timestamp" in data
        assert data["passes_validation"] is True


class TestEpicAcceptanceValidationResult:
    """Tests for EpicAcceptanceValidationResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> EpicAcceptanceValidationResult:
        """Create a sample validation result."""
        epic_result_pass = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=10,
            total_criteria=10,
            success_rate=1.0,
            failing_criteria=[],
            criterion_results={},
        )

        epic_result_fail = EpicValidationResult(
            epic_id="Epic-01",
            criteria_met=8,
            total_criteria=10,
            success_rate=0.80,
            failing_criteria=["Epic-01-AC-001", "Epic-01-AC-002"],
            criterion_results={},
        )

        return EpicAcceptanceValidationResult(
            epic_validations={"Epic-00": epic_result_pass, "Epic-01": epic_result_fail},
            overall_success=False,
            overall_success_rate=0.90,
            total_criteria=20,
            total_passed=18,
        )

    def test_get_failing_epics(self, sample_result: EpicAcceptanceValidationResult) -> None:
        """Test get_failing_epics method."""
        failing = sample_result.get_failing_epics()

        assert "Epic-01" in failing
        assert "Epic-00" not in failing

    def test_get_passing_epics(self, sample_result: EpicAcceptanceValidationResult) -> None:
        """Test get_passing_epics method."""
        passing = sample_result.get_passing_epics()

        assert "Epic-00" in passing
        assert "Epic-01" not in passing

    def test_get_all_failing_criteria(self) -> None:
        """Test get_all_failing_criteria method."""
        crit_result = CriterionValidationResult(
            criterion_id="Epic-01-AC-001",
            passed=False,
            notes="Test failed",
            validation_type=ValidationType.AUTOMATED,
        )

        epic_result = EpicValidationResult(
            epic_id="Epic-01",
            criteria_met=0,
            total_criteria=1,
            success_rate=0.0,
            failing_criteria=["Epic-01-AC-001"],
            criterion_results={"Epic-01-AC-001": crit_result},
        )

        result = EpicAcceptanceValidationResult(
            epic_validations={"Epic-01": epic_result},
            overall_success=False,
            overall_success_rate=0.0,
            total_criteria=1,
            total_passed=0,
        )

        all_failing = result.get_all_failing_criteria()

        assert len(all_failing) == 1
        assert all_failing[0]["epic_id"] == "Epic-01"
        assert all_failing[0]["criterion_id"] == "Epic-01-AC-001"

    def test_to_dict(self, sample_result: EpicAcceptanceValidationResult) -> None:
        """Test to_dict method."""
        data = sample_result.to_dict()

        assert "epic_validations" in data
        assert data["overall_success"] is False
        assert data["overall_success_rate"] == 0.90
        assert data["total_criteria"] == 20
        assert data["total_passed"] == 18
        assert data["total_failed"] == 2
        assert "failing_epics" in data
        assert "passing_epics" in data


class TestGapAnalysisResult:
    """Tests for GapAnalysisResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic result creation."""
        result = GapAnalysisResult(
            total_gaps=5,
            critical_gaps=2,
            gaps_by_epic={"Epic-00": [{"criterion_id": "AC-001"}]},
            remediation_recommendations=["Fix Epic-00"],
        )

        assert result.total_gaps == 5
        assert result.critical_gaps == 2
        assert "Epic-00" in result.gaps_by_epic
        assert len(result.remediation_recommendations) == 1

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        result = GapAnalysisResult(
            total_gaps=3,
            critical_gaps=1,
            gaps_by_epic={"Epic-00": []},
            remediation_recommendations=["Rec1", "Rec2"],
        )

        data = result.to_dict()

        assert data["total_gaps"] == 3
        assert data["critical_gaps"] == 1
        assert "gaps_by_epic" in data
        assert len(data["remediation_recommendations"]) == 2


class TestEpicAcceptanceCriteriaValidator:
    """Tests for EpicAcceptanceCriteriaValidator class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        validator = EpicAcceptanceCriteriaValidator()

        assert validator.config is not None
        assert len(validator.config.completed_epics) > 0
        assert validator.epic_criteria == {}
        assert validator._validation_cache == {}

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = AcceptanceValidationConfig(
            completed_epics=["Epic-00"],
            success_threshold=0.90,
        )
        validator = EpicAcceptanceCriteriaValidator(config)

        assert validator.config == config
        assert validator.config.success_threshold == 0.90

    def test_determine_validation_type_automated(self) -> None:
        """Test validation type determination for automated."""
        validator = EpicAcceptanceCriteriaValidator()

        assert validator._determine_validation_type("Module implemented") == ValidationType.AUTOMATED
        assert validator._determine_validation_type("Function supports X") == ValidationType.AUTOMATED
        assert validator._determine_validation_type("Plugin class created") == ValidationType.AUTOMATED

    def test_determine_validation_type_evidence(self) -> None:
        """Test validation type determination for evidence."""
        validator = EpicAcceptanceCriteriaValidator()

        assert validator._determine_validation_type("Documentation exists") == ValidationType.EVIDENCE
        assert validator._determine_validation_type("Report generated") == ValidationType.EVIDENCE
        assert validator._determine_validation_type("Test coverage > 80%") == ValidationType.EVIDENCE

    def test_determine_validation_type_manual(self) -> None:
        """Test validation type determination for manual."""
        validator = EpicAcceptanceCriteriaValidator()

        assert validator._determine_validation_type("Review completed") == ValidationType.MANUAL
        assert validator._determine_validation_type("Approved by team") == ValidationType.MANUAL

    def test_create_default_criteria(self) -> None:
        """Test default criteria creation."""
        validator = EpicAcceptanceCriteriaValidator()

        criteria = validator._create_default_criteria("Epic-99")

        assert len(criteria) == 1
        assert criteria[0].epic_id == "Epic-99"
        assert criteria[0].is_checked is True
        assert criteria[0].status == CriterionStatus.PASSED

    def test_validate_epic_criteria_empty(self) -> None:
        """Test validation with no criteria."""
        validator = EpicAcceptanceCriteriaValidator()

        result = validator._validate_epic_criteria("Epic-99")

        assert result.epic_id == "Epic-99"
        assert result.total_criteria == 0
        assert result.success_rate == 1.0  # Empty considered passing

    def test_validate_epic_criteria_with_criteria(self) -> None:
        """Test validation with criteria."""
        validator = EpicAcceptanceCriteriaValidator()

        # Manually add criteria
        validator.epic_criteria["Epic-00"] = [
            AcceptanceCriterion(
                criterion_id="Epic-00-AC-001",
                epic_id="Epic-00",
                description="Test criterion",
                is_checked=True,
            ),
            AcceptanceCriterion(
                criterion_id="Epic-00-AC-002",
                epic_id="Epic-00",
                description="Another criterion",
                validation_type=ValidationType.MANUAL,
                is_checked=False,
            ),
        ]

        result = validator._validate_epic_criteria("Epic-00")

        assert result.epic_id == "Epic-00"
        assert result.total_criteria == 2
        assert result.criteria_met == 1  # Only checked criterion passes
        assert result.success_rate == 0.5

    def test_validation_caching(self) -> None:
        """Test that validation results are cached."""
        validator = EpicAcceptanceCriteriaValidator()

        call_count = 0
        def counting_func() -> bool:
            nonlocal call_count
            call_count += 1
            return True

        validator.epic_criteria["Epic-00"] = [
            AcceptanceCriterion(
                criterion_id="Epic-00-AC-001",
                epic_id="Epic-00",
                description="Test criterion",
                validation_type=ValidationType.AUTOMATED,
                validation_func=counting_func,
            ),
        ]

        # First validation
        validator._validate_epic_criteria("Epic-00")
        assert call_count == 1

        # Second validation should use cache
        validator._validate_epic_criteria("Epic-00")
        assert call_count == 1  # Should not increase

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        validator = EpicAcceptanceCriteriaValidator()

        validator._validation_cache["test"] = MagicMock()
        assert len(validator._validation_cache) == 1

        validator.clear_cache()

        assert len(validator._validation_cache) == 0

    def test_is_critical_criterion(self) -> None:
        """Test critical criterion detection."""
        validator = EpicAcceptanceCriteriaValidator()

        validator.epic_criteria["Epic-00"] = [
            AcceptanceCriterion(
                criterion_id="Epic-00-AC-001",
                epic_id="Epic-00",
                description="Security validation must pass",
            ),
            AcceptanceCriterion(
                criterion_id="Epic-00-AC-002",
                epic_id="Epic-00",
                description="Optional feature",
            ),
        ]

        assert validator._is_critical_criterion("Epic-00-AC-001") is True
        assert validator._is_critical_criterion("Epic-00-AC-002") is False

    def test_generate_recommendation(self) -> None:
        """Test recommendation generation."""
        validator = EpicAcceptanceCriteriaValidator()

        gaps = [
            {"criterion_id": "AC-001", "validation_type": "automated"},
            {"criterion_id": "AC-002", "validation_type": "evidence"},
            {"criterion_id": "AC-003", "validation_type": "manual"},
        ]

        rec = validator._generate_recommendation("Epic-00", gaps)

        assert "Epic-00" in rec
        assert "automated" in rec.lower()
        assert "evidence" in rec.lower()
        assert "manual" in rec.lower()

    def test_generate_recommendation_empty(self) -> None:
        """Test recommendation generation with no gaps."""
        validator = EpicAcceptanceCriteriaValidator()

        rec = validator._generate_recommendation("Epic-00", [])

        assert rec == ""

    def test_analyze_gaps(self) -> None:
        """Test gap analysis."""
        validator = EpicAcceptanceCriteriaValidator()

        # Set up criteria with critical keyword
        validator.epic_criteria["Epic-01"] = [
            AcceptanceCriterion(
                criterion_id="Epic-01-AC-001",
                epic_id="Epic-01",
                description="Security validation must pass",
            ),
        ]

        crit_result = CriterionValidationResult(
            criterion_id="Epic-01-AC-001",
            passed=False,
            notes="Failed validation",
            validation_type=ValidationType.AUTOMATED,
        )

        epic_result = EpicValidationResult(
            epic_id="Epic-01",
            criteria_met=0,
            total_criteria=1,
            success_rate=0.0,
            failing_criteria=["Epic-01-AC-001"],
            criterion_results={"Epic-01-AC-001": crit_result},
        )

        validation_result = EpicAcceptanceValidationResult(
            epic_validations={"Epic-01": epic_result},
            overall_success=False,
            overall_success_rate=0.0,
            total_criteria=1,
            total_passed=0,
        )

        gap_analysis = validator.analyze_gaps(validation_result)

        assert gap_analysis.total_gaps == 1
        assert gap_analysis.critical_gaps == 1
        assert "Epic-01" in gap_analysis.gaps_by_epic
        assert len(gap_analysis.remediation_recommendations) >= 1

    def test_validate_all_epic_criteria(self) -> None:
        """Test full validation across all epics."""
        config = AcceptanceValidationConfig(
            completed_epics=["Epic-00", "Epic-01"],
        )
        validator = EpicAcceptanceCriteriaValidator(config)

        # Manually populate criteria
        validator.epic_criteria = {
            "Epic-00": [
                AcceptanceCriterion(
                    criterion_id="Epic-00-AC-001",
                    epic_id="Epic-00",
                    description="Implementation complete",
                    is_checked=True,
                ),
            ],
            "Epic-01": [
                AcceptanceCriterion(
                    criterion_id="Epic-01-AC-001",
                    epic_id="Epic-01",
                    description="Tests passing",
                    is_checked=True,
                ),
            ],
        }

        result = validator.validate_all_epic_criteria()

        assert result.total_criteria == 2
        assert result.total_passed == 2
        assert result.overall_success_rate == 1.0
        assert result.overall_success is True
        assert "Epic-00" in result.epic_validations
        assert "Epic-01" in result.epic_validations

    def test_validate_single_epic(self) -> None:
        """Test single epic validation."""
        validator = EpicAcceptanceCriteriaValidator()

        # Manually populate criteria
        validator.epic_criteria["Epic-00"] = [
            AcceptanceCriterion(
                criterion_id="Epic-00-AC-001",
                epic_id="Epic-00",
                description="Implementation complete",
                is_checked=True,
            ),
        ]

        result = validator.validate_single_epic("Epic-00")

        assert result.epic_id == "Epic-00"
        assert result.total_criteria == 1
        assert result.criteria_met == 1

    def test_generate_report(self) -> None:
        """Test report generation."""
        config = AcceptanceValidationConfig(
            completed_epics=["Epic-00"],
        )
        validator = EpicAcceptanceCriteriaValidator(config)

        epic_result = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=10,
            total_criteria=10,
            success_rate=1.0,
            failing_criteria=[],
            criterion_results={},
        )

        validation_result = EpicAcceptanceValidationResult(
            epic_validations={"Epic-00": epic_result},
            overall_success=True,
            overall_success_rate=1.0,
            total_criteria=10,
            total_passed=10,
            config=config,
        )

        report = validator.generate_report(validation_result)

        assert "EPIC ACCEPTANCE CRITERIA VALIDATION REPORT" in report
        assert "Overall Success: PASS" in report
        assert "100.0%" in report
        assert "Epic-00" in report

    def test_generate_report_with_failures(self) -> None:
        """Test report generation with failures."""
        config = AcceptanceValidationConfig(
            completed_epics=["Epic-00"],
        )
        validator = EpicAcceptanceCriteriaValidator(config)

        # Set up criteria for gap analysis
        validator.epic_criteria["Epic-00"] = [
            AcceptanceCriterion(
                criterion_id="Epic-00-AC-001",
                epic_id="Epic-00",
                description="Critical security check",
            ),
        ]

        crit_result = CriterionValidationResult(
            criterion_id="Epic-00-AC-001",
            passed=False,
            notes="Security validation failed",
            validation_type=ValidationType.AUTOMATED,
        )

        epic_result = EpicValidationResult(
            epic_id="Epic-00",
            criteria_met=5,
            total_criteria=10,
            success_rate=0.50,
            failing_criteria=["Epic-00-AC-001"],
            criterion_results={"Epic-00-AC-001": crit_result},
        )

        validation_result = EpicAcceptanceValidationResult(
            epic_validations={"Epic-00": epic_result},
            overall_success=False,
            overall_success_rate=0.50,
            total_criteria=10,
            total_passed=5,
            config=config,
        )

        report = validator.generate_report(validation_result)

        assert "Overall Success: FAIL" in report
        assert "GAP ANALYSIS" in report
        assert "Epic-00-AC-001" in report

    def test_generate_report_to_file(self) -> None:
        """Test report generation to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"

            config = AcceptanceValidationConfig(
                completed_epics=["Epic-00"],
            )
            validator = EpicAcceptanceCriteriaValidator(config)

            epic_result = EpicValidationResult(
                epic_id="Epic-00",
                criteria_met=10,
                total_criteria=10,
                success_rate=1.0,
                failing_criteria=[],
                criterion_results={},
            )

            validation_result = EpicAcceptanceValidationResult(
                epic_validations={"Epic-00": epic_result},
                overall_success=True,
                overall_success_rate=1.0,
                total_criteria=10,
                total_passed=10,
                config=config,
            )

            report = validator.generate_report(validation_result, str(output_path))

            assert output_path.exists()
            assert output_path.read_text() == report


class TestAcceptanceCriterionParsing:
    """Tests for acceptance criteria parsing from markdown."""

    @pytest.fixture
    def temp_epic_file(self) -> Path:
        """Create a temporary epic file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            epic_file = Path(tmpdir) / "Epic-00.md"
            epic_file.write_text("""# Epic-00: Project Setup

## Description
Initial project setup and configuration.

## 🎯 Acceptance Criteria

- [x] Project structure implemented
- [x] CI/CD pipeline configured
- [ ] Documentation completed
- [x] Test coverage > 80%

## Notes
Additional notes here.
""")
            yield epic_file

    def test_parse_acceptance_criteria(self, temp_epic_file: Path) -> None:
        """Test parsing acceptance criteria from markdown."""
        config = AcceptanceValidationConfig(
            epic_docs_path=str(temp_epic_file.parent),
            completed_epics=["Epic-00"],
        )
        validator = EpicAcceptanceCriteriaValidator(config)

        criteria = validator._parse_acceptance_criteria(temp_epic_file, "Epic-00")

        assert len(criteria) == 4
        assert criteria[0].is_checked is True
        assert criteria[2].is_checked is False
        assert "Test coverage" in criteria[3].description

    def test_parse_acceptance_criteria_no_emoji(self) -> None:
        """Test parsing acceptance criteria without emoji."""
        with tempfile.TemporaryDirectory() as tmpdir:
            epic_file = Path(tmpdir) / "Epic-00.md"
            epic_file.write_text("""# Epic-00

## Acceptance Criteria

- [x] Feature A implemented
- [ ] Feature B pending

""")
            validator = EpicAcceptanceCriteriaValidator()
            criteria = validator._parse_acceptance_criteria(epic_file, "Epic-00")

            assert len(criteria) == 2

    def test_parse_acceptance_criteria_no_section(self) -> None:
        """Test parsing with no acceptance criteria section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            epic_file = Path(tmpdir) / "Epic-00.md"
            epic_file.write_text("""# Epic-00

## Description
No acceptance criteria section.
""")
            validator = EpicAcceptanceCriteriaValidator()
            criteria = validator._parse_acceptance_criteria(epic_file, "Epic-00")

            # Should return default criteria
            assert len(criteria) == 1
            assert criteria[0].is_checked is True


class TestValidationHelpers:
    """Tests for validation helper methods."""

    def test_validate_test_coverage_with_file(self) -> None:
        """Test coverage validation when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cov_file = Path(tmpdir) / ".coverage"
            cov_file.touch()

            with patch("src.validation.epic_acceptance_validator.Path") as mock_path:
                mock_path.return_value.exists.return_value = True

                validator = EpicAcceptanceCriteriaValidator()
                result = validator._validate_test_coverage(0.80)

                assert result is True

    def test_validate_performance_benchmarks(self) -> None:
        """Test performance benchmark validation."""
        validator = EpicAcceptanceCriteriaValidator()

        # Without any benchmark files
        with patch("src.validation.epic_acceptance_validator.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = validator._validate_performance_benchmarks()
            assert result is False

    def test_validate_security_scan(self) -> None:
        """Test security scan validation."""
        validator = EpicAcceptanceCriteriaValidator()

        # Without security report
        with patch("src.validation.epic_acceptance_validator.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = validator._validate_security_scan()
            assert result is False

    def test_find_evidence_path_with_file_pattern(self) -> None:
        """Test finding evidence path from description."""
        validator = EpicAcceptanceCriteriaValidator()

        description = "See `reports/output.html` for details"
        path = validator._find_evidence_path(description)

        assert path == "reports/output.html"

    def test_find_evidence_path_no_match(self) -> None:
        """Test finding evidence path with no file pattern."""
        validator = EpicAcceptanceCriteriaValidator()

        description = "Simple description without file reference"
        path = validator._find_evidence_path(description)

        assert path is None

    def test_create_validation_func_for_coverage(self) -> None:
        """Test validation function creation for coverage criteria."""
        validator = EpicAcceptanceCriteriaValidator()

        criterion = AcceptanceCriterion(
            criterion_id="AC-001",
            epic_id="Epic-00",
            description="Test coverage > 85%",
        )

        func = validator._create_validation_func(criterion)

        assert func is not None

    def test_create_validation_func_for_checked(self) -> None:
        """Test validation function for already checked criterion."""
        validator = EpicAcceptanceCriteriaValidator()

        criterion = AcceptanceCriterion(
            criterion_id="AC-001",
            epic_id="Epic-00",
            description="Already done",
            is_checked=True,
        )

        func = validator._create_validation_func(criterion)

        assert func is not None
        assert func() is True

    def test_create_validation_func_for_path(self) -> None:
        """Test validation function for path-based criteria."""
        validator = EpicAcceptanceCriteriaValidator()

        criterion = AcceptanceCriterion(
            criterion_id="AC-001",
            epic_id="Epic-00",
            description="File `src/module.py` exists",
        )

        func = validator._create_validation_func(criterion)

        assert func is not None
