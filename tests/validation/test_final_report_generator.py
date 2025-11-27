"""Tests for the final validation report generator.

Tests cover:
- ApprovalStatus and RiskLevel enums
- ReportConfig dataclass
- ReportMetadata dataclass
- ExecutiveSummary dataclass
- ValidationComponentSummary dataclass
- FinalValidationReport dataclass
- Simple validation dataclasses
- FinalValidationReportGenerator class methods
- HTML, JSON, and text export functionality
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.validation.final_report_generator import (
    ApprovalStatus,
    ExecutiveSummary,
    FinalValidationReport,
    FinalValidationReportGenerator,
    ReportConfig,
    ReportMetadata,
    RiskLevel,
    SimpleEpicValidation,
    SimplePerformanceValidation,
    SimpleQualityValidation,
    SimpleRobustnessValidation,
    SimpleSecurityValidation,
    ValidationComponentSummary,
)


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_approval_status_values(self) -> None:
        """Test ApprovalStatus enum values."""
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.CONDITIONAL.value == "conditional"
        assert ApprovalStatus.NOT_READY.value == "not_ready"

    def test_approval_status_members(self) -> None:
        """Test that all expected members exist."""
        members = [m.value for m in ApprovalStatus]
        assert len(members) == 3


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_level_values(self) -> None:
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_risk_level_members(self) -> None:
        """Test that all expected members exist."""
        members = [m.value for m in RiskLevel]
        assert len(members) == 4


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ReportConfig()

        assert config.generated_by == "PrevCarga QA System"
        assert config.validation_period == "MVP Validation"
        assert len(config.stakeholders) > 0
        assert "Project Sponsor" in config.stakeholders
        assert config.output_dir == "reports"
        assert config.approval_threshold == 0.95
        assert config.conditional_threshold == 0.85

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ReportConfig(
            generated_by="Test System",
            validation_period="Test Period",
            stakeholders=["User A", "User B"],
            output_dir="custom/reports",
            approval_threshold=0.90,
            conditional_threshold=0.80,
        )

        assert config.generated_by == "Test System"
        assert config.validation_period == "Test Period"
        assert config.stakeholders == ["User A", "User B"]
        assert config.output_dir == "custom/reports"
        assert config.approval_threshold == 0.90
        assert config.conditional_threshold == 0.80

    def test_config_to_dict(self) -> None:
        """Test configuration to_dict method."""
        config = ReportConfig(
            generated_by="Test",
            approval_threshold=0.90,
        )

        result = config.to_dict()

        assert result["generated_by"] == "Test"
        assert result["approval_threshold"] == 0.90
        assert "stakeholders" in result
        assert "output_dir" in result


class TestReportMetadata:
    """Tests for ReportMetadata dataclass."""

    def test_basic_metadata(self) -> None:
        """Test basic metadata creation."""
        now = datetime.now()
        metadata = ReportMetadata(
            report_id="TEST-001",
            generated_date=now,
            generated_by="Test System",
            validation_period="Test Period",
        )

        assert metadata.report_id == "TEST-001"
        assert metadata.generated_date == now
        assert metadata.generated_by == "Test System"
        assert metadata.validation_period == "Test Period"
        assert metadata.report_version == "1.0"
        assert metadata.stakeholders == []

    def test_metadata_with_stakeholders(self) -> None:
        """Test metadata with stakeholders."""
        metadata = ReportMetadata(
            report_id="TEST-002",
            generated_date=datetime.now(),
            generated_by="Test",
            validation_period="Test",
            stakeholders=["User A", "User B"],
        )

        assert metadata.stakeholders == ["User A", "User B"]

    def test_metadata_to_dict(self) -> None:
        """Test metadata to_dict method."""
        now = datetime.now()
        metadata = ReportMetadata(
            report_id="TEST-003",
            generated_date=now,
            generated_by="Test",
            validation_period="Test",
        )

        result = metadata.to_dict()

        assert result["report_id"] == "TEST-003"
        assert result["generated_by"] == "Test"
        assert "generated_date" in result


class TestExecutiveSummary:
    """Tests for ExecutiveSummary dataclass."""

    def test_basic_summary(self) -> None:
        """Test basic executive summary creation."""
        summary = ExecutiveSummary(
            overall_status=ApprovalStatus.APPROVED,
            overall_score=0.97,
            key_achievements=["Achievement 1", "Achievement 2"],
            critical_issues=[],
            risk_assessment="LOW - Ready for production",
            deployment_recommendation="APPROVED",
        )

        assert summary.overall_status == ApprovalStatus.APPROVED
        assert summary.overall_score == 0.97
        assert len(summary.key_achievements) == 2
        assert len(summary.critical_issues) == 0

    def test_summary_with_issues(self) -> None:
        """Test executive summary with critical issues."""
        summary = ExecutiveSummary(
            overall_status=ApprovalStatus.NOT_READY,
            overall_score=0.75,
            key_achievements=["Achievement 1"],
            critical_issues=["Issue 1", "Issue 2"],
            risk_assessment="HIGH - Issues found",
            deployment_recommendation="NOT APPROVED",
        )

        assert summary.overall_status == ApprovalStatus.NOT_READY
        assert len(summary.critical_issues) == 2

    def test_summary_to_dict(self) -> None:
        """Test executive summary to_dict method."""
        summary = ExecutiveSummary(
            overall_status=ApprovalStatus.CONDITIONAL,
            overall_score=0.90,
            key_achievements=[],
            critical_issues=[],
            risk_assessment="MEDIUM",
            deployment_recommendation="CONDITIONAL",
        )

        result = summary.to_dict()

        assert result["overall_status"] == "conditional"
        assert result["overall_score"] == 0.90


class TestValidationComponentSummary:
    """Tests for ValidationComponentSummary dataclass."""

    def test_basic_summary(self) -> None:
        """Test basic component summary creation."""
        summary = ValidationComponentSummary(
            component_name="Test Component",
            passed=True,
            score=0.95,
        )

        assert summary.component_name == "Test Component"
        assert summary.passed is True
        assert summary.score == 0.95
        assert summary.issues == []
        assert summary.details == {}

    def test_summary_with_issues(self) -> None:
        """Test component summary with issues."""
        summary = ValidationComponentSummary(
            component_name="Test Component",
            passed=False,
            score=0.60,
            issues=["Issue 1", "Issue 2"],
            details={"key": "value"},
        )

        assert summary.passed is False
        assert len(summary.issues) == 2
        assert summary.details["key"] == "value"

    def test_summary_to_dict(self) -> None:
        """Test component summary to_dict method."""
        summary = ValidationComponentSummary(
            component_name="Test",
            passed=True,
            score=0.90,
            issues=["Issue"],
        )

        result = summary.to_dict()

        assert result["component_name"] == "Test"
        assert result["passed"] is True
        assert result["score"] == 0.90
        assert result["issues"] == ["Issue"]


class TestSimpleValidationDataclasses:
    """Tests for simple validation dataclasses."""

    def test_simple_epic_validation(self) -> None:
        """Test SimpleEpicValidation dataclass."""
        validation = SimpleEpicValidation()

        assert validation.overall_success is True
        assert validation.overall_success_rate == 1.0
        assert validation.get_failing_epics() == []

    def test_simple_epic_validation_with_failures(self) -> None:
        """Test SimpleEpicValidation with failing epics."""
        validation = SimpleEpicValidation(
            overall_success=False,
            overall_success_rate=0.80,
            _failing_epics=["Epic-01", "Epic-02"],
        )

        assert validation.overall_success is False
        assert validation.get_failing_epics() == ["Epic-01", "Epic-02"]

    def test_simple_performance_validation(self) -> None:
        """Test SimplePerformanceValidation dataclass."""
        validation = SimplePerformanceValidation()

        assert validation.overall_score == 1.0
        assert validation.ready_for_production is True
        assert validation.deployment_approval is True
        assert validation.get_failed_checks() == []

    def test_simple_quality_validation(self) -> None:
        """Test SimpleQualityValidation dataclass."""
        validation = SimpleQualityValidation()

        assert validation.meets_standards is True
        assert validation.overall_coverage == 0.85
        assert validation.quality_issues == []

    def test_simple_security_validation(self) -> None:
        """Test SimpleSecurityValidation dataclass."""
        validation = SimpleSecurityValidation()

        assert validation.passes_validation is True
        assert validation.vulnerability_summary["critical"] == 0
        assert validation.remediation_recommendations == []

    def test_simple_robustness_validation(self) -> None:
        """Test SimpleRobustnessValidation dataclass."""
        validation = SimpleRobustnessValidation()

        assert validation.overall_score == 0.90


class TestFinalValidationReport:
    """Tests for FinalValidationReport dataclass."""

    @pytest.fixture
    def sample_report(self) -> FinalValidationReport:
        """Create a sample report for testing."""
        executive_summary = ExecutiveSummary(
            overall_status=ApprovalStatus.APPROVED,
            overall_score=0.97,
            key_achievements=["Achievement 1"],
            critical_issues=[],
            risk_assessment="LOW",
            deployment_recommendation="APPROVED",
        )

        component_summaries = {
            "epic": ValidationComponentSummary(
                component_name="Epic Acceptance",
                passed=True,
                score=0.98,
            )
        }

        metadata = ReportMetadata(
            report_id="TEST-001",
            generated_date=datetime.now(),
            generated_by="Test",
            validation_period="Test Period",
        )

        return FinalValidationReport(
            executive_summary=executive_summary,
            component_summaries=component_summaries,
            recommendations=["Recommendation 1"],
            approval_status=True,
            next_steps=["Step 1", "Step 2"],
            report_metadata=metadata,
        )

    def test_report_creation(self, sample_report: FinalValidationReport) -> None:
        """Test report creation."""
        assert sample_report.approval_status is True
        assert len(sample_report.recommendations) == 1
        assert len(sample_report.next_steps) == 2
        assert "epic" in sample_report.component_summaries

    def test_report_to_dict(self, sample_report: FinalValidationReport) -> None:
        """Test report to_dict method."""
        result = sample_report.to_dict()

        assert "executive_summary" in result
        assert "component_summaries" in result
        assert "recommendations" in result
        assert "approval_status" in result
        assert "next_steps" in result
        assert "report_metadata" in result


class TestFinalValidationReportGenerator:
    """Tests for FinalValidationReportGenerator class."""

    @pytest.fixture
    def generator(self) -> FinalValidationReportGenerator:
        """Create a generator instance."""
        config = ReportConfig(
            generated_by="Test System",
            validation_period="Test Period",
        )
        return FinalValidationReportGenerator(config)

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        generator = FinalValidationReportGenerator()

        assert generator.config is not None
        assert generator.config.generated_by == "PrevCarga QA System"

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ReportConfig(
            generated_by="Custom",
            approval_threshold=0.90,
        )
        generator = FinalValidationReportGenerator(config)

        assert generator.config.generated_by == "Custom"
        assert generator.config.approval_threshold == 0.90

    def test_generate_report_defaults(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test report generation with default values."""
        report = generator.generate_final_validation_report()

        assert report is not None
        assert report.executive_summary is not None
        assert report.executive_summary.overall_status == ApprovalStatus.APPROVED
        assert report.approval_status is True

    def test_generate_report_with_all_passing(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test report generation with all passing validations."""
        report = generator.generate_final_validation_report(
            epic_validation=SimpleEpicValidation(
                overall_success=True, overall_success_rate=0.98
            ),
            performance_validation=SimplePerformanceValidation(overall_score=0.99),
            quality_validation=SimpleQualityValidation(
                meets_standards=True, overall_coverage=0.90
            ),
            security_validation=SimpleSecurityValidation(passes_validation=True),
            robustness_validation=SimpleRobustnessValidation(overall_score=0.95),
        )

        assert report.executive_summary.overall_status == ApprovalStatus.APPROVED
        assert report.executive_summary.overall_score > 0.95
        assert report.approval_status is True

    def test_generate_report_with_security_failures(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test report generation with security failures."""
        report = generator.generate_final_validation_report(
            security_validation=SimpleSecurityValidation(
                passes_validation=False,
                vulnerability_summary={"critical": 2, "high": 3, "medium": 1, "low": 0},
            )
        )

        assert report.executive_summary.overall_status == ApprovalStatus.NOT_READY
        assert report.approval_status is False
        assert len(report.executive_summary.critical_issues) > 0

    def test_generate_report_with_epic_failures(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test report generation with epic validation failures."""
        report = generator.generate_final_validation_report(
            epic_validation=SimpleEpicValidation(
                overall_success=False,
                overall_success_rate=0.80,
                _failing_epics=["Epic-01", "Epic-02"],
            )
        )

        assert report.approval_status is False
        assert len(report.executive_summary.critical_issues) > 0
        assert any("Epic" in issue for issue in report.executive_summary.critical_issues)

    def test_generate_report_with_performance_not_ready(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test report generation with performance not ready."""
        report = generator.generate_final_validation_report(
            performance_validation=SimplePerformanceValidation(
                overall_score=0.70,
                ready_for_production=False,
                _failed_checks=["Check 1", "Check 2"],
            )
        )

        assert report.executive_summary.overall_status == ApprovalStatus.NOT_READY

    def test_calculate_overall_score(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test overall score calculation."""
        score = generator._calculate_overall_score(
            SimpleEpicValidation(overall_success_rate=0.96),
            SimplePerformanceValidation(overall_score=0.98),
            SimpleQualityValidation(meets_standards=True),
            SimpleSecurityValidation(passes_validation=True),
            SimpleRobustnessValidation(overall_score=0.85),
        )

        assert 0.9 <= score <= 1.0

    def test_determine_overall_status_approved(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test overall status determination for approved."""
        status = generator._determine_overall_status(
            0.97,
            SimpleSecurityValidation(vulnerability_summary={"critical": 0}),
            SimplePerformanceValidation(ready_for_production=True),
        )

        assert status == ApprovalStatus.APPROVED

    def test_determine_overall_status_conditional(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test overall status determination for conditional."""
        status = generator._determine_overall_status(
            0.90,
            SimpleSecurityValidation(vulnerability_summary={"critical": 0}),
            SimplePerformanceValidation(ready_for_production=True),
        )

        assert status == ApprovalStatus.CONDITIONAL

    def test_determine_overall_status_not_ready_critical_vuln(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test overall status determination with critical vulnerability."""
        status = generator._determine_overall_status(
            0.97,
            SimpleSecurityValidation(vulnerability_summary={"critical": 1}),
            SimplePerformanceValidation(ready_for_production=True),
        )

        assert status == ApprovalStatus.NOT_READY

    def test_determine_overall_status_not_ready_low_score(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test overall status determination with low score."""
        status = generator._determine_overall_status(
            0.70,
            SimpleSecurityValidation(vulnerability_summary={"critical": 0}),
            SimplePerformanceValidation(ready_for_production=True),
        )

        assert status == ApprovalStatus.NOT_READY

    def test_generate_component_summaries(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test component summary generation."""
        summaries = generator._generate_component_summaries(
            SimpleEpicValidation(overall_success_rate=0.95),
            SimplePerformanceValidation(overall_score=0.90),
            SimpleQualityValidation(overall_coverage=0.85),
            SimpleSecurityValidation(passes_validation=True),
            SimpleRobustnessValidation(overall_score=0.88),
        )

        assert "epic_acceptance" in summaries
        assert "performance" in summaries
        assert "quality" in summaries
        assert "security" in summaries
        assert "robustness" in summaries

        assert summaries["epic_acceptance"].score == 0.95
        assert summaries["performance"].score == 0.90
        assert summaries["quality"].score == 0.85
        assert summaries["security"].passed is True
        assert summaries["robustness"].score == 0.88

    def test_identify_key_achievements(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test key achievement identification."""
        achievements = generator._identify_key_achievements(
            SimpleEpicValidation(overall_success_rate=0.98),
            SimplePerformanceValidation(overall_score=0.95),
            SimpleQualityValidation(overall_coverage=0.88),
            SimpleRobustnessValidation(overall_score=0.90),
        )

        assert len(achievements) == 4
        assert any("98.0%" in a for a in achievements)
        assert any("88.0%" in a for a in achievements)

    def test_identify_critical_issues_none(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test critical issue identification with no issues."""
        issues = generator._identify_critical_issues(
            SimpleEpicValidation(overall_success=True),
            SimplePerformanceValidation(ready_for_production=True),
            SimpleSecurityValidation(
                passes_validation=True, vulnerability_summary={"critical": 0}
            ),
        )

        assert len(issues) == 0

    def test_identify_critical_issues_with_security(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test critical issue identification with security issues."""
        issues = generator._identify_critical_issues(
            SimpleEpicValidation(overall_success=True),
            SimplePerformanceValidation(ready_for_production=True),
            SimpleSecurityValidation(
                passes_validation=False, vulnerability_summary={"critical": 3}
            ),
        )

        assert len(issues) > 0
        assert any("critical" in i.lower() for i in issues)

    def test_assess_deployment_risk_low(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test deployment risk assessment - low."""
        risk = generator._assess_deployment_risk(
            0.97,
            [],
            SimpleSecurityValidation(),
        )

        assert "LOW" in risk

    def test_assess_deployment_risk_high_with_issues(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test deployment risk assessment - high with critical issues."""
        risk = generator._assess_deployment_risk(
            0.97,
            ["Critical security vulnerability found"],
            SimpleSecurityValidation(),
        )

        assert "HIGH" in risk

    def test_generate_deployment_recommendation_approved(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test deployment recommendation - approved."""
        rec = generator._generate_deployment_recommendation(
            ApprovalStatus.APPROVED,
            [],
            0.97,
        )

        assert "RECOMMEND APPROVAL" in rec
        assert "97.0%" in rec

    def test_generate_deployment_recommendation_conditional(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test deployment recommendation - conditional."""
        rec = generator._generate_deployment_recommendation(
            ApprovalStatus.CONDITIONAL,
            [],
            0.90,
        )

        assert "CONDITIONAL" in rec

    def test_generate_deployment_recommendation_not_ready(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test deployment recommendation - not ready."""
        rec = generator._generate_deployment_recommendation(
            ApprovalStatus.NOT_READY,
            ["Issue 1"],
            0.70,
        )

        assert "NOT RECOMMENDED" in rec

    def test_generate_recommendations(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test recommendations generation."""
        recommendations = generator._generate_recommendations(
            SimpleEpicValidation(
                overall_success=False, _failing_epics=["Epic-01"]
            ),
            SimplePerformanceValidation(
                ready_for_production=False, _failed_checks=["Check 1"]
            ),
            SimpleQualityValidation(
                meets_standards=False, quality_issues=["Issue 1"]
            ),
            SimpleSecurityValidation(
                remediation_recommendations=["Fix vuln 1"]
            ),
        )

        assert len(recommendations) > 0

    def test_determine_approval_status_true(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test approval status determination - approved."""
        approved = generator._determine_approval_status(
            SimpleEpicValidation(overall_success=True),
            SimplePerformanceValidation(deployment_approval=True),
            SimpleSecurityValidation(passes_validation=True),
        )

        assert approved is True

    def test_determine_approval_status_false(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test approval status determination - not approved."""
        approved = generator._determine_approval_status(
            SimpleEpicValidation(overall_success=False),
            SimplePerformanceValidation(deployment_approval=True),
            SimpleSecurityValidation(passes_validation=True),
        )

        assert approved is False

    def test_define_next_steps_approved(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test next steps for approved status."""
        steps = generator._define_next_steps(True, [])

        assert len(steps) > 0
        assert any("stakeholder" in s.lower() for s in steps)
        assert any("deployment" in s.lower() for s in steps)

    def test_define_next_steps_not_approved(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test next steps for not approved status."""
        steps = generator._define_next_steps(False, ["Recommendation 1"])

        assert len(steps) > 0
        assert any("remediation" in s.lower() for s in steps)

    def test_generate_report_id(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test report ID generation."""
        report_id = generator._generate_report_id()

        assert report_id.startswith("VALIDATION-REPORT-")
        assert len(report_id) > 25

    def test_generate_report_summary(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test report summary generation."""
        report = generator.generate_final_validation_report()
        summary = generator.generate_report_summary(report)

        assert "Status:" in summary
        assert "Score:" in summary


class TestReportExport:
    """Tests for report export functionality."""

    @pytest.fixture
    def generator(self) -> FinalValidationReportGenerator:
        """Create a generator instance."""
        return FinalValidationReportGenerator()

    @pytest.fixture
    def sample_report(
        self, generator: FinalValidationReportGenerator
    ) -> FinalValidationReport:
        """Create a sample report."""
        return generator.generate_final_validation_report()

    def test_export_report_html(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test HTML export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result = generator.export_report_html(sample_report, output_path)

            assert Path(result).exists()
            content = Path(result).read_text()
            assert "<!DOCTYPE html>" in content
            assert "PrevCarga" in content
            assert sample_report.report_metadata.report_id in content

    def test_export_report_html_default_path(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test HTML export with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.config.output_dir = tmpdir
            result = generator.export_report_html(sample_report)

            assert Path(result).exists()
            assert result.endswith(".html")

    def test_export_report_json(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            result = generator.export_report_json(sample_report, output_path)

            assert Path(result).exists()
            with open(result) as f:
                data = json.load(f)
            assert "executive_summary" in data
            assert "component_summaries" in data

    def test_export_report_json_default_path(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test JSON export with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.config.output_dir = tmpdir
            result = generator.export_report_json(sample_report)

            assert Path(result).exists()
            assert result.endswith(".json")

    def test_export_report_text(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test text export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            result = generator.export_report_text(sample_report, output_path)

            assert Path(result).exists()
            content = Path(result).read_text()
            assert "PREVCARGA" in content
            assert "EXECUTIVE SUMMARY" in content
            assert "VALIDATION COMPONENTS" in content

    def test_export_report_text_default_path(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test text export with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.config.output_dir = tmpdir
            result = generator.export_report_text(sample_report)

            assert Path(result).exists()
            assert result.endswith(".txt")

    def test_generate_html_content(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test HTML content generation."""
        html = generator._generate_html_content(sample_report)

        assert "<!DOCTYPE html>" in html
        assert "<style>" in html
        assert "</html>" in html
        assert sample_report.report_metadata.report_id in html

    def test_generate_text_content(
        self,
        generator: FinalValidationReportGenerator,
        sample_report: FinalValidationReport,
    ) -> None:
        """Test text content generation."""
        text = generator._generate_text_content(sample_report)

        assert "PREVCARGA" in text
        assert "EXECUTIVE SUMMARY" in text
        assert "Key Achievements" in text
        assert "NEXT STEPS" in text

    def test_generate_html_content_with_issues(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test HTML content with issues."""
        report = generator.generate_final_validation_report(
            security_validation=SimpleSecurityValidation(
                passes_validation=False,
                vulnerability_summary={"critical": 2, "high": 0, "medium": 0, "low": 0},
            )
        )
        html = generator._generate_html_content(report)

        assert "Critical Issues" in html or "critical" in html.lower()

    def test_generate_text_content_with_issues(
        self, generator: FinalValidationReportGenerator
    ) -> None:
        """Test text content with issues."""
        report = generator.generate_final_validation_report(
            epic_validation=SimpleEpicValidation(
                overall_success=False,
                overall_success_rate=0.80,
                _failing_epics=["Epic-01"],
            )
        )
        text = generator._generate_text_content(report)

        assert "Critical Issues" in text


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_validation_results(self) -> None:
        """Test with empty validation results."""
        generator = FinalValidationReportGenerator()
        report = generator.generate_final_validation_report()

        assert report is not None
        assert report.executive_summary is not None

    def test_score_boundaries(self) -> None:
        """Test score boundary conditions."""
        generator = FinalValidationReportGenerator()

        # Test exactly at approval threshold
        config = ReportConfig(approval_threshold=0.95, conditional_threshold=0.85)
        generator = FinalValidationReportGenerator(config)

        status = generator._determine_overall_status(
            0.95,
            SimpleSecurityValidation(vulnerability_summary={"critical": 0}),
            SimplePerformanceValidation(ready_for_production=True),
        )
        assert status == ApprovalStatus.APPROVED

        # Test just below approval threshold
        status = generator._determine_overall_status(
            0.94,
            SimpleSecurityValidation(vulnerability_summary={"critical": 0}),
            SimplePerformanceValidation(ready_for_production=True),
        )
        assert status == ApprovalStatus.CONDITIONAL

        # Test exactly at conditional threshold
        status = generator._determine_overall_status(
            0.85,
            SimpleSecurityValidation(vulnerability_summary={"critical": 0}),
            SimplePerformanceValidation(ready_for_production=True),
        )
        assert status == ApprovalStatus.CONDITIONAL

        # Test below conditional threshold
        status = generator._determine_overall_status(
            0.84,
            SimpleSecurityValidation(vulnerability_summary={"critical": 0}),
            SimplePerformanceValidation(ready_for_production=True),
        )
        assert status == ApprovalStatus.NOT_READY

    def test_score_clamping(self) -> None:
        """Test score clamping to 0-1 range."""
        generator = FinalValidationReportGenerator()

        # Create mock validation with scores that might cause overflow
        score = generator._calculate_overall_score(
            SimpleEpicValidation(overall_success_rate=1.5),  # > 1
            SimplePerformanceValidation(overall_score=1.2),  # > 1
            SimpleQualityValidation(meets_standards=True),
            SimpleSecurityValidation(passes_validation=True),
            SimpleRobustnessValidation(overall_score=1.0),
        )

        assert 0.0 <= score <= 1.0

    def test_multiple_report_generation(self) -> None:
        """Test generating multiple reports."""
        generator = FinalValidationReportGenerator()

        report1 = generator.generate_final_validation_report()
        report2 = generator.generate_final_validation_report()

        # Reports should have different IDs
        assert report1.report_metadata.report_id != report2.report_metadata.report_id
