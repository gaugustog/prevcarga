"""Final validation report generator for PrevCarga system.

This module provides comprehensive final validation report generation that
consolidates all quality assurance results into a single executive-level
report for stakeholder approval and deployment certification.

Key Features:
- Executive summary generation with key findings
- Integration of all validation component results
- Gap analysis and recommendations section
- Deployment readiness certification
- Multiple output formats (HTML, JSON, text)
- Report versioning and archival

Example:
    ```python
    from src.validation.final_report_generator import (
        FinalValidationReportGenerator,
        ReportConfig,
    )

    config = ReportConfig(
        generated_by="QA System",
        validation_period="Week 28-29",
    )
    generator = FinalValidationReportGenerator(config)

    report = generator.generate_final_validation_report(
        epic_validation=epic_result,
        performance_validation=perf_result,
        quality_validation=quality_result,
        security_validation=security_result,
        robustness_validation=robustness_result,
    )

    generator.export_report_html(report, "reports/final_report.html")
    ```
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ApprovalStatus(Enum):
    """Approval status for final validation."""

    APPROVED = "approved"
    CONDITIONAL = "conditional"
    NOT_READY = "not_ready"


class RiskLevel(Enum):
    """Risk level assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Attributes:
        generated_by: Name of the generator/system.
        validation_period: Period covered by validation.
        stakeholders: List of stakeholder names.
        output_dir: Directory for report output.
        include_detailed_metrics: Whether to include detailed metrics.
        approval_threshold: Minimum score for approval.
        conditional_threshold: Minimum score for conditional approval.
    """

    generated_by: str = "PrevCarga QA System"
    validation_period: str = "MVP Validation"
    stakeholders: list[str] = field(default_factory=list)
    output_dir: str = "reports"
    include_detailed_metrics: bool = True
    approval_threshold: float = 0.95
    conditional_threshold: float = 0.85

    def __post_init__(self) -> None:
        """Initialize with default stakeholders if none provided."""
        if not self.stakeholders:
            self.stakeholders = [
                "Project Sponsor",
                "Technical Lead",
                "Operations Team",
                "QA Team",
            ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "generated_by": self.generated_by,
            "validation_period": self.validation_period,
            "stakeholders": self.stakeholders,
            "output_dir": self.output_dir,
            "include_detailed_metrics": self.include_detailed_metrics,
            "approval_threshold": self.approval_threshold,
            "conditional_threshold": self.conditional_threshold,
        }


@dataclass
class ReportMetadata:
    """Metadata for validation report.

    Attributes:
        report_id: Unique identifier for the report.
        generated_date: When the report was generated.
        generated_by: Name of the generator.
        validation_period: Period covered by validation.
        report_version: Version of the report format.
        stakeholders: List of stakeholders.
    """

    report_id: str
    generated_date: datetime
    generated_by: str
    validation_period: str
    report_version: str = "1.0"
    stakeholders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "report_id": self.report_id,
            "generated_date": self.generated_date.isoformat(),
            "generated_by": self.generated_by,
            "validation_period": self.validation_period,
            "report_version": self.report_version,
            "stakeholders": self.stakeholders,
        }


@dataclass
class ExecutiveSummary:
    """Executive summary of validation results.

    Attributes:
        overall_status: Overall validation status.
        overall_score: Overall validation score (0-1).
        key_achievements: List of key achievements.
        critical_issues: List of critical issues found.
        risk_assessment: Risk assessment description.
        deployment_recommendation: Recommendation for deployment.
    """

    overall_status: ApprovalStatus
    overall_score: float
    key_achievements: list[str]
    critical_issues: list[str]
    risk_assessment: str
    deployment_recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "key_achievements": self.key_achievements,
            "critical_issues": self.critical_issues,
            "risk_assessment": self.risk_assessment,
            "deployment_recommendation": self.deployment_recommendation,
        }


@dataclass
class ValidationComponentSummary:
    """Summary of a validation component.

    Attributes:
        component_name: Name of the validation component.
        passed: Whether the component passed validation.
        score: Component score (0-1).
        issues: List of issues found.
        details: Additional details.
    """

    component_name: str
    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "component_name": self.component_name,
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues,
            "details": self.details,
        }


@dataclass
class FinalValidationReport:
    """Comprehensive final validation report.

    Attributes:
        executive_summary: Executive summary.
        component_summaries: Summaries of validation components.
        recommendations: List of recommendations.
        approval_status: Whether system is approved.
        next_steps: List of next steps.
        report_metadata: Report metadata.
        raw_results: Raw validation results.
    """

    executive_summary: ExecutiveSummary
    component_summaries: dict[str, ValidationComponentSummary]
    recommendations: list[str]
    approval_status: bool
    next_steps: list[str]
    report_metadata: ReportMetadata
    raw_results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "executive_summary": self.executive_summary.to_dict(),
            "component_summaries": {
                k: v.to_dict() for k, v in self.component_summaries.items()
            },
            "recommendations": self.recommendations,
            "approval_status": self.approval_status,
            "next_steps": self.next_steps,
            "report_metadata": self.report_metadata.to_dict(),
        }


class EpicValidationProtocol(Protocol):
    """Protocol for epic validation results."""

    @property
    def overall_success(self) -> bool:
        """Whether overall validation was successful."""
        ...

    @property
    def overall_success_rate(self) -> float:
        """Overall success rate."""
        ...

    def get_failing_epics(self, threshold: float = 0.95) -> list[str]:
        """Get list of failing epics."""
        ...


class PerformanceValidationProtocol(Protocol):
    """Protocol for performance validation results."""

    @property
    def overall_score(self) -> float:
        """Overall performance score."""
        ...


class QualityValidationProtocol(Protocol):
    """Protocol for code quality validation results."""

    @property
    def meets_standards(self) -> bool:
        """Whether quality meets standards."""
        ...

    @property
    def overall_coverage(self) -> float:
        """Overall test coverage."""
        ...

    @property
    def quality_issues(self) -> list[str]:
        """List of quality issues."""
        ...


class SecurityValidationProtocol(Protocol):
    """Protocol for security validation results."""

    @property
    def passes_validation(self) -> bool:
        """Whether security validation passed."""
        ...

    @property
    def vulnerability_summary(self) -> dict[str, int]:
        """Summary of vulnerabilities by severity."""
        ...

    @property
    def remediation_recommendations(self) -> list[str]:
        """List of remediation recommendations."""
        ...


class RobustnessValidationProtocol(Protocol):
    """Protocol for robustness validation results."""

    @property
    def overall_score(self) -> float:
        """Overall robustness score."""
        ...


@dataclass
class SimpleEpicValidation:
    """Simple epic validation result for testing."""

    overall_success: bool = True
    overall_success_rate: float = 1.0
    _failing_epics: list[str] = field(default_factory=list)

    def get_failing_epics(self, threshold: float = 0.95) -> list[str]:
        """Get list of failing epics."""
        return self._failing_epics


@dataclass
class SimplePerformanceValidation:
    """Simple performance validation result for testing."""

    overall_score: float = 1.0
    ready_for_production: bool = True
    deployment_approval: bool = True
    _failed_checks: list[str] = field(default_factory=list)

    def get_failed_checks(self) -> list[str]:
        """Get list of failed checks."""
        return self._failed_checks


@dataclass
class SimpleQualityValidation:
    """Simple quality validation result for testing."""

    meets_standards: bool = True
    overall_coverage: float = 0.85
    quality_issues: list[str] = field(default_factory=list)


@dataclass
class SimpleSecurityValidation:
    """Simple security validation result for testing."""

    passes_validation: bool = True
    vulnerability_summary: dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0}
    )
    remediation_recommendations: list[str] = field(default_factory=list)


@dataclass
class SimpleRobustnessValidation:
    """Simple robustness validation result for testing."""

    overall_score: float = 0.90


class FinalValidationReportGenerator:
    """Generates comprehensive final validation reports.

    Consolidates all validation results into a single executive-level
    report for stakeholder approval and deployment certification.

    Attributes:
        config: Configuration for report generation.
    """

    # Weights for calculating overall score
    SCORE_WEIGHTS = {
        "epic_acceptance": 0.25,
        "production_readiness": 0.25,
        "quality": 0.15,
        "security": 0.20,
        "robustness": 0.15,
    }

    def __init__(self, config: ReportConfig | None = None) -> None:
        """Initialize final validation report generator.

        Args:
            config: Configuration for report generation.
        """
        self.config = config or ReportConfig()

    def generate_final_validation_report(
        self,
        epic_validation: Any | None = None,
        performance_validation: Any | None = None,
        quality_validation: Any | None = None,
        security_validation: Any | None = None,
        robustness_validation: Any | None = None,
    ) -> FinalValidationReport:
        """Generate comprehensive final validation report.

        Args:
            epic_validation: Epic acceptance validation results.
            performance_validation: Performance validation results.
            quality_validation: Code quality validation results.
            security_validation: Security validation results.
            robustness_validation: Robustness validation results.

        Returns:
            FinalValidationReport with all consolidated results.
        """
        logger.info("Generating final validation report")

        # Use defaults if not provided
        epic_val = epic_validation or SimpleEpicValidation()
        perf_val = performance_validation or SimplePerformanceValidation()
        quality_val = quality_validation or SimpleQualityValidation()
        security_val = security_validation or SimpleSecurityValidation()
        robustness_val = robustness_validation or SimpleRobustnessValidation()

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            epic_val, perf_val, quality_val, security_val, robustness_val
        )

        # Determine overall status
        overall_status = self._determine_overall_status(
            overall_score, security_val, perf_val
        )

        # Generate component summaries
        component_summaries = self._generate_component_summaries(
            epic_val, perf_val, quality_val, security_val, robustness_val
        )

        # Identify key achievements and critical issues
        key_achievements = self._identify_key_achievements(
            epic_val, perf_val, quality_val, robustness_val
        )
        critical_issues = self._identify_critical_issues(
            epic_val, perf_val, security_val
        )

        # Generate risk assessment
        risk_assessment = self._assess_deployment_risk(
            overall_score, critical_issues, security_val
        )

        # Generate deployment recommendation
        deployment_recommendation = self._generate_deployment_recommendation(
            overall_status, critical_issues, overall_score
        )

        # Generate executive summary
        executive_summary = ExecutiveSummary(
            overall_status=overall_status,
            overall_score=overall_score,
            key_achievements=key_achievements,
            critical_issues=critical_issues,
            risk_assessment=risk_assessment,
            deployment_recommendation=deployment_recommendation,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            epic_val, perf_val, quality_val, security_val
        )

        # Determine approval status
        approval_status = self._determine_approval_status(
            epic_val, perf_val, security_val
        )

        # Define next steps
        next_steps = self._define_next_steps(approval_status, recommendations)

        # Create report metadata
        report_metadata = ReportMetadata(
            report_id=self._generate_report_id(),
            generated_date=datetime.now(),
            generated_by=self.config.generated_by,
            validation_period=self.config.validation_period,
            stakeholders=self.config.stakeholders,
        )

        report = FinalValidationReport(
            executive_summary=executive_summary,
            component_summaries=component_summaries,
            recommendations=recommendations,
            approval_status=approval_status,
            next_steps=next_steps,
            report_metadata=report_metadata,
        )

        logger.info(
            f"Final validation report generated. "
            f"Status: {overall_status.value}, Score: {overall_score:.1%}"
        )

        return report

    def _calculate_overall_score(
        self,
        epic_validation: Any,
        performance_validation: Any,
        quality_validation: Any,
        security_validation: Any,
        robustness_validation: Any,
    ) -> float:
        """Calculate overall validation score.

        Args:
            epic_validation: Epic validation results.
            performance_validation: Performance validation results.
            quality_validation: Quality validation results.
            security_validation: Security validation results.
            robustness_validation: Robustness validation results.

        Returns:
            Overall score between 0 and 1.
        """
        scores = {
            "epic_acceptance": getattr(epic_validation, "overall_success_rate", 1.0),
            "production_readiness": getattr(performance_validation, "overall_score", 1.0),
            "quality": 1.0 if getattr(quality_validation, "meets_standards", True) else 0.7,
            "security": 1.0 if getattr(security_validation, "passes_validation", True) else 0.0,
            "robustness": getattr(robustness_validation, "overall_score", 1.0),
        }

        overall_score = sum(
            scores[key] * self.SCORE_WEIGHTS[key] for key in self.SCORE_WEIGHTS
        )

        return min(max(overall_score, 0.0), 1.0)

    def _determine_overall_status(
        self,
        overall_score: float,
        security_validation: Any,
        performance_validation: Any,
    ) -> ApprovalStatus:
        """Determine overall validation status.

        Args:
            overall_score: Overall score.
            security_validation: Security validation results.
            performance_validation: Performance validation results.

        Returns:
            ApprovalStatus enum value.
        """
        # Check for blocking conditions
        vuln_summary = getattr(
            security_validation, "vulnerability_summary", {"critical": 0}
        )
        if vuln_summary.get("critical", 0) > 0:
            return ApprovalStatus.NOT_READY

        ready = getattr(performance_validation, "ready_for_production", True)
        if not ready:
            return ApprovalStatus.NOT_READY

        # Score-based status
        if overall_score >= self.config.approval_threshold:
            return ApprovalStatus.APPROVED
        elif overall_score >= self.config.conditional_threshold:
            return ApprovalStatus.CONDITIONAL
        else:
            return ApprovalStatus.NOT_READY

    def _generate_component_summaries(
        self,
        epic_validation: Any,
        performance_validation: Any,
        quality_validation: Any,
        security_validation: Any,
        robustness_validation: Any,
    ) -> dict[str, ValidationComponentSummary]:
        """Generate summaries for each validation component.

        Args:
            epic_validation: Epic validation results.
            performance_validation: Performance validation results.
            quality_validation: Quality validation results.
            security_validation: Security validation results.
            robustness_validation: Robustness validation results.

        Returns:
            Dictionary of component summaries.
        """
        summaries = {}

        # Epic acceptance summary
        epic_rate = getattr(epic_validation, "overall_success_rate", 1.0)
        epic_success = getattr(epic_validation, "overall_success", True)
        failing_epics = (
            epic_validation.get_failing_epics()
            if hasattr(epic_validation, "get_failing_epics")
            else []
        )
        summaries["epic_acceptance"] = ValidationComponentSummary(
            component_name="Epic Acceptance Criteria",
            passed=epic_success,
            score=epic_rate,
            issues=[f"Failing epic: {e}" for e in failing_epics],
            details={"success_rate": epic_rate, "failing_epics": failing_epics},
        )

        # Performance summary
        perf_score = getattr(performance_validation, "overall_score", 1.0)
        perf_ready = getattr(performance_validation, "ready_for_production", True)
        failed_checks = (
            performance_validation.get_failed_checks()
            if hasattr(performance_validation, "get_failed_checks")
            else []
        )
        summaries["performance"] = ValidationComponentSummary(
            component_name="Performance Validation",
            passed=perf_ready,
            score=perf_score,
            issues=[f"Failed check: {c}" for c in failed_checks],
            details={"overall_score": perf_score, "ready": perf_ready},
        )

        # Quality summary
        quality_meets = getattr(quality_validation, "meets_standards", True)
        quality_coverage = getattr(quality_validation, "overall_coverage", 0.85)
        quality_issues = getattr(quality_validation, "quality_issues", [])
        summaries["quality"] = ValidationComponentSummary(
            component_name="Code Quality",
            passed=quality_meets,
            score=quality_coverage,
            issues=quality_issues,
            details={"coverage": quality_coverage, "meets_standards": quality_meets},
        )

        # Security summary
        security_passes = getattr(security_validation, "passes_validation", True)
        vuln_summary = getattr(
            security_validation, "vulnerability_summary", {"critical": 0, "high": 0}
        )
        security_score = 1.0 if security_passes else 0.0
        security_issues = []
        if vuln_summary.get("critical", 0) > 0:
            security_issues.append(
                f"{vuln_summary['critical']} critical vulnerabilities"
            )
        if vuln_summary.get("high", 0) > 0:
            security_issues.append(f"{vuln_summary['high']} high vulnerabilities")
        summaries["security"] = ValidationComponentSummary(
            component_name="Security Validation",
            passed=security_passes,
            score=security_score,
            issues=security_issues,
            details={"vulnerability_summary": vuln_summary},
        )

        # Robustness summary
        robustness_score = getattr(robustness_validation, "overall_score", 0.90)
        summaries["robustness"] = ValidationComponentSummary(
            component_name="System Robustness",
            passed=robustness_score >= 0.80,
            score=robustness_score,
            issues=[],
            details={"overall_score": robustness_score},
        )

        return summaries

    def _identify_key_achievements(
        self,
        epic_validation: Any,
        performance_validation: Any,
        quality_validation: Any,
        robustness_validation: Any,
    ) -> list[str]:
        """Identify key achievements from validation results.

        Args:
            epic_validation: Epic validation results.
            performance_validation: Performance validation results.
            quality_validation: Quality validation results.
            robustness_validation: Robustness validation results.

        Returns:
            List of key achievements.
        """
        achievements = []

        epic_rate = getattr(epic_validation, "overall_success_rate", 1.0)
        achievements.append(
            f"Epic acceptance criteria: {epic_rate * 100:.1f}% completion"
        )

        coverage = getattr(quality_validation, "overall_coverage", 0.85)
        achievements.append(f"Test coverage: {coverage * 100:.1f}%")

        perf_score = getattr(performance_validation, "overall_score", 1.0)
        achievements.append(f"Production readiness: {perf_score * 100:.1f}%")

        robustness_score = getattr(robustness_validation, "overall_score", 0.90)
        achievements.append(f"System robustness: {robustness_score * 100:.1f}%")

        return achievements

    def _identify_critical_issues(
        self,
        epic_validation: Any,
        performance_validation: Any,
        security_validation: Any,
    ) -> list[str]:
        """Identify critical issues from validation results.

        Args:
            epic_validation: Epic validation results.
            performance_validation: Performance validation results.
            security_validation: Security validation results.

        Returns:
            List of critical issues.
        """
        critical_issues = []

        # Security issues
        vuln_summary = getattr(
            security_validation, "vulnerability_summary", {"critical": 0}
        )
        if vuln_summary.get("critical", 0) > 0:
            critical_issues.append(
                f"{vuln_summary['critical']} critical security vulnerabilities"
            )

        # Epic acceptance issues
        epic_success = getattr(epic_validation, "overall_success", True)
        if not epic_success:
            failing_epics = (
                epic_validation.get_failing_epics()
                if hasattr(epic_validation, "get_failing_epics")
                else []
            )
            if failing_epics:
                critical_issues.append(
                    f"Acceptance criteria not met for epics: {', '.join(failing_epics)}"
                )

        # Production readiness issues
        perf_ready = getattr(performance_validation, "ready_for_production", True)
        if not perf_ready:
            failed_checks = (
                performance_validation.get_failed_checks()
                if hasattr(performance_validation, "get_failed_checks")
                else []
            )
            if failed_checks:
                critical_issues.append(
                    f"{len(failed_checks)} production readiness checks failed"
                )

        return critical_issues

    def _assess_deployment_risk(
        self,
        overall_score: float,
        critical_issues: list[str],
        security_validation: Any,
    ) -> str:
        """Assess deployment risk level.

        Args:
            overall_score: Overall validation score.
            critical_issues: List of critical issues.
            security_validation: Security validation results.

        Returns:
            Risk assessment description.
        """
        if critical_issues:
            if any("critical" in issue.lower() for issue in critical_issues):
                return "HIGH - Critical issues identified that must be resolved before deployment"
            else:
                return "MEDIUM - Some issues identified that should be addressed"

        if overall_score >= self.config.approval_threshold:
            return "LOW - System meets all quality standards and is ready for production"
        elif overall_score >= self.config.conditional_threshold:
            return "MEDIUM - System meets most standards with minor gaps"
        else:
            return "HIGH - Significant gaps in validation require remediation"

    def _generate_deployment_recommendation(
        self,
        overall_status: ApprovalStatus,
        critical_issues: list[str],
        overall_score: float,
    ) -> str:
        """Generate deployment recommendation.

        Args:
            overall_status: Overall validation status.
            critical_issues: List of critical issues.
            overall_score: Overall validation score.

        Returns:
            Deployment recommendation string.
        """
        if overall_status == ApprovalStatus.APPROVED:
            return (
                f"RECOMMEND APPROVAL FOR PRODUCTION DEPLOYMENT. "
                f"System has achieved {overall_score * 100:.1f}% validation score "
                f"and meets all critical requirements."
            )
        elif overall_status == ApprovalStatus.CONDITIONAL:
            return (
                f"CONDITIONAL APPROVAL. System may proceed to production with "
                f"monitoring and planned remediation of identified issues. "
                f"Overall score: {overall_score * 100:.1f}%"
            )
        else:
            issues_str = "; ".join(critical_issues) if critical_issues else "score below threshold"
            return (
                f"NOT RECOMMENDED FOR PRODUCTION. Issues must be resolved: {issues_str}"
            )

    def _generate_recommendations(
        self,
        epic_validation: Any,
        performance_validation: Any,
        quality_validation: Any,
        security_validation: Any,
    ) -> list[str]:
        """Generate prioritized recommendations.

        Args:
            epic_validation: Epic validation results.
            performance_validation: Performance validation results.
            quality_validation: Quality validation results.
            security_validation: Security validation results.

        Returns:
            List of recommendations.
        """
        recommendations = []

        # Security recommendations first (highest priority)
        security_recs = getattr(security_validation, "remediation_recommendations", [])
        recommendations.extend(security_recs)

        # Epic acceptance recommendations
        epic_success = getattr(epic_validation, "overall_success", True)
        if not epic_success:
            failing_epics = (
                epic_validation.get_failing_epics()
                if hasattr(epic_validation, "get_failing_epics")
                else []
            )
            if failing_epics:
                recommendations.append(
                    f"Address failing acceptance criteria in: {', '.join(failing_epics)}"
                )

        # Production readiness recommendations
        perf_ready = getattr(performance_validation, "ready_for_production", True)
        if not perf_ready:
            failed_checks = (
                performance_validation.get_failed_checks()
                if hasattr(performance_validation, "get_failed_checks")
                else []
            )
            if failed_checks:
                recommendations.append(
                    f"Complete {len(failed_checks)} production readiness checks"
                )

        # Quality recommendations
        quality_meets = getattr(quality_validation, "meets_standards", True)
        if not quality_meets:
            quality_issues = getattr(quality_validation, "quality_issues", [])
            for issue in quality_issues[:5]:  # Limit to top 5
                recommendations.append(f"Code Quality: {issue}")

        return recommendations

    def _determine_approval_status(
        self,
        epic_validation: Any,
        performance_validation: Any,
        security_validation: Any,
    ) -> bool:
        """Determine if system is approved for deployment.

        Args:
            epic_validation: Epic validation results.
            performance_validation: Performance validation results.
            security_validation: Security validation results.

        Returns:
            True if approved, False otherwise.
        """
        epic_success = getattr(epic_validation, "overall_success", True)
        deployment_approval = getattr(performance_validation, "deployment_approval", True)
        security_passes = getattr(security_validation, "passes_validation", True)

        return epic_success and deployment_approval and security_passes

    def _define_next_steps(
        self,
        approval_status: bool,
        recommendations: list[str],
    ) -> list[str]:
        """Define next steps based on approval status.

        Args:
            approval_status: Whether system is approved.
            recommendations: List of recommendations.

        Returns:
            List of next steps.
        """
        if approval_status:
            return [
                "Obtain final stakeholder sign-off",
                "Schedule production deployment",
                "Prepare deployment communication",
                "Execute deployment runbook",
                "Monitor initial production performance",
            ]
        else:
            return [
                "Address identified recommendations",
                "Re-run validation after remediation",
                "Schedule follow-up review",
                "Update stakeholders on timeline",
            ]

    def _generate_report_id(self) -> str:
        """Generate unique report ID.

        Returns:
            Unique report ID string.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"VALIDATION-REPORT-{timestamp}-{unique_id}"

    def export_report_html(
        self,
        report: FinalValidationReport,
        output_path: str | Path | None = None,
    ) -> str:
        """Export report as HTML.

        Args:
            report: Final validation report.
            output_path: Optional output path.

        Returns:
            Path to generated HTML report.
        """
        if output_path is None:
            output_path = Path(self.config.output_dir) / f"{report.report_metadata.report_id}.html"
        else:
            output_path = Path(output_path)

        html_content = self._generate_html_content(report)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)

        logger.info(f"HTML report exported to: {output_path}")
        return str(output_path)

    def export_report_json(
        self,
        report: FinalValidationReport,
        output_path: str | Path | None = None,
    ) -> str:
        """Export report as JSON.

        Args:
            report: Final validation report.
            output_path: Optional output path.

        Returns:
            Path to generated JSON report.
        """
        if output_path is None:
            output_path = Path(self.config.output_dir) / f"{report.report_metadata.report_id}.json"
        else:
            output_path = Path(output_path)

        report_dict = report.to_dict()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"JSON report exported to: {output_path}")
        return str(output_path)

    def export_report_text(
        self,
        report: FinalValidationReport,
        output_path: str | Path | None = None,
    ) -> str:
        """Export report as plain text.

        Args:
            report: Final validation report.
            output_path: Optional output path.

        Returns:
            Path to generated text report.
        """
        if output_path is None:
            output_path = Path(self.config.output_dir) / f"{report.report_metadata.report_id}.txt"
        else:
            output_path = Path(output_path)

        text_content = self._generate_text_content(report)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text_content)

        logger.info(f"Text report exported to: {output_path}")
        return str(output_path)

    def _generate_html_content(self, report: FinalValidationReport) -> str:
        """Generate HTML content for report.

        Args:
            report: Final validation report.

        Returns:
            HTML content string.
        """
        status = report.executive_summary.overall_status.value
        status_class = status.lower().replace("_", "-")
        score_pct = report.executive_summary.overall_score * 100

        achievements_html = "\n".join(
            f"<li>{a}</li>" for a in report.executive_summary.key_achievements
        )

        issues_html = ""
        if report.executive_summary.critical_issues:
            issues_items = "\n".join(
                f'<li class="issue">{i}</li>'
                for i in report.executive_summary.critical_issues
            )
            issues_html = f"""
            <h3>Critical Issues</h3>
            <ul>{issues_items}</ul>
            """

        components_html = ""
        for name, summary in report.component_summaries.items():
            status_indicator = "pass" if summary.passed else "fail"
            issues_list = ""
            if summary.issues:
                issues_items = "\n".join(f"<li>{i}</li>" for i in summary.issues[:3])
                issues_list = f"<ul>{issues_items}</ul>"
            components_html += f"""
            <div class="component">
                <h4>{summary.component_name}</h4>
                <p>Status: <span class="{status_indicator}">{status_indicator.upper()}</span></p>
                <p>Score: {summary.score * 100:.1f}%</p>
                {issues_list}
            </div>
            """

        recommendations_html = ""
        if report.recommendations:
            rec_items = "\n".join(f"<li>{r}</li>" for r in report.recommendations)
            recommendations_html = f"""
            <div class="section">
                <h2>Recommendations</h2>
                <ol>{rec_items}</ol>
            </div>
            """

        next_steps_html = "\n".join(f"<li>{s}</li>" for s in report.next_steps)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Final Validation Report - {report.report_metadata.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; }}
        .header {{ background-color: #1976d2; color: white; padding: 20px; border-radius: 5px; }}
        .approved {{ color: #2e7d32; font-weight: bold; }}
        .conditional {{ color: #f57c00; font-weight: bold; }}
        .not-ready {{ color: #c62828; font-weight: bold; }}
        .pass {{ color: #2e7d32; font-weight: bold; }}
        .fail {{ color: #c62828; font-weight: bold; }}
        .section {{ margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
        .component {{ display: inline-block; margin: 10px; padding: 15px; background-color: white;
                     border-radius: 5px; vertical-align: top; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .issue {{ color: #c62828; }}
        h1 {{ margin: 0 0 10px 0; }}
        h2 {{ color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 5px; }}
        h3 {{ color: #424242; }}
        .score {{ font-size: 2em; font-weight: bold; color: #1976d2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PrevCarga System - Final Validation Report</h1>
        <p>Report ID: {report.report_metadata.report_id}</p>
        <p>Generated: {report.report_metadata.generated_date.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Period: {report.report_metadata.validation_period}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>Overall Status: <span class="{status_class}">{status.upper()}</span></p>
        <p class="score">{score_pct:.1f}%</p>

        <h3>Key Achievements</h3>
        <ul>{achievements_html}</ul>

        {issues_html}

        <h3>Risk Assessment</h3>
        <p>{report.executive_summary.risk_assessment}</p>

        <h3>Deployment Recommendation</h3>
        <p><strong>{report.executive_summary.deployment_recommendation}</strong></p>
    </div>

    <div class="section">
        <h2>Validation Components</h2>
        {components_html}
    </div>

    {recommendations_html}

    <div class="section">
        <h2>Next Steps</h2>
        <ol>{next_steps_html}</ol>
    </div>

    <div class="section" style="font-size: 0.9em; color: #757575;">
        <p>Generated by: {report.report_metadata.generated_by}</p>
        <p>Stakeholders: {', '.join(report.report_metadata.stakeholders)}</p>
    </div>
</body>
</html>"""

    def _generate_text_content(self, report: FinalValidationReport) -> str:
        """Generate plain text content for report.

        Args:
            report: Final validation report.

        Returns:
            Text content string.
        """
        lines = [
            "=" * 70,
            "PREVCARGA SYSTEM - FINAL VALIDATION REPORT",
            "=" * 70,
            "",
            f"Report ID: {report.report_metadata.report_id}",
            f"Generated: {report.report_metadata.generated_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Period: {report.report_metadata.validation_period}",
            "",
            "-" * 70,
            "EXECUTIVE SUMMARY",
            "-" * 70,
            "",
            f"Overall Status: {report.executive_summary.overall_status.value.upper()}",
            f"Overall Score: {report.executive_summary.overall_score * 100:.1f}%",
            "",
            "Key Achievements:",
        ]

        for achievement in report.executive_summary.key_achievements:
            lines.append(f"  - {achievement}")

        if report.executive_summary.critical_issues:
            lines.extend(["", "Critical Issues:"])
            for issue in report.executive_summary.critical_issues:
                lines.append(f"  ! {issue}")

        lines.extend([
            "",
            f"Risk Assessment: {report.executive_summary.risk_assessment}",
            "",
            f"Recommendation: {report.executive_summary.deployment_recommendation}",
            "",
            "-" * 70,
            "VALIDATION COMPONENTS",
            "-" * 70,
        ])

        for name, summary in report.component_summaries.items():
            status = "PASS" if summary.passed else "FAIL"
            lines.extend([
                "",
                f"{summary.component_name}:",
                f"  Status: {status}",
                f"  Score: {summary.score * 100:.1f}%",
            ])
            if summary.issues:
                for issue in summary.issues[:3]:
                    lines.append(f"  - {issue}")

        if report.recommendations:
            lines.extend([
                "",
                "-" * 70,
                "RECOMMENDATIONS",
                "-" * 70,
                "",
            ])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        lines.extend([
            "",
            "-" * 70,
            "NEXT STEPS",
            "-" * 70,
            "",
        ])
        for i, step in enumerate(report.next_steps, 1):
            lines.append(f"  {i}. {step}")

        lines.extend([
            "",
            "=" * 70,
            f"Generated by: {report.report_metadata.generated_by}",
            f"Stakeholders: {', '.join(report.report_metadata.stakeholders)}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def generate_report_summary(self, report: FinalValidationReport) -> str:
        """Generate a brief summary of the report.

        Args:
            report: Final validation report.

        Returns:
            Brief summary string.
        """
        status = report.executive_summary.overall_status.value.upper()
        score = report.executive_summary.overall_score * 100
        issues_count = len(report.executive_summary.critical_issues)

        summary = f"Status: {status} | Score: {score:.1f}%"
        if issues_count > 0:
            summary += f" | Critical Issues: {issues_count}"

        return summary
