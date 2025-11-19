# PC-091-10B: Security Vulnerability Scanner Implementation

**Ticket ID:** PC-091-10B  
**Epic:** Epic-10B (Quality Assurance & Acceptance Validation)  
**Story:** User Story 2 - Code Quality and Test Coverage Validation  
**Assignee:** TBD  
**Status:** Not Started  
**Priority:** Critical  
**Estimated Effort:** 3 days  
**Sprint:** Week 29

---

## 📋 Description

Implement comprehensive security vulnerability scanning framework to identify and remediate security issues in the codebase, dependencies, and configurations. The scanner must integrate with multiple security tools and enforce zero high-severity vulnerability policy for production deployment.

---

## 🎯 Acceptance Criteria

- [ ] `SecurityScanner` class with multi-tool integration
- [ ] Dependency vulnerability scanning (pip, requirements)
- [ ] Code security scanning (Bandit, safety checks)
- [ ] Configuration security validation (secrets, credentials)
- [ ] OWASP Top 10 vulnerability checking
- [ ] Severity classification (critical, high, medium, low)
- [ ] Zero high-severity vulnerabilities for production
- [ ] Automated remediation recommendations
- [ ] Security report generation with actionable items
- [ ] CI/CD pipeline integration for continuous scanning
- [ ] Unit tests achieve >80% coverage
- [ ] Integration tests validate end-to-end security workflows

---

## 🏗️ Technical Implementation

### **1. SecurityScanner Class**

**Location:** `src/validation/security_scanner.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import subprocess
import json
from enum import Enum
from pathlib import Path

class Severity(Enum):
    """Security vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SecurityIssue:
    """Represents a security vulnerability."""
    issue_id: str
    title: str
    severity: Severity
    description: str
    affected_file: str
    line_number: Optional[int]
    cwe_id: Optional[str]
    remediation: str
    
    def is_blocking(self) -> bool:
        """Check if issue blocks production deployment."""
        return self.severity in [Severity.CRITICAL, Severity.HIGH]

@dataclass
class SecurityValidationResult:
    """Results from security validation."""
    vulnerability_summary: Dict[str, int]
    detailed_issues: List[SecurityIssue]
    passes_validation: bool
    remediation_recommendations: List[str]
    scan_timestamp: float
    
    def get_blocking_issues(self) -> List[SecurityIssue]:
        """Get issues that block production deployment."""
        return [issue for issue in self.detailed_issues if issue.is_blocking()]

@dataclass
class SecurityConfig:
    """Configuration for security scanning."""
    max_high_vulnerabilities: int = 0
    max_medium_vulnerabilities: int = 5
    scan_dependencies: bool = True
    scan_code: bool = True
    scan_configurations: bool = True
    ignore_list: List[str] = None
    
    def __post_init__(self):
        if self.ignore_list is None:
            self.ignore_list = []

class SecurityScanner:
    """Comprehensive security vulnerability scanner."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize security scanner.
        
        Args:
            config: Configuration for security scanning
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.issues: List[SecurityIssue] = []
    
    def scan_codebase(self) -> SecurityValidationResult:
        """Execute comprehensive security scan of codebase.
        
        Returns:
            SecurityValidationResult with all identified issues
        """
        import time
        
        self.logger.info("Starting comprehensive security scan")
        self.issues = []
        
        # Scan dependencies for known vulnerabilities
        if self.config.scan_dependencies:
            self.logger.info("Scanning dependencies for vulnerabilities")
            self._scan_dependencies()
        
        # Scan code for security issues
        if self.config.scan_code:
            self.logger.info("Scanning code for security issues")
            self._scan_code_security()
        
        # Scan configurations for security issues
        if self.config.scan_configurations:
            self.logger.info("Scanning configurations for security issues")
            self._scan_configurations()
        
        # Filter ignored issues
        filtered_issues = [
            issue for issue in self.issues
            if issue.issue_id not in self.config.ignore_list
        ]
        
        # Calculate vulnerability summary
        vulnerability_summary = {
            'critical': sum(1 for i in filtered_issues if i.severity == Severity.CRITICAL),
            'high': sum(1 for i in filtered_issues if i.severity == Severity.HIGH),
            'medium': sum(1 for i in filtered_issues if i.severity == Severity.MEDIUM),
            'low': sum(1 for i in filtered_issues if i.severity == Severity.LOW),
            'info': sum(1 for i in filtered_issues if i.severity == Severity.INFO)
        }
        
        # Validate against security policy
        passes_validation = (
            vulnerability_summary['critical'] == 0 and
            vulnerability_summary['high'] <= self.config.max_high_vulnerabilities and
            vulnerability_summary['medium'] <= self.config.max_medium_vulnerabilities
        )
        
        # Generate remediation recommendations
        remediation_recommendations = self._generate_remediation_recommendations(
            filtered_issues
        )
        
        return SecurityValidationResult(
            vulnerability_summary=vulnerability_summary,
            detailed_issues=filtered_issues,
            passes_validation=passes_validation,
            remediation_recommendations=remediation_recommendations,
            scan_timestamp=time.time()
        )
    
    def _scan_dependencies(self):
        """Scan dependencies for known vulnerabilities using safety."""
        try:
            # Run safety check on dependencies
            result = subprocess.run(
                ['safety', 'check', '--json', '--file', 'requirements.txt'],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                
                for vuln in safety_data:
                    self.issues.append(SecurityIssue(
                        issue_id=f"SAFETY-{vuln.get('vulnerability_id', 'UNKNOWN')}",
                        title=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                        severity=self._map_cvss_to_severity(vuln.get('cvss', 0)),
                        description=vuln.get('advisory', 'No description available'),
                        affected_file='requirements.txt',
                        line_number=None,
                        cwe_id=None,
                        remediation=f"Update {vuln.get('package')} to version {vuln.get('fixed_version', 'latest')}"
                    ))
        
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Safety check failed: {e}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse safety output: {e}")
    
    def _scan_code_security(self):
        """Scan code for security issues using Bandit."""
        try:
            # Run Bandit security scanner
            result = subprocess.run(
                ['bandit', '-r', 'prevcarga/', '-f', 'json'],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                for issue in bandit_data.get('results', []):
                    self.issues.append(SecurityIssue(
                        issue_id=f"BANDIT-{issue.get('test_id', 'UNKNOWN')}",
                        title=issue.get('issue_text', 'Security issue detected'),
                        severity=self._map_bandit_severity(issue.get('issue_severity', 'LOW')),
                        description=issue.get('issue_text', ''),
                        affected_file=issue.get('filename', 'unknown'),
                        line_number=issue.get('line_number'),
                        cwe_id=issue.get('issue_cwe', {}).get('id'),
                        remediation=self._get_bandit_remediation(issue.get('test_id'))
                    ))
        
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Bandit scan failed: {e}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse Bandit output: {e}")
    
    def _scan_configurations(self):
        """Scan configurations for security issues."""
        # Check for hardcoded secrets
        self._scan_for_secrets()
        
        # Check for insecure configurations
        self._scan_insecure_configs()
    
    def _scan_for_secrets(self):
        """Scan for hardcoded secrets and credentials."""
        import re
        
        # Common patterns for secrets
        secret_patterns = {
            'AWS_KEY': r'AKIA[0-9A-Z]{16}',
            'PRIVATE_KEY': r'-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
            'API_KEY': r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{32,})',
            'PASSWORD': r'password["\']?\s*[:=]\s*["\']?([^\s"\';]+)',
            'SECRET': r'secret["\']?\s*[:=]\s*["\']?([^\s"\';]+)'
        }
        
        # Scan Python files
        for py_file in Path('prevcarga').rglob('*.py'):
            content = py_file.read_text()
            
            for pattern_name, pattern in secret_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    
                    self.issues.append(SecurityIssue(
                        issue_id=f"SECRET-{pattern_name}-{py_file.name}-{line_num}",
                        title=f"Potential hardcoded {pattern_name} detected",
                        severity=Severity.HIGH,
                        description=f"Hardcoded secret detected in {py_file}",
                        affected_file=str(py_file),
                        line_number=line_num,
                        cwe_id="CWE-798",
                        remediation="Remove hardcoded secrets and use environment variables or secrets management"
                    ))
    
    def _scan_insecure_configs(self):
        """Scan for insecure configuration settings."""
        # Check for SSL/TLS settings
        self._check_ssl_settings()
        
        # Check for debug mode in production configs
        self._check_debug_settings()
    
    def _check_ssl_settings(self):
        """Check for insecure SSL/TLS settings."""
        for py_file in Path('prevcarga').rglob('*.py'):
            content = py_file.read_text()
            
            # Check for SSL verification disabled
            if 'verify=False' in content or 'verify = False' in content:
                line_num = content[:content.find('verify=False')].count('\n') + 1 if 'verify=False' in content else 0
                
                self.issues.append(SecurityIssue(
                    issue_id=f"SSL-VERIFY-{py_file.name}",
                    title="SSL certificate verification disabled",
                    severity=Severity.HIGH,
                    description="SSL certificate verification is disabled, making connections vulnerable to MITM attacks",
                    affected_file=str(py_file),
                    line_number=line_num,
                    cwe_id="CWE-295",
                    remediation="Enable SSL certificate verification or use proper certificate handling"
                ))
    
    def _check_debug_settings(self):
        """Check for debug mode enabled in production."""
        config_files = list(Path('.').rglob('config*.py')) + list(Path('.').rglob('settings*.py'))
        
        for config_file in config_files:
            if config_file.exists():
                content = config_file.read_text()
                
                if 'DEBUG = True' in content or "DEBUG=True" in content:
                    self.issues.append(SecurityIssue(
                        issue_id=f"DEBUG-{config_file.name}",
                        title="Debug mode enabled",
                        severity=Severity.MEDIUM,
                        description="Debug mode is enabled, which may expose sensitive information",
                        affected_file=str(config_file),
                        line_number=None,
                        cwe_id="CWE-489",
                        remediation="Disable debug mode in production configurations"
                    ))
    
    def _map_cvss_to_severity(self, cvss_score: float) -> Severity:
        """Map CVSS score to severity level."""
        if cvss_score >= 9.0:
            return Severity.CRITICAL
        elif cvss_score >= 7.0:
            return Severity.HIGH
        elif cvss_score >= 4.0:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _map_bandit_severity(self, bandit_severity: str) -> Severity:
        """Map Bandit severity to our severity enum."""
        mapping = {
            'HIGH': Severity.HIGH,
            'MEDIUM': Severity.MEDIUM,
            'LOW': Severity.LOW
        }
        return mapping.get(bandit_severity.upper(), Severity.LOW)
    
    def _get_bandit_remediation(self, test_id: str) -> str:
        """Get remediation advice for Bandit test ID."""
        remediation_map = {
            'B101': 'Avoid using assert statements in production code',
            'B102': 'Avoid using exec() function',
            'B103': 'Set proper file permissions',
            'B201': 'Avoid using Flask debug mode in production',
            'B301': 'Use pickle safely with trusted data only',
            'B501': 'Use strong cryptographic protocols',
            'B601': 'Avoid using shell=True in subprocess calls'
        }
        return remediation_map.get(test_id, 'Review and fix security issue')
    
    def _generate_remediation_recommendations(
        self,
        issues: List[SecurityIssue]
    ) -> List[str]:
        """Generate prioritized remediation recommendations.
        
        Args:
            issues: List of security issues
            
        Returns:
            List of remediation recommendations
        """
        recommendations = []
        
        # Group by severity
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        
        if critical_issues:
            recommendations.append(
                f"CRITICAL: Address {len(critical_issues)} critical vulnerabilities immediately"
            )
        
        if high_issues:
            recommendations.append(
                f"HIGH: Fix {len(high_issues)} high-severity vulnerabilities before production"
            )
        
        # Group by type
        dependency_issues = [i for i in issues if 'SAFETY' in i.issue_id]
        if dependency_issues:
            recommendations.append(
                f"Update {len(dependency_issues)} vulnerable dependencies"
            )
        
        secret_issues = [i for i in issues if 'SECRET' in i.issue_id]
        if secret_issues:
            recommendations.append(
                f"Remove {len(secret_issues)} hardcoded secrets from codebase"
            )
        
        return recommendations
    
    def generate_security_report(
        self,
        result: SecurityValidationResult
    ) -> str:
        """Generate comprehensive security report.
        
        Args:
            result: Security validation results
            
        Returns:
            Path to generated security report
        """
        from jinja2 import Template
        import datetime
        
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Scan Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .critical { color: #d32f2f; font-weight: bold; }
                .high { color: #f57c00; font-weight: bold; }
                .medium { color: #fbc02d; font-weight: bold; }
                .low { color: #388e3c; }
                .pass { color: green; font-weight: bold; }
                .fail { color: red; font-weight: bold; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #1976d2; color: white; }
                .summary { background-color: #f5f5f5; padding: 15px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Security Scan Report</h1>
            <p>Generated: {{ timestamp }}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Overall Status: <span class="{{ 'pass' if result.passes_validation else 'fail' }}">
                    {{ 'PASS' if result.passes_validation else 'FAIL' }}
                </span></p>
                <ul>
                    <li class="critical">Critical: {{ result.vulnerability_summary.critical }}</li>
                    <li class="high">High: {{ result.vulnerability_summary.high }}</li>
                    <li class="medium">Medium: {{ result.vulnerability_summary.medium }}</li>
                    <li class="low">Low: {{ result.vulnerability_summary.low }}</li>
                </ul>
            </div>
            
            {% if result.remediation_recommendations %}
            <h2>Remediation Recommendations</h2>
            <ul>
                {% for recommendation in result.remediation_recommendations %}
                <li>{{ recommendation }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            
            <h2>Detailed Issues</h2>
            <table>
                <tr>
                    <th>Severity</th>
                    <th>Title</th>
                    <th>File</th>
                    <th>Line</th>
                    <th>Remediation</th>
                </tr>
                {% for issue in result.detailed_issues %}
                <tr>
                    <td class="{{ issue.severity.value }}">{{ issue.severity.value.upper() }}</td>
                    <td>{{ issue.title }}</td>
                    <td>{{ issue.affected_file }}</td>
                    <td>{{ issue.line_number or 'N/A' }}</td>
                    <td>{{ issue.remediation }}</td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """)
        
        html_content = template.render(
            result=result,
            timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        report_path = Path('reports/security_report.html')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html_content)
        
        return str(report_path)
```

---

## 🧪 Testing Requirements

### **Unit Tests**

**Location:** `tests/validation/test_security_scanner.py`

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from prevcarga.validation.security_scanner import (
    SecurityScanner,
    SecurityConfig,
    Severity,
    SecurityIssue
)

class TestSecurityScanner:
    """Test suite for SecurityScanner."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SecurityConfig(
            max_high_vulnerabilities=0,
            max_medium_vulnerabilities=5
        )
    
    @pytest.fixture
    def scanner(self, config):
        """Create scanner instance."""
        return SecurityScanner(config)
    
    def test_scan_codebase(self, scanner):
        """Test complete security scan."""
        with patch.object(scanner, '_scan_dependencies'):
            with patch.object(scanner, '_scan_code_security'):
                with patch.object(scanner, '_scan_configurations'):
                    result = scanner.scan_codebase()
                    
                    assert result.vulnerability_summary is not None
                    assert 'critical' in result.vulnerability_summary
                    assert 'high' in result.vulnerability_summary
    
    def test_map_cvss_to_severity(self, scanner):
        """Test CVSS to severity mapping."""
        assert scanner._map_cvss_to_severity(9.5) == Severity.CRITICAL
        assert scanner._map_cvss_to_severity(7.5) == Severity.HIGH
        assert scanner._map_cvss_to_severity(5.0) == Severity.MEDIUM
        assert scanner._map_cvss_to_severity(2.0) == Severity.LOW
    
    def test_security_issue_is_blocking(self):
        """Test security issue blocking logic."""
        critical_issue = SecurityIssue(
            issue_id='TEST-1',
            title='Test',
            severity=Severity.CRITICAL,
            description='Test',
            affected_file='test.py',
            line_number=1,
            cwe_id=None,
            remediation='Fix it'
        )
        
        assert critical_issue.is_blocking() is True
        
        low_issue = SecurityIssue(
            issue_id='TEST-2',
            title='Test',
            severity=Severity.LOW,
            description='Test',
            affected_file='test.py',
            line_number=1,
            cwe_id=None,
            remediation='Fix it'
        )
        
        assert low_issue.is_blocking() is False
```

---

## 📊 Success Metrics

- [ ] Zero critical vulnerabilities detected
- [ ] Zero high-severity vulnerabilities detected
- [ ] All dependency vulnerabilities identified
- [ ] No hardcoded secrets in codebase
- [ ] Security scan completes in <5 minutes

---

## 🔗 Dependencies

**Requires:**
- safety (dependency scanning)
- bandit (code security scanning)
- Python codebase

**Blocks:**
- PC-092-10B (Production Readiness Validator)

---

## 📚 Documentation

- [ ] Security scanning framework documented
- [ ] Vulnerability remediation guide created
- [ ] Security policy and thresholds documented
- [ ] CI/CD integration guide

---

## 🔄 Implementation Steps

1. **Day 1: Core Security Scanner**
   - Implement `SecurityScanner` class
   - Implement dependency scanning
   - Create severity classification

2. **Day 2: Code and Config Scanning**
   - Implement code security scanning
   - Implement configuration scanning
   - Create secret detection

3. **Day 3: Testing and Reporting**
   - Complete unit tests
   - Implement security report generation
   - Write documentation

---

**Estimated Completion:** Week 29, Day 3  
**Review Required:** Security Team, DevOps Team
