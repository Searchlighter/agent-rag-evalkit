from pathlib import Path
import unittest

from app.main import create_app


ROOT = Path(__file__).resolve().parents[1]


class P406CommunityTests(unittest.TestCase):
    def test_license_and_version_metadata_are_consistent(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        self.assertIn("MIT License", license_text)
        self.assertIn('version = "0.5.0"', pyproject)
        self.assertEqual("0.5.0", create_app().version)
        self.assertIn("## 0.5.0 - 2026-08-11", changelog)
        self.assertIn("### Compatibility", changelog)

    def test_issue_forms_require_reproduction_compatibility_and_data_safety(self) -> None:
        bug = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(
            encoding="utf-8"
        )
        feature = (ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").read_text(
            encoding="utf-8"
        )
        config = (ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("id: reproduction", bug)
        self.assertIn("id: environment", bug)
        self.assertIn("Data safety confirmation", bug)
        self.assertIn("id: compatibility", feature)
        self.assertIn("required: true", feature)
        self.assertIn("blank_issues_enabled: false", config)

    def test_contribution_conduct_security_and_pr_guidance_are_actionable(self) -> None:
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        pull_request = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("python -m unittest discover", contributing)
        self.assertIn("Adapter 与外部契约", contributing)
        self.assertIn("不可接受行为", conduct)
        self.assertIn("报告与处理", conduct)
        self.assertIn("Do not open a public Issue", security)
        self.assertIn("Compatibility and safety", pull_request)


if __name__ == "__main__":
    unittest.main()
