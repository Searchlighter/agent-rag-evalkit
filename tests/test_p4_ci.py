from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class P404ContinuousIntegrationTests(unittest.TestCase):
    def test_ci_has_separate_quality_contract_and_docker_jobs(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

        self.assertIn("quality-and-unit:", workflow)
        self.assertIn("adapter-contracts:", workflow)
        self.assertIn("docker-build:", workflow)
        self.assertIn('python-version: ["3.11", "3.12"]', workflow)
        self.assertIn("python -m ruff check", workflow)
        self.assertIn("python -m unittest discover -s tests -t . -v", workflow)
        self.assertIn("python -m pip wheel . --no-deps", workflow)
        self.assertIn("tests.test_adapter_contracts", workflow)
        self.assertIn("docker compose config --quiet", workflow)
        self.assertIn("docker build --tag agent-rag-evalkit:ci", workflow)

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('select = ["E4", "E7", "E9", "F"]', pyproject)
        self.assertIn('include = ["app*"]', pyproject)
        self.assertIn('readme = "README.md"', pyproject)

    def test_ci_applies_read_only_permissions_concurrency_and_timeouts(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("cancel-in-progress: true", workflow)
        self.assertEqual(3, workflow.count("timeout-minutes:"))
        self.assertNotIn("pull_request_target", workflow)

    def test_contribution_guide_contains_ci_equivalent_commands(self) -> None:
        guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

        self.assertIn("python -m ruff check", guide)
        self.assertIn("python -m unittest discover -s tests -t . -v", guide)
        self.assertIn("tests.test_adapter_contracts", guide)
        self.assertIn("docker compose config --quiet", guide)


if __name__ == "__main__":
    unittest.main()
