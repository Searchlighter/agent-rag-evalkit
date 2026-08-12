from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class P401DocumentationTests(unittest.TestCase):
    def test_readme_contains_quickstart_boundaries_visual_and_faq(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## Quickstart", readme)
        self.assertIn("## 能力边界", readme)
        self.assertIn("docs/assets/dashboard-overview.png", readme)
        self.assertIn("docs/faq.md", readme)
        self.assertIn("```mermaid", readme)
        self.assertIn("python -m unittest discover -s tests -v", readme)

    def test_architecture_and_visual_asset_are_present(self) -> None:
        architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
        visual = ROOT / "docs" / "assets" / "dashboard-overview.png"

        self.assertGreaterEqual(architecture.count("```mermaid"), 4)
        self.assertIn("InMemoryRepository", architecture)
        self.assertIn("Dify Adapter", architecture)
        self.assertIn("RAGFlow Adapter", architecture)
        self.assertTrue(visual.is_file())
        self.assertGreater(visual.stat().st_size, 10_000)

    def test_faq_states_demo_limitations(self) -> None:
        faq = (ROOT / "docs" / "faq.md").read_text(encoding="utf-8")

        self.assertIn("不是", faq)
        self.assertIn("Fake HTTP", faq)
        self.assertIn("MCP 风格", faq)
        self.assertIn("生产性能", faq)


if __name__ == "__main__":
    unittest.main()
