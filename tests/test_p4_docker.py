from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class P403DockerTests(unittest.TestCase):
    def test_dockerfile_runs_as_non_root_and_exposes_both_demo_ports(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("FROM python:3.11-slim", dockerfile)
        self.assertIn("USER evalkit", dockerfile)
        self.assertIn("APP_UID=10001", dockerfile)
        self.assertIn("EXPOSE 8000 8001", dockerfile)
        self.assertNotIn("COPY . .", dockerfile)

    def test_compose_defines_two_hardened_healthy_services(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("evalkit-api:", compose)
        self.assertIn("mock-rag:", compose)
        self.assertIn("condition: service_healthy", compose)
        self.assertEqual(2, compose.count("healthcheck:"))
        self.assertIn("read_only: true", compose)
        self.assertIn("no-new-privileges:true", compose)
        self.assertIn("mem_limit: 256m", compose)
        self.assertIn("cpus: 0.50", compose)

    def test_deployment_document_states_resources_and_persistence_boundary(self) -> None:
        deployment = (ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn("1 CPU / 512 MB", deployment)
        self.assertIn("docker compose up --build -d", deployment)
        self.assertIn("InMemoryRepository", deployment)
        self.assertIn("Docker daemon 未启动", deployment)
        self.assertIn(".env", dockerignore)
        self.assertIn("tests/", dockerignore)


if __name__ == "__main__":
    unittest.main()
