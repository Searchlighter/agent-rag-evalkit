"""无需外部服务的合成评测命令行示例。"""

from app.adapter_contract import MockRagAdapter
from app.repository import InMemoryRepository
from app.service import EvalKitService


def demo() -> None:
    """构造两条样本并打印评测摘要与 CSV 结果。"""
    service = EvalKitService(InMemoryRepository())
    dataset = service.create_dataset("enterprise-knowledge-demo", "demo-user")
    version = service.import_dataset_version(
        dataset.id,
        [
            {
                "id": "case_001",
                "question": "How do I submit a travel expense?",
                "expected_evidence": ["mock-document#chunk-001"],
                "metadata": {"require_citations": True},
            },
            {
                "id": "case_002",
                "question": "What is the contract approval flow?",
                "expected_evidence": ["mock-document#chunk-001"],
            },
        ],
    )
    config = service.create_config("baseline-rag", retrieval_k=5)
    run = service.create_eval_run(version.id, config.id, "mock-rag", 5.0)
    summary = service.execute_eval_run(run.id, MockRagAdapter())
    print(summary)
    print(service.export_case_results_csv(run.id))


if __name__ == "__main__":
    demo()
