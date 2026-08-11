from pathlib import Path
import json
import unittest

from app.dify_adapter import DifyAdapterError, DifyChatAdapter
from app.http_adapter import AdapterContractError, HttpTargetAgentAdapter
from app.mock_rag_service import build_mock_response
from app.ragflow_adapter import RagFlowAdapterError, RagFlowChatAdapter


ROOT = Path(__file__).resolve().parents[1]


class AdapterContractTests(unittest.TestCase):
    def test_dify_example_maps_to_evalkit_contract(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "dify_blocking_response.json").read_text(encoding="utf-8")
        )
        response = DifyChatAdapter._parse_response(payload, "dify-request")

        self.assertEqual("dify-request", response.request_id)
        self.assertEqual("policy-document#chunk-001", response.citations[0])
        self.assertEqual("policy-document", response.retrievals[0].document_id)
        self.assertEqual("dify", response.events[0]["source"])

    def test_dify_rejects_missing_answer(self) -> None:
        with self.assertRaises(DifyAdapterError):
            DifyChatAdapter._parse_response({"metadata": {}}, "request")

    def test_ragflow_example_maps_to_evalkit_contract(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "ragflow_chat_completion.json").read_text(encoding="utf-8")
        )
        response = RagFlowChatAdapter._parse_response(payload, "ragflow-request")

        self.assertEqual("ragflow-request", response.request_id)
        self.assertEqual("ragflow-document#chunk-001", response.citations[0])
        self.assertEqual(0.88, response.retrievals[0].score)
        self.assertEqual(0.91, response.retrievals[0].metadata["vector_similarity"])

    def test_ragflow_rejects_missing_choices(self) -> None:
        with self.assertRaises(RagFlowAdapterError):
            RagFlowChatAdapter._parse_response({"choices": []}, "request")

    def test_generic_http_contract_accepts_synthetic_mock_service(self) -> None:
        payload = build_mock_response("如何申请差旅报销？", "http-request")
        response = HttpTargetAgentAdapter._parse_response(payload, "http-request")

        self.assertEqual("http-request", response.request_id)
        self.assertEqual("expense-policy#travel", response.citations[0])
        self.assertEqual("synthetic", response.retrievals[0].metadata["source"])

    def test_generic_http_contract_rejects_invalid_retrieval(self) -> None:
        with self.assertRaises(AdapterContractError):
            HttpTargetAgentAdapter._parse_response(
                {"answer": "invalid", "citations": [], "retrievals": [{}], "events": []},
                "request",
            )


if __name__ == "__main__":
    unittest.main()
