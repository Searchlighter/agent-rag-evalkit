"""浏览器数据集上传与格式校验测试。"""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from app.api import UploadDatasetRequest, create_router
from app.ingestion import parse_dataset_content
from app.repository import InMemoryRepository
from app.service import EvalKitService


class DatasetUploadTests(unittest.TestCase):
    def test_parse_dataset_content_supports_jsonl_and_csv(self) -> None:
        jsonl = parse_dataset_content(
            "cases.JSONL",
            '{"id":"json-1","question":"如何报销？","tags":["finance"]}\n',
        )
        csv_cases = parse_dataset_content(
            "cases.csv",
            "id,question,expected_evidence,tags\n"
            "csv-1,如何请假？,policy#leave,hr|policy\n",
        )

        self.assertEqual("json-1", jsonl[0]["id"])
        self.assertEqual(["policy#leave"], csv_cases[0]["expected_evidence"])
        self.assertEqual(["hr", "policy"], csv_cases[0]["tags"])

    def test_upload_api_creates_dataset_and_first_version(self) -> None:
        repository = InMemoryRepository()
        service = EvalKitService(repository)
        router = create_router(service)
        upload = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/datasets/upload"
        )

        result = upload(
            UploadDatasetRequest(
                name="客服回归集",
                filename="support.jsonl",
                content=(
                    '{"id":"case-1","question":"退款规则是什么？",'
                    '"expected_answers":["七天"],"expected_evidence":[],"tags":[]}\n'
                ),
            )
        )

        self.assertEqual("客服回归集", result["dataset"]["name"])
        self.assertEqual(1, result["version"]["version_number"])
        self.assertEqual(1, result["version"]["case_count"])
        self.assertEqual(1, len(repository.datasets))
        self.assertEqual(1, len(repository.dataset_versions))

    def test_upload_api_rejects_invalid_content_without_orphan_dataset(self) -> None:
        repository = InMemoryRepository()
        service = EvalKitService(repository)
        router = create_router(service)
        upload = next(
            route.endpoint
            for route in router.routes
            if route.path == "/api/v1/datasets/upload"
        )

        invalid_requests = (
            UploadDatasetRequest(
                name="错误 JSONL", filename="bad.jsonl", content="{not-json}\n"
            ),
            UploadDatasetRequest(
                name="错误 CSV", filename="bad.csv", content="id,tags\ncase-1,test\n"
            ),
            UploadDatasetRequest(
                name="错误格式", filename="bad.txt", content="question"
            ),
            UploadDatasetRequest(
                name="重复 ID",
                filename="duplicate.jsonl",
                content=(
                    '{"id":"same","question":"问题一"}\n'
                    '{"id":"same","question":"问题二"}\n'
                ),
            ),
        )
        for request in invalid_requests:
            with self.subTest(filename=request.filename), self.assertRaises(
                HTTPException
            ) as context:
                upload(request)
            self.assertEqual(400, context.exception.status_code)

        self.assertEqual({}, repository.datasets)
        self.assertEqual({}, repository.dataset_versions)


if __name__ == "__main__":
    unittest.main()
