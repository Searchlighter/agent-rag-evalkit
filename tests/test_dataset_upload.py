"""浏览器数据集上传与格式校验测试。"""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from app.api import UploadDatasetRequest, create_router
from app.ingestion import DatasetUploadValidationError, parse_dataset_content
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
            (
                UploadDatasetRequest(
                    name="错误 JSONL", filename="bad.jsonl", content="{not-json}\n"
                ),
                "invalid_json",
            ),
            (
                UploadDatasetRequest(
                    name="错误 CSV",
                    filename="bad.csv",
                    content="id,tags\ncase-1,test\n",
                ),
                "missing_field",
            ),
            (
                UploadDatasetRequest(
                    name="错误格式", filename="bad.txt", content="question"
                ),
                "unsupported_format",
            ),
            (
                UploadDatasetRequest(
                    name="空数据集", filename="empty.jsonl", content=""
                ),
                "empty_dataset",
            ),
            (
                UploadDatasetRequest(
                    name="重复 ID",
                    filename="duplicate.jsonl",
                    content=(
                        '{"id":"same","question":"问题一"}\n'
                        '{"id":"same","question":"问题二"}\n'
                    ),
                ),
                "duplicate_id",
            ),
        )
        for request, expected_code in invalid_requests:
            with self.subTest(filename=request.filename), self.assertRaises(
                HTTPException
            ) as context:
                upload(request)
            self.assertEqual(400, context.exception.status_code)
            self.assertEqual(
                "dataset_validation_failed", context.exception.detail["code"]
            )
            self.assertEqual(
                expected_code, context.exception.detail["issues"][0]["code"]
            )

        self.assertEqual({}, repository.datasets)
        self.assertEqual({}, repository.dataset_versions)

    def test_jsonl_validation_reports_lines_missing_fields_and_duplicate_ids(self) -> None:
        content = (
            "{not-json}\n"
            '{"id":"same","question":"问题一"}\n'
            '{"id":"same","question":"问题二"}\n'
            '{"id":"missing-question"}\n'
        )

        with self.assertRaises(DatasetUploadValidationError) as context:
            parse_dataset_content("issues.jsonl", content)

        issues = context.exception.issues
        self.assertEqual(
            ["invalid_json", "missing_field", "duplicate_id"],
            [issue["code"] for issue in issues],
        )
        self.assertEqual(1, issues[0]["line"])
        self.assertEqual("question", issues[1]["field"])
        self.assertEqual(4, issues[1]["line"])
        self.assertEqual([2, 3], issues[2]["lines"])
        self.assertEqual("same", issues[2]["case_id"])


if __name__ == "__main__":
    unittest.main()
