import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "payload"))
import refusal_policy

fake_vllm = ModuleType("vllm")
fake_vllm.refusal_policy = refusal_policy
spec = importlib.util.spec_from_file_location(
    "refusal_api_under_test", ROOT / "payload" / "refusal_api.py"
)
refusal_api = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"vllm": fake_vllm}):
    spec.loader.exec_module(refusal_api)


class RefusalApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifact = self.root / "synthetic-directions.safetensors"
        self.artifact.write_bytes(b"synthetic directions fixture")
        self.token_path = self.root / "admin-token"
        self.token_path.write_text("test-only-token\n")
        self.environment = patch.dict(os.environ, {
            "VLLM_REFUSAL_DIRS": str(self.artifact),
            "STATICDUO_REFUSAL_ADMIN_TOKEN_FILE": str(self.token_path),
        })
        self.environment.start()
        self.artifact_hash = patch.object(
            refusal_policy, "_hash_artifact", return_value=refusal_policy.ARTIFACT_SHA256
        )
        self.artifact_hash.start()
        with refusal_policy._lock:
            refusal_policy._default_lambda = 1.0
            refusal_policy._epoch = 0
        self.app = FastAPI()
        refusal_api.attach_router(self.app)
        self.client = TestClient(self.app, client=("127.0.0.1", 50000))

    def tearDown(self):
        self.client.close()
        self.artifact_hash.stop()
        self.environment.stop()
        self.temp.cleanup()

    def test_loopback_admin_changes_default_for_future_stamps(self):
        headers = {"Authorization": "Bearer test-only-token"}
        before = refusal_policy.stamp_cache_salt("tenant")
        response = self.client.post(
            refusal_api.ROUTE, headers=headers, json={"lambda": 2.5}
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["lambda"], 2.5)
        self.assertEqual(payload["epoch"], 1)
        self.assertIn("future request admissions only", payload["effect"])
        after = refusal_policy.stamp_cache_salt("tenant")
        self.assertEqual(refusal_policy.parse_effective_lambda(before), 1.0)
        self.assertEqual(refusal_policy.parse_effective_lambda(after), 2.5)
        self.assertEqual(
            self.client.get(refusal_api.ROUTE, headers=headers).status_code, 200
        )

    def test_auth_loopback_and_body_validation_fail_closed(self):
        self.assertEqual(self.client.get(refusal_api.ROUTE).status_code, 401)
        self.assertEqual(
            self.client.get(
                refusal_api.ROUTE, headers={"Authorization": "Bearer wrong"}
            ).status_code,
            401,
        )
        unauthorized = TestClient(self.app, client=("192.0.2.9", 50000))
        try:
            response = unauthorized.get(
                refusal_api.ROUTE,
                headers={"Authorization": "Bearer test-only-token"},
            )
            self.assertEqual(response.status_code, 403)
        finally:
            unauthorized.close()
        headers = {"Authorization": "Bearer test-only-token"}
        for value in (-1, 4.1, "NaN", "Infinity"):
            with self.subTest(value=value):
                self.assertEqual(
                    self.client.post(
                        refusal_api.ROUTE, headers=headers, json={"lambda": value}
                    ).status_code,
                    422,
                )
        self.assertEqual(
            self.client.post(
                refusal_api.ROUTE,
                headers=headers,
                json={"lambda": 2, "extra": 1},
            ).status_code,
            422,
        )

    def test_missing_admin_token_is_service_unavailable(self):
        with patch.dict(
            os.environ,
            {"STATICDUO_REFUSAL_ADMIN_TOKEN_FILE": str(self.root / "missing")},
        ):
            response = self.client.get(
                refusal_api.ROUTE,
                headers={"Authorization": "Bearer test-only-token"},
            )
        self.assertEqual(response.status_code, 503)

    def test_router_is_empty_when_refusal_feature_is_disabled(self):
        with patch.dict(os.environ, {"VLLM_REFUSAL_DIRS": ""}):
            self.assertEqual(len(refusal_api.create_router().routes), 0)


if __name__ == "__main__":
    unittest.main()
