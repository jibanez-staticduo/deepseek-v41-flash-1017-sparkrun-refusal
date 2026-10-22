import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "vllm/models/deepseek_v4_1/nvidia/model.py": "a063dc4e14e9e0037a57b203ba9e621abb525ec451749799bd368e4bffdd9177",
    "vllm/models/deepseek_v4_1/nvidia/vl_model.py": "88e241ebe5050f771352dd5239d253748f83c48093badcc3f0c1eb62674be8d6",
    "vllm/models/deepseek_v4_1/nvidia/model_state.py": "d69f0a24ae3cf11d9a65eb23fe103addaff12f4148391e0dd9c4fbbbfc78eeb6",
    "vllm/v1/core/sched/output.py": "fbebfde451e8d8036d0d456612f44257aa9f1d92361637c2a0638cc378680cfa",
    "vllm/v1/worker/gpu/model_runner.py": "c09f536bdf383c9f17f4db63fcaf231de33daeacfdb3a37cc02f278326256400",
    "vllm/v1/worker/gpu/warmup.py": "496793148716cd377f8a9e61bca9c1ad6bf6e8e6e70ac2d2599741a1e68879c3",
    "vllm/v1/engine/input_processor.py": "f6518bc64a648c7fce1ecffb40cd195f07487941203cfc26050cf7539ef3f66f",
    "vllm/entrypoints/serve/__init__.py": "326c5db237f5f2da4a156248fb97547f34c1666a7aae77dcc089bf43983300cc",
    "vllm/entrypoints/cli/serve.py": "5a213f408bd25327cc65a3546feef1777b5f7b1a33e36c5dcea68022ae14c4a1",
}


def overlay_root():
    value = os.environ.get("SPARKRUN_REFUSAL_OVERLAY")
    if not value:
        return None
    return Path(value).expanduser().resolve()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patched_source_sha256(path):
    text = path.read_text(encoding="utf-8")
    marker = "# Modified by Staticduo contributors, 2026-10-22: request-scoped refusal integration.\n"
    if not text.startswith(marker):
        raise AssertionError(f"prepared overlay is missing its modification receipt: {path}")
    return hashlib.sha256(text[len(marker):].encode("utf-8")).hexdigest()


@unittest.skipUnless(
    os.environ.get("SPARKRUN_REFUSAL_OVERLAY"),
    "set SPARKRUN_REFUSAL_OVERLAY to a prepared, ignored vLLM source overlay",
)
class RuntimeOverlayTests(unittest.TestCase):
    def setUp(self):
        self.overlay = overlay_root()
        self.assertIsNotNone(self.overlay)

    def test_all_overlay_files_parse_and_match_expected_patch(self):
        import ast

        for relative, expected in REQUIRED_FILES.items():
            path = self.overlay / relative
            with self.subTest(path=relative):
                self.assertTrue(path.is_file())
                self.assertEqual(patched_source_sha256(path), expected)
                ast.parse(path.read_text(encoding="utf-8"))

    def test_request_admission_and_worker_wiring(self):
        processor = (self.overlay / "vllm/v1/engine/input_processor.py").read_text()
        self.assertLess(
            processor.index("refusal_salt = stamp_cache_salt("),
            processor.index("return EngineCoreRequest("),
        )
        self.assertIn("cache_salt=refusal_salt,", processor)

        scheduler = (self.overlay / "vllm/v1/core/sched/output.py").read_text()
        self.assertIn("parse_effective_lambda(request.cache_salt)", scheduler)
        self.assertIn("refusal_lambda=refusal_lambda", scheduler)

        state = (self.overlay / "vllm/models/deepseek_v4_1/nvidia/model_state.py").read_text()
        self.assertIn("self.refusal_rows.fill(input_batch)", state)
        self.assertIn("self.refusal_rows.dummy(num_tokens)", state)
        self.assertIn("self.refusal_rows.remove_request(req_id)", state)

    def test_frontend_routes_lambda_rows_through_backbone_and_multimodal_wrapper(self):
        model = (self.overlay / "vllm/models/deepseek_v4_1/nvidia/model.py").read_text()
        self.assertIn("x = self.refusal_projection(x, local_lambdas)", model)
        self.assertIn("else sp_rows.local_slice(refusal_lambdas)", model)
        self.assertEqual(
            model.count("refusal_lambdas = gather_rows(refusal_lambdas, ced_indices)"),
            3,
        )
        self.assertLess(
            model.index("x = self.refusal_projection(x, local_lambdas)"),
            model.index("self.mhc.post_pre("),
        )
        vl = (self.overlay / "vllm/models/deepseek_v4_1/nvidia/vl_model.py").read_text()
        self.assertIn("refusal_lambdas=refusal_lambdas", vl)

    def test_patch_rejects_unsupported_runner_layouts(self):
        runner = (self.overlay / "vllm/v1/worker/gpu/model_runner.py").read_text()
        self.assertIn(
            "self.pcp_manager is not None or self.ubatch_runner is not None", runner
        )
        self.assertIn("without PCP or ubatching", runner)

    def test_streaming_strength_guard_runs_before_request_removal(self):
        import ast

        runner = (self.overlay / "vllm/v1/worker/gpu/model_runner.py").read_text()
        tree = ast.parse(runner)
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
        )
        method = next(
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "add_requests"
        )
        calls = [
            (node.func.attr, node.lineno)
            for node in ast.walk(method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        check = next(line for name, line in calls if name == "check_update")
        remove = next(line for name, line in calls if name == "_remove_request")
        self.assertLess(check, remove)


if __name__ == "__main__":
    unittest.main()
