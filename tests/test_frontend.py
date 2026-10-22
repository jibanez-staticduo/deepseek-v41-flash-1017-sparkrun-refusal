import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FrontendTests(unittest.TestCase):
    def test_patch_stamps_before_engine_request_and_preserves_cache_hasher(self):
        patch = (ROOT / "patches/0001-request-scoped-refusal.patch").read_text()
        self.assertLess(
            patch.index("refusal_salt = stamp_cache_salt("),
            patch.index("return EngineCoreRequest("),
        )
        self.assertIn("cache_salt=refusal_salt,", patch)
        self.assertIn("vllm/v1/core/sched/output.py", patch)
        # The patch contains no hunks for kv_cache_utils, so original cache hashing remains.
        self.assertNotIn("kv_cache_utils.py", patch)

    def test_admin_router_registration_patch_is_single(self):
        patch = (ROOT / "patches/0001-request-scoped-refusal.patch").read_text()
        self.assertIn(
            "from vllm.refusal_api import attach_router as attach_refusal_router", patch
        )
        self.assertEqual(patch.count("attach_refusal_router(app)"), 1)

    def test_hot_default_is_single_python_frontend_and_headless_workers_are_allowed(self):
        patch = (ROOT / "patches/0001-request-scoped-refusal.patch").read_text()
        marker = "+        if feature_enabled() and not args.headless:"
        start = patch.index(marker)
        end = patch.index("         # Detect LB mode", start)
        body = patch[start:end]
        self.assertIn("args.api_server_count not in (None, 1)", body)
        self.assertIn("rust_frontend_path", body)
        self.assertIn('getattr(args, "grpc", False)', body)
        self.assertIn("args.api_server_count = 1", body)
        self.assertIn("args.headless", marker)


if __name__ == "__main__":
    unittest.main()
