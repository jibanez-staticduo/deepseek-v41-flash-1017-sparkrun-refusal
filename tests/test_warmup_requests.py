import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WarmupRequestTests(unittest.TestCase):
    def test_internal_warmup_requests_use_explicit_zero_policy_stamp(self):
        patch = (ROOT / "patches/0001-request-scoped-refusal.patch").read_text()
        self.assertIn('refusal_warmup_salt = stamp_cache_salt("refusal:0")', patch)
        self.assertEqual(patch.count("cache_salt=refusal_warmup_salt"), 2)
        warmup_patch = patch.split("--- a/vllm/v1/worker/gpu/warmup.py", 1)[1].split(
            "--- a/", 1
        )[0]
        self.assertEqual(warmup_patch.count("cache_salt=refusal_warmup_salt"), 2)


if __name__ == "__main__":
    unittest.main()
