import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "payload"))
import refusal_policy as policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.artifact = Path(self.temp.name) / "directions.safetensors"
        self.artifact.write_bytes(b"synthetic directions fixture")
        self.env = patch.dict(os.environ, {
            "VLLM_REFUSAL_DIRS": str(self.artifact),
            "VLLM_REFUSAL_LAMBDA_INIT": "1",
        })
        self.env.start()
        self.artifact_hash = patch.object(
            policy, "_hash_artifact", return_value=policy.ARTIFACT_SHA256
        )
        self.artifact_hash.start()
        with policy._lock:
            policy._default_lambda = 1.0
            policy._epoch = 0

    def tearDown(self):
        self.artifact_hash.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_request_stamp_is_canonical_and_preserves_tenant_salt(self):
        stamped = policy.stamp_cache_salt("tenant:private-salt")
        self.assertTrue(stamped.startswith(policy.PREFIX))
        envelope = json.loads(stamped[len(policy.PREFIX):])
        self.assertEqual(envelope["tenant_salt"], "tenant:private-salt")
        self.assertEqual(envelope["lambda"], float(1).hex())
        self.assertEqual(envelope["artifact_sha256"], policy.ARTIFACT_SHA256)
        self.assertEqual(policy.parse_effective_lambda(stamped), 1.0)
        self.assertEqual(policy.stamp_cache_salt(stamped), stamped)

    def test_override_zero_and_precise_float_values_have_distinct_stamps(self):
        zero = policy.stamp_cache_salt("refusal:0")
        precise = policy.stamp_cache_salt("refusal:1.0000001")
        self.assertEqual(policy.parse_effective_lambda(zero), 0.0)
        self.assertNotEqual(zero, precise)
        self.assertEqual(policy.parse_effective_lambda(precise), 1.0000001)

    def test_default_change_only_affects_future_stamps(self):
        before = policy.stamp_cache_salt("tenant")
        self.assertEqual(policy.set_default_lambda(2.5), (2.5, 1))
        after = policy.stamp_cache_salt("tenant")
        self.assertEqual(policy.parse_effective_lambda(before), 1.0)
        self.assertEqual(policy.parse_effective_lambda(after), 2.5)
        self.assertNotEqual(before, after)

    def test_bad_values_and_tampered_stamps_are_rejected(self):
        for value in (-0.01, 4.01, math.nan, math.inf, "nan", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.set_default_lambda(value)
        stamped = policy.stamp_cache_salt("tenant")
        envelope = json.loads(stamped[len(policy.PREFIX):])
        envelope["version"] += 1
        changed = policy.PREFIX + json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ValueError):
            policy.stamp_cache_salt(changed)

    def test_enabled_worker_requires_a_valid_stamp_and_artifact_identity(self):
        for value in (None, "tenant-salt"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "missing its policy envelope"
            ):
                policy.parse_effective_lambda(value)
        stamped = policy.stamp_cache_salt("tenant")
        envelope = json.loads(stamped[len(policy.PREFIX):])
        envelope["artifact_sha256"] = "0" * 64
        forged = policy.PREFIX + json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(ValueError, "artifact does not match"):
            policy.parse_effective_lambda(forged)

    def test_disabled_feature_is_inert_and_legacy_salt_is_preserved(self):
        with patch.dict(os.environ, {"VLLM_REFUSAL_DIRS": ""}):
            legacy = "refusal:2.0"
            self.assertEqual(policy.stamp_cache_salt(legacy), legacy)
            self.assertEqual(policy.parse_effective_lambda(None), 0.0)

    def test_only_ip_loopback_is_accepted(self):
        self.assertTrue(policy.is_loopback_host("127.0.0.1"))
        self.assertTrue(policy.is_loopback_host("::1"))
        self.assertFalse(policy.is_loopback_host("localhost"))
        self.assertFalse(policy.is_loopback_host("10.0.0.1"))


if __name__ == "__main__":
    unittest.main()
