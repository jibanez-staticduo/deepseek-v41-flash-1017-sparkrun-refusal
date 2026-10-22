import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "payload"))
import refusal_policy


class RequestToProjectionTests(unittest.TestCase):
    def test_admission_reorder_mixed_rows_projection_and_default_change(self):
        import test_projection

        projection_module = test_projection.rp
        rows_spec = importlib.util.spec_from_file_location(
            "rows_for_cpu_integration", ROOT / "payload/refusal_rows.py"
        )
        rows_module = importlib.util.module_from_spec(rows_spec)
        rows_spec.loader.exec_module(rows_module)

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "synthetic-directions.safetensors"
            test_projection.write_synthetic_directions(artifact)
            with patch.dict(os.environ, {"VLLM_REFUSAL_DIRS": str(artifact)}), \
                 patch.object(
                     projection_module,
                     "_sha256_file",
                     return_value=projection_module.EXPECTED_SHA256,
                 ):
                projection_module._load_directions.cache_clear()
                projection_module._configured_artifact.cache_clear()
                with patch.object(
                    refusal_policy, "_hash_artifact",
                    return_value=refusal_policy.ARTIFACT_SHA256,
                ):
                    with refusal_policy._lock:
                        refusal_policy._default_lambda = 1.0
                        refusal_policy._epoch = 0
                    stamp_one = refusal_policy.stamp_cache_salt("tenant-a")
                    stamp_zero = refusal_policy.stamp_cache_salt("refusal:0")
                    refusal_policy.set_default_lambda(2.5)
                    stamp_default = refusal_policy.stamp_cache_salt(None)
                    self.assertEqual(refusal_policy.parse_effective_lambda(stamp_one), 1)
                    self.assertEqual(refusal_policy.parse_effective_lambda(stamp_zero), 0)
                    self.assertEqual(
                        refusal_policy.parse_effective_lambda(stamp_default), 2.5
                    )

                state = rows_module.RefusalRows(4, 8, "cpu")
                state.set_request(
                    2, refusal_policy.parse_effective_lambda(stamp_one), "a"
                )
                state.set_request(
                    0, refusal_policy.parse_effective_lambda(stamp_zero), "b"
                )
                batch = SimpleNamespace(
                    num_reqs=2,
                    num_tokens_after_padding=6,
                    query_start_loc=torch.tensor([0, 3, 5]),
                    idx_mapping=torch.tensor([0, 2]),
                    is_padding=torch.tensor([False] * 5 + [True]),
                )
                strengths = state.fill(batch)
                project = projection_module.make_projection(
                    projection_module.FIRST_LAYER, projection_module.HIDDEN_SIZE
                )
                values = torch.randn(
                    6, projection_module.HIDDEN_SIZE, dtype=torch.bfloat16
                )
                result = project(values, strengths)
                self.assertTrue(
                    torch.equal(result[:3].view(torch.uint16), values[:3].view(torch.uint16))
                )
                self.assertTrue(
                    torch.equal(result[5].view(torch.uint16), values[5].view(torch.uint16))
                )
                self.assertFalse(torch.equal(result[3:5], values[3:5]))
                projection_module._load_directions.cache_clear()
                projection_module._configured_artifact.cache_clear()


if __name__ == "__main__":
    unittest.main()
