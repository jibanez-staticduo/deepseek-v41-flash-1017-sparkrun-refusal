import hashlib
import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refusal_projection_under_test", ROOT / "payload" / "refusal_projection.py"
)
rp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rp)


def write_synthetic_directions(path):
    """Write valid unit-vector directions without distributing model artifacts."""
    header = {
        "__metadata__": {
            "model": rp.EXPECTED_MODEL,
            "revision": rp.EXPECTED_REVISION,
            "scope": rp.EXPECTED_SCOPE,
            "status": "synthetic test fixture",
        }
    }
    chunks = []
    offset = 0
    for index, layer in enumerate(range(rp.FIRST_LAYER, rp.LAST_LAYER + 1)):
        direction = torch.zeros(rp.HIDDEN_SIZE, dtype=torch.float32)
        direction[index] = 1.0
        chunk = direction.numpy().tobytes()
        header[f"layers.{layer}.attn.wo_b"] = {
            "dtype": "F32",
            "shape": [rp.HIDDEN_SIZE],
            "data_offsets": [offset, offset + len(chunk)],
        }
        offset += len(chunk)
        chunks.append(chunk)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


class SyntheticArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.artifact = Path(self.temp.name) / "synthetic-directions.safetensors"
        write_synthetic_directions(self.artifact)
        self.environment = patch.dict(
            os.environ, {"VLLM_REFUSAL_DIRS": str(self.artifact)}, clear=True
        )
        self.environment.start()
        self.sha_patch = patch.object(
            rp, "_sha256_file", return_value=rp.EXPECTED_SHA256
        )
        self.sha_patch.start()
        rp._load_directions.cache_clear()
        rp._configured_artifact.cache_clear()

    def tearDown(self):
        rp._load_directions.cache_clear()
        rp._configured_artifact.cache_clear()
        self.sha_patch.stop()
        self.environment.stop()
        self.temp.cleanup()

    def projection(self, layer=rp.FIRST_LAYER):
        return rp.make_projection(layer, rp.HIDDEN_SIZE)

    def test_factory_constructs_only_target_layers(self):
        projections = [
            rp.make_projection(layer, rp.HIDDEN_SIZE)
            for layer in range(rp.FIRST_LAYER, rp.LAST_LAYER + 1)
        ]
        self.assertEqual(len(projections), 38)
        self.assertTrue(all(isinstance(item, rp.RefusalProjection) for item in projections))
        self.assertTrue(all(item.direction.shape == (rp.HIDDEN_SIZE,) for item in projections))
        self.assertIsNone(rp.make_projection(rp.FIRST_LAYER - 1, rp.HIDDEN_SIZE))
        self.assertIsNone(rp.make_projection(rp.LAST_LAYER + 1, rp.HIDDEN_SIZE))

    def test_header_and_layer_validation_fail_closed(self):
        self.assertEqual(rp.artifact_identity().split("sha256=")[-1], rp.EXPECTED_SHA256)
        with self.assertRaises(ValueError):
            rp.make_projection(rp.FIRST_LAYER, rp.HIDDEN_SIZE - 1)
        with patch.dict(os.environ, {"VLLM_REFUSAL_DIRS": ""}):
            rp._configured_artifact.cache_clear()
            self.assertIsNone(rp.make_projection(rp.FIRST_LAYER, rp.HIDDEN_SIZE))

    def test_mixed_rows_zero_one_and_two_point_five_match_reference(self):
        projection = self.projection()
        direction = projection.direction
        values = torch.arange(
            4 * rp.HIDDEN_SIZE, dtype=torch.float32
        ).reshape(4, rp.HIDDEN_SIZE).to(torch.bfloat16)
        lambdas = torch.tensor([0.0, 1.0, 2.5, 0.0], dtype=torch.float32)
        actual = projection(values, lambdas)
        as_float = values.float()
        dot = torch.sum(as_float * direction, dim=-1, keepdim=True)
        calculated = (as_float - lambdas[:, None] * dot * direction).to(torch.bfloat16)
        expected = torch.where(lambdas[:, None] == 0, values, calculated)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), tuple(values.shape))
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_zero_rows_preserve_signed_zero_and_nan_payload_bits(self):
        projection = self.projection()
        values = torch.zeros((2, rp.HIDDEN_SIZE), dtype=torch.bfloat16)
        raw = values.view(torch.uint16)
        raw[0, 3] = 0x8000
        raw[0, 4] = 0x7FC1
        raw[1, 3] = 0xFFFF
        result = projection(values, torch.zeros(2, dtype=torch.float32))
        self.assertTrue(torch.equal(result.view(torch.uint16), values.view(torch.uint16)))

    def test_noncontiguous_rows_and_lambdas_are_supported(self):
        projection = self.projection()
        values = torch.randn((3, 2 * rp.HIDDEN_SIZE)).to(torch.bfloat16)[:, ::2]
        lambdas = torch.tensor([0.0, 8.0, 1.0, 8.0, 2.5, 8.0])[::2]
        result = projection(values, lambdas)
        self.assertEqual(tuple(result.shape), (3, rp.HIDDEN_SIZE))
        self.assertTrue(
            torch.equal(result[0].view(torch.uint16), values[0].view(torch.uint16))
        )

    def test_invalid_shapes_dtypes_and_gradients_reject(self):
        projection = self.projection()
        good = torch.zeros((2, rp.HIDDEN_SIZE), dtype=torch.bfloat16)
        for lambdas in (
            torch.zeros((1,), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.float16),
            torch.zeros((2, 1), dtype=torch.float32),
        ):
            with self.subTest(shape=lambdas.shape, dtype=lambdas.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    projection(good, lambdas)
        with self.assertRaises(TypeError):
            projection(good.float(), torch.ones(2, dtype=torch.float32))
        with self.assertRaisesRegex(RuntimeError, "does not support autograd"):
            projection(good.requires_grad_(), torch.ones(2, dtype=torch.float32))

    def test_custom_op_is_torch_compile_fullgraph_compatible_on_cpu(self):
        projection = self.projection()
        compiled = torch.compile(projection, backend="eager", fullgraph=True)
        values = torch.randn((3, rp.HIDDEN_SIZE)).to(torch.bfloat16)
        lambdas = torch.tensor([0.0, 1.0, 2.5], dtype=torch.float32)
        torch.testing.assert_close(
            compiled(values, lambdas), projection(values, lambdas), rtol=0, atol=0
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime verification is environment-gated")
class CudaProjectionTests(unittest.TestCase):
    def test_cuda_rows_read_mutated_lambda_values(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "synthetic-directions.safetensors"
            write_synthetic_directions(artifact)
            with patch.dict(os.environ, {"VLLM_REFUSAL_DIRS": str(artifact)}, clear=True), \
                 patch.object(rp, "_sha256_file", return_value=rp.EXPECTED_SHA256):
                rp._load_directions.cache_clear()
                rp._configured_artifact.cache_clear()
                projection = rp.make_projection(rp.FIRST_LAYER, rp.HIDDEN_SIZE).to("cuda")
                values = torch.randn(
                    (3, rp.HIDDEN_SIZE), dtype=torch.float32, device="cuda"
                ).to(torch.bfloat16)
                lambdas = torch.zeros((3,), dtype=torch.float32, device="cuda")
                pointer = lambdas.data_ptr()
                zero = projection(values, lambdas)
                lambdas.copy_(torch.tensor([0.0, 1.0, 2.5], device="cuda"))
                mixed = projection(values, lambdas)
                self.assertEqual(lambdas.data_ptr(), pointer)
                self.assertTrue(
                    torch.equal(zero[0].view(torch.uint16), values[0].view(torch.uint16))
                )
                self.assertFalse(torch.equal(mixed[1], values[1]))
                rp._load_directions.cache_clear()
                rp._configured_artifact.cache_clear()


if __name__ == "__main__":
    unittest.main()
