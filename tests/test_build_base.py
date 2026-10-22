# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""CPU-only tests for the public pinned upstream base build wrapper."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_base", ROOT / "scripts/build_base.py")
assert SPEC is not None and SPEC.loader is not None
build_base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_base)


def fixture_inputs():
    inputs = {"base_image": build_base.BASE_IMAGE}
    for name in ("vllm", "b12x", "nccl", "tilelang", "tile_kernels", "sparknet"):
        inputs[name] = {
            "upstream": f"https://github.com/example/{name}.git",
            "tracking_ref": "refs/heads/main",
            "revision": "1" * 40,
            "patch_head": "2" * 40,
            "expected_tree": "3" * 40,
            "version": "1.2.3",
        }
    return inputs


class FakeSpark:
    BUILD_LAYOUT = 2

    @staticmethod
    def build_contexts(_inputs):
        return ("vllm-source", "b12x-source", "cutlass-source", "cutlass-dsl-wheels", "vllm-deletions")

    @staticmethod
    def build_projects(_inputs):
        return ("vllm", "b12x", "nccl", "tilelang", "tile_kernels", "sparknet")

    @staticmethod
    def runtime_stage(_inputs):
        return "runtime-tilelang-sparknet"


class BuildBaseTests(unittest.TestCase):
    def test_compile_job_sizing_matches_upstream_limits(self):
        self.assertEqual(build_base.compile_jobs(64, 8), 2)
        self.assertEqual(build_base.compile_jobs(104, 64), 10)
        self.assertEqual(build_base.compile_jobs(80, 8), 2)
        with self.assertRaisesRegex(ValueError, "64 GiB"):
            build_base.compile_jobs(63.9, 64)

    def test_only_safe_local_tags_are_accepted(self):
        self.assertEqual(build_base.validate_tag("local/example:revision-1"), "local/example:revision-1")
        for tag in ("example:latest", "docker.io/example/image:tag", "local/name", "local/../bad:tag"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                build_base.validate_tag(tag)

    def test_workload_idle_check_reads_labels_as_well_as_names(self):
        build_base.check_idle_build_host(["ordinary\tcom.example=unrelated"])
        with self.assertRaisesRegex(ValueError, "SparkRun workloads"):
            build_base.check_idle_build_host(["worker\tio.spark-arena.sparkrun.job=active"])

    def test_build_command_has_locked_contexts_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "upstreams.lock.json").write_text(json.dumps({
                "sources": {"vllm_base_image": {
                    "upstream": "docker.io/vllm/vllm-openai",
                    "recorded_base_digest": build_base.BASE_IMAGE.rsplit("@", 1)[1],
                }}
            }))
            inputs = fixture_inputs()
            command = build_base.build_command(
                FakeSpark(), source, source / ".work/build", inputs, "local/test:locked", 3,
                environ={},
            )
        self.assertIn("--platform", command)
        self.assertIn("linux/arm64", command)
        self.assertIn("--load", command)
        self.assertIn("VLLM_COMPILE_JOBS=3", command)
        self.assertIn("TILELANG_COMPILE_JOBS=3", command)
        self.assertNotIn("push", command)
        self.assertNotIn("--gpus", command)
        self.assertIn("--build-context", command)

    def test_built_image_requires_arm64_and_all_tree_labels(self):
        inputs = fixture_inputs()
        labels = {f"local.spark3.{name}.tree": record["expected_tree"]
                  for name, record in inputs.items() if isinstance(record, dict) and "expected_tree" in record}
        labels["org.opencontainers.image.revision"] = build_base.UPSTREAM_REVISION
        info = {"Os": "linux", "Architecture": "arm64", "Config": {"Labels": labels}}
        self.assertEqual(set(build_base.validate_built_image(info, inputs)), set(labels) - {"org.opencontainers.image.revision"})
        info["Architecture"] = "amd64"
        with self.assertRaisesRegex(ValueError, "arm64"):
            build_base.validate_built_image(info, inputs)

    def test_apply_build_is_local_and_does_not_run_a_gpu_smoke(self):
        inputs = fixture_inputs()
        labels = {f"local.spark3.{name}.tree": record["expected_tree"]
                  for name, record in inputs.items() if isinstance(record, dict) and "expected_tree" in record}
        labels["org.opencontainers.image.revision"] = build_base.UPSTREAM_REVISION
        image = {"Id": "sha256:" + "a" * 64, "Os": "linux", "Architecture": "arm64", "Config": {"Labels": labels}}
        calls = []

        def fake_build(command, reserve_gib=24):
            calls.append(command)
            self.assertEqual(reserve_gib, 24)
            return 0, 70.0

        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            result_receipt_path = Path(directory) / "other-receipt.json"
            lock_path = Path(directory) / "build.lock"
            source = Path(directory) / "fresh-source"
            source.mkdir()
            (source / "upstreams.lock.json").write_text(json.dumps({
                "sources": {"vllm_base_image": {
                    "upstream": "docker.io/vllm/vllm-openai",
                    "recorded_base_digest": build_base.BASE_IMAGE.rsplit("@", 1)[1],
                }}
            }))
            with (
                patch.object(build_base.platform, "machine", return_value="aarch64"),
                patch.object(build_base, "available_memory_gib", return_value=80.0),
                patch.object(build_base.os, "sched_getaffinity", return_value=set(range(16))),
                patch.object(build_base, "checked", side_effect=[CompletedProcess([], 0, "buildx v1", ""), CompletedProcess([], 0, "", ""), *[CompletedProcess([], 0, "4" * 40, "") for _ in range(6)]]),
                patch.object(build_base, "write_receipt", wraps=build_base.write_receipt),
                patch.object(build_base, "check_idle_build_host"),
                patch.object(build_base, "inspect_local_image", side_effect=[None, image]),
                patch.object(build_base, "prepare_only", return_value=(FakeSpark(), source / ".work/build", inputs)),
                patch.object(build_base, "memory_guarded_build", side_effect=fake_build),
                patch.object(build_base, "BUILD_LOCK", lock_path),
            ):
                result = build_base.apply_build(source, "local/test:locked", result_receipt_path)
            self.assertTrue(result_receipt_path.is_file())

        self.assertEqual(result["image_id"], image["Id"])
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertIn("--load", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("push", command)
        self.assertIn("--tag", command)


if __name__ == "__main__":
    unittest.main()
