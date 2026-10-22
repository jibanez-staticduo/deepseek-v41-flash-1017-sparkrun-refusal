# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Render a portable SparkRun v2 recipe from the sanitized runtime contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DEFAULT_RUNTIME = ROOT / "runtime.json"
DEFAULT_SITE = ROOT / "examples/site.example.json"
DEFAULT_OUTPUT = ROOT / "recipes/deepseek-v4.1-flash-1017-tilelang-tp3-refusal.yaml"
MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
IMAGE_EXAMPLE = "registry.example.com/deepseek1017-refusal@sha256:" + "1" * 64


def _validate_runtime(runtime: dict) -> tuple[dict[str, str], list[str]]:
    if runtime.get("source_revision") != "c38a24f2c9e714d2648769b12cd71aa6f0834d00":
        raise ValueError("runtime source_revision does not match the pinned public source")
    environment = runtime.get("environment")
    args = runtime.get("serve_args")
    if not isinstance(environment, dict) or not all(
        isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in environment.items()
    ):
        raise ValueError("runtime environment must be a mapping of scalar values")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("runtime serve_args must be an argument list")
    if args.count("/model") != 1:
        raise ValueError("runtime serve_args must have exactly one /model placeholder")
    pairs = {
        args[i]: args[i + 1]
        for i, arg in enumerate(args[:-1])
        if arg.startswith("--") and args[i + 1].isdigit()
    }
    required = {
        "--tensor-parallel-size": "3",
        "--max-model-len": "1048576",
        "--max-num-seqs": "8",
        "--kv-cache-memory-bytes": "5268045824",
        "--max-num-batched-tokens": "4096",
    }
    for flag, expected in required.items():
        if pairs.get(flag) != expected:
            raise ValueError(f"runtime serve_args must set {flag}={expected}")
    if pairs.get("--pipeline-parallel-size", "1") != "1":
        raise ValueError("maximum1M requires pipeline parallel size 1")
    return {str(k): str(v) for k, v in environment.items()}, list(args)


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _validate_image(image: str) -> None:
    if "@sha256:" not in image or not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
        raise ValueError("image must be an immutable registry reference ending in @sha256:<64 lowercase hex>")


def render(runtime: dict, site: dict, image: str = IMAGE_EXAMPLE) -> str:
    environment, _serve_args = _validate_runtime(runtime)
    _validate_image(image)
    _validate_site(site)

    # Render only variables SparkRun does not supply or the wrapper does not bind per rank.
    env = dict(environment)
    env.update(
        {
            "VLLM_REFUSAL_LAMBDA_INIT": "2.5",
            "VLLM_REFUSAL_DIRS": "/opt/sparkrun-refusal/refusal_dirs_1017.safetensors",
            "STATICDUO_REFUSAL_ADMIN_TOKEN_FILE": "/srv/secrets/refusal-admin-token",
            "SPARKRUN_NO_RUNTIME_CACHE": "1",
        }
    )
    # Values are rendered as JSON-escaped scalars, valid in YAML's plain/quoted forms.
    command = [
        "python3 /opt/sparkrun-refusal/launch.py --model {model} --revision " + EXPECTED_REVISION,
        "--profile maximum1M",
        "--served-model-name deepseek-v4.1-flash-1017",
        "--site /etc/sparkrun-refusal/site.json",
        "--runtime-config /opt/sparkrun-refusal/runtime.json",
        # SparkRun vllm-distributed appends the native distributed flags to the recipe command.
    ]
    lines = [
        'recipe_version: "2"',
        "runtime: vllm-distributed",
        f"container: {image}",
        'model_revision: "' + EXPECTED_REVISION + '"',
        'model: ' + _quote(site['model_path']),
        "min_nodes: 3",
        "max_nodes: 3",
        "builder: docker-pull",
        "defaults:",
        "  tensor_parallel: 3",
        "  pipeline_parallel: 1",
        "  max_model_len: 1048576",
        "  gpu_memory_utilization: 0.82",
        "metadata:",
        '  description: "Example only: edit immutable image and local mounts before use"',
        "executor_config:",
        '  entrypoint: ""',
        "  ipc: host",
        "  shm_size: 16gb",
        "  network: host",
        "  privileged: false",
        '  user: "0:0"',
        "  cap_add: [IPC_LOCK]",
        '  devices: ["/dev/dri/card0:/dev/dri/card0:rwm", "/dev/infiniband:/dev/infiniband:rwm"]',
        '  gpus: "all"',
        '  memory_limit: "112g"',
        '  security_opt: ["seccomp=unconfined"]',
        '  ulimit: ["memlock=-1:-1", "stack=67108864:67108864", "core=0"]',
        "  volumes:",
        '    - ' + _quote(site['model_path'] + ':' + site['model_path'] + ':ro'),
        '    - "/srv/cache/deepseek1017:/cache/runtime:rw"',
        '    - "/srv/secrets/refusal-admin-token:/srv/secrets/refusal-admin-token:ro"',
        '    - "/srv/sparkrun-refusal/site.json:/etc/sparkrun-refusal/site.json:ro"',
        "distribution_config:",
        "  models:",
        "    enabled: false",
        "    entries: []",
        "runtime_cache:",
        "  enabled: false",
        "env:",
    ]
    lines.extend(f"  {key}: {_quote(value)}" for key, value in sorted(env.items()))
    lines.extend(["command: >-"] + [f"  {line}" for line in command])
    return "\n".join(lines) + "\n"


def _validate_site(site: dict) -> None:
    from launch import _validate_site as validate
    validate(site)
    return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--image", required=True, help="immutable registry/image@sha256:digest")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    site = json.loads(args.site.read_text(encoding="utf-8"))
    recipe = render(runtime, site, args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(recipe, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
