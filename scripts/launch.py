# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Validate a three-node site and assemble one safe, per-rank vLLM command."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_SITE = Path("/etc/sparkrun-refusal/site.json")
DEFAULT_RUNTIME = Path("/opt/sparkrun-refusal/runtime.json")
DEFAULT_MODEL = Path("/srv/models/deepseek-v4.1-flash-1017")
MIN_AVAILABLE_BYTES = 5 * 1024**3
MIN_PAGE_SIZE = 65536
EXPECTED_MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
PROBES = {
    "page_size": Path("/proc/sys/kernel/osrelease"),
    "uvm_pack": Path("/sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables"),
    "drm_modeset": Path("/sys/module/nvidia_drm/parameters/modeset"),
    "mem_available": Path("/proc/meminfo"),
    "device": Path("/dev/dri/card0"),
}


class LaunchError(ValueError):
    """Input or host preflight violates the launch contract."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"{path} must contain a JSON object")
    return value


def _validate_site(site: dict[str, Any]) -> list[dict[str, Any]]:
    if site.get("schema_version") != 1:
        raise LaunchError("site schema_version must be 1")
    if site.get("topology", "direct") != "direct":
        raise LaunchError("site topology must be direct")
    if site.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise LaunchError("site model_revision must be the pinned official 1017 revision")
    model_path = site.get("model_path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise LaunchError("site model_path must be an absolute path to the pinned local model snapshot")
    nodes = site.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 3:
        raise LaunchError("site must define exactly three nodes")
    if sorted(node.get("rank") for node in nodes if isinstance(node, dict)) != [0, 1, 2]:
        raise LaunchError("site ranks must be exactly 0, 1, and 2")
    by_rank = sorted(nodes, key=lambda node: node["rank"])
    for node in by_rank:
        if "control_iface" not in node and "control_interface" in node:
            node["control_iface"] = node["control_interface"]
        for key in ("hostname", "ip", "management_ip", "control_iface", "roce_peer_hcas"):
            if key not in node:
                raise LaunchError(f"node rank {node['rank']} is missing {key}")
        if not isinstance(node["ip"], str) or not node["ip"]:
            raise LaunchError(f"node rank {node['rank']} ip must be a non-empty string")
        if not isinstance(node["roce_peer_hcas"], dict) or not node["roce_peer_hcas"]:
            raise LaunchError(f"node rank {node['rank']} roce_peer_hcas must be a non-empty mapping")
        if not all(isinstance(peers, list) and peers for peers in node["roce_peer_hcas"].values()):
            raise LaunchError(f"node rank {node['rank']} roce_peer_hcas entries must be non-empty lists")
    ips = [node["ip"] for node in by_rank]
    if len(set(ips)) != 3:
        raise LaunchError("site node IP addresses must be unique")
    return by_rank


def _validate_profile(runtime: dict[str, Any], model: Path) -> list[str]:
    if runtime.get("source_revision") != "c38a24f2c9e714d2648769b12cd71aa6f0834d00":
        raise LaunchError("runtime source revision is not the pinned c38a24f2 contract")
    env = runtime.get("environment")
    args = runtime.get("serve_args")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in env.items()):
        raise LaunchError("runtime environment must map strings to scalar values")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise LaunchError("runtime serve_args must be a list of strings")
    if args.count("/model") != 1:
        raise LaunchError("runtime serve_args must contain exactly one /model placeholder")
    args = list(args)
    args[args.index("/model")] = str(model)
    if "--served-model-name" not in args:
        raise LaunchError("runtime serve_args must specify a served model name")
    name_index = args.index("--served-model-name")
    args[name_index + 1] = "deepseek-v4.1-flash-1017"
    pairs = {args[i]: args[i + 1] for i, arg in enumerate(args[:-1]) if arg.startswith("--")}
    expected = {
        "--tensor-parallel-size": "3",
        "--max-model-len": "1048576",
        "--max-num-seqs": "8",
        "--max-num-batched-tokens": "4096",
        "--kv-cache-memory-bytes": "5268045824",
    }
    for key, value in expected.items():
        if pairs.get(key) != value:
            raise LaunchError(f"maximum1M profile requires {key} {value}")
    return list(args)


def _validate_model_config(model: Path) -> None:
    config_path = model / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read model config {config_path}: {exc}") from exc
    text = config.get("text_config", {})
    quant = config.get("quantization_config", {})
    if config.get("model_type") != "deepseek_v41" or "DeepseekV41ForCausalLM" not in config.get("architectures", []):
        raise LaunchError("model config does not identify the official DeepSeek V4.1 architecture")
    if text.get("hidden_size") != 5120 or text.get("num_hidden_layers") != 40:
        raise LaunchError("model config dimensions do not match DeepSeek V4.1 Flash 1017")
    if quant.get("quant_method") != "fp8" or quant.get("expert_dtype") != "fp4":
        raise LaunchError("model quantization metadata does not match the official FP8/FP4 checkpoint")


def _parse_mem_available(meminfo: str) -> int:
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB\s*$", meminfo, re.MULTILINE)
    if not match:
        raise LaunchError("cannot read MemAvailable from /proc/meminfo")
    return int(match.group(1)) * 1024


def preflight(probes: dict[str, str]) -> dict[str, Any]:
    try:
        page_size = int(probes["page_size"]) if "page_size" in probes else os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError) as exc:
        raise LaunchError("cannot determine host kernel page size") from exc
    if page_size != MIN_PAGE_SIZE:
        raise LaunchError(f"host kernel page size must be 65536 bytes, got {page_size}")
    if probes.get("uvm_pack", "").strip().upper() != "Y":
        raise LaunchError("NVIDIA UVM pack_sysmem_leaf_tables must be Y")
    if probes.get("drm_modeset", "").strip() != "Y":
        raise LaunchError("NVIDIA DRM modeset must be Y")
    available = _parse_mem_available(probes.get("mem_available", ""))
    if available < MIN_AVAILABLE_BYTES:
        raise LaunchError(f"MemAvailable must be at least 5 GiB, got {available} bytes")
    if probes.get("device") != "present":
        raise LaunchError("/dev/dri/card0 must be present in the container")
    if probes.get("model_readonly") != "yes":
        raise LaunchError("the mounted model path must be read-only")
    if probes.get("cache_writable") != "yes":
        raise LaunchError("the rank-local runtime cache must be writable")
    return {
        "kernel_page_size_bytes": page_size,
        "uvm_pack_sysmem_leaf_tables": "Y",
        "nvidia_drm_modeset": "Y",
        "mem_available_bytes": available,
        "drm_device": "/dev/dri/card0",
        "model_readonly": True,
        "rank_cache_writable": True,
        "driver_615_qualified": False,
    }


def build_plan(
    site: dict[str, Any],
    runtime: dict[str, Any],
    *,
    model: Path = DEFAULT_MODEL,
    node_rank: int,
    model_revision: str = EXPECTED_MODEL_REVISION,
    probes: dict[str, str],
    distributed_args: list[str] | None = None,
) -> dict[str, Any]:
    nodes = _validate_site(site)
    if node_rank not in (0, 1, 2):
        raise LaunchError("--node-rank must be 0, 1, or 2")
    if not model.is_absolute():
        raise LaunchError("--model must be an absolute path")
    if not model.is_dir():
        raise LaunchError(f"model path is not a directory: {model}")
    if model_revision != EXPECTED_MODEL_REVISION or model != Path(site.get("model_path", "")):
        raise LaunchError("model path or revision disagrees with the editable site config")
    _validate_model_config(model)
    cache_root = str(site.get("cache_root", "/cache/runtime"))
    cache_rank = f"{cache_root}/rank{node_rank}"
    env = {
        str(k): str(v).replace("/cache/kkref", cache_rank, 1)
        if str(v).startswith("/cache/kkref")
        else str(v)
        for k, v in runtime["environment"].items()
    }
    argv = ["vllm", "serve", *_validate_profile(runtime, model)]
    node = nodes[node_rank]
    head = nodes[0]
    master_port = int(site.get("master_port", 25000))
    if not 1 <= master_port <= 65535:
        raise LaunchError("master_port must be between 1 and 65535")
    gid_index = int(node.get("gid_index", site.get("gid_index", 3)))
    if gid_index != 3:
        raise LaunchError("this profile requires RoCE GID index 3")
    env.update(
        {
            "HOME": cache_rank + "/home",
            "TMPDIR": cache_rank + "/tmp",
            "HF_HOME": "/cache/huggingface",
            "XDG_CACHE_HOME": cache_rank + "/jit",
            "CUDA_CACHE_PATH": cache_rank + "/jit/nv-compute",
            "VLLM_CACHE_DIR": cache_rank + "/jit/vllm-r5o",
            "VLLM_CACHE_ROOT": cache_rank + "/jit/vllm-r5o",
            "TILELANG_CACHE_DIR": cache_rank + "/jit/tilelang",
            "TRITON_CACHE_DIR": cache_rank + "/jit/triton",
            "FLASHINFER_WORKSPACE_BASE": cache_rank + "/flashinfer",
            "B12X_COMPILE_CACHE_DIR": cache_rank + "/jit/b12x/compile",
            "B12X_CUTE_COMPILE_CACHE_DIR": cache_rank + "/jit/b12x/cute",
            "TORCHINDUCTOR_CACHE_DIR": cache_rank + "/jit/torchinductor",
            "CUTE_DSL_CACHE_DIR": cache_rank + "/jit/cute-dsl",
            "VLLM_REFUSAL_LAMBDA_INIT": os.environ.get("VLLM_REFUSAL_LAMBDA_INIT", "2.5"),
            "STATICDUO_REFUSAL_ADMIN_TOKEN_FILE": os.environ.get(
                "STATICDUO_REFUSAL_ADMIN_TOKEN_FILE", "/srv/secrets/refusal-admin-token"
            ),
            "VLLM_REFUSAL_DIRS": "/opt/sparkrun-refusal/refusal_dirs_1017.safetensors",
            "NCCL_SOCKET_IFNAME": str(node["control_iface"]),
            "GLOO_SOCKET_IFNAME": str(node["control_iface"]),
            "NCCL_IB_HCA": ",".join(
                sorted(
                    {
                        hca
                        for peers in node["roce_peer_hcas"].values()
                        for hca in peers
                    }
                )
            ),
            "NCCL_IB_GID_INDEX": str(gid_index),
            "SPARKNET_ROCE_TOPOLOGY": "direct",
            "SPARKNET_ROCE_GID_INDEX": str(gid_index),
            "VLLM_HOST_IP": str(node["management_ip"]),
            "TP_SOCKET_IFNAME": str(node["control_iface"]),
            "SPARKNET_ROCE_PEER_HCAS": json.dumps(
                node.get("roce_peer_hcas", node.get("peer_hcas")), separators=(",", ":")
            ),
        }
    )
    distributed_args = distributed_args or [
        "--nnodes", "3", "--node-rank", str(node_rank),
        "--master-addr", str(head["ip"]), "--master-port", str(master_port),
    ] + (["--headless"] if node_rank > 0 else [])
    distributed = argparse.ArgumentParser(add_help=False)
    distributed.add_argument("--nnodes", type=int, required=True)
    distributed.add_argument("--node-rank", type=int, required=True)
    distributed.add_argument("--master-addr", required=True)
    distributed.add_argument("--master-port", type=int, required=True)
    distributed.add_argument("--headless", action="store_true")
    try:
        native = distributed.parse_args(distributed_args)
    except SystemExit as exc:
        raise LaunchError("missing or malformed SparkRun distributed flags") from exc
    if native.nnodes != 3 or native.node_rank != node_rank:
        raise LaunchError("SparkRun distributed node count/rank disagrees with site")
    if native.master_addr != str(head["ip"]) or native.master_port != master_port:
        raise LaunchError("SparkRun master endpoint disagrees with site")
    if native.headless != (node_rank > 0):
        raise LaunchError("worker ranks require --headless; rank 0 must not be headless")
    if native.headless:
        for flag in ("--host", "--port"):
            if flag in argv:
                at = argv.index(flag)
                del argv[at : at + 2]
    argv.extend(distributed_args)
    return {
        "rank": node_rank,
        "node": node["hostname"],
        "master_addr": head["ip"],
        "master_port": master_port,
        "preflight": preflight(probes),
        "environment": env,
        "argv": argv,
    }


def _host_probes(
    node_rank: int | None = None,
    model_path: Path = DEFAULT_MODEL,
    site_path: Path = DEFAULT_SITE,
) -> dict[str, str]:
    meminfo = _read(PROBES["mem_available"])
    device = "present" if PROBES["device"].exists() else "missing"
    return {
        "page_size": str(os.sysconf("SC_PAGE_SIZE")),
        "uvm_pack": _read(PROBES["uvm_pack"]),
        "drm_modeset": _read(PROBES["drm_modeset"]),
        "mem_available": meminfo,
        "device": device,
        "model_readonly": _mount_readonly(model_path),
        "cache_writable": "yes" if os.access("/cache/runtime", os.W_OK) else "no",
    }


def _mount_readonly(path: Path) -> str:
    try:
        resolved = path.resolve()
        best: tuple[int, str] | None = None
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            mountpoint = Path(fields[4].replace("\\040", " "))
            if resolved == mountpoint or mountpoint in resolved.parents:
                options = fields[5].split(",")
                if best is None or len(str(mountpoint)) > best[0]:
                    best = (len(str(mountpoint)), "yes" if "ro" in options else "no")
        return best[1] if best else "no"
    except OSError:
        return "no"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=("maximum1M",), required=True)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--served-model-name", choices=("deepseek-v4.1-flash-1017",),
                        default="deepseek-v4.1-flash-1017")
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    native = argparse.ArgumentParser(add_help=False)
    native.add_argument("--nnodes", type=int, required=True)
    native.add_argument("--node-rank", type=int, required=True)
    native.add_argument("--master-addr", required=True)
    native.add_argument("--master-port", type=int, required=True)
    native.add_argument("--headless", action="store_true")
    args.native = native.parse_args(unknown)
    args.distributed_args = unknown
    return args


def main(argv: list[str] | None = None, *, probes_provider: Callable[[], dict[str, str]] = _host_probes) -> int:
    args = _parse_args(argv)
    try:
        site = _load_json(args.site)
        runtime = _load_json(args.runtime_config)
        if probes_provider is _host_probes:
            probes = _host_probes(args.native.node_rank, args.model, args.site)
        else:
            probes = probes_provider()
        plan = build_plan(
            site,
            runtime,
            model=args.model,
            node_rank=args.native.node_rank,
            model_revision=args.revision,
            probes=probes,
            distributed_args=args.distributed_args,
        )
    except (LaunchError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"launch preflight failed: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    for value in plan["environment"].values():
        if value.startswith(cache_root := str(site.get("cache_root", "/cache/runtime")) + "/rank"):
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(plan["environment"])
    os.execvp(plan["argv"][0], plan["argv"])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
