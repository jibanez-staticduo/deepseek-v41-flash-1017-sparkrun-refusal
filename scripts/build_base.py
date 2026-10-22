#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Build the public, pinned r6x ARM64 base locally without a GPU smoke test.

The default is a read-only plan. Source preparation writes only under the
selected upstream checkout's ignored .work/build directory. --apply is the
only mode that invokes Docker Buildx, and it never pushes an image.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REPOSITORY = "https://github.com/christopherowen/spark-ds41f.git"
UPSTREAM_REVISION = "c38a24f2c9e714d2648769b12cd71aa6f0834d00"
UPSTREAM_LOCK_SHA256 = "e901dfab23a7f39a4c634fdf646e8fcec7a0dfc1feb68574acce94833a4cefcc"
SOURCE_MANIFEST = "manifests/sources/2026-10-19-r6x-source.json"
SOURCE_MANIFEST_SHA256 = "b92fc14f8ea1870f4176afd6726019c61ce7ae72a65eef44a4d4739b029e5e8b"
BASE_IMAGE = "docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf"
DEFAULT_TAG = "local/deepseek-v41-flash-1017-r6x:c38a24f2c9e7"
MIN_AVAILABLE_GIB = 64
RESERVE_GIB = 24
MEMORY_PER_JOB_GIB = 8
MAX_JOBS = 10
CPU_RESERVE = 4
NVCC_THREADS = 2
GIB = 1024**3
BUILD_LOCK = ROOT / "build/.build_base.lock"
TAG_PATTERN = re.compile(r"^local/[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def checked(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return runner(
        list(argv), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tag(tag: str) -> str:
    """Accept a local Docker tag only; this script has no registry push path."""
    if not TAG_PATTERN.fullmatch(tag) or "//" in tag or ".." in tag:
        raise ValueError("tag must be a safe local/<name>:<version> image reference")
    return tag


def validate_upstream_metadata(source: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        raise ValueError(f"source is not a Git checkout: {source}")
    revision = checked(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=source).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"upstream HEAD {revision}, expected pinned {UPSTREAM_REVISION}")
    dirty = checked(["git", "status", "--porcelain", "--untracked-files=all"], cwd=source).stdout.strip()
    if dirty:
        raise ValueError("upstream checkout has local changes; use a clean pinned checkout")

    remotes = checked(["git", "remote"], cwd=source).stdout.split()
    remote_urls: list[str] = []
    for remote in remotes:
        result = checked(["git", "remote", "get-url", remote], cwd=source)
        remote_urls.append(result.stdout.strip())
    normalized = {url.removesuffix(".git").rstrip("/") for url in remote_urls}
    public_https = UPSTREAM_REPOSITORY.removesuffix(".git")
    public_ssh = "git@github.com:christopherowen/spark-ds41f"
    if not normalized.intersection({public_https, public_ssh}):
        raise ValueError("checkout remote must be the public christopherowen/spark-ds41f repository")

    lock_path = source / "upstreams.lock.json"
    manifest_path = source / SOURCE_MANIFEST
    if sha256(lock_path) != UPSTREAM_LOCK_SHA256:
        raise ValueError("upstreams.lock.json does not match the pinned r6x lock hash")
    if sha256(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("the r6x source manifest does not match its pinned hash")
    with lock_path.open(encoding="utf-8") as stream:
        lock = json.load(stream)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = {
        "vllm": "4d6851c8bb1397a55254ba88905a87d9785b98dc",
        "b12x": "1a647a311ca55d45070ab83d60668d9c8fb8886c",
        "nccl": "6af10aa730a64fcee8781d62340a9c9184e79469",
        "tilelang": "222b895d04ee6cd75179155d06e0c0d450636f51",
        "tile_kernels": "cb2fa41c8af290204c81c547ed678ebf957cca53",
        "sparknet": "f72599fa212e894d67ee114553e628ab05daf2da",
    }
    for component, tree in expected.items():
        if manifest.get(component, {}).get("expected_tree") != tree:
            raise ValueError(f"r6x source manifest has an unexpected {component} tree")
    if lock.get("source_manifest") != SOURCE_MANIFEST:
        raise ValueError("upstreams.lock.json does not point to the pinned r6x source manifest")
    return lock


def ensure_upstream(source: Path) -> Path:
    source = source.expanduser().resolve()
    if source.exists():
        validate_upstream_metadata(source)
        return source

    source.parent.mkdir(parents=True, exist_ok=True)
    checked(["git", "clone", "--no-checkout", UPSTREAM_REPOSITORY, str(source)])
    checked(["git", "-C", str(source), "checkout", "--detach", UPSTREAM_REVISION])
    validate_upstream_metadata(source)
    return source


def load_spark_module(source: Path):
    """Load the pinned upstream build helpers without invoking site config."""
    script = source / "bin/spark"
    if not script.is_file():
        raise ValueError(f"pinned upstream build CLI is missing: {script}")
    loader = importlib.machinery.SourceFileLoader("spark_ds41f_build", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ValueError("cannot load pinned upstream bin/spark build helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def prepare_locked_inputs(spark: Any, upstreams: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Use upstream's locked source preparation, tree checks and wheel checks."""
    inputs = spark.build_inputs(upstreams)
    directory = spark.build_directory(inputs)
    problems = spark.build_problems(directory, inputs)
    if not problems:
        print(f"pinned upstream build inputs ready: {directory}", flush=True)
        return directory, inputs

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs.json").unlink(missing_ok=True)
    print("preparing the pinned upstream sources and verified BuildKit contexts", flush=True)
    original_problems = spark.project_problems

    def verify_source(source: Path, source_record: dict[str, Any]) -> list[str]:
        problems = original_problems(source, source_record)
        head_errors = [problem for problem in problems if ": patch head " in problem]
        if not head_errors or len(head_errors) != len(problems):
            return problems
        # Git may produce a different commit ID for the same reviewed patch
        # tree. Accept metadata drift only when the complete clean source tree
        # and any pinned submodule commits still match the upstream manifest.
        return []

    spark.project_problems = verify_source
    try:
        for name in spark.build_projects(inputs):
            spark.prepare_project(directory, name, inputs[name])
        spark.prepare_cutlass(directory, inputs["cutlass"])
        spark.prepare_wheels(directory, upstreams)

        context = directory / "context"
        for name in (*spark.build_projects(inputs), "cutlass"):
            spark.export_source(directory / "src" / name, context / f"{name}-source")
        deleted = spark.vllm_deletions(directory, inputs)
        deletions = context / "vllm-deletions"
        deletions.mkdir(exist_ok=True)
        (deletions / "deleted.txt").write_text("".join(f"{path}\n" for path in deleted), encoding="utf-8")
        (context / "empty").mkdir(exist_ok=True)
        spark.normalize_modes(context)
        record = {
            "schema": spark.BUILD_LAYOUT,
            "inputs": inputs,
            "contexts": {name: spark.tree_digest(context / name) for name in spark.build_contexts(inputs)},
            "vllm_deletions": len(deleted),
        }
        temporary = directory / ".inputs.json"
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(directory / "inputs.json")

        problems = spark.build_problems(directory, inputs)
        if problems:
            raise ValueError("upstream build context verification failed: " + "; ".join(problems))
    finally:
        spark.project_problems = original_problems
    print(f"pinned upstream build inputs ready: {directory}", flush=True)
    return directory, inputs


def compile_jobs(available_gib: float, cpu_count: int) -> int:
    """Match upstream's compile sizing: 8 GiB/job, 24 GiB reserve, 4 CPUs reserved."""
    if available_gib < MIN_AVAILABLE_GIB:
        raise ValueError(f"build needs at least {MIN_AVAILABLE_GIB} GiB MemAvailable; found {available_gib:.1f} GiB")
    cpu_jobs = max(1, (cpu_count - CPU_RESERVE) // NVCC_THREADS)
    memory_jobs = int((available_gib - RESERVE_GIB) // MEMORY_PER_JOB_GIB)
    return min(MAX_JOBS, cpu_jobs, memory_jobs)


def available_memory_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise ValueError("MemAvailable is missing from /proc/meminfo")


def build_command(
    spark: Any,
    source: Path,
    directory: Path,
    inputs: dict[str, Any],
    tag: str,
    jobs: int | str,
    *,
    environ: dict[str, str] | None = None,
) -> list[str]:
    """Render the exact Buildx shape used by upstream's image builder."""
    validate_tag(tag)
    upstreams = json.loads((source / "upstreams.lock.json").read_text(encoding="utf-8"))
    base = upstreams["sources"]["vllm_base_image"]
    locked_base = f"{base['upstream']}@{base['recorded_base_digest']}"
    if locked_base != BASE_IMAGE:
        raise ValueError("upstream base image differs from the pinned public ARM64 digest")

    context = directory / "context"
    command = [
        "docker", "buildx", "build", "--platform", "linux/arm64", "--network", "host", "--load",
        "--file", str(source / "docker/Dockerfile"), "--tag", tag,
    ]
    for name in spark.build_contexts(inputs):
        command += ["--build-context", f"{name}={context / name}"]
    build_args = {
        "BASE_IMAGE": inputs["base_image"],
        "BASE_IMAGE_NAME": base["upstream"],
        "BASE_IMAGE_DIGEST": base["recorded_base_digest"],
        "DEPLOYMENT_COMMIT": UPSTREAM_REVISION,
        "VLLM_COMPILE_JOBS": str(jobs),
        "VLLM_NVCC_THREADS": str(NVCC_THREADS),
    }
    for name in spark.build_projects(inputs):
        record = inputs[name]
        prefix = name.upper()
        build_args.update({
            f"{prefix}_UPSTREAM": record["upstream"].removesuffix(".git"),
            f"{prefix}_REF": record["tracking_ref"].removeprefix("refs/heads/"),
            f"{prefix}_REVISION": record["revision"],
            f"{prefix}_PATCH_HEAD": record["patch_head"],
            f"{prefix}_TREE": record["expected_tree"],
        })
        if "version" in record:
            build_args[f"{prefix}_VERSION"] = record["version"]
    stage = spark.runtime_stage(inputs)
    if stage is not None:
        build_args.update(RUNTIME_STAGE=stage, TILELANG_COMPILE_JOBS=str(jobs))
    for key, value in build_args.items():
        command += ["--build-arg", f"{key}={value}"]
    env = os.environ if environ is None else environ
    if env.get("GITHUB_TOKEN"):
        command += ["--secret", "id=github_token,env=GITHUB_TOKEN"]
    return command + [str(context / "empty")]


def check_idle_build_host(rows: Sequence[str]) -> None:
    active = []
    for row in rows:
        name, _, labels = row.partition("\t")
        if any(marker in f"{name} {labels}".lower() for marker in ("sparkrun", "spark.arena", "spark.cluster-config")):
            active.append(name)
    if active:
        raise ValueError("build host must be idle; SparkRun workloads are running: " + ", ".join(active))


def inspect_local_image(tag: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["docker", "image", "inspect", tag], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def validate_built_image(info: dict[str, Any], inputs: dict[str, Any]) -> dict[str, str]:
    if info.get("Os") != "linux" or info.get("Architecture") != "arm64":
        raise ValueError("Buildx result is not a linux/arm64 image")
    labels = (info.get("Config") or {}).get("Labels") or {}
    expected: dict[str, str] = {}
    for name, record in inputs.items():
        if isinstance(record, dict) and "expected_tree" in record:
            expected[f"local.spark3.{name}.tree"] = record["expected_tree"]
    for key, value in expected.items():
        if labels.get(key) != value:
            raise ValueError(f"built image label {key} is {labels.get(key)!r}, expected {value!r}")
    if labels.get("org.opencontainers.image.revision") != UPSTREAM_REVISION:
        raise ValueError("built image does not identify the pinned upstream revision")
    return expected


def memory_guarded_build(command: list[str], reserve_gib: float = RESERVE_GIB) -> tuple[int, float]:
    minimum = available_memory_gib()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            try:
                status = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                available = available_memory_gib()
                minimum = min(minimum, available)
                if available < reserve_gib:
                    stop_process_group(process)
                    raise RuntimeError(
                        f"build canceled: MemAvailable fell to {available:.1f} GiB, below {reserve_gib} GiB"
                    )
                continue
            if status:
                raise RuntimeError(f"docker buildx build failed with exit status {status}")
            return status, minimum
    except KeyboardInterrupt:
        stop_process_group(process)
        raise RuntimeError("build canceled by user") from None
    except BaseException:
        stop_process_group(process)
        raise


def stop_process_group(process: subprocess.Popen[Any]) -> None:
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"receipt already exists; choose a new path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def default_source() -> Path:
    return ROOT / "build/upstream/spark-ds41f"


def plan_text(source: Path, tag: str) -> str:
    rows = [
        "Read-only plan; no clone, checkout, build, registry, GPU or host mutation.",
        f"Upstream: {UPSTREAM_REPOSITORY}@{UPSTREAM_REVISION}",
        f"Public ARM64 base: {BASE_IMAGE}",
        f"Local image tag: {tag}",
        f"Upstream checkout: {source}",
    ]
    if source.exists():
        validate_upstream_metadata(source)
        spark = load_spark_module(source)
        upstreams = json.loads((source / "upstreams.lock.json").read_text(encoding="utf-8"))
        inputs = spark.build_inputs(upstreams)
        directory = spark.build_directory(inputs)
        problems = spark.build_problems(directory, inputs)
        if problems and all(": patch head " in problem for problem in problems):
            original = spark.project_problems
            spark.project_problems = lambda _source, _record: []
            try:
                problems = spark.build_problems(directory, inputs)
            finally:
                spark.project_problems = original
        if not problems:
            command = build_command(spark, source, directory, inputs, tag, "<sized from MemAvailable>")
            rows.extend(["", "Buildx plan:", shlex.join(command)])
        else:
            rows.extend(["", "Run --prepare to create and verify the locked source contexts."])
    else:
        rows.extend(["", "Run --apply on a dedicated ARM64 CPU build host to clone, prepare, check, and build."])
    return "\n".join(rows)


def prepare_only(source: Path) -> tuple[Any, Path, dict[str, Any]]:
    source = ensure_upstream(source)
    spark = load_spark_module(source)
    upstreams = validate_upstream_metadata(source)
    directory, inputs = prepare_locked_inputs(spark, upstreams)
    return spark, directory, inputs


def apply_build(source: Path, tag: str, receipt_path: Path) -> dict[str, Any]:
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise ValueError(f"--apply requires an ARM64 CPU build host, found {platform.machine()}")
    initial_memory = available_memory_gib()
    cpus = len(os.sched_getaffinity(0))
    jobs = compile_jobs(initial_memory, cpus)
    checked(["docker", "buildx", "version"])
    if receipt_path.exists():
        raise ValueError(f"receipt already exists; choose a new path: {receipt_path}")
    source = source.expanduser().resolve()
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = BUILD_LOCK.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise ValueError("another public base build is already running on this host") from error
    try:
        ps = checked(["docker", "ps", "--no-trunc", "--format", "{{.Names}}\t{{.Labels}}"])
        check_idle_build_host(ps.stdout.splitlines())
        if inspect_local_image(tag) is not None:
            raise ValueError(f"{tag} already exists; choose a new local tag")

        spark, directory, inputs = prepare_only(source)
        available = available_memory_gib()
        jobs = compile_jobs(available, cpus)
        command = build_command(spark, source, directory, inputs, tag, jobs)
        print(
            f"Building {tag}: linux/arm64, {jobs} compile jobs x {NVCC_THREADS} NVCC threads, "
            f"{available:.1f} GiB available; cancel below {RESERVE_GIB} GiB.",
            flush=True,
        )
        started = time.monotonic()
        _, minimum = memory_guarded_build(command)
        elapsed = round(time.monotonic() - started)
        info = inspect_local_image(tag)
        if info is None:
            raise ValueError("build finished but the local image is not present")
        trees = validate_built_image(info, inputs)
        receipt = {
            "schema_version": 1,
            "tag": tag,
            "image_id": info["Id"],
            "platform": "linux/arm64",
            "source_repository": UPSTREAM_REPOSITORY,
            "source_revision": UPSTREAM_REVISION,
            "source_patch_heads_actual": {
                name: checked(["git", "rev-parse", "HEAD"], cwd=directory / "src" / name).stdout.strip()
                for name in spark.build_projects(inputs)
            },
            "base_image": BASE_IMAGE,
            "source_tree_labels": trees,
            "compile_jobs": jobs,
            "nvcc_threads": NVCC_THREADS,
            "min_available_gib": round(minimum, 1),
            "elapsed_seconds": elapsed,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_receipt(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return receipt
    finally:
        lock.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="clone/verify pinned sources and prepare upstream build contexts")
    mode.add_argument("--apply", action="store_true", help="build the local ARM64 image; requires an ARM64 CPU host and 64 GiB MemAvailable")
    result.add_argument("--source", type=Path, default=default_source(), help="pinned upstream checkout path")
    result.add_argument("--tag", default=DEFAULT_TAG, help="local image tag; remote registry tags are rejected")
    result.add_argument("--receipt", type=Path, default=ROOT / "build/base-image-receipt.json")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_tag(args.tag)
    source = args.source.expanduser().resolve()
    if args.prepare:
        _, directory, _ = prepare_only(source)
        print(f"Prepared and verified upstream contexts: {directory}")
        return 0
    if args.apply:
        apply_build(source, args.tag, args.receipt.expanduser().resolve())
        return 0
    print(plan_text(source, args.tag))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
