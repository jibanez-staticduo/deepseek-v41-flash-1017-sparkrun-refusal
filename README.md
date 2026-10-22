# DeepSeek V4.1 Flash 1017 refusal projection

This repository, [jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal](https://github.com/jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal), adds an optional, request-scoped refusal-direction projection to the official DeepSeek V4.1 Flash 1017 inference path. It keeps the published model checkpoint intact and applies a small runtime projection to selected hidden states. The projection strength can be selected for each request or changed as the default for new requests.

The implementation targets the r6x TileLang/vLLM stack on NVIDIA GB10 systems. It is an inference integration and an experimental control, not a claim that refusal behavior has been calibrated or improved on a benchmark.

## Credits And Declared Lineage

According to the lineage declared by the mechanism's author, **Dani designed this runtime integration approach: a rank-1 projection with a switchable lambda dial, allowing the original checkpoint to be served without baking the ablation into its weights.** It originated in his qwen38 work, and Dani also ported it to DeepSeek V4.0-Flash (0731). His reported milestones are:

- **2026-08-12:** `deepseek-v4-rank1`, hot lambda switching, lambda-0 bit-exactness, and 46 F32[4096] directions.
- **2026-08-18:** `rank1g`, an in-place device lambda buffer so CUDA Graphs read its updated value instead of baking in the strength.
- **2026-09-03:** a per-request lambda dial in qwen38, working value 1.0 and default 0.
- **2026-09-24:** a token-by-token probe and calibration `cal-20260924a`.
- **2026-10-15:** handover to Staticduo.

These milestones are attributed author statements, not an independent audit of Dani's Git history or current service. This credits the lineage of this runtime integration, without claiming that the underlying mathematical projection or directional-ablation research originated here. **Our 1017 implementation is a third generation in that declared lineage**, with separate request/row plumbing and different directions published by **pqhaz**, not recalibrated by us. Credit also remains with **christopherowen/spark-ds41f**, **local-inference-lab/vLLM and B12X**, their dependencies, and **Infatoshi** for the referenced checkpoint comparison. See [attribution](ATTRIBUTION.md#runtime-mechanism-lineage) and [direction provenance](docs/provenance.md).

## What it does

- Loads the official `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint and a separately pinned set of 38 refusal directions.
- Applies `y = x - lambda * (r^T x) * r` to the attention `WO-B` output in backbone layers 2 through 39. Each direction has width 5,120. MTP and DSpark layers are not changed.
- Allows lambda from 0 through 4, with a per-process default and an optional per-request override. Lambda 0 returns the input BF16 values bit-for-bit.
- Keeps request policy stable after admission and separates prefix-cache identity by the effective lambda, direction artifact, and caller cache salt.
- Provides an optional loopback-only admin endpoint for changing the default for future requests.

The code does not rewrite model weights. Direction data is a separate artifact fetched from its published origin and checked against the SHA-256 recorded in the runtime code. The pinned source repository declares MIT (`license: mit`) and includes an MIT license. This repository does not redistribute the artifact; review upstream terms before redistributing it or an image containing it.

### Provenance And Projection Scope

The 38 F32[5120] vectors are rows 2-39 of the author's `projected` storage in `pqhaz/DeepSeek-V4.1-Flash-1017-heretic@12d58ad0878b5008e80d720251adb969a0069412/ablation_directions.pt`. They are not directions from the earlier 0731 family and were not calibrated by us. The [provenance audit](docs/provenance.md) records source files, immutable revisions, hashes, extraction and limits of the local tensor comparison, with links to the author's README and specification.

**This implementation projects only target backbone WO-B layers 2-39. It does not project MTP/DSpark layers 40-42.** A changed target with an unchanged drafter could reduce speculative acceptance and add verification work; that possible acceptance/throughput cost has not been measured here. The earlier Oppo 0731 handover describes projecting both target and drafter using different, width-4096 directions. This compares scopes with that handover, not Oppo's current service. See [target/drafter boundaries](docs/provenance.md#target-and-drafter-scope).

## Model and runtime

The runtime serves DeepSeek V4.1 Flash 1017 using the official mixed-precision checkpoint: native MXFP4 expert weights, FP8 dense weights, and BF16 activations. The tested runtime configuration uses TileLang kernels, SSD-backed Engram tables, DSpark with five speculative tokens, and a configured 1,048,576-token maximum. These are runtime configuration choices; they do not change the checkpoint.

The upstream TP3 profile reports KV capacity equivalent to about 4.55 full 1M-token windows after its memory sizing. This does not mean eight full-context requests fit simultaneously: the recipe also caps sequences at eight, and graph captures, runtime buffers, prompt lengths, and available SSD/HBM affect usable capacity.

## Refusal strength

For a row `x`, unit direction `r`, and strength `lambda`, the projection is:

```text
y = x - lambda * dot(r, x) * r
```

- `lambda = 0` leaves the official model's hidden state unchanged.
- `lambda = 1` is one full simple projection.
- `lambda = 2.5` is the measured author-intensity reference for these directions. It is not an optimum, a calibrated operating point, or a promised refusal rate.

The directions are identified as the original 1017 directions, rather than SVD-derived substitutes. The reported mean SVD energy retention of 81.15% describes a vector approximation analysis; it is not a refusal rate or a quality score. CPU measurements of the recovered intensity were 2.499929 to 2.500011. The residual is 40.77% to 48.00% of the source delta norm after FP8/BF16 rounding; those values do not establish a quality impact.

## Evidence and limitations

The recorded integration validation includes the full TP3 model starting, CUDA graph capture, API inference, concurrent requests using different refusal values, and an 11,210-token prefill. A projector-level GPU run with lambda 2.5 also completed. The host used an experimental driver/kernel combination; this does not qualify other driver versions or systems.

The available evidence does **not** include a fixed refusal-quality corpus, a measured refusal-rate comparison, a model-quality comparison, a controlled performance comparison, or a repeated 1M-token-context test. Those outcomes remain unmeasured. A successful API response or a projection norm is not evidence of a refusal-quality improvement.

The publication's omitted test was `CudaProjectionTests.test_cuda_rows_read_mutated_lambda_values`, skipped because CUDA was unavailable in the CPU test environment. It checks eager calls before/after an in-place lambda update; it does **not** capture or replay a CUDA Graph. An isolated capture -> mutate lambda -> replay test was not in that suite and is not claimed as passed. See [the exact test and evidence boundary](docs/provenance.md#exact-omitted-cuda-test).

See [architecture](docs/architecture.md), [validation](docs/validation.md), and [security](docs/security.md) for implementation details and operational limits.

## Run with SparkRun

The sample deployment recipe explicitly starts with lambda 2.5. A request can override that value, or an operator can change the global default for future admissions through the protected local admin route. Lambda 2.5 is an author-intensity reference, not a quality-calibrated setting. The included recipe targets the 1M profile, which requires the site prerequisites described in [validation](docs/validation.md); the public package does not include the separate local 615 kernel fix.

Download the official model snapshot to the same read-only path on each node and prepare a private site map from the public example. Configure cache directories and a token file for the admin route. Do not commit the site map or token:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir /srv/models/deepseek-v4.1-flash-1017
install -d -m 0700 /srv/cache/deepseek1017 /srv/secrets /srv/sparkrun-refusal
install -d -m 0700 config
cp examples/site.example.json config/site.json
```

Create a token without printing it or putting it in shell history:

```bash
umask 077
python3 -c 'import secrets; from pathlib import Path; Path("/srv/secrets/refusal-admin-token").write_text(secrets.token_urlsafe(32) + chr(10))'
```

Edit `config/site.json` with your own three node names, routable addresses, control interface, ConnectX-7 HCA names, and peer mappings. Use the same checkpoint and cache mount paths on all three nodes. Render the recipe using your immutable image digest, then start it with the site node names in rank order:

Copy the completed site file to `/srv/sparkrun-refusal/site.json` on each node; that is the path mounted by the recipe. Keep this site file private because it contains network and fabric details.

```bash
python3 scripts/render_recipe.py \
  --site config/site.json \
  --image REGISTRY/IMAGE@sha256:YOUR_64_HEX_DIGEST
sparkrun run recipes/deepseek-v4.1-flash-1017-tilelang-tp3-refusal.yaml \
  --hosts NODE_RANK0,NODE_RANK1,NODE_RANK2
```

SparkRun performs the distributed rendezvous. The launch wrapper checks the kernel page size, UVM packing, DRM modeset, available host memory, model mount, cache permissions, and device mapping before executing vLLM. Passing preflight does not certify refusal quality or the 1M context on a new host.

On the API host, the authenticated helper changes the default for future admissions:

```bash
python3 scripts/refusal.py 0  # lambda 0
python3 scripts/refusal.py 1  # lambda 2.5
```

It reads the mounted token file without printing it. Existing requests keep their admitted value. Per-request `cache_salt: "refusal:0"` or `"refusal:2.5"` also works without changing the default.

## Build the image

The public build uses a locally built ARM64 upstream base image with verified component tree labels. The base builder defaults to a read-only plan. On an ARM64 CPU build host, use `--apply` to prepare the pinned sources and build the local base image:

```bash
python3 scripts/build_base.py
python3 scripts/build_base.py --apply
```

The builder requires at least 64 GiB `MemAvailable`, reserves 24 GiB throughout the build, records the local image id and source labels in a receipt, and does not push to a registry or run a GPU smoke. Use an unused local tag if you override the default. Then fetch and convert the direction artifact, prepare the copy-only overlay, and build the derived image from the emitted context:

```bash
python scripts/fetch_directions.py --output assets/refusal_dirs_1017.safetensors
python3 scripts/prepare_build.py --base-image local/deepseek-v41-flash-1017-r6x:c38a24f2c9e7 --output build/context
docker build --platform linux/arm64 \
  --build-arg BASE_IMAGE=local/deepseek-refusal-base:BASE_IMAGE_ID_HEX \
  -t local/deepseek-refusal:v1 build/context
```

Replace `BASE_IMAGE_ID_HEX` with the full image id printed by `scripts/build_base.py --apply`. `prepare_build.py` verifies and tags that local image by its id; use the exact build command it prints. The derived build does not include model weights or credentials. This repository does not redistribute the direction artifact or a built OCI image. See [the build guide](docs/validation.md) for the input checks and [the LiteLLM guide](docs/litellm.md) for the six aliases.

Do not expose the inference endpoint to an untrusted network without a separately designed authentication and network boundary. The refusal admin endpoint is not an inference authentication mechanism.

## Upstream references

- [DeepSeek V4.1 Flash 1017 model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [DeepSeek V4.1 Flash 1017 announcement](https://api-docs.deepseek.com/news/news261017/)
- [DeepSeek API changelog](https://api-docs.deepseek.com/updates/)

The upstream model and inference stack remain subject to their own licenses and notices. See [licenses](licenses/) in this repository before redistribution.

## Run tests

The CPU suite uses a CPU-only PyTorch wheel and keeps temporary files on local SSD storage:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
mkdir -p build/test-tmp
export TMPDIR="$PWD/build/test-tmp"
python -m unittest discover -s tests -v
```

To include the five source-overlay checks, point them at a prepared, ignored overlay directory:

```bash
SPARKRUN_REFUSAL_OVERLAY=build/context/overlay python -m unittest discover -s tests -p test_runtime_overlay.py -v
```

CUDA tests are skipped when no CUDA device is available. A passing CPU suite does not qualify the full model on GPU.
