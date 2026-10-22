# Validation and build

## What has been exercised

See [direction provenance and exact CUDA-test coverage](provenance.md) for the audited source revisions, artifact hashes, target/drafter scope and the distinction between eager lambda mutation and CUDA Graph replay. The skipped CUDA test is not a graph replay test and remains unexecuted; separate graph-enabled service probes do not establish which execution path served each request.

The recorded target run used a three-system GB10 / SM121 TP3 deployment. Evidence includes a full model start, CUDA graph capture, API inference, simultaneous requests with different refusal values, and a successful 11,210-token prefill. The projector was also exercised on GPU at lambda 2.5. The earlier integration suite had 39 cases: 38 passed and one CUDA-isolated case remained skipped; that count describes the earlier code state and is not a result for this public checkout. These results establish that those paths ran in that environment; they do not establish model quality or broad hardware compatibility.

The recorded 1M context value is a configured maximum, not a repeated full-context benchmark. No controlled refusal-quality corpus, refusal-rate delta, quality regression study, or performance comparison is available. The reported SVD energy retention (81.15%) and residual of 40.77%-48.00% of the source delta norm are numerical artifact comparisons and must not be presented as behavioral measurements.

The recorded target used experimental NVIDIA driver 615.71.09 and Linux kernel `7.0.0-1019-nvidia-64k` with MemorySaver mode. This is evidence for that host, not a universal driver requirement or compatibility claim for other systems. The upstream host instructions list the 580 driver family as a standard recommendation; follow the current upstream instructions and run preflight on the actual target. The local 615 guard/fix path has not been included or qualified by this public package. The public launcher rejects a page size below 64 KiB; do not assume that every host using the upstream recommended driver family meets this requirement.

## Reproducible build inputs

The base for this project is the ARM64 image produced from the pinned public [spark-ds41f](https://github.com/christopherowen/spark-ds41f) source at revision `c38a24f2c9e714d2648769b12cd71aa6f0834d00`. Its locked foundation image is `docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf`. The upstream lock also pins compiled components and dependencies; use its lock and patch series instead of assembling an unpinned source tree.

`scripts/build_base.py` runs a read-only plan by default. Its `--prepare` option clones or verifies the pinned upstream checkout and prepares the upstream's locked sources without building an image. `--apply` prepares inputs if needed and runs Docker Buildx on an ARM64 CPU builder. It requires at least 64 GiB `MemAvailable`, reserves at least 24 GiB while building, sizes compile jobs using the upstream rules, records an image receipt, and does not run a GPU smoke or push to a registry. Choose an unused local tag if you override the default:

```bash
python3 scripts/build_base.py
python3 scripts/build_base.py --prepare
python3 scripts/build_base.py --apply
```

`scripts/prepare_build.py` then inspects the resulting ARM64 upstream image and requires labels for the expected vLLM, B12X, NCCL, TileLang, Tile kernels, and SparkNet source trees. It extracts the patched source files from that image, verifies every pre-patch and post-patch file hash, and writes an overlay into a new empty output directory. It uses `docker create` and `docker cp`; it does not run a build step inside the base image. A local derived image adds the request-scoped refusal code and fetched direction artifact; it does not copy model weights or credentials. This repository does not publish an OCI image; check the artifact license before redistributing a local image that contains it.

`scripts/fetch_directions.py` downloads the pinned author artifact or accepts it via `--source`. It checks the source revision and checksums, reads the ZIP entries as raw data, and never unpickles the file. It extracts the original layer directions into the runtime safetensors format, then checks the final digest. The pinned source repository declares MIT and includes an MIT license. This repository does not redistribute the artifact; the fetcher downloads the pinned file and verifies its SHA-256 before extracting it.

After building the upstream ARM64 image, fetch directions and prepare the derived context. Pass the local tag produced by `build_base.py`:

```bash
python scripts/fetch_directions.py --output assets/refusal_dirs_1017.safetensors
python3 scripts/prepare_build.py --base-image local/deepseek-v41-flash-1017-r6x:c38a24f2c9e7 --output build/context
docker build --platform linux/arm64 --build-arg BASE_IMAGE=local/deepseek-refusal-base:BASE_IMAGE_ID_HEX -t local/deepseek-refusal:v1 build/context
```

`prepare_build.py` verifies the upstream image id and labels, then creates a local tag named `local/deepseek-refusal-base:<full-image-id>` and verifies that it resolves to the same image. Run the `build` command it prints, which uses this content-named tag. Do not replace it with a bare `sha256:` value or a mutable registry tag. The build context receipt includes source revision, patch input/output hashes, runtime configuration hash, and directions hash. Preserve it with the final image digest. The rendered recipe does not change host configuration; SparkRun supplies distributed rank and rendezvous arguments at launch.

To render a portable SparkRun v2 recipe, provide a local site file with three node entries and an immutable image reference. SparkRun 0.3.10 supplies node ranks and rendezvous arguments at launch; the wrapper does not duplicate those flags.

```bash
python scripts/render_recipe.py \
  --site config/site.json \
  --image registry.example.com/deepseek-refusal@sha256:YOUR_64_HEX_DIGEST \
  --output recipes/deepseek-v4.1-flash-1017-tilelang-tp3-refusal.yaml
```

Create `config/site.json` from the sanitized schema in `examples/site.example.json`, using the site's own node addresses and interface names. Copy that private file to `/srv/sparkrun-refusal/site.json` on each node, the path mounted by the recipe. Do not publish the site file or its private addresses and hostnames. Install the checkpoint at the same read-only model path on each node and provision the admin token at the mounted secret path with owner-only permissions. Review the generated recipe, build manifest, source hashes, artifact hash, and final image digest before using it.

Start using the three site hostnames in rank order:

```bash
sparkrun run recipes/deepseek-v4.1-flash-1017-tilelang-tp3-refusal.yaml \
  --hosts NODE_RANK0,NODE_RANK1,NODE_RANK2
```

The recipe only declares container/runtime settings. The launch wrapper runs read-only preflight checks and starts vLLM; it does not configure host kernel, drivers, network, or RDMA. The maximum1M profile requires 64 KiB pages, UVM leaf packing, DRM modeset, and sufficient available host memory. The local 615 kernel fix is not included in this package.

Do not treat a mutable tag as an image identity. Keep the build receipt and record the final image digest used by each deployment.

## Runtime checks

### Public Package Checks On 2026-10-22

The source package passed 48 CPU tests with one CUDA test skipped, including the five checks against the prepared source overlay. Fetching the author's pinned artifact reproduced the expected safetensors digest. All nine patch input/output hashes matched. A local COPY-only derived image built successfully against the available ARM64 source image. That existing image includes the experimental local driver adaptation; this check validates packaging, not the behavior of a freshly compiled public base.

The public upstream source preparation and emitted Buildx plan were exercised. A complete fresh ARM64 compilation and a TP3 launch of this public launcher have not been run. The earlier live TP3 results describe the original integration, not a qualification of this packaging adapter.

SparkRun 0.3.10 validates the recipe with zero errors and four warnings. The warnings concern explicit communication settings, cache settings, and local model/cache/secret/site mounts. These are intentional operator inputs: the launcher binds per-rank interfaces from the site file and puts compile caches under the persistent SSD mount. This is a site-configured template, not a registry recipe that works without local preparation. Select the NVIDIA DRM character device for each host; it may be `card1` rather than `card0` on the host, while the container destination remains `card0`.

The actual SparkRun node-command generator was exercised for ranks 0, 1 and 2, and its output was accepted by the launcher with synthetic host probes. Each plan carries the rank, rendezvous address/port and worker headless flag exactly once. No GPU was used by that check.

The base builder accepts a generated patch commit hash differing from upstream's recorded `patch_head` only when all other upstream source checks, including the exact expected tree, clean checkout, submodules and exported context hashes, pass. This accounts for Git patch commit metadata differences; the receipt records actual source commit IDs.

Before serving traffic, verify that:

- The official model id and revision match the intended 1017 checkpoint.
- The direction artifact has exactly 38 F32[5120] vectors for backbone `WO-B` layers 2–39 and passes its pinned SHA-256 check.
- The configured process default is the intended lambda. The bare code default is 1.0 when no environment override is supplied; the generated deployment recipe explicitly selects 2.5.
- Lambda 0 is bit-exact at the projection site, while nonzero strengths use the documented BF16 projection.
- Request overrides survive the full request path, including multimodal wrapping, CED selection, SP selection, and graph execution.
- Simultaneous requests with distinct values remain isolated and prefix caching uses distinct policy identities.
- The admin endpoint is reachable only through the intended loopback/tunnel path and rejects missing or invalid credentials.

For a public or production claim about refusal behavior, add a fixed, versioned evaluation set and report counts, sampling settings, model/image/artifact hashes, and uncertainty. Keep this separate from integration and kernel tests.

## CPU test suite

Use a Python 3.12 virtual environment and the CPU-only PyTorch wheel. Put temporary files on local SSD storage:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
mkdir -p build/test-tmp
export TMPDIR="$PWD/build/test-tmp"
python -m unittest discover -s tests -v
```

The CUDA projection test skips when CUDA is unavailable. To run the five structural upstream overlay checks as well, point them at the prepared overlay:

```bash
SPARKRUN_REFUSAL_OVERLAY=build/context/overlay python -m unittest tests.test_runtime_overlay -v
```

A passing CPU suite does not qualify the full model on GPU.
