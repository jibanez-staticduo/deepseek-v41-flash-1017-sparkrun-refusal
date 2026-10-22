# Attributions and pinned third-party components

This repository contains original Staticduo integration code and patches for
an existing inference stack. Pinned source repositories below are references
and build inputs; this repository does not vendor complete third-party source
trees. The root [LICENSE](LICENSE) applies to original project code. Third-party
files retain their respective licenses and notices.

## Runtime Mechanism Lineage

According to the lineage declared by the mechanism's author, Dani designed the
runtime rank-1 integration with a switchable lambda dial used as the antecedent
to this project: it began in qwen38 and he ported it to DeepSeek V4.0-Flash
(0731). The declared milestones are `deepseek-v4-rank1` on 2026-08-12 (hot
lambda, lambda-0 bit-exactness, 46 F32[4096] directions), `rank1g` on 2026-08-18
(an in-place device lambda buffer for CUDA Graphs), the qwen38 per-request dial
on 2026-09-03 (working lambda 1.0, default 0), the token-by-token probe and
`cal-20260924a` calibration on 2026-09-24, and the handover on 2026-10-15.

These are attributed author statements; this project has not independently
audited that Git history or the author's current service. The credit concerns
this runtime integration's declared lineage, not original authorship of the
mathematical projection or directional-ablation research. The present 1017
implementation is a third generation in that lineage, using different
pqhaz-published directions without recalibrating them ourselves. This
acknowledgment does not replace the component, model or direction-data credits
and licenses below. Infatoshi is credited for the referenced native-checkpoint
comparison documented in [provenance](docs/provenance.md).

### Pocharlies Reference Implementation

According to the lineage declared by its authors, Pocharlies
(`@oppocharliesllmbot`, `pocharlies-org`) authored the reference implementation
ported here and its CUDA Graph fix documented as `rank1g`; Dani retains credit
for the mechanism's design. The supplied reference is the private handover
`pocharlies-org/qwen38-recipe-handover`, directory `deepseek-rank1-refusal/`,
declared commit `0af2f04da7545f1687fe7aae85250ca89929304b`. Its identifiers
include `patches/0001-rank1-projection.patch`,
`patches/0002-per-request-lambda.patch`, `extract_refusal_dirs.py`, and
`FASE1.md`, `FASE2.md`, `FASE3.md`. Those identifiers exist in the supplied
material; the handover is not publicly accessible or redistributed here.
This is not an independent audit of the authors' Git history or current service.

The reference supplies the adapted patterns of an in-place device lambda
tensor (never rebinding it) for CUDA Graphs, distinct target/drafter roles,
lambda in the prefix-cache hash identity, and per-request `cache_salt`.
Other prior refusal implementations attributed to Pocharlies are the resident
qwen38 per-request dial (working lambda 1.0) and the image source artifact
`images/deepseek-v4-rank1` for DeepSeek V4.0-Flash 0731 (46 F32[4096]
directions and author-reported lambda-0 bit-exactness). The latter identifier
is documented in the private handover. No public source URL for either prior
implementation has been verified; no private registry address is published.

The present third-generation 1017 integration adapts that reference to a
different runtime and request/row path. It uses pqhaz directions without our
recalibration and projects only target layers 2-39, not the drafter. This credit
does not extend validation claims or replace the existing third-party licenses.

## Sources used by the recipe/runtime

| Component | Pinned source | License | License SHA-256 | Required handling |
|---|---|---|---|---|
| SparkRun upstream recipe (`spark-ds41f`) | [`christopherowen/spark-ds41f@c38a24f2c9e714d2648769b12cd71aa6f0834d00`](https://github.com/christopherowen/spark-ds41f/tree/c38a24f2c9e714d2648769b12cd71aa6f0834d00) | Apache-2.0 | `c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c` | Root license text is in [licenses/Apache-2.0-spark-ds41f.txt](licenses/Apache-2.0-spark-ds41f.txt). Preserve upstream notices; mark changed files. |
| vLLM fork | [`local-inference-lab/vllm@19f2c20ed4d64ae22bd009ceacadd85e2a81e354`](https://github.com/local-inference-lab/vllm/tree/19f2c20ed4d64ae22bd009ceacadd85e2a81e354) | Apache-2.0 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | License text in [licenses/Apache-2.0-vllm.txt](licenses/Apache-2.0-vllm.txt). Overlay files retain their vLLM SPDX/copyright and get a prominent modification notice in locally prepared build contexts. |
| Canonical vLLM base revision | [`vllm-project/vllm@0f8fa53acf43a9a6a7a1f7e6b196950c78cd6d7a`](https://github.com/vllm-project/vllm/tree/0f8fa53acf43a9a6a7a1f7e6b196950c78cd6d7a) | Apache-2.0 | Same Apache-2.0 text as the vLLM fork | The pinned base revision is recorded by the upstream lock. Retain relevant source notices in any redistribution. |
| B12X | [`local-inference-lab/b12x@236ddff04a5f5064084b46464800b70b34ec7d8c`](https://github.com/local-inference-lab/b12x/tree/236ddff04a5f5064084b46464800b70b34ec7d8c) | Apache-2.0 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | Preserve the upstream license, copyright/attribution notices, and notices for changed files if B12X source or binaries are redistributed. |
| TileLang | [`tile-ai/tilelang@b95ee4ffb52ae26e15619cad38c6f41e7e0a2b46`](https://github.com/tile-ai/tilelang/tree/b95ee4ffb52ae26e15619cad38c6f41e7e0a2b46) | MIT | `f70e5cdc24c3797af741930440086f9a0ad36930b05a3dfc25b420936dc2cfa1` | License text in [licenses/MIT-TileLang.txt](licenses/MIT-TileLang.txt); preserve copyright and permission notice in copies/substantial portions. |
| DeepSeek TileKernels | [`deepseek-ai/TileKernels@66258df6175d2f630ffecb04c5ab66bff8a2ae6a`](https://github.com/deepseek-ai/TileKernels/tree/66258df6175d2f630ffecb04c5ab66bff8a2ae6a) | MIT | `ebb4f09972aee8608be255debaf78451a68e95c290f55c240dec2ecfa16ea6be` | License text in [licenses/MIT-TileKernels.txt](licenses/MIT-TileKernels.txt); retain `Copyright (c) 2026 DeepSeek` and the MIT permission notice. TileKernels remains separately licensed from vLLM. |
| NCCL | [`NVIDIA/nccl@73cf112295c33aee2b895f329f592f2a9b4b0f97`](https://github.com/NVIDIA/nccl/tree/73cf112295c33aee2b895f329f592f2a9b4b0f97) | Mixed: most Apache-2.0, some BSD, plus file-specific upstream licenses | `c1f53beabe4dbf05bd87c00f7ca6084c0cb541c3f7bf8edab7913f266018b7be` | Complete pinned `LICENSE.txt` is in [licenses/NCCL-LICENSE.txt](licenses/NCCL-LICENSE.txt). Keep file-level notices; do not treat the entire project as Apache-2.0 only. |
| SparkNet | [`christopherowen/dgx-spark-networking@3bc8d10104a5012244a0424d7dfc1107a72eafab`](https://github.com/christopherowen/dgx-spark-networking/tree/3bc8d10104a5012244a0424d7dfc1107a72eafab) | Apache-2.0 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | License text in [licenses/Apache-2.0-sparknet.txt](licenses/Apache-2.0-sparknet.txt); preserve notices and mark modified files if source is redistributed. |
| NVIDIA CUDA/vLLM ARM64 base image | `docker.io/vllm/vllm-openai:nightly-af1c01499b289be555c475669ba50a88e96d846e` (source digest recorded in the upstream lock) | Multiple upstream and vendor licenses | Not represented by one license hash | This repo does not publish a base or derived image. Inspect and include the image's applicable notices before redistributing an image or included binaries. |
| SparkRun CLI package, if separately redistributed | [`jibanez-staticduo/sparkrun`](https://github.com/jibanez-staticduo/sparkrun); locally inspected package metadata: `sparkrun 0.3.10` | Package metadata declares `Apache-2.0 AND AGPL-3.0-only`; SparkRoute has an additional permission notice | Apache `e3e14546b427c2ac05b8d3435477c63f148732513f2a52c763f52e3057ac2bcd`; AGPL `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`; exception `5544c1a6f482ba90b37b992f658763384ccacf5ec550541433f3649fde47e176` | SparkRun is invoked as an external tool; this repo does not include its CLI/plugin source. License texts and exception are copied under `licenses/`. If redistributing the package or SparkRoute code, include all three; the exception does not remove AGPL source obligations for SparkRoute. |

The hashes above are SHA-256 of the license files at the pinned raw source URLs,
not hashes of the repositories or source trees. The exact pinned links identify
the applicable license snapshots.

## Model and refusal-direction data

The runtime references the official model snapshot
[`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
Its pinned LICENSE is MIT, Copyright (c) 2023 DeepSeek; the license text is in
[licenses/MIT-DeepSeek.txt](licenses/MIT-DeepSeek.txt), SHA-256
`f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040`. The
repository references the snapshot; it does not distribute model weights.

The upstream source for refusal directions is
[`pqhaz/DeepSeek-V4.1-Flash-1017-heretic@12d58ad0878b5008e80d720251adb969a0069412`](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/tree/12d58ad0878b5008e80d720251adb969a0069412).
That pinned model card declares `license: mit` and identifies
`ablation_directions.pt` as the file containing the directions. Its LICENSE
is MIT with DeepSeek copyright text. The source file SHA-256 is
`87e81bffab36b046786eb7a3888c8d12c69ddef145235e363e0d4fb2c8fe31b4`. The
repository does not redistribute this file or the converted
`refusal_dirs_1017.safetensors` artifact. The local fetcher downloads only the
pinned file and verifies its checksum before extracting the raw tensor storage
without executing pickle instructions. The converted artifact is then checked
against its expected SHA-256 by the local build preparer.

The source card's MIT metadata and LICENSE are recorded as published by the
source repository. This project makes no separate claim about ownership of the
standalone derived direction tensor. Do not add either direction artifact to a
public source release or public image distribution based on this attribution
record alone.

## Distribution boundary

The public source repository contains the fetcher and source-derived
integration, but no model weights, direction binaries, full third-party source
trees, registry credentials, or private handover assets. A user may build an
image locally after fetching the direction file; the build context and image
then contain that direction data. No public image is published by this
repository. **Do not redistribute an image or build context containing
`refusal_dirs_1017.safetensors` without independently establishing the rights
to redistribute the direction data and checking all included third-party
notices.**

The original upstream source also has an experiment-specific
`experiments/2026-10-02-rocenante-mesh4/native/NOTICE` for a SparkRing-derived
marker. That experiment's marker is not part of the public refusal recipe
source selected here. If it is copied into a distribution later, include its
adjacent `LICENSE` and `NOTICE` and preserve the SparkRing attribution.
