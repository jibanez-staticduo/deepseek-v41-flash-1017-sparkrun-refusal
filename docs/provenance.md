# Direction Provenance And Evidence Boundaries

This audit refers to the initial publication, commit [`c35de60f58e849df26e9c8d3acd4be6d1c53ab80`](https://github.com/jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal/tree/c35de60f58e849df26e9c8d3acd4be6d1c53ab80). No direction binary, checkpoint, private log or private infrastructure configuration is distributed here.

## Author Source And Extraction

The directions are **author-published data**, not an SVD reconstruction or a calibration performed by this project. The source is [pqhaz/DeepSeek-V4.1-Flash-1017-heretic, revision 12d58ad0878b5008e80d720251adb969a0069412](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/tree/12d58ad0878b5008e80d720251adb969a0069412), file `ablation_directions.pt`, specifically its `projected` tensor storage.

The pinned [author README](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/blob/12d58ad0878b5008e80d720251adb969a0069412/README.md) names 1017 as its base. It describes one direction per block: mean residual streams for 400 harmful versus 400 harmless instructions at the last prompt token, reasoning closed, with the harmless mean projected out. That is the author's account; we did not reproduce those activation measurements. Exact sample identities, sampling order and a complete independent calibration reproduction are not established by our audit.

The pinned [ablation_spec.txt](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/blob/12d58ad0878b5008e80d720251adb969a0069412/ablation_spec.txt) states `m:2.5:0:0:2:1`: attention WO-B strength 2.5, no shared/routed expert edit, starting at block 2, projected directions, FP8 rounding awareness. The separate [patch/heretic_patch.safetensors](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/blob/12d58ad0878b5008e80d720251adb969a0069412/patch/heretic_patch.safetensors), SHA-256 `16e63cd3e79c3590855aad340b5b9dc37f27562bf2d0a40f85268915da221ce2`, contains the 38 edited weight matrices. `patch/ds_patch.py` applies them to the base checkpoint. These patch matrices are comparison inputs, not the direction fetcher's input.

The extraction implementation is [scripts/fetch_directions.py at the audited commit](https://github.com/jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal/blob/c35de60f58e849df26e9c8d3acd4be6d1c53ab80/scripts/fetch_directions.py), SHA-256 `a559ef3c3a243711b8dda0023e79a5db89724723be2e33e3a437995e0a0b2f1b`.

| Item | Identity |
| --- | --- |
| Official comparison checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Author `.pt` SHA-256 / bytes | `87e81bffab36b046786eb7a3888c8d12c69ddef145235e363e0d4fb2c8fe31b4` / 3,279,227 |
| ZIP metadata / SHA-256 | `directions/data.pkl` / `2bb142115470f795d985a6c1f51a9b7423281a569d5dc9aeb957830ddeca7b99` |
| Raw projected storage / SHA-256 | `directions/data/3` / `98d65378f2167fed790b92ffb8c6b9aa2ac866b128126b0efc7803e10108ea13` |
| Raw layout | little-endian F32 `[40, 5120]`, 819,200 bytes |
| Selected rows / generated keys | zero-based layers 2-39 / `layers.N.attn.wo_b`, 38 F32 `[5120]` vectors |
| Output safetensors SHA-256 / bytes | `d9820fff33a289ed9b0ce8a08669512250bae35d13fba8598acb521a25a2daaa` / 781,776 |

The fetcher verifies the entire input file, reviewed pickle-metadata digest and raw storage digest, then selects rows 2-39. It does not evaluate pickle or call `torch.load`. Model/revision metadata in the output is assigned by our converter; that metadata alone is not independent proof of origin.

## Evidence Associating The Artifact With 1017

Three separate observations support this association:

1. The pinned author's model card names 1017 and describes the layer scope and generation method.
2. The artifact's 40-block, width-5,120 layout agrees with the 1017 backbone. Shape alone would not prove provenance.
3. A local CPU comparison against the pinned official checkpoint and pqhaz patch found 38 edited `layers.2..39.attn.wo_b.weight` matrices, each `[5120,8192]`. Regressing the original published directions against their deltas gave effective strength 2.499928953-2.500011442, mean 2.499986910. A separate reconstruction matched all 48 target shard hashes against the pinned [Infatoshi native-format release](https://huggingface.co/Infatoshi/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored/tree/f07d8148c7432ba5b713af890892fac0d4ae4379), which cites the same pqhaz edit.

These are **1017-associated author directions**, not the 46 width-4,096 directions from the earlier 0731 handover. They were not resized, padded, interpolated or recalibrated from that family. This supports correspondence to the 1017 tensor edit; it does not independently reproduce the author's calibration, prove semantic specificity, or validate refusal/quality scores in this adapter.

The residual of our runtime-projection approximation was 40.77%-48.00% of the edited-delta norm. The FP8 weight edit and BF16/FP32 runtime projection are not bit-identical. Retained private comparison evidence is identified by basename and SHA-256:

| Private Record | SHA-256 |
| --- | --- |
| `RESULTADO.md` | `8f8bf7531f4be4e6dff6f747103398824df5fa953175f497b4517ce27a76a689` |
| `directions-verification.json` | `c7320ce209e8fdf612cb5aab07ff9c645d524d23c813535dd7c3bce400bf4d10` |
| `archive-verification.json` | `9964841fab025f1b2fa8b407b4676eb1d32175e113f00180139d6981f7446d0b` |

Those records are not publicly downloadable and their hashes are not an independent public reproduction. The public author files and converter permit checking extraction separately. Public availability and numerical agreement do not settle separate ownership of derived direction data: see [ATTRIBUTION](../ATTRIBUTION.md). The repository still distributes no vector binary.

## Exact Omitted CUDA Test

The omitted test was [`test_projection.CudaProjectionTests.test_cuda_rows_read_mutated_lambda_values`](https://github.com/jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal/blob/c35de60f58e849df26e9c8d3acd4be6d1c53ab80/tests/test_projection.py). At that commit the test file SHA-256 is `241ac8ecc5f9253571b89d0b8f779de5d156a57ae55b2dac2ab53fb3083d793e`.

Its decorator skips when `torch.cuda.is_available()` is false, with reason `CUDA runtime verification is environment-gated`. Publication ran in an x86 CPU-only environment with PyTorch `2.13.0+cpu`: **48 discovered, 47 passed, 1 skipped**. Retained suite log SHA-256: `f199bf94dfba40d78485aae4f8b100cc3d170b706b8881963e0f8dd06d18191f`.

The test creates synthetic directions, makes one eager CUDA projector call with zero lambda, mutates the same device tensor in place to `[0,1,2.5]`, then makes a second eager call. It checks pointer stability, original bits for the zero row and a changed nonzero row. It contains **no `torch.cuda.CUDAGraph`, capture context or `replay()`**. Thus it does not cover capture -> mutate lambda -> replay. It was skipped, not passed.

Earlier TP3 service runs captured CUDA Graphs and later served different lambdas, but the request probes do not identify whether those specific generations used FULL graph replay. This is separate integration evidence, not an isolated capture/replay regression test. No such isolated test was in the publication suite or established as executed by this audit. The CPU `torch.compile(..., backend="eager")` test is not CUDA Graph coverage either.

No GPU tests, benchmark, service change or lambda change were performed in this documentation follow-up. GPU resources were not reserved for this audit; the isolated replay check remains unverified.

## Target And Drafter Scope

The [published factory](https://github.com/jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal/blob/c35de60f58e849df26e9c8d3acd4be6d1c53ab80/payload/refusal_projection.py) and [integration patch](https://github.com/jibanez-staticduo/deepseek-v41-flash-1017-sparkrun-refusal/blob/c35de60f58e849df26e9c8d3acd4be6d1c53ab80/patches/0001-request-scoped-refusal.patch) project **only target backbone WO-B layers 2-39**. Layers 0-1, other target writers and MTP/DSpark layers 40-42 are outside this projection. The pinned author's weight edit also leaves MTP/DSpark unchanged.

A changed target and unchanged drafter can disagree more often, potentially lowering speculative acceptance and increasing verification work. The direction, size and end-to-end throughput effect have **not been measured** in a controlled lambda-0 versus lambda-2.5 comparison here. This is a possible cost, not a measured regression or a claim that acceptance must fall.

The earlier Oppo 0731 handover describes a different scope: 43 target blocks plus 3 MTP/DSpark stages, 46 width-4,096 directions, and mappings `layers.43 -> mtp.0` through `layers.45 -> mtp.2`, with the same dial for target and drafter. This is a statement about the supplied `deepseek-rank1-refusal/FASE3.md` handover (declared commit `0af2f04da7545f1687fe7aae85250ca89929304b`), not independent verification of Oppo's current service or its per-request graph behavior. The private handover is not redistributed. Our target-only 1017 scope differs intentionally; projecting its drafter needs separate compatible directions, request-row plumbing and validation.
