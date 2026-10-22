# Architecture

## Base inference path

The runtime serves the official `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint with the pinned r6x vLLM/TileLang integration. The tested deployment profile uses tensor parallelism across three GB10 systems and configures a maximum context length of 1,048,576 tokens.

The checkpoint's mixed precision is retained: MXFP4 expert weights, FP8 dense weights, and BF16 activations. The runtime configuration uses SSD-backed Engram tables, five DSpark speculative tokens, and TileLang kernels for GB10 / SM121. These settings describe the inference stack, not a conversion of the model or a modification of its learned weights.

The upstream TP3 profile reports KV capacity equivalent to about 4.55 full 1M-token windows after its memory sizing. This is a KV capacity figure, not eight full-context requests served simultaneously. Request concurrency and maximum usable context also depend on prompt lengths, HBM, SSD throughput, graph capture sizes, and runtime buffers.

## Direction artifact

The optional artifact contains exactly 38 original directions from the 1017 comparison, one for each `WO-B` output in backbone layers 2 through 39. Each direction is an F32 vector of width 5,120. Runtime validation checks the model id and revision, layer keys, tensor shape and type, finite values, unit norm, file layout, and the pinned SHA-256 before creating the projection. The pinned source repository declares MIT and includes an MIT license. This repository does not redistribute the artifact; review upstream terms before redistributing it or an image containing it.

The projection is not applied to MTP or DSpark layers. It is applied to the selected backbone attention output before the following mHC post/pre path. The checkpoint files are not rewritten; the direction file is a separately fetched input.

## Request projection

For each target row, the runtime computes:

```text
y = x - lambda * dot(x, r) * r
```

The operation accumulates in FP32 and returns BF16. A zero lambda selects the original BF16 bits directly, including signed zeros and NaNs. Nonzero values are constrained to the supported range `[0, 4]`.

The implementation carries lambda as request state through admission, batching, CED row selection, SP row selection, and TP execution. CUDA row buffers are populated from final GPU query boundaries and padding masks. CPU query bounds are not used as an upper bound for GPU rows. The implementation handles zero/padded rows without claiming that every graph, scheduler, or parallelism mode has been qualified.

At admission, the effective lambda is frozen into an internal cache-salt envelope with the exact float hexadecimal value, direction artifact SHA, and any tenant salt. Active requests keep their admitted value when the global default changes. Prefix-cache namespaces therefore distinguish different effective values and direction artifacts.

## Strength interpretation

`lambda=0` means no projection and is bit-exact at the projection site. `lambda=1` applies one simple projection. `lambda=2.5` matches the intensity measured in the source comparison (CPU range 2.499929-2.500011), subject to runtime rounding. It is an author-intensity reference, not an optimal or calibrated setting. The generated deployment recipe explicitly sets the process default to 2.5; a bare runtime defaults to 1.0 when `VLLM_REFUSAL_LAMBDA_INIT` is unset.

The original directions are used. The reported average SVD energy retention of 81.15% describes how much vector energy a low-rank approximation retained in a separate analysis; it does not measure refusal frequency, answer quality, or safety. The residual from approximating the source delta with the runtime projection is 40.77%-48.00% of the source delta norm after FP8/BF16 rounding. No quality effect can be inferred from that numerical comparison alone.

## Reproducibility anchors

The implementation pins the upstream source revision in `runtime.json`, per-file patch before/after hashes in `patches/manifest.json`, and the direction artifact hash and model revision in the projection code. The original direction artifact comes from [pqhaz/DeepSeek-V4.1-Flash-1017-heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic/tree/12d58ad0878b5008e80d720251adb969a0069412). Build outputs should additionally record and deploy an immutable base-image digest and final image digest. Review the generated build receipt before using a new image; a tag alone is not an immutable identity.

## References

- [DeepSeek V4.1 Flash announcement and model report](https://api-docs.deepseek.com/news/news260910/)
- [spark-ds41f r6x source revision](https://github.com/christopherowen/spark-ds41f/tree/c38a24f2c9e714d2648769b12cd71aa6f0834d00)
