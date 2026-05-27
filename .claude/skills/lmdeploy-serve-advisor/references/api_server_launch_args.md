# api_server Launch Guidance

This file is not an exhaustive argument list. Exact flags, defaults, choices, and deprecations must come from the user's installed LMDeploy, usually `lmdeploy serve api_server -h`. See `runtime_discovery.md`.

Use this file for stable tuning concepts and parameter relationships.

## Command Shape

```bash
lmdeploy serve api_server <model_path> [server args] [engine args] [feature args]
```

Build commands from the user's discovered flags. Avoid adding unsupported flags just because they appear in this reference.

## Tuning Knob Families

Server exposure and client access:

- Bind/listen address and port control where the service runs.
- Model name controls what clients pass as `model`; confirm with `/v1/models`.
- Auth, SSL, CORS, docs exposure, and log length are production-safety knobs, not throughput knobs.
- A server-side concurrent request cap can protect latency, but it can also hide engine capacity in benchmarks.

Engine selection:

- `turbomind` is usually the default high-performance LLM path.
- `pytorch` is often needed for newer model support or PyTorch-specific features.
- Verify backend availability and feature support in the installed version.

Parallelism:

- Tensor parallelism helps fit or accelerate a single model replica across GPUs.
- Data parallelism or multiple API servers help request throughput when each replica can fit.
- Expert/disaggregated/distributed executor options are version-sensitive. Discover them from installed help before recommending them.
- Do not recommend deprecated `--distributed-executor-backend mp`.

Context and KV cache:

- The maximum session/context length should match the real workload, not the largest theoretical model context.
- Larger context, larger active batch, and longer outputs all consume KV cache.
- Cache memory allocation affects how many concurrent long sessions fit.
- KV quantization can reduce cache memory, but should be validated for quality and performance.

Batching and concurrency:

- Active batch capacity should be aligned with expected client concurrency for throughput tests.
- If benchmark `request_parallel` is greater than the engine's active batch size, extra requests wait.
- If server concurrent-request cap is below benchmark concurrency, requests queue before the engine.
- Larger batches can improve throughput but may hurt p95 latency and memory headroom.

Prefill:

- Long prompts increase TTFT.
- Prefill token limits and prefill iteration knobs are version/backend-sensitive; discover exact names from help output.
- Prefix caching helps when many requests share a long system prompt, retrieval prefix, or conversation prefix.

VLM:

- Text-side batch capacity and vision encoder batch capacity are separate bottlenecks.
- Increase the vision batch knob only if image/video preprocessing or vision encoder work is the bottleneck and memory allows.
- If a VLM checkpoint is used for text-only serving, use the installed-version flag that disables the vision encoder, commonly `--disable-vision-encoder`. Never use it for image/video workloads.

DP/EP MoE serving:

- DP, TP, and EP support is backend- and version-sensitive. Discover the installed flags and source behavior before saying a backend does or does not support a layout.
- For PyTorch MoE serving with `dp > 1`, prefer the Ray executor when supported by the installed version. Do not recommend deprecated `mp`.
- In common PyTorch MoE layouts, `dp=N, ep=N, tp=1` uses roughly `max(dp, tp, ep)` devices, not `dp * ep` devices. Confirm this against the installed runtime.
- Ray DP setups require `LMDEPLOY_DP_MASTER_ADDR` and `LMDEPLOY_DP_MASTER_PORT` so workers across DP ranks can initialize distributed process groups. Set them before launching the API server.
- Do not tell users to stop or restart Ray unless they asked to recreate the Ray cluster or their current Ray state is the diagnosed problem. Existing cluster/job managers may already own Ray lifecycle.
- Use a proxy when multiple API server processes should be exposed as one stable endpoint. If users can target per-rank endpoints directly, a proxy is optional.
- In same-host/same-container tests, scripts may use `0.0.0.0` for proxy binding and local registration. For remote or multi-node clients, use an address that other processes can actually reach.
- For DeepEP-based MoE kernels, deployment scripts may set `DEEPEP_MAX_BATCH_SIZE`, `DEEPEP_MAX_TOKENS_PER_RANK`, `DEEPEP_MODE`, or related env vars. Treat these as optional implementation/runtime knobs tied to the installed DeepEP stack, not generic LMDeploy serve requirements.

Feature gates:

- Tool calling usually needs a model-appropriate tool parser launch flag.
- Parsed reasoning content usually needs a model-appropriate reasoning parser launch flag.
- Logprobs usually need a launch-time logprobs mode.
- Returning routed experts usually needs a launch-time enable flag.
- Parser names and choices are version-sensitive; discover before recommending exact values.

## Practical Setup Rules

- First pick the smallest set of flags that meets the user's goal.
- Start with the backend and parallelism needed to load the model.
- Set context length to the real max request need.
- Set active batch size near expected concurrency when optimizing throughput.
- Set a server concurrent-request cap only when protecting latency or preventing overload.
- Enable prefix caching only when the workload has meaningful prefix reuse.
- Add feature flags only when the request body actually uses those features.
- Validate with the user's real prompt length, output length, concurrency, and latency target.
