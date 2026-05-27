# Diagnosis and Tuning Playbooks

Use this reference to turn symptoms into a fast fix, then guide the user toward a better serve configuration.

Before recommending exact flags, trust the user's installed LMDeploy version. Ask for `lmdeploy --version` and `lmdeploy serve api_server -h` when version differences may matter. If the user named an environment and LMDeploy is missing or broken there, confirm the environment before falling back to another install.

## Output Template

```text
Likely issue:
<one or two concrete causes>

Fast fix:
<one change the user can try now>

Recommended command/request:
<runnable command or JSON diff>

Why:
<short explanation of the important knobs>

Validate:
<small benchmark or check with metrics>

If still bad:
<next two or three adjustments>
```

## Setup Workflow for Best Serve Parameters

1. Choose backend:
   - Start with `--backend turbomind` for default high-performance LLM serving.
   - Use `--backend pytorch` for PyTorch-only model support or features such as some logprobs, routed experts, LoRA/kernel behavior, or specific new models.
2. Choose GPU parallelism:
   - If the model does not fit, increase `--tp`, use quantization, or reduce memory pressure.
   - If one replica fits and traffic is high, consider `--dp`/multiple servers/proxy for request throughput.
3. Set context and cache:
   - Set `--session-len` to the real maximum context, not the largest theoretical context.
   - Start `--cache-max-entry-count 0.8`.
   - If OOM under concurrency, reduce `--max-batch-size`, `--session-len`, or cache allocation.
4. Set active batch capacity:
   - Start `--max-batch-size` near expected concurrency.
   - For benchmark `request_parallel=N`, use `--max-batch-size >= N` if memory allows.
   - If `--max-concurrent-requests` is set, keep it aligned with the intended concurrency.
5. Set prefill behavior:
   - Start `--max-prefill-token-num 8192`.
   - Raise for long prompts if TTFT is high and memory allows.
   - Enable `--enable-prefix-caching` when requests share a long prefix.
6. Add feature flags only when needed:
   - `--tool-call-parser` for tool calling.
   - `--reasoning-parser` for parsed reasoning content.
   - `--logprobs-mode` for logprobs requests.
   - `--vision-max-batch-size` tuning for VLM.
   - `--disable-vision-encoder` only when serving a VLM checkpoint as text-only language model.
7. Validate with the user's real traffic shape.

## Starting Commands

Throughput-first, single model:

```bash
lmdeploy serve api_server <model> \
  --backend turbomind \
  --tp <gpu_count_needed_to_fit> \
  --session-len <real_context_need> \
  --max-batch-size <expected_concurrency_or_slightly_higher> \
  --cache-max-entry-count 0.8 \
  --max-prefill-token-num 8192
```

Latency-first:

```bash
lmdeploy serve api_server <model> \
  --backend turbomind \
  --tp <gpu_count_needed_to_fit> \
  --session-len <real_context_need> \
  --max-batch-size <moderate_concurrency> \
  --max-concurrent-requests <latency_budget_concurrency> \
  --cache-max-entry-count 0.8
```

Shared-prefix workload:

```bash
lmdeploy serve api_server <model> \
  --backend turbomind \
  --tp <gpu_count_needed_to_fit> \
  --session-len <real_context_need> \
  --max-batch-size <expected_concurrency> \
  --cache-max-entry-count 0.8 \
  --enable-prefix-caching
```

VLM throughput:

```bash
lmdeploy serve api_server <vlm_model> \
  --backend turbomind \
  --tp <gpu_count_needed_to_fit> \
  --session-len <real_context_need> \
  --max-batch-size <text_concurrency> \
  --vision-max-batch-size <image_batch_capacity> \
  --cache-max-entry-count 0.8
```

Tool or reasoning model:

```bash
lmdeploy serve api_server <model> \
  --backend turbomind \
  --tool-call-parser <parser_name_if_tools_are_used> \
  --reasoning-parser <parser_name_if_reasoning_is_needed>
```

## Benchmark Sweep

Recommend small sweeps, not huge grids:

```text
Hold fixed:
- model, backend, tp/dp
- average input/output length
- session_len

Sweep:
- client concurrency: 0.5x, 1x, 1.25x expected load
- max_batch_size: expected concurrency, 1.25x expected concurrency
- cache_max_entry_count: 0.7, 0.8, 0.9 only if memory/capacity is unclear
- max_prefill_token_num: 4096, 8192, 16384 for long prompts

Pick:
- no OOM
- stable p95 latency
- best tokens/s or requests/s for the user's target
```

Track these metrics:

- request/s
- output tokens/s and total tokens/s
- TTFT
- p50/p95/p99 latency
- GPU utilization and memory
- queueing/rejection/errors
- OOM rate

## DP/EP MoE Serving Pattern

When the user asks how to serve an MoE model with DP/EP, first inspect their installed flags, backend support, and existing deployment scripts. Do not assume every backend has the same DP/EP semantics.

A common PyTorch Ray pattern is:

```bash
export LMDEPLOY_DP_MASTER_ADDR=<reachable_host_or_local_test_addr>
export LMDEPLOY_DP_MASTER_PORT=<free_port>

lmdeploy serve api_server <moe_model> \
  --backend pytorch \
  --dp <dp> \
  --ep <ep> \
  --tp <tp> \
  --distributed-executor-backend ray \
  --max-batch-size <max_batch_size> \
  --max-prefill-token-num <prefill_tokens> \
  --cache-max-entry-count <cache_ratio> \
  --proxy-url http://<proxy_host>:<proxy_port>
```

Advice:

- Do not recommend `--distributed-executor-backend mp` for DP; use Ray when supported.
- In common PyTorch layouts, `dp=N, ep=N, tp=1` needs about `N` devices, not `N*N`; verify with the installed version.
- Do not include Ray lifecycle commands like `ray stop` or `ray start` as required serve steps. Only mention Ray cluster setup when the user has no Ray cluster or asks how to create one.
- `LMDEPLOY_DP_MASTER_ADDR` and `LMDEPLOY_DP_MASTER_PORT` are required for Ray DP setups.
- Add `DEEPEP_*` env vars only when the installed DeepEP stack and model kernel path need them.
- Start a proxy only when the user wants one stable endpoint across multiple API server processes.
- If users care about TTFT, do not blindly raise `--max-batch-size`; higher batch capacity can improve throughput but worsen TTFT/p95 latency.
- Keep proxy bind address and proxy URL straight: `0.0.0.0` is a bind address; a real reachable host/IP is safer for remote clients.

## Symptom Rules

### Concurrency exceeds active batch capacity

Signal:

- User benchmark `request_parallel` or client concurrency is greater than `--max-batch-size`.
- Example: `request_parallel=150`, `--max-batch-size 128`.

Diagnosis:

- The engine can actively batch only `--max-batch-size` sequences; extra requests wait.

Fast fix:

- If memory allows, set `--max-batch-size` at or above target concurrency, such as `160` for `150`.
- Otherwise lower client concurrency to the current batch capacity.

Tradeoff:

- Larger batch size consumes more KV cache and can hurt p95 latency.

### Server concurrency gate is too low

Signal:

- `--max-concurrent-requests` is set lower than desired client concurrency.

Diagnosis:

- Requests queue at the API server before reaching the engine.

Fast fix:

- Raise `--max-concurrent-requests` for throughput tests, or intentionally keep it low for latency protection.

### OOM at startup

Likely causes:

- Model weights do not fit.
- Too few GPUs or wrong `--tp`.
- Quantized model format not selected/detected.

Fast fixes:

- Increase `--tp`.
- Use a quantized checkpoint and set `--model-format` if needed.
- Use a smaller model.

### OOM under load

Likely causes:

- KV cache pressure from high `--max-batch-size`, high `--session-len`, long prompts/outputs, or high cache allocation.

Fast fixes:

- Lower `--max-batch-size`.
- Lower `--session-len` to real need.
- Lower request output cap.
- Consider KV quantization via `--quant-policy`.

### Low throughput with low GPU utilization

Likely causes:

- Client concurrency too low.
- `--max-batch-size` too low.
- Requests are short and overhead-bound.
- Server concurrency gate too low.

Fast fixes:

- Increase client concurrency and `--max-batch-size` together.
- Remove or raise `--max-concurrent-requests`.
- Enable prefix caching if prompts share prefixes.

### High TTFT

Likely causes:

- Long prompts, VLM preprocessing, insufficient prefill capacity, queueing, or no prefix reuse.

Fast fixes:

- Reduce prompt length if possible.
- Raise `--max-prefill-token-num` for long prompts if memory allows.
- Enable `--enable-prefix-caching` for shared prefixes.
- Increase `--vision-max-batch-size` for VLM if image encoder is the bottleneck.
- Lower concurrency if queueing dominates.

### High p95/p99 latency

Likely causes:

- Overloaded concurrency, too-large active batch, long-tail prompts/outputs, or unbounded output tokens.

Fast fixes:

- Cap `max_completion_tokens` in the request.
- Lower client concurrency or `--max-concurrent-requests`.
- Lower `--max-batch-size` for latency-first serving.
- Separate very long requests from normal traffic.

### Long context fails or capacity is poor

Likely causes:

- `--session-len` too small for the request or too large for available KV capacity.

Fast fixes:

- Set `--session-len` to the actual needed maximum.
- Reduce `--max-batch-size` for long-context workloads.
- Consider KV quantization or more GPUs.
- Use RoPE scaling only when model support/docs require it.

### Tool calling does not work

Signal:

- Request uses `tools`, but server returns an error asking for `--tool-call-parser`, or tool calls appear as plain text.

Fast fix:

- Relaunch with a model-appropriate `--tool-call-parser`.
- Set request `tool_choice` deliberately: `none`, `auto`, `required`, or a specific function.

### Reasoning content missing or mixed into text

Fast fix:

- Relaunch with a model-appropriate `--reasoning-parser`.
- Check request `reasoning_effort` only if the model/parser supports it.

### Logprobs request fails

Fast fix:

- Relaunch PyTorch engine with `--logprobs-mode raw_logits` or `--logprobs-mode raw_logprobs`.
- Keep `top_logprobs >= 0` and set `logprobs=true` when using `top_logprobs`.

### `session_id is occupied`

Cause:

- The client reused the same `session_id` while a previous request with that session is still active.

Fast fix:

- Use default `session_id=-1`, unique session ids, or wait for the previous request to finish.

### VLM is slow

Likely causes:

- Vision encoder bottleneck, image resolution/count, low `--vision-max-batch-size`, long text context, or media IO.

Fast fixes:

- Increase `--vision-max-batch-size` if GPU memory allows.
- Reduce image/video size or frame count.
- Tune text-side `--max-batch-size` separately from vision batch size.
- Inspect `media_io_kwargs`/`mm_processor_kwargs` if preprocessing is the issue.

### VLM checkpoint used for text-only serving

Fast fix:

- Relaunch with `--disable-vision-encoder` if the user does not need image/video inputs.
- Do not suggest this flag for normal multimodal serving because it disables the vision part.

### OpenAI client cannot find the model

Fast fix:

- Call `/v1/models` and use the returned id.
- Relaunch with `--model-name <stable_name>` if clients expect a specific model string.

### Auth errors

Fast fix:

- If server launched with `--api-keys`, client must send a matching bearer token.
- If no auth is desired, remove `--api-keys`.
