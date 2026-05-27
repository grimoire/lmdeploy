---
name: lmdeploy-serve-advisor
description: Diagnose and tune LMDeploy serve/api_server usage for non-expert users, including launch parameters, OpenAI-compatible request parameters, throughput, latency, OOM, long context, VLM, tools, reasoning, prefix caching, and request_parallel versus max_batch_size issues.
disable-model-invocation: true
---

# LMDeploy Serve Advisor

Help users turn a serving problem into a practical `lmdeploy serve api_server ...` command, request-body fix, and validation plan.

Use this skill when a user asks about:

- choosing the best serve parameters
- bad throughput, bad latency, high TTFT, OOM, queueing, or unstable p95
- benchmark concurrency, `request_parallel`, `--max-batch-size`, or `--max-concurrent-requests`
- `api_server` launch flags, OpenAI-compatible request parameters, VLM serving, tool calling, reasoning output, logprobs, prefix caching, LoRA, quantized models, long context, or distributed serving

## Version Authority

First trust the LMDeploy code the user actually installed or is running. Flags, defaults, parser names, and supported features can differ by version.

When exact behavior matters, ask for or inspect:

- `lmdeploy --version`
- `lmdeploy serve api_server -h`
- the user's actual launch command and server logs
- the installed package source, if available, via Python/module path inspection

If the user provides a specific Python/conda/container environment and `lmdeploy` cannot be found or imported there, stop and confirm the intended environment before using another environment or this repo checkout.

Use the current repo and bundled references as fallback guidance only when they match the user's installed version or when the user is working directly in this checkout.

## Repo Source of Truth

When the user is working from this repo, prefer code over memory:

- CLI launch flags: `lmdeploy/cli/serve.py`, `lmdeploy/cli/utils.py`
- OpenAI-compatible request models: `lmdeploy/serve/openai/protocol.py`
- Request validation and hidden JSON extras: `lmdeploy/serve/openai/api_server.py`, `lmdeploy/serve/openai/serving_*.py`
- User docs: `docs/en/llm/api_server.md`, `docs/en/multi_modal/api_server_vl.md`, `docs/en/llm/api_server_tools.md`, `docs/en/llm/api_server_reasoning.md`, `docs/en/llm/api_server_lora.md`

Load references only as needed:

- `references/runtime_discovery.md` for discovering the user's installed CLI flags and running server schema.
- `references/api_server_launch_args.md` for launch parameters and serve command setup.
- `references/openai_request_params.md` for request-body parameters and endpoint issues.
- `references/diagnosis_and_tuning.md` for symptom-to-fix rules and tuning playbooks.

## Operating Workflow

1. Classify the user's target:
   - throughput-first
   - latency-first
   - long-context
   - memory-constrained
   - VLM
   - tool/reasoning correctness
   - general production serving
2. Extract or ask for only the missing critical context:
   - LMDeploy version or `lmdeploy serve api_server -h` output when version may differ
   - model path/name and model size
   - backend: `turbomind` or `pytorch`
   - GPU type/count/memory
   - current `lmdeploy serve api_server` command
   - expected concurrency, QPS, or benchmark `request_parallel`
   - average/max input tokens and output tokens
   - streaming or non-streaming
   - whether requests share a long common prefix
   - relevant request JSON and error logs
3. Discover runtime parameters before exact recommendations:
   - Prefer the user's `lmdeploy serve api_server -h` and running server `/openapi.json`.
   - Use `scripts/inspect_lmdeploy_serve.py` when available to collect version, flags, and optional OpenAPI schema.
   - Treat bundled parameter notes as fallback semantics, not an exhaustive or authoritative arg list.
4. Diagnose obvious mismatches before deep tuning:
   - `request_parallel` or client concurrency greater than `--max-batch-size`
   - `--max-concurrent-requests` lower than intended concurrency
   - oversized `--session-len` causing KV cache pressure
   - long prompts with too-small prefill capacity
   - logprobs/tools/reasoning/routed experts requested without matching launch flags
   - VLM image workload limited by vision batch size or image preprocessing
5. Give a fast fix first, then a better setup path.
6. Recommend a small benchmark sweep instead of claiming one universal best config.

## Answer Format

For diagnosis, answer with:

1. Likely issue
2. Fast fix
3. Recommended serve command or request JSON change
4. Why these parameters
5. Validation plan
6. What to change if OOM, low throughput, or poor latency remains

For best-config setup, produce one starting command plus a short sweep. Keep explanations concrete and friendly; assume the user is not an inference-engine expert.

## Quick Example

User: "I benchmark request parallel 150, but my server uses `--max-batch-size 128`."

Response pattern:

```text
Likely issue:
Your client can send 150 concurrent requests, but the engine can actively batch at most 128 sequences. The extra requests wait, so throughput and p95 latency can look worse than expected.

Fast fix:
If GPU memory allows, set `--max-batch-size 160`; otherwise lower benchmark concurrency to 128.

Try:
lmdeploy serve api_server <model> --max-batch-size 160 --cache-max-entry-count 0.8

Validate:
Run the same prompt/output workload and compare tokens/s, request/s, TTFT, p50/p95 latency, and OOM rate.
```

## Guardrails

- Do not promise a globally "best" config. Say "best for this model, GPU, and traffic shape."
- Do not recommend flags that are absent from the user's installed `lmdeploy serve api_server -h`.
- Do not silently switch environments when the user provided one. Confirm first if LMDeploy is missing or broken there.
- Do not recommend deprecated `--distributed-executor-backend mp`; prefer non-deprecated choices supported by the user's installed version.
- Do not tune only one flag when the symptom involves capacity; consider batch size, session length, cache memory, prompt/output length, and concurrency together.
- Prefer `max_completion_tokens` in OpenAI-compatible requests; mention `max_tokens` only for compatibility or legacy examples.
- If exact supported parsers, model formats, or backend defaults matter, inspect the repo or ask the user to run `lmdeploy serve api_server -h`.
- Keep the final recommendation runnable.
