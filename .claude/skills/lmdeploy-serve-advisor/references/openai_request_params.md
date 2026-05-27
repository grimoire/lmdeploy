# Request Schema Guidance

This file is not an exhaustive request schema. Exact request and response fields should come from the running server's Swagger UI or `GET /openapi.json`. See `runtime_discovery.md`.

Use this file for stable relationships and common request-side mistakes.

## Discover the Running Schema

Prefer:

```bash
curl -sS http://<server>:<port>/openapi.json
curl -sS http://<server>:<port>/v1/models
```

or use:

```bash
python .claude/skills/lmdeploy-serve-advisor/scripts/inspect_lmdeploy_serve.py \
  --base-url http://<server>:<port> \
  --format markdown
```

Trust the running server over this reference when fields differ.

## Stable Endpoint Concepts

- `/v1/models` tells clients which `model` names are valid.
- `/v1/chat/completions` is the main OpenAI-compatible chat endpoint.
- `/v1/completions` is for plain prompt completion.
- `/generate` is LMDeploy-native generation and can expose fields that differ from OpenAI-compatible endpoints.
- `/v1/encode` is useful for estimating prompt token length.
- `/health` is useful for readiness checks.
- `/openapi.json` is the best machine-readable schema for the running server.

## Stable Request Relationships

Model name:

- Client `model` must match a served model id from `/v1/models`.
- If clients expect a stable alias, launch the server with the installed-version model-name flag.

Output length:

- Cap generated tokens in production to protect latency and memory.
- Prefer the modern output-token field exposed by the running schema. In recent LMDeploy/OpenAI-compatible APIs this is usually `max_completion_tokens`; `max_tokens` may exist for compatibility.

Sampling:

- Temperature, top-p, top-k, min-p, penalties, and seed affect output behavior, not server capacity.
- Invalid sampling ranges are common causes of 400 responses; trust validation errors from the server.

Streaming:

- Streaming improves perceived latency for users but does not remove compute work.
- Compare TTFT and full completion latency separately.

Sessions:

- Avoid reusing the same session id concurrently.
- A "session_id is occupied" error usually means the client reused a session before the previous request finished.

Tools:

- If the request uses tools and the server errors or returns plain-text tool calls, check whether the server was launched with a matching tool parser.
- Parser names are model- and version-sensitive.

Reasoning:

- Parsed reasoning content usually needs a matching reasoning parser.
- Request fields such as reasoning effort or thinking controls are model- and version-sensitive; trust the running schema.

Logprobs:

- Logprob request fields usually require a launch-time logprobs mode.
- If logprobs fail, inspect both request fields and server launch flags.

Structured output:

- JSON or regex schema support can depend on backend, parser, and model behavior.
- Validate with a small request before benchmarking.

Multimodal:

- For normal VLM chat, put images/videos in the message format expected by the running schema.
- Do not combine mutually exclusive raw token/image fields with non-empty chat messages unless the running schema explicitly supports it.
- Media IO and multimodal processor kwargs are advanced knobs; use them only when preprocessing behavior is the issue.

Cache and disaggregation extras:

- Fields such as cache preservation, cache migration, or disaggregated serving metadata are advanced and version-sensitive.
- Use them only when the serving architecture expects them.

## Common Request Problems

- Model not found: call `/v1/models` and update client `model`.
- Tool calling broken: add the correct launch parser and use the schema-supported tool fields.
- Reasoning not separated: add the correct launch parser and confirm request fields.
- Logprobs rejected: add the correct launch logprobs mode and valid request fields.
- VLM input rejected: confirm the exact multimodal message schema from `/openapi.json` or docs for that version.
- High p95 latency: cap output tokens and separate very long requests from normal traffic.
