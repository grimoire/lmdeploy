# Runtime Discovery

Use this reference before giving exact LMDeploy serve flags or request schemas. The user's installed LMDeploy version is the authority.

If the user provided a specific environment, inspect that environment first. If `lmdeploy` is not found there, or importing/running it fails because the environment is incomplete, ask the user to confirm the intended environment before trying another environment.

## Fast Path

If this skill directory is available locally, run:

```bash
python .claude/skills/lmdeploy-serve-advisor/scripts/inspect_lmdeploy_serve.py --format markdown
```

If `lmdeploy` is not on `PATH` but the repo checkout is importable, use:

```bash
python .claude/skills/lmdeploy-serve-advisor/scripts/inspect_lmdeploy_serve.py \
  --lmdeploy-cmd "python -m lmdeploy" \
  --format markdown
```

If LMDeploy lives in a specific conda environment, run the inspector with that environment's Python and pass that environment's `lmdeploy` executable:

```bash
/path/to/conda/env/bin/python .claude/skills/lmdeploy-serve-advisor/scripts/inspect_lmdeploy_serve.py \
  --lmdeploy-cmd /path/to/conda/env/bin/lmdeploy \
  --format markdown
```

If a server is already running, include its base URL:

```bash
python .claude/skills/lmdeploy-serve-advisor/scripts/inspect_lmdeploy_serve.py \
  --lmdeploy-cmd "python -m lmdeploy" \
  --base-url http://127.0.0.1:23333 \
  --format markdown
```

Use the output to decide which flags and request fields are actually supported.

## Manual Fallback

Ask the user to provide, or run locally:

```bash
lmdeploy --version
lmdeploy serve api_server -h
curl -sS http://<server>:<port>/openapi.json
curl -sS http://<server>:<port>/v1/models
```

For Python environment ambiguity, inspect the installed package path:

```bash
python -c "import lmdeploy, inspect; print(lmdeploy.__version__); print(inspect.getfile(lmdeploy))"
```

## How to Use Discovery Results

- Use discovered flags and defaults over bundled reference notes.
- If a flag is missing from help output, do not recommend it.
- If `/openapi.json` differs from this skill's request notes, trust `/openapi.json`.
- When the user is working in this repo checkout, repo source files can be treated as authoritative for that checkout.
- When the user runs a released package, do not assume this repo's current branch matches their installation.

## What to Report

When giving a final diagnosis, include the source used for exact parameters:

```text
I used your `lmdeploy serve api_server -h` output from LMDeploy <version> for the flags below.
```

or:

```text
I could not inspect your installed version, so this is a version-agnostic starting point. Please confirm the flags with `lmdeploy serve api_server -h`.
```
