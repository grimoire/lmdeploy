#!/usr/bin/env python3
"""Inspect the installed LMDeploy serve API surface.

This script intentionally uses the `lmdeploy` executable and the running
server's `/openapi.json` instead of importing the repo checkout. That keeps the
skill aligned with the user's installed version.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


FLAG_PATTERN = re.compile(r'(?<![\w-])--[A-Za-z][A-Za-z0-9-]*[A-Za-z0-9](?![\w-])')
OPTION_START_PATTERN = re.compile(r'^\s+(?:-\w,\s*)?(--[A-Za-z][A-Za-z0-9-]*[A-Za-z0-9](?![\w-]))')


def run_command(cmd: list[str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault('COLUMNS', '240')
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {'cmd': cmd, 'ok': False, 'error': str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {'cmd': cmd, 'ok': False, 'error': f'timed out after {timeout}s', 'output': exc.stdout}

    return {
        'cmd': cmd,
        'ok': completed.returncode == 0,
        'returncode': completed.returncode,
        'output': completed.stdout,
    }


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def extract_flags(help_text: str) -> list[str]:
    return ordered_unique(FLAG_PATTERN.findall(help_text or ''))


def extract_option_lines(help_text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    current: str | None = None
    for line in (help_text or '').splitlines():
        option_start = OPTION_START_PATTERN.search(line)
        if option_start:
            current = option_start.group(1)
            options[current] = line.strip()
            continue
        if current and line.startswith(' ') and line.strip():
            options[current] += ' ' + line.strip()
            continue
        current = None
    return options


def fetch_openapi(base_url: str, timeout: int) -> dict[str, Any]:
    url = base_url.rstrip('/') + '/openapi.json'
    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {'url': url, 'ok': True, 'schema': json.load(response)}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {'url': url, 'ok': False, 'error': str(exc)}


def summarize_openapi(schema: dict[str, Any]) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    for path, methods in sorted(schema.get('paths', {}).items()):
        if not isinstance(methods, dict):
            continue
        for method, spec in sorted(methods.items()):
            if method.lower() not in {'get', 'post', 'put', 'patch', 'delete'}:
                continue
            spec = spec if isinstance(spec, dict) else {}
            endpoints.append({
                'method': method.upper(),
                'path': path,
                'summary': spec.get('summary'),
                'operation_id': spec.get('operationId'),
                'request_body': summarize_request_body(spec.get('requestBody')),
            })
    return endpoints


def summarize_request_body(request_body: Any) -> Any:
    if not isinstance(request_body, dict):
        return None
    content = request_body.get('content') or {}
    result: dict[str, Any] = {}
    for media_type, media_spec in content.items():
        schema = (media_spec or {}).get('schema') if isinstance(media_spec, dict) else None
        result[media_type] = summarize_schema_ref(schema)
    return result or None


def summarize_schema_ref(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return None
    if '$ref' in schema:
        return schema['$ref'].split('/')[-1]
    if 'items' in schema:
        return {'type': schema.get('type'), 'items': summarize_schema_ref(schema.get('items'))}
    return {key: schema.get(key) for key in ('type', 'title') if key in schema}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    lmdeploy_cmd = get_lmdeploy_cmd(args)
    version = run_command(lmdeploy_cmd + ['--version'], args.timeout)
    help_result = run_command(lmdeploy_cmd + ['serve', 'api_server', '-h'], args.timeout)
    help_text = help_result.get('output') or ''
    option_lines = extract_option_lines(help_text)

    report: dict[str, Any] = {
        'lmdeploy_cmd': lmdeploy_cmd,
        'version': version,
        'api_server_help': {
            'ok': help_result.get('ok', False),
            'returncode': help_result.get('returncode'),
            'error': help_result.get('error'),
            'failure_output': help_text if not help_result.get('ok', False) else None,
            'flags': list(option_lines.keys()) or extract_flags(help_text),
            'option_lines': option_lines,
        },
    }
    if args.include_raw_help:
        report['api_server_help']['raw_output'] = help_text

    if args.base_url:
        openapi = fetch_openapi(args.base_url, args.timeout)
        if openapi.get('ok'):
            openapi['endpoints'] = summarize_openapi(openapi['schema'])
            if not args.include_raw_openapi:
                openapi.pop('schema', None)
        report['openapi'] = openapi

    return report


def print_markdown(report: dict[str, Any]) -> None:
    print('# LMDeploy Runtime Inspection')
    print()
    print(f"lmdeploy command: `{' '.join(report['lmdeploy_cmd'])}`")
    print()

    version = report['version']
    print('## Version')
    if version.get('ok'):
        print('```text')
        print((version.get('output') or '').strip())
        print('```')
    else:
        print(f"Could not run version command: {version.get('error') or 'return code ' + str(version.get('returncode'))}")
        if version.get('output'):
            print('```text')
            print((version.get('output') or '').strip())
            print('```')
    print()

    help_info = report['api_server_help']
    print('## api_server Flags')
    if not help_info.get('ok'):
        print(
            'Could not inspect `lmdeploy serve api_server -h`: '
            f"{help_info.get('error') or 'return code ' + str(help_info.get('returncode'))}")
        if help_info.get('failure_output'):
            print('```text')
            print((help_info.get('failure_output') or '').strip())
            print('```')
    else:
        flags = help_info.get('flags') or []
        print(f'Discovered {len(flags)} flags.')
        for flag in flags:
            line = (help_info.get('option_lines') or {}).get(flag)
            if line:
                print(f'- `{flag}`: {line}')
            else:
                print(f'- `{flag}`')
    print()

    openapi = report.get('openapi')
    if openapi is not None:
        print('## OpenAPI')
        if not openapi.get('ok'):
            print(f"Could not fetch `{openapi.get('url')}`: {openapi.get('error')}")
        else:
            print(f"Schema URL: `{openapi.get('url')}`")
            for endpoint in openapi.get('endpoints') or []:
                summary = f" - {endpoint['summary']}" if endpoint.get('summary') else ''
                print(f"- `{endpoint['method']} {endpoint['path']}`{summary}")
                if endpoint.get('request_body'):
                    print(f"  request_body: `{endpoint['request_body']}`")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect installed LMDeploy serve parameters.')
    parser.add_argument('--lmdeploy-cmd',
                        default=None,
                        help='lmdeploy command to inspect, for example "lmdeploy" or "python -m lmdeploy"')
    parser.add_argument('--lmdeploy-bin',
                        default=None,
                        help='deprecated alias for --lmdeploy-cmd when the command is a single executable')
    parser.add_argument('--base-url', default=None, help='running server base URL, for example http://127.0.0.1:23333')
    parser.add_argument('--timeout', type=int, default=20, help='command and HTTP timeout in seconds')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json')
    parser.add_argument('--include-raw-help', action='store_true', help='include raw api_server help output in JSON')
    parser.add_argument('--include-raw-openapi', action='store_true', help='include raw OpenAPI schema in JSON')
    return parser.parse_args(argv)


def get_lmdeploy_cmd(args: argparse.Namespace) -> list[str]:
    if args.lmdeploy_cmd:
        return shlex.split(args.lmdeploy_cmd)
    if args.lmdeploy_bin:
        return [args.lmdeploy_bin]
    return ['lmdeploy']


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.format == 'markdown':
        print_markdown(report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
