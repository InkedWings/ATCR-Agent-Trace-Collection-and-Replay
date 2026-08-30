#!/usr/bin/env python3
"""Secret-safe connectivity checks for OpenClaw + GAIA services."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


ALCF_ROOT = "https://inference-api.alcf.anl.gov/resource_server"
CHAT_URL = f"{ALCF_ROOT}/minerva/api/v1/chat/completions"


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def safe_json(response: requests.Response) -> dict[str, Any]:
    if not response.headers.get("content-type", "").startswith("application/json"):
        return {}
    value = response.json()
    return value if isinstance(value, dict) else {"data": value}


def check_brave(api_key: str) -> dict[str, Any]:
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": "OpenClaw", "count": 1},
        timeout=30,
    )
    payload = safe_json(response)
    result_count = len(payload.get("web", {}).get("results", [])) if response.ok else 0
    return {
        "ok": response.ok and result_count > 0,
        "http_status": response.status_code,
        "result_count": result_count,
    }


def check_alcf_catalog(access_token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {access_token}"}
    endpoints = requests.get(f"{ALCF_ROOT}/list-endpoints", headers=headers, timeout=60)
    jobs = requests.get(f"{ALCF_ROOT}/minerva/jobs", headers=headers, timeout=60)
    endpoint_text = endpoints.text.lower()
    jobs_text = jobs.text.lower()
    return {
        "ok": endpoints.ok
        and jobs.ok
        and "minerva" in endpoint_text
        and "nemotron-3-ultra" in endpoint_text
        and "nemotron-3-ultra" in jobs_text,
        "endpoints_http_status": endpoints.status_code,
        "jobs_http_status": jobs.status_code,
        "minerva_present": "minerva" in endpoint_text,
        "nemotron_ultra_present": "nemotron-3-ultra" in endpoint_text,
        "nemotron_ultra_listed": "nemotron-3-ultra" in jobs_text,
    }


def common_request() -> dict[str, Any]:
    return {
        "model": "nemotron-3-ultra",
        "temperature": 0.2,
        "top_p": 0.95,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "force_nonempty_content": True,
        },
    }


def check_plain_inference(access_token: str) -> dict[str, Any]:
    response = requests.post(
        CHAT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            **common_request(),
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with exactly READY."}],
        },
        timeout=300,
    )
    payload = safe_json(response)
    choices = payload.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    nonempty = bool((message.get("content") or "").strip())
    return {
        "ok": response.ok and bool(choices) and nonempty,
        "http_status": response.status_code,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "nonempty": nonempty,
        "usage_present": bool(payload.get("usage")),
    }


def check_tool_inference(access_token: str) -> dict[str, Any]:
    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    response = requests.post(
        CHAT_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            **common_request(),
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the get_weather tool to check the weather in Chicago. "
                        "Do not answer from memory."
                    ),
                }
            ],
            "tools": [tool_schema],
            "tool_choice": "auto",
        },
        timeout=300,
    )
    payload = safe_json(response)
    choices = payload.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    tool_calls = message.get("tool_calls") or []
    names = [
        (item.get("function") or {}).get("name")
        for item in tool_calls
        if isinstance(item, dict)
    ]
    arguments_valid = False
    if tool_calls:
        raw_arguments = (tool_calls[0].get("function") or {}).get("arguments", "")
        try:
            arguments_valid = bool(json.loads(raw_arguments).get("city"))
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments_valid = False
    return {
        "ok": response.ok
        and names == ["get_weather"]
        and arguments_valid
        and choices[0].get("finish_reason") == "tool_calls",
        "http_status": response.status_code,
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "tool_call_count": len(tool_calls),
        "tool_names": names,
        "arguments_valid": arguments_valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Check credentials and catalogs without issuing model requests.",
    )
    args = parser.parse_args()

    try:
        brave_key = require_env("BRAVE_API_KEY")
        alcf_token = require_env("ALCF_AI_TOKEN")
        results: dict[str, Any] = {
            "brave": check_brave(brave_key),
            "alcf_catalog": check_alcf_catalog(alcf_token),
        }
        if not args.skip_inference:
            results["plain_inference"] = check_plain_inference(alcf_token)
            results["tool_inference"] = check_tool_inference(alcf_token)
    except Exception as error:
        print(json.dumps({"ok": False, "error_type": type(error).__name__}, indent=2))
        return 1

    all_ok = all(result.get("ok") for result in results.values())
    print(json.dumps({"ok": all_ok, "checks": results}, indent=2, sort_keys=True))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
