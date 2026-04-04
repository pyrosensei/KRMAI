#!/usr/bin/env python3
"""KRMAI Project Audit Agent.

Runs a full project-level audit in one command:
- Directory intelligence scan (project shape + entrypoints)
- Backend model/system test suite (test_system.py)
- API latency benchmark (/chat)
- Frontend build performance snapshot (web-app build artifacts)

Outputs:
- evaluation/agent_audit_report.json
- evaluation/agent_audit_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation"

SCAN_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "chroma_db",
}

DEFAULT_API_URL = "http://localhost:8000"


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    cwd: str
    exit_code: int
    duration_sec: float
    stdout: str
    stderr: str


@dataclass
class APICallResult:
    ok: bool
    status_code: int
    latency_sec: float
    answer_chars: int
    sources_count: int
    error: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(command: list[str], cwd: Path, timeout_sec: int) -> CommandResult:
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    duration = time.time() - start
    return CommandResult(
        ok=proc.returncode == 0,
        command=command,
        cwd=str(cwd),
        exit_code=proc.returncode,
        duration_sec=duration,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def scan_project_tree(root: Path) -> dict[str, Any]:
    file_count = 0
    dir_count = 0
    ext_counts: dict[str, int] = {}
    largest_files: list[tuple[int, str]] = []

    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SCAN_EXCLUDE_DIRS]
        dir_count += len(dirnames)

        for filename in filenames:
            file_count += 1
            full_path = Path(current_root) / filename
            ext = full_path.suffix.lower() or "<no_ext>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

            try:
                size = full_path.stat().st_size
            except OSError:
                size = 0
            rel = str(full_path.relative_to(root))
            largest_files.append((size, rel))

    largest_files.sort(reverse=True)

    key_files = [
        "api.py",
        "rag_engine.py",
        "ingest.py",
        "test_system.py",
        "web-app/package.json",
        "web-app/src/App.jsx",
    ]

    key_file_status = {
        path: (root / path).exists() for path in key_files
    }

    top_ext = sorted(ext_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]

    return {
        "root": str(root),
        "files_scanned": file_count,
        "dirs_scanned": dir_count,
        "top_extensions": [{"extension": ext, "count": count} for ext, count in top_ext],
        "largest_files": [
            {"path": path, "size_bytes": size} for size, path in largest_files[:10]
        ],
        "key_file_status": key_file_status,
    }


def parse_test_summary(output: str) -> dict[str, Any]:
    total_match = re.search(r"TOTAL:\s+(\d+) passed,\s+(\d+) failed,\s+(\d+) total", output)
    pass_rate_match = re.search(r"PASS RATE:\s+([0-9.]+)%", output)

    summary: dict[str, Any] = {
        "parsed": False,
        "passed": None,
        "failed": None,
        "total": None,
        "pass_rate_percent": None,
    }

    if total_match:
        summary["passed"] = int(total_match.group(1))
        summary["failed"] = int(total_match.group(2))
        summary["total"] = int(total_match.group(3))
        summary["parsed"] = True

    if pass_rate_match:
        summary["pass_rate_percent"] = float(pass_rate_match.group(1))

    return summary


def run_backend_tests(python_exec: str, timeout_sec: int) -> dict[str, Any]:
    result = run_cmd([python_exec, "test_system.py"], cwd=ROOT, timeout_sec=timeout_sec)
    parsed = parse_test_summary(result.stdout)
    return {
        "ok": result.ok,
        "duration_sec": round(result.duration_sec, 2),
        "exit_code": result.exit_code,
        "summary": parsed,
        "stdout_tail": result.stdout[-2500:],
        "stderr_tail": result.stderr[-1000:],
    }


def check_api_health(api_url: str) -> dict[str, Any]:
    endpoint = f"{api_url.rstrip('/')}/health"
    try:
        start = time.time()
        resp = requests.get(endpoint, timeout=10)
        latency = time.time() - start
        payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
            "latency_sec": round(latency, 3),
            "payload": payload,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "status_code": 0,
            "latency_sec": None,
            "payload": {},
            "error": str(exc),
        }


def single_chat_call(api_url: str, question: str, timeout_sec: int) -> APICallResult:
    endpoint = f"{api_url.rstrip('/')}/chat"
    start = time.time()
    try:
        resp = requests.post(endpoint, json={"message": question}, timeout=timeout_sec)
        latency = time.time() - start
        if resp.status_code != 200:
            return APICallResult(
                ok=False,
                status_code=resp.status_code,
                latency_sec=latency,
                answer_chars=0,
                sources_count=0,
                error=f"HTTP {resp.status_code}",
            )

        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        return APICallResult(
            ok=True,
            status_code=resp.status_code,
            latency_sec=latency,
            answer_chars=len(answer),
            sources_count=len(sources),
        )
    except Exception as exc:
        latency = time.time() - start
        return APICallResult(
            ok=False,
            status_code=0,
            latency_sec=latency,
            answer_chars=0,
            sources_count=0,
            error=str(exc),
        )


def run_api_benchmark(api_url: str, rounds: int, concurrency: int, timeout_sec: int) -> dict[str, Any]:
    prompts = [
        "What are the bus routes available at KRMU?",
        "What is the highest package and average placement package at KRMU?",
        "How can I apply for scholarships at KR Mangalam University?",
        "What is the fee structure for BTech CSE?",
        "What are hostel facilities at KRMU?",
    ]

    jobs: list[str] = []
    for i in range(rounds):
        jobs.append(prompts[i % len(prompts)])

    calls: list[APICallResult] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(single_chat_call, api_url, q, timeout_sec) for q in jobs]
        for future in as_completed(futures):
            calls.append(future.result())

    successes = [c for c in calls if c.ok]
    latencies = [c.latency_sec for c in successes]
    source_counts = [c.sources_count for c in successes]
    answer_chars = [c.answer_chars for c in successes]

    return {
        "requests": len(calls),
        "success": len(successes),
        "failed": len(calls) - len(successes),
        "success_rate_percent": round((len(successes) / len(calls) * 100) if calls else 0, 2),
        "avg_latency_sec": round(statistics.mean(latencies), 3) if latencies else None,
        "p95_latency_sec": round(percentile(latencies, 95), 3) if latencies else None,
        "max_latency_sec": round(max(latencies), 3) if latencies else None,
        "avg_sources_count": round(statistics.mean(source_counts), 2) if source_counts else None,
        "avg_answer_chars": round(statistics.mean(answer_chars), 1) if answer_chars else None,
        "errors": [c.error for c in calls if not c.ok][:8],
    }


def get_dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def run_frontend_performance(web_dir: Path, install_timeout_sec: int, build_timeout_sec: int) -> dict[str, Any]:
    if not web_dir.exists():
        return {
            "ok": False,
            "skipped": True,
            "reason": "web-app directory missing",
        }

    install_result = None
    if not (web_dir / "node_modules").exists():
        install_result = run_cmd(["npm", "install"], cwd=web_dir, timeout_sec=install_timeout_sec)
        if not install_result.ok:
            return {
                "ok": False,
                "skipped": False,
                "install_ok": False,
                "install_duration_sec": round(install_result.duration_sec, 2),
                "install_stderr_tail": install_result.stderr[-1200:],
                "reason": "npm install failed",
            }

    build_result = run_cmd(["npm", "run", "build"], cwd=web_dir, timeout_sec=build_timeout_sec)

    dist_dir = web_dir / "dist"
    dist_size = get_dir_size_bytes(dist_dir) if dist_dir.exists() else 0

    assets: list[dict[str, Any]] = []
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        for file in assets_dir.rglob("*"):
            if file.is_file():
                assets.append({
                    "path": str(file.relative_to(web_dir)),
                    "size_bytes": file.stat().st_size,
                })
    assets.sort(key=lambda item: item["size_bytes"], reverse=True)

    return {
        "ok": build_result.ok,
        "skipped": False,
        "install_ok": True if install_result is None else install_result.ok,
        "install_duration_sec": None if install_result is None else round(install_result.duration_sec, 2),
        "build_duration_sec": round(build_result.duration_sec, 2),
        "dist_size_bytes": dist_size,
        "largest_assets": assets[:8],
        "build_stdout_tail": build_result.stdout[-2500:],
        "build_stderr_tail": build_result.stderr[-1200:],
    }


def write_reports(report: dict[str, Any]) -> dict[str, str]:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / "agent_audit_report.json"
    md_path = EVAL_DIR / "agent_audit_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    backend = report.get("backend_tests", {})
    api = report.get("api_benchmark", {})
    fe = report.get("frontend_performance", {})

    lines = [
        "# KRMAI Agent Audit Report",
        "",
        f"Generated: {report.get('generated_at_utc')}",
        "",
        "## Project Scan",
        f"- Files scanned: {report['project_scan']['files_scanned']}",
        f"- Directories scanned: {report['project_scan']['dirs_scanned']}",
        "",
        "## Backend Model/System Tests",
        f"- Ran: {backend.get('ran')}",
        f"- Success: {backend.get('ok')}",
        f"- Duration (s): {backend.get('duration_sec')}",
    ]

    summary = backend.get("summary") or {}
    if summary.get("parsed"):
        lines.extend([
            f"- Passed: {summary.get('passed')}",
            f"- Failed: {summary.get('failed')}",
            f"- Pass rate (%): {summary.get('pass_rate_percent')}",
        ])

    lines.extend([
        "",
        "## API Performance",
        f"- Health reachable: {report['api_health'].get('reachable')}",
        f"- Benchmark ran: {api.get('ran')}",
        f"- Success rate (%): {api.get('success_rate_percent')}",
        f"- Avg latency (s): {api.get('avg_latency_sec')}",
        f"- P95 latency (s): {api.get('p95_latency_sec')}",
        "",
        "## Frontend Performance",
        f"- Ran: {fe.get('ran')}",
        f"- Build success: {fe.get('ok')}",
        f"- Build duration (s): {fe.get('build_duration_sec')}",
        f"- Dist size (bytes): {fe.get('dist_size_bytes')}",
        "",
        "## Overall",
        f"- Overall OK: {report.get('overall_ok')}",
    ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run KRMAI full project audit and performance checks.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base API URL (default: http://localhost:8000)")
    parser.add_argument("--api-rounds", type=int, default=6, help="Number of API benchmark requests")
    parser.add_argument("--api-concurrency", type=int, default=2, help="Parallel API benchmark workers")
    parser.add_argument("--skip-backend-tests", action="store_true", help="Skip test_system.py")
    parser.add_argument("--skip-api-benchmark", action="store_true", help="Skip /chat benchmark")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip web-app build checks")
    parser.add_argument("--backend-timeout", type=int, default=1800, help="Timeout for test_system.py (seconds)")
    parser.add_argument("--frontend-install-timeout", type=int, default=1200, help="Timeout for npm install (seconds)")
    parser.add_argument("--frontend-build-timeout", type=int, default=1200, help="Timeout for npm run build (seconds)")
    args = parser.parse_args()

    print("[Audit] Starting KRMAI project audit...")

    report: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "project_scan": scan_project_tree(ROOT),
        "api_health": check_api_health(args.api_url),
    }

    python_exec = sys.executable

    if args.skip_backend_tests:
        report["backend_tests"] = {"ran": False, "ok": None, "reason": "skipped by flag"}
    else:
        print("[Audit] Running backend model/system tests (test_system.py)...")
        backend = run_backend_tests(python_exec=python_exec, timeout_sec=args.backend_timeout)
        backend["ran"] = True
        report["backend_tests"] = backend

    if args.skip_api_benchmark:
        report["api_benchmark"] = {"ran": False, "reason": "skipped by flag"}
    elif not report["api_health"].get("reachable"):
        report["api_benchmark"] = {
            "ran": False,
            "reason": "API health endpoint not reachable",
        }
    else:
        print("[Audit] Running API latency benchmark...")
        api = run_api_benchmark(
            api_url=args.api_url,
            rounds=max(1, args.api_rounds),
            concurrency=max(1, args.api_concurrency),
            timeout_sec=180,
        )
        api["ran"] = True
        report["api_benchmark"] = api

    if args.skip_frontend:
        report["frontend_performance"] = {"ran": False, "reason": "skipped by flag"}
    else:
        print("[Audit] Running frontend build performance check...")
        frontend = run_frontend_performance(
            web_dir=ROOT / "web-app",
            install_timeout_sec=args.frontend_install_timeout,
            build_timeout_sec=args.frontend_build_timeout,
        )
        frontend["ran"] = True
        report["frontend_performance"] = frontend

    backend_ok = report["backend_tests"].get("ok") in (True, None)
    api_ok = report["api_benchmark"].get("success_rate_percent", 100) >= 80 if report["api_benchmark"].get("ran") else True
    fe_ok = report["frontend_performance"].get("ok") in (True, None)

    report["overall_ok"] = bool(backend_ok and api_ok and fe_ok)

    paths = write_reports(report)

    print("\n[Audit] Completed.")
    print(f"[Audit] Overall OK: {report['overall_ok']}")
    print(f"[Audit] JSON report: {paths['json']}")
    print(f"[Audit] Markdown report: {paths['markdown']}")

    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
