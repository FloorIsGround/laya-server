#!/usr/bin/env python3
"""Small external HTTP load test for the Laya API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


async def run(args: argparse.Namespace) -> dict[str, float | int]:
    payload = json.loads(args.payload.read_text())
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.timeout)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers) as client:
        for _ in range(args.warmup):
            response = await client.post(args.url, json=payload)
            response.raise_for_status()

        semaphore = asyncio.Semaphore(args.concurrency)
        latencies: list[float] = []
        errors = 0

        async def request_once() -> None:
            nonlocal errors
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(args.url, json=payload)
                    response.raise_for_status()
                except (httpx.HTTPError, httpx.TimeoutException):
                    errors += 1
                else:
                    latencies.append(time.perf_counter() - started)

        started = time.perf_counter()
        await asyncio.gather(*(request_once() for _ in range(args.requests)))
        elapsed = time.perf_counter() - started

    result: dict[str, float | int] = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "succeeded": len(latencies),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(len(latencies) / elapsed, 3),
    }
    if latencies:
        result.update(
            {
                "latency_p50_ms": round(percentile(latencies, 0.50) * 1_000, 3),
                "latency_p95_ms": round(percentile(latencies, 0.95) * 1_000, 3),
                "latency_max_ms": round(max(latencies) * 1_000, 3),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/systemone",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=Path("tests/fixtures/request.json"),
    )
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--api-key")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), indent=2, sort_keys=True))
