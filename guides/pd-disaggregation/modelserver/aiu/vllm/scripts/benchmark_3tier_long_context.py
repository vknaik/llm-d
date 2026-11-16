#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
"""Phase 3 Benchmarking Suite: Long-Context (16k - 32k) & 3-Tier Hierarchical Memory Evaluation on 4P1D Hybrid."""

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import statistics
import time
from typing import Any

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_3tier")

DEFAULT_ENDPOINT = "http://172.30.116.228:80/hybrid/v1/completions"
OUTPUT_DIR = "/share/test-prefill/workspace/performance-data/hybrid/11162026/4P1D/3tier-mem-hierarchy"


def generate_synthetic_prompt(num_tokens: int) -> str:
    """Generate synthetic text of approximately num_tokens words."""
    base_text = (
        "IBM Spyre accelerator and NVIDIA A100 GPU heterogeneous disaggregated prefill-decode serving "
        "with three-tier memory hierarchy offloading KV cache to local NVMe solid-state storage. "
    )
    words_per_base = len(base_text.split())
    repeats = max(1, math.ceil(num_tokens / words_per_base))
    full_text = (base_text * repeats).split()[:num_tokens]
    return " ".join(full_text)


async def send_single_streaming_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    max_tokens: int,
    req_id: str,
) -> dict[str, Any]:
    """Send a single streaming completion request and record fine-grained timestamps."""
    payload = {
        "model": "granite-8b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    timestamps = []
    token_count = 0

    try:
        async with session.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as response:
            if response.status != 200:
                text = await response.text()
                logger.error("Request %s failed with status %d: %s", req_id, response.status, text)
                return {"success": False, "error": f"HTTP {response.status}", "req_id": req_id}

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data:"):
                    continue
                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    token_count += 1
                    t_now = time.perf_counter()
                    timestamps.append(t_now)
                    if ttft is None:
                        ttft = t_now - t_start
                except Exception:
                    continue

        t_end = time.perf_counter()
        total_time = t_end - t_start
        itls = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

        tok_s_total = token_count / max(total_time, 1e-6)
        time_after_ttft = total_time - (ttft if ttft is not None else 0.0)
        tok_s_post_ttft = (max(0, token_count - 1)) / max(time_after_ttft, 1e-6)

        return {
            "success": True,
            "req_id": req_id,
            "tokens": token_count,
            "ttft": ttft if ttft is not None else total_time,
            "total_time": total_time,
            "tok_s_total": tok_s_total,
            "tok_s_post_ttft": tok_s_post_ttft,
            "itls": itls,
        }
    except Exception as e:
        logger.error("Request %s exception: %s", req_id, e)
        return {"success": False, "error": str(e), "req_id": req_id}


async def run_batch_experiment(
    endpoint: str,
    prompt_len: int,
    max_tokens: int,
    batch_size: int,
    num_runs: int = 10,
) -> dict[str, Any]:
    """Execute concurrent batch requests across multiple iterations."""
    prompt = generate_synthetic_prompt(prompt_len)
    run_summaries = []

    logger.info(
        "Starting Batch Grid: P=%d tokens, M=%d tokens, B=%d streams (%d runs)",
        prompt_len, max_tokens, batch_size, num_runs,
    )

    connector = aiohttp.TCPConnector(limit=max(128, batch_size * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        for run_idx in range(1, num_runs + 1):
            t_run_start = time.perf_counter()
            tasks = [
                send_single_streaming_request(
                    session,
                    endpoint,
                    prompt,
                    max_tokens,
                    f"run_{run_idx}_req_{b}",
                )
                for b in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)
            t_run_end = time.perf_counter()
            run_wall_time = t_run_end - t_run_start

            successful = [r for r in results if r.get("success")]
            if not successful:
                logger.error("Run %d failed: 0 successful responses", run_idx)
                continue

            run_ttfts = [r["ttft"] for r in successful]
            run_total_tokens = sum(r["tokens"] for r in successful)
            run_tok_s_post = [r["tok_s_post_ttft"] for r in successful]

            run_summary = {
                "run": run_idx,
                "wall_time": run_wall_time,
                "successful_requests": len(successful),
                "total_tokens": run_total_tokens,
                "aggregate_throughput": run_total_tokens / max(run_wall_time, 1e-6),
                "mean_stream_post_ttft": statistics.mean(run_tok_s_post) if run_tok_s_post else 0.0,
                "mean_ttft": statistics.mean(run_ttfts),
                "p50_ttft": float(np.percentile(run_ttfts, 50)),
                "p90_ttft": float(np.percentile(run_ttfts, 90)),
                "raw_results": results,
            }
            run_summaries.append(run_summary)
            logger.info(
                "  Run %2d/%2d: Agg Tok/s=%.2f, Mean TTFT=%.3fs, WallTime=%.2fs",
                run_idx, num_runs, run_summary["aggregate_throughput"],
                run_summary["mean_ttft"], run_wall_time,
            )

    if not run_summaries:
        return {"status": "failed"}

    agg_th_list = [r["aggregate_throughput"] for r in run_summaries]
    ttft_list = [r["mean_ttft"] for r in run_summaries]
    stream_post_list = [r["mean_stream_post_ttft"] for r in run_summaries]
    wall_time_list = [r["wall_time"] for r in run_summaries]

    # Calculate 3-tier memory allocation footprint for Granite-4.2-8B:
    # Logical block size = 10.24 MB (64 tokens)
    # Total context = P + M tokens
    total_tokens_per_stream = prompt_len + max_tokens
    total_blocks_per_stream = math.ceil(total_tokens_per_stream / 64)
    total_blocks_all_streams = total_blocks_per_stream * batch_size
    total_kv_memory_gb = (total_blocks_all_streams * 10.24 * 1024 * 1024) / (1024**3)

    # 3-Tier Allocation breakdown:
    # Tier 1 (Device AIU SRAM/DRAM): Max 256 blocks (~2.5 GB active decode window)
    # Tier 2 (Host DRAM Staging): Max 1024 blocks (~10.0 GB warm staging)
    # Tier 3 (NVMe SSD): Remaining blocks stored persistently
    t1_blocks = min(total_blocks_all_streams, 256)
    t2_blocks = min(max(0, total_blocks_all_streams - t1_blocks), 1024)
    t3_blocks = max(0, total_blocks_all_streams - t1_blocks - t2_blocks)

    return {
        "prompt_tokens": prompt_len,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "num_runs": len(run_summaries),
        "mean_aggregate_throughput": statistics.mean(agg_th_list),
        "stdev_aggregate_throughput": statistics.stdev(agg_th_list) if len(agg_th_list) > 1 else 0.0,
        "mean_ttft": statistics.mean(ttft_list),
        "stdev_ttft": statistics.stdev(ttft_list) if len(ttft_list) > 1 else 0.0,
        "mean_stream_post_ttft": statistics.mean(stream_post_list),
        "stdev_stream_post_ttft": statistics.stdev(stream_post_list) if len(stream_post_list) > 1 else 0.0,
        "mean_wall_time": statistics.mean(wall_time_list),
        "total_kv_memory_gb": total_kv_memory_gb,
        "tier1_device_gb": (t1_blocks * 10.24 * 1024 * 1024) / (1024**3),
        "tier2_host_gb": (t2_blocks * 10.24 * 1024 * 1024) / (1024**3),
        "tier3_nvme_gb": (t3_blocks * 10.24 * 1024 * 1024) / (1024**3),
        "runs": run_summaries,
    }


async def run_multiturn_prefix_experiment(
    endpoint: str,
    p_sys: int,
    num_sessions: int = 10,
) -> dict[str, Any]:
    """Execute 5-turn conversational sessions to measure 3-tier prefix caching speedup."""
    logger.info("Starting Multi-Turn Prefix Caching: P_sys=%d tokens (%d sessions)", p_sys, num_sessions)
    sys_prompt = generate_synthetic_prompt(p_sys)

    turn_ttfts = {f"turn_{t}": [] for t in range(1, 6)}
    session_summaries = []

    connector = aiohttp.TCPConnector(limit=16)
    async with aiohttp.ClientSession(connector=connector) as session:
        for s_idx in range(1, num_sessions + 1):
            history = sys_prompt
            sess_results = []
            for turn in range(1, 6):
                turn_prompt = history + f"\nUser Query Turn {turn}: " + generate_synthetic_prompt(32)
                res = await send_single_streaming_request(
                    session,
                    endpoint,
                    turn_prompt,
                    max_tokens=100,
                    req_id=f"sess_{s_idx}_turn_{turn}",
                )
                sess_results.append(res)
                if res.get("success"):
                    turn_ttfts[f"turn_{turn}"].append(res["ttft"])
                history = turn_prompt + f"\nAssistant Response Turn {turn}: Output generated token stream."

            session_summaries.append({"session": s_idx, "turns": sess_results})

    turn1_mean = statistics.mean(turn_ttfts["turn_1"])
    turn2_mean = statistics.mean(turn_ttfts["turn_2"])
    reduction_pct = ((turn1_mean - turn2_mean) / turn1_mean) * 100.0
    speedup = turn1_mean / max(turn2_mean, 1e-4)

    return {
        "p_sys": p_sys,
        "num_sessions": num_sessions,
        "turn_1_cold_ttft_mean": turn1_mean,
        "turn_1_cold_ttft_stdev": statistics.stdev(turn_ttfts["turn_1"]) if len(turn_ttfts["turn_1"]) > 1 else 0.0,
        "turn_2_warm_ttft_mean": turn2_mean,
        "turn_2_warm_ttft_stdev": statistics.stdev(turn_ttfts["turn_2"]) if len(turn_ttfts["turn_2"]) > 1 else 0.0,
        "turn_3_ttft_mean": statistics.mean(turn_ttfts["turn_3"]),
        "turn_4_ttft_mean": statistics.mean(turn_ttfts["turn_4"]),
        "turn_5_ttft_mean": statistics.mean(turn_ttfts["turn_5"]),
        "ttft_reduction_pct": reduction_pct,
        "ttft_speedup_factor": speedup,
        "sessions": session_summaries,
    }


async def main():
    parser = argparse.ArgumentParser(description="Phase 3 3-Tier Long-Context Benchmarks")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Gateway completion endpoint")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="Output directory on PVC")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logger.info("Output directory: %s", args.outdir)

    # 1. Long-Context Scaling Grid: 16k and 32k contexts
    # Grid: P in {16384, 32768}, M in {200, 1024}, B in {1, 4, 8, 16}
    grid_configs = [
        (16384, 200, 1),
        (16384, 200, 4),
        (16384, 200, 8),
        (16384, 200, 16),
        (16384, 1024, 1),
        (16384, 1024, 4),
        (16384, 1024, 8),
        (16384, 1024, 16),
        (32768, 200, 1),
        (32768, 200, 4),
        (32768, 200, 8),
        (32768, 200, 16),
        (32768, 1024, 1),
        (32768, 1024, 4),
        (32768, 1024, 8),
        (32768, 1024, 16),
    ]

    long_context_results = []
    for p_len, m_tok, b_size in grid_configs:
        n_runs = 5 if (p_len == 32768 and b_size >= 8 and m_tok == 1024) else 10
        res = await run_batch_experiment(args.endpoint, p_len, m_tok, b_size, num_runs=n_runs)
        long_context_results.append(res)

    # Save JSON raw measurements
    lc_path = os.path.join(args.outdir, "long_context_scaling_results.json")
    with open(lc_path, "w") as f:
        json.dump(long_context_results, f, indent=2)
    logger.info("Saved long-context scaling results to %s", lc_path)

    # 2. Multi-Turn Prefix Caching with 3-Tier Hierarchy
    prefix_configs = [2048, 4096, 8192, 16384]
    prefix_results = []
    for p_sys in prefix_configs:
        res = await run_multiturn_prefix_experiment(args.endpoint, p_sys, num_sessions=10)
        prefix_results.append(res)

    pfx_path = os.path.join(args.outdir, "prefix_caching_3tier_results.json")
    with open(pfx_path, "w") as f:
        json.dump(prefix_results, f, indent=2)
    logger.info("Saved 3-tier prefix caching results to %s", pfx_path)

    # 3. Save consolidated CSV summary
    csv_path = os.path.join(args.outdir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Category", "Prompt_Tokens", "Max_Tokens", "Batch_Size",
            "Mean_Agg_Tok_s", "Stdev_Agg_Tok_s", "Mean_TTFT_s", "Stdev_TTFT_s",
            "Mean_Stream_Post_TTFT", "Total_KV_Memory_GB", "Tier1_Device_GB",
            "Tier2_Host_GB", "Tier3_NVMe_GB"
        ])
        for r in long_context_results:
            writer.writerow([
                "Long_Context_Scaling", r["prompt_tokens"], r["max_tokens"], r["batch_size"],
                f"{r['mean_aggregate_throughput']:.2f}", f"{r['stdev_aggregate_throughput']:.2f}",
                f"{r['mean_ttft']:.3f}", f"{r['stdev_ttft']:.3f}",
                f"{r['mean_stream_post_ttft']:.2f}", f"{r['total_kv_memory_gb']:.2f}",
                f"{r['tier1_device_gb']:.2f}", f"{r['tier2_host_gb']:.2f}", f"{r['tier3_nvme_gb']:.2f}"
            ])
    logger.info("Saved CSV summary to %s", csv_path)
    logger.info("Phase 3 Benchmarking Suite Finished Successfully! 🎉")


if __name__ == "__main__":
    asyncio.run(main())
