#!/usr/bin/env python3
"""
Concurrent batch streaming benchmark script.
Measures TTFT, total request time, per-stream tok/s, and aggregate batch tok/s.
"""

import json
import time
import sys
import urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18001/v1/completions"
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 2
N_ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 50

PROMPT = "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate?"

PAYLOAD = {
    "model": "granite-8b",
    "prompt": PROMPT,
    "max_tokens": MAX_TOKENS,
    "temperature": 0,
    "stream": True,
    "ignore_eos": True
}

def send_single_stream(req_id, url, payload, max_retries=5):
    """Sends a single streaming request with retry logic on empty/transient errors."""
    for attempt in range(max_retries):
        t0 = time.monotonic()
        ttft = None
        token_count = 0
        error_msg = None
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3600) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        tok = chunk["choices"][0].get("text", "")
                        if tok:
                            token_count += 1
                            if ttft is None:
                                ttft = time.monotonic() - t0
                    except Exception:
                        pass
        except Exception as e:
            error_msg = str(e)
            
        t1 = time.monotonic()
        total_time = t1 - t0
        
        if ttft is not None and token_count > 0:
            tps_total = token_count / total_time if total_time > 0 else 0
            tps_after = (token_count - 1) / (total_time - ttft) if (total_time - ttft) > 0 else 0
            return {
                "req_id": req_id,
                "token_count": token_count,
                "ttft": ttft,
                "total_time": total_time,
                "tok_s_total": tps_total,
                "tok_s_post_ttft": tps_after,
                "success": True,
                "attempts": attempt + 1
            }
        time.sleep(1.0)
        
    return {
        "req_id": req_id,
        "token_count": 0,
        "ttft": None,
        "total_time": 0,
        "tok_s_total": 0,
        "tok_s_post_ttft": 0,
        "success": False,
        "error": error_msg,
        "attempts": max_retries
    }

def run_batch_iteration(iteration_num, batch_size, url, payload):
    batch_t0 = time.monotonic()
    futures = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for b in range(batch_size):
            futures.append(executor.submit(send_single_stream, b + 1, url, payload))
            
        stream_results = [f.result() for f in futures]
    batch_t1 = time.monotonic()
    batch_wall_time = batch_t1 - batch_t0
    
    successful_streams = [s for s in stream_results if s["success"]]
    if not successful_streams:
        print(f"Iteration {iteration_num}: ALL STREAMS FAILED", flush=True)
        return None
        
    total_tokens = sum(s["token_count"] for s in successful_streams)
    agg_tok_s = total_tokens / batch_wall_time if batch_wall_time > 0 else 0
    mean_ttft = np.mean([s["ttft"] for s in successful_streams])
    mean_post_tps = np.mean([s["tok_s_post_ttft"] for s in successful_streams])
    mean_total_tps = np.mean([s["tok_s_total"] for s in successful_streams])
    
    print(f"Iter {iteration_num:2d}: {len(successful_streams)}/{batch_size} streams, "
          f"Total Toks={total_tokens}, Wall={batch_wall_time:.2f}s, "
          f"Agg tok/s={agg_tok_s:.4f}, Mean TTFT={mean_ttft:.3f}s, "
          f"Stream tok/s(post-TTFT)={mean_post_tps:.4f}", flush=True)
          
    return {
        "iteration": iteration_num,
        "batch_size": batch_size,
        "streams_completed": len(successful_streams),
        "total_tokens": total_tokens,
        "wall_time_s": batch_wall_time,
        "agg_tok_s": agg_tok_s,
        "mean_ttft_s": mean_ttft,
        "mean_post_tps": mean_post_tps,
        "mean_total_tps": mean_total_tps,
        "streams": stream_results
    }

def main():
    print(f"================================================================")
    print(f"Benchmark: URL={URL}")
    print(f"Batch Size={BATCH_SIZE}, Iterations={N_ITERATIONS}, max_tokens={MAX_TOKENS}")
    print(f"================================================================")
    
    all_iterations = []
    for i in range(1, N_ITERATIONS + 1):
        res = run_batch_iteration(i, BATCH_SIZE, URL, PAYLOAD)
        if res:
            all_iterations.append(res)
            
    if not all_iterations:
        print("ERROR: No iterations succeeded.")
        sys.exit(1)
        
    wall_times = [it["wall_time_s"] for it in all_iterations]
    agg_tps = [it["agg_tok_s"] for it in all_iterations]
    ttfts = [it["mean_ttft_s"] for it in all_iterations]
    post_tps = [it["mean_post_tps"] for it in all_iterations]
    total_tps = [it["mean_total_tps"] for it in all_iterations]
    
    ddof = 1 if len(all_iterations) > 1 else 0
    print("\n" + "="*60)
    print(f"FINAL SUMMARY: Batch Size = {BATCH_SIZE}, max_tokens = {MAX_TOKENS} ({len(all_iterations)} iterations)")
    print("="*60)
    print(f"Mean TTFT (s):               {np.mean(ttfts):.3f} ± {np.std(ttfts, ddof=ddof):.3f}")
    print(f"Mean Batch Wall Time (s):    {np.mean(wall_times):.3f} ± {np.std(wall_times, ddof=ddof):.3f}")
    print(f"Aggregate Throughput (tok/s): {np.mean(agg_tps):.4f} ± {np.std(agg_tps, ddof=ddof):.4f}")
    print(f"Per-Stream Post-TTFT (tok/s): {np.mean(post_tps):.4f} ± {np.std(post_tps, ddof=ddof):.4f}")
    print(f"Per-Stream Total (tok/s):     {np.mean(total_tps):.4f} ± {np.std(total_tps, ddof=ddof):.4f}")
    print("="*60)
    
    output = {
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "num_iterations": len(all_iterations),
        "summary": {
            "mean_ttft_s": round(float(np.mean(ttfts)), 3),
            "std_ttft_s": round(float(np.std(ttfts, ddof=ddof)), 3),
            "mean_wall_time_s": round(float(np.mean(wall_times)), 3),
            "std_wall_time_s": round(float(np.std(wall_times, ddof=ddof)), 3),
            "mean_agg_tok_s": round(float(np.mean(agg_tps)), 4),
            "std_agg_tok_s": round(float(np.std(agg_tps, ddof=ddof)), 4),
            "mean_post_tps": round(float(np.mean(post_tps)), 4),
            "std_post_tps": round(float(np.std(post_tps, ddof=ddof)), 4),
            "mean_total_tps": round(float(np.mean(total_tps)), 4),
            "std_total_tps": round(float(np.std(total_tps, ddof=ddof)), 4)
        },
        "iterations": all_iterations
    }
    
    with open(f"/tmp/batch_results_b{BATCH_SIZE}_t{MAX_TOKENS}.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
