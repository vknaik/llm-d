#!/usr/bin/env python3
"""
Experiment 2: Continuous Open-Loop Traffic (Poisson Arrival) Benchmark
Evaluates (P-Spyre AIU Prefill + 1D-A100 GPU Decode)
Dispatches streaming inference requests with Exponential inter-arrival times ~ Exp(lambda)
Evaluates arrival rates: lambda in {5, 10, 20, 30} req/s with granite-4.2-8b.
Prompt: 128 tokens, Max Tokens: 200 tokens.
10 independent Poisson runs per rate.
"""
import json
import time
import sys
import os
import random
import urllib.request
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

PROMPT_128 = (
    "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate? "
    "Artificial intelligence and deep learning models run efficiently on dedicated hardware accelerators such as the high-performance IBM Spyre AIU processor. "
    "Disaggregated prefill and decode architecture enables high throughput distributed serving by decoupling compute-intensive prompt processing from memory-bandwidth-bound autoregressive token generation. "
    "In this comprehensive benchmark we evaluate the end-to-end performance and latency characteristics of modern large language models across tensor parallel configurations, batch sizes, and varying prompt context lengths to thoroughly understand the overall scaling limits of distributed inference on accelerators accurately today."
)

def send_poisson_stream(req_id, url, payload, submit_time):
    t0 = time.monotonic()
    queue_delay = t0 - submit_time
    ttft = None
    token_count = 0
    token_timestamps = []
    error_msg = None
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
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
                        now = time.monotonic()
                        token_count += 1
                        token_timestamps.append(now)
                        if ttft is None:
                            ttft = now - t0
                except Exception:
                    pass
    except Exception as e:
        error_msg = str(e)
        
    t_end = time.monotonic()
    e2e_time = t_end - submit_time
    serv_time = t_end - t0
    
    itls = []
    if len(token_timestamps) > 1:
        for i in range(1, len(token_timestamps)):
            itls.append(token_timestamps[i] - token_timestamps[i-1])
            
    success = (ttft is not None and token_count > 0)
    tok_s = (token_count - 1) / (serv_time - ttft) if (success and serv_time > ttft) else 0.0
    
    return {
        "req_id": req_id,
        "submit_time": submit_time,
        "start_time": t0,
        "end_time": t_end,
        "queue_delay": queue_delay,
        "ttft": ttft,
        "e2e_time": e2e_time,
        "serv_time": serv_time,
        "token_count": token_count,
        "itls": itls,
        "tok_s_post_ttft": tok_s,
        "success": success,
        "error": error_msg
    }

def run_poisson_workload(target_rate_rps, num_requests, url, payload, run_num=1):
    print(f"  [Run {run_num:2d}] Dispatching Poisson Traffic: Target = {target_rate_rps} req/s ({num_requests} requests)...", flush=True)
    results = []
    futures = []
    
    start_bench = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(64, int(target_rate_rps * 10))) as executor:
        for i in range(1, num_requests + 1):
            sub_t = time.monotonic()
            f = executor.submit(send_poisson_stream, i, url, payload, sub_t)
            futures.append(f)
            
            inter_arrival = random.expovariate(target_rate_rps)
            time.sleep(inter_arrival)
            
        for f in futures:
            res = f.result()
            results.append(res)
            
    total_duration = time.monotonic() - start_bench
    successful = [r for r in results if r["success"]]
    
    if not successful:
        print(f"  [Run {run_num:2d}] ALL REQUESTS FAILED", flush=True)
        return None
        
    ttfts = [r["ttft"] for r in successful]
    e2e_times = [r["e2e_time"] for r in successful]
    all_itls = [itl for r in successful for itl in r["itls"]]
    total_tokens = sum(r["token_count"] for r in successful)
    
    actual_rate = len(successful) / total_duration
    sustained_tps = total_tokens / total_duration
    
    def pct(data, p):
        if not data: return 0.0
        sorted_d = sorted(data)
        idx = int(len(sorted_d) * (p / 100.0))
        return sorted_d[min(idx, len(sorted_d) - 1)]
        
    summary = {
        "run_num": run_num,
        "target_rate_rps": target_rate_rps,
        "actual_rate_rps": actual_rate,
        "num_requests": num_requests,
        "success_count": len(successful),
        "total_tokens": total_tokens,
        "total_duration_s": total_duration,
        "sustained_tok_s": sustained_tps,
        "ttft_p50_s": pct(ttfts, 50),
        "ttft_p90_s": pct(ttfts, 90),
        "ttft_p99_s": pct(ttfts, 99),
        "itl_p50_ms": pct(all_itls, 50) * 1000.0 if all_itls else 0.0,
        "itl_p90_ms": pct(all_itls, 90) * 1000.0 if all_itls else 0.0,
        "itl_p99_ms": pct(all_itls, 99) * 1000.0 if all_itls else 0.0,
        "e2e_p50_s": pct(e2e_times, 50),
        "e2e_p90_s": pct(e2e_times, 90),
        "e2e_p99_s": pct(e2e_times, 99)
    }
    
    print(f"  [Run {run_num:2d}] Sustained Tok/s={sustained_tps:.2f} | "
          f"TTFT p50={summary['ttft_p50_s']:.3f}s / p90={summary['ttft_p90_s']:.3f}s | "
          f"ITL p50={summary['itl_p50_ms']:.2f}ms | Duration={total_duration:.2f}s", flush=True)
          
    return {
        "summary": summary,
        "requests": results
    }

def run_experiment_grid(setup_name="4P-Hybrid", url="http://172.30.116.228:80/hybrid/v1/completions", out_file="/tmp/hybrid_exp2_poisson_results.json", model_name="granite-8b", n_runs=10):
    rates = [5, 10, 20, 30]
    num_requests_per_rate = 60
    
    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    print(f"\n{'='*75}")
    print(f"STARTING HYBRID EXP 2: CONTINUOUS POISSON ARRIVAL BENCHMARK ({setup_name})")
    print(f"URL: {url} | Model: {model_name} | Rates: {rates} req/s | Runs per rate: {n_runs}")
    print(f"{'='*75}\n")
    
    payload = {
        "model": model_name,
        "prompt": PROMPT_128,
        "max_tokens": 200,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True
    }
    
    for r in rates:
        r_key = str(r)
        if r_key not in results:
            results[r_key] = {"runs": [], "overall_summary": {}}
            
        existing_runs = results[r_key].get("runs", [])
        start_run = len(existing_runs) + 1
        
        if start_run > n_runs:
            print(f"Skipping already completed rate={r} req/s ({len(existing_runs)} runs)")
            continue
            
        print(f"\n--- Rate: {r} req/s ({n_runs} independent runs of {num_requests_per_rate} reqs each) ---")
        for run_idx in range(start_run, n_runs + 1):
            res = run_poisson_workload(r, num_requests_per_rate, url, payload, run_num=run_idx)
            if res:
                results[r_key]["runs"].append(res)
                
                # Compute running mean and std
                all_tps = [rn["summary"]["sustained_tok_s"] for rn in results[r_key]["runs"]]
                all_p50_ttft = [rn["summary"]["ttft_p50_s"] for rn in results[r_key]["runs"]]
                all_p90_ttft = [rn["summary"]["ttft_p90_s"] for rn in results[r_key]["runs"]]
                all_p50_itl = [rn["summary"]["itl_p50_ms"] for rn in results[r_key]["runs"]]
                
                results[r_key]["overall_summary"] = {
                    "mean_sustained_tok_s": statistics.mean(all_tps),
                    "std_sustained_tok_s": statistics.stdev(all_tps) if len(all_tps) > 1 else 0.0,
                    "mean_ttft_p50_s": statistics.mean(all_p50_ttft),
                    "mean_ttft_p90_s": statistics.mean(all_p90_ttft),
                    "mean_itl_p50_ms": statistics.mean(all_p50_itl)
                }
                
                os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)
            time.sleep(1.0)
            
        print(f"=== RATE {r} REQ/S FINAL: Sustained TPS = {results[r_key]['overall_summary']['mean_sustained_tok_s']:.2f} +/- {results[r_key]['overall_summary']['std_sustained_tok_s']:.2f} | "
              f"TTFT p50={results[r_key]['overall_summary']['mean_ttft_p50_s']:.3f}s / p90={results[r_key]['overall_summary']['mean_ttft_p90_s']:.3f}s ===")
        
    print(f"\nExperiment 2 complete. Saved to {out_file}")
    return results

if __name__ == "__main__":
    setup = sys.argv[1] if len(sys.argv) > 1 else "4P-Hybrid"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hybrid_exp2_poisson_results.json"
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run_experiment_grid(setup_name=setup, out_file=out_path, n_runs=n_runs)
