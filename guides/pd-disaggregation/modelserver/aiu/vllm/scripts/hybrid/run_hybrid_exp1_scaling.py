#!/usr/bin/env python3
"""
Experiment 1: High Concurrency & Long Context Batch Scaling Benchmark
Evaluates (P-Spyre AIU Prefill + 1D-A100 GPU Decode)
Grid:
  Prompts (P): {32, 128} tokens
  Max Tokens (M): {1024, 4096} tokens
  Batch Sizes (B): {32, 64, 128}
  Iterations: 10 runs per setting
"""
import json
import time
import sys
import os
import urllib.request
import statistics
from concurrent.futures import ThreadPoolExecutor

PROMPT_32 = (
    "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate "
    "in modern natural language benchmarks and distributed AI accelerator clusters across "
    "high-performance serving nodes?"
)

PROMPT_128 = (
    "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate? "
    "Artificial intelligence and deep learning models run efficiently on dedicated hardware accelerators such as the high-performance IBM Spyre AIU processor. "
    "Disaggregated prefill and decode architecture enables high throughput distributed serving by decoupling compute-intensive prompt processing from memory-bandwidth-bound autoregressive token generation. "
    "In this comprehensive benchmark we evaluate the end-to-end performance and latency characteristics of modern large language models across tensor parallel configurations, batch sizes, and varying prompt context lengths to thoroughly understand the overall scaling limits of distributed inference on accelerators accurately today."
)

PROMPTS = {
    32: PROMPT_32,
    128: PROMPT_128
}

def send_single_stream(req_id, url, payload, max_retries=3):
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
            with urllib.request.urlopen(req, timeout=900) as resp:
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
            
        total_time = time.monotonic() - t0
        
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
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(send_single_stream, b + 1, url, payload) for b in range(batch_size)]
        stream_results = [f.result() for f in futures]
    batch_wall_time = time.monotonic() - batch_t0
    
    successful_streams = [s for s in stream_results if s["success"]]
    if not successful_streams:
        print(f"  Iter {iteration_num:2d}: ALL STREAMS FAILED", flush=True)
        return None
        
    total_tokens = sum(s["token_count"] for s in successful_streams)
    agg_tok_s = total_tokens / batch_wall_time if batch_wall_time > 0 else 0
    mean_ttft = statistics.mean([s["ttft"] for s in successful_streams])
    mean_post_tps = statistics.mean([s["tok_s_post_ttft"] for s in successful_streams])
    mean_total_tps = statistics.mean([s["tok_s_total"] for s in successful_streams])
    
    print(f"  Iter {iteration_num:2d}: {len(successful_streams)}/{batch_size} streams | "
          f"Total Toks={total_tokens} | Wall={batch_wall_time:.3f}s | "
          f"Agg tok/s={agg_tok_s:.3f} | TTFT={mean_ttft:.3f}s | "
          f"Post-TTFT tok/s={mean_post_tps:.3f}", flush=True)
          
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

def run_experiment_grid(setup_name="4P-Hybrid", url="http://172.30.116.228:80/hybrid/v1/completions", out_file="/tmp/hybrid_exp1_results.json", model_name="granite-8b", n_iterations=10):
    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    print(f"\n{'='*75}")
    print(f"STARTING HYBRID EXP 1: HIGH CONCURRENCY & LONG CONTEXT BENCHMARK ({setup_name})")
    print(f"URL: {url} | Model: {model_name} | Iterations: {n_iterations}")
    print(f"Grid: P in [32, 128], M in [1024, 4096], B in [32, 64, 128]")
    print(f"{'='*75}\n")
    
    # Warmup
    print("Performing warmup request...")
    send_single_stream(0, url, {
        "model": model_name,
        "prompt": PROMPT_32,
        "max_tokens": 50,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True
    })
    print("Warmup complete.\n")
    
    for prompt_len in [32, 128]:
        prompt_text = PROMPTS[prompt_len]
        p_key = str(prompt_len)
        if p_key not in results:
            results[p_key] = {}
            
        for max_tokens in [1024, 4096]:
            m_key = str(max_tokens)
            if m_key not in results[p_key]:
                results[p_key][m_key] = {}
                
            for batch_size in [32, 64, 128]:
                b_key = str(batch_size)
                if b_key in results[p_key][m_key] and len(results[p_key][m_key][b_key].get("iterations", [])) >= n_iterations:
                    print(f"Skipping already completed P={prompt_len}, M={max_tokens}, B={batch_size}")
                    continue
                    
                print(f"\n--- Running: Prompt={prompt_len} tok, max_tokens={max_tokens}, B={batch_size} ({n_iterations} runs) ---")
                payload = {
                    "model": model_name,
                    "prompt": prompt_text,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "ignore_eos": True
                }
                
                iterations = []
                for i in range(1, n_iterations + 1):
                    res = run_batch_iteration(i, batch_size, url, payload)
                    if res:
                        iterations.append(res)
                    time.sleep(0.5)
                        
                if not iterations:
                    print(f"ERROR: All iterations failed for P={prompt_len}, M={max_tokens}, B={batch_size}")
                    continue
                    
                wall_times = [it["wall_time_s"] for it in iterations]
                agg_tps = [it["agg_tok_s"] for it in iterations]
                ttfts = [it["mean_ttft_s"] for it in iterations]
                post_tps = [it["mean_post_tps"] for it in iterations]
                
                mean_ttft = statistics.mean(ttfts)
                std_ttft = statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
                mean_wall = statistics.mean(wall_times)
                std_wall = statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0
                mean_agg = statistics.mean(agg_tps)
                std_agg = statistics.stdev(agg_tps) if len(agg_tps) > 1 else 0.0
                mean_post = statistics.mean(post_tps)
                std_post = statistics.stdev(post_tps) if len(post_tps) > 1 else 0.0
                
                results[p_key][m_key][b_key] = {
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens,
                    "batch_size": batch_size,
                    "iterations": iterations,
                    "summary": {
                        "mean_ttft_s": mean_ttft,
                        "std_ttft_s": std_ttft,
                        "mean_wall_time_s": mean_wall,
                        "std_wall_time_s": std_wall,
                        "mean_agg_tok_s": mean_agg,
                        "std_agg_tok_s": std_agg,
                        "mean_post_ttft_tok_s": mean_post,
                        "std_post_ttft_tok_s": std_post
                    }
                }
                
                print(f"=== SUMMARY [P={prompt_len}, M={max_tokens}, B={batch_size}]: "
                      f"Agg tok/s={mean_agg:.2f} +/- {std_agg:.2f} | "
                      f"TTFT={mean_ttft:.3f} +/- {std_ttft:.3f}s | "
                      f"Post-TTFT tok/s={mean_post:.2f} +/- {std_post:.2f} | "
                      f"Wall Time={mean_wall:.2f}s ===")
                
                os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nBenchmark finished. Saved to {out_file}")
    return results

if __name__ == "__main__":
    setup = sys.argv[1] if len(sys.argv) > 1 else "4P-Hybrid"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hybrid_exp1_results.json"
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run_experiment_grid(setup_name=setup, out_file=out_path, n_iterations=n_iters)
