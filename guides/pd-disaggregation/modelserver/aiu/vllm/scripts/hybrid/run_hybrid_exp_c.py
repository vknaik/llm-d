#!/usr/bin/env python3
"""
Hybrid Disaggregated Prefill-Decode (Spyre Prefill + A100 GPU Decode)
Experiment C Benchmark Suite via Istio Gateway
Evaluates granite-4.2-8b with Prompt 16 & Prompt 128 across max_tokens in {50, 200}
and batch sizes in {1, 2, 4, 8, 16} with 10 iterations per setting.
"""
import json
import time
import sys
import os
import urllib.request
import statistics
from concurrent.futures import ThreadPoolExecutor

PROMPT_16 = "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate?"
PROMPT_128 = (
    "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate? "
    "Artificial intelligence and deep learning models run efficiently on dedicated hardware accelerators such as the high-performance IBM Spyre AIU processor. "
    "Disaggregated prefill and decode architecture enables high throughput distributed serving by decoupling compute-intensive prompt processing from memory-bandwidth-bound autoregressive token generation. "
    "In this comprehensive benchmark we evaluate the end-to-end performance and latency characteristics of modern large language models across tensor parallel configurations, batch sizes, and varying prompt context lengths to thoroughly understand the overall scaling limits of distributed inference on accelerators accurately today."
)

PROMPTS = {
    16: PROMPT_16,
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
        time.sleep(0.5)
        
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

def run_experiment_grid(url="http://172.30.116.228:80/hybrid/v1/completions", out_file="/tmp/hybrid_exp_c_results.json", model_name="granite-8b", n_iterations=10):
    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    print(f"\n{'='*70}")
    print(f"STARTING HYBRID EXP-C BENCHMARK (Spyre Prefill + A100 GPU Decode)")
    print(f"URL: {url} | Model: {model_name} | Iterations: {n_iterations}")
    print(f"{'='*70}\n")
    
    # Warmup
    print("Performing warmup request...")
    send_single_stream(0, url, {
        "model": model_name,
        "prompt": PROMPT_16,
        "max_tokens": 10,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True
    })
    print("Warmup complete.\n")
    
    for prompt_len, prompt_text in PROMPTS.items():
        p_key = str(prompt_len)
        if p_key not in results:
            results[p_key] = {}
            
        for max_tokens in [50, 200]:
            m_key = str(max_tokens)
            if m_key not in results[p_key]:
                results[p_key][m_key] = {}
                
            for batch_size in [1, 2, 4, 8, 16]:
                b_key = str(batch_size)
                if b_key in results[p_key][m_key]:
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
                    time.sleep(0.1)
                        
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
                
                summary = {
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens,
                    "batch_size": batch_size,
                    "num_runs": len(iterations),
                    "mean_ttft": mean_ttft,
                    "std_ttft": std_ttft,
                    "mean_wall": mean_wall,
                    "std_wall": std_wall,
                    "mean_agg_tok_s": mean_agg,
                    "std_agg_tok_s": std_agg,
                    "mean_post_tok_s": mean_post,
                    "std_post_tok_s": std_post,
                }
                results[p_key][m_key][b_key] = summary
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                print(f"Summary: TTFT={mean_ttft:.3f}±{std_ttft:.3f}s | Wall={mean_wall:.3f}±{std_wall:.3f}s | "
                      f"Agg tok/s={mean_agg:.3f}±{std_agg:.3f} | Post-TTFT tok/s={mean_post:.3f}±{std_post:.3f}")
                      
    return results

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hybrid_exp_c_results.json"
    url = sys.argv[2] if len(sys.argv) > 2 else "http://172.30.116.228:80/hybrid/v1/completions"
    run_experiment_grid(url=url, out_file=out_file, n_iterations=10)
    print(f"\nAll results saved to {out_file}")
