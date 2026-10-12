#!/usr/bin/env python3
"""
Granular TTFT & KV-Transfer Profiler for IBM Spyre AIU vLLM Disaggregation.
Evaluates prompt=16 and prompt=128 across TP=1 and TP=2 disaggregated configurations.
"""
import json
import time
import urllib.request
import numpy as np
import sys

PROMPTS = {
    16: "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate?",
    128: (
        "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate? "
        "Artificial intelligence and deep learning models run efficiently on dedicated hardware accelerators such as the high-performance IBM Spyre AIU processor. "
        "Disaggregated prefill and decode architecture enables high throughput distributed serving by decoupling compute-intensive prompt processing from memory-bandwidth-bound autoregressive token generation. "
        "In this comprehensive benchmark we evaluate the end-to-end performance and latency characteristics of modern large language models across tensor parallel configurations, batch sizes, and varying prompt context lengths to thoroughly understand the overall scaling limits of distributed inference on accelerators accurately today."
    )
}

def measure_stream(url, prompt_text, max_tokens, n_runs=10):
    payload = {
        "model": "granite-8b",
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True
    }
    
    runs = []
    for r in range(n_runs):
        t0 = time.monotonic()
        ttft = None
        token_count = 0
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
        total_time = time.monotonic() - t0
        post_ttft_tps = (token_count - 1) / (total_time - ttft) if (total_time - ttft) > 0 else 0
        total_tps = token_count / total_time if total_time > 0 else 0
        runs.append({
            "run": r + 1,
            "token_count": token_count,
            "ttft_s": ttft,
            "total_time_s": total_time,
            "post_ttft_tps": post_ttft_tps,
            "total_tps": total_tps
        })
        time.sleep(0.5)
        
    ttfts = [x["ttft_s"] for x in runs]
    totals = [x["total_time_s"] for x in runs]
    post_tpss = [x["post_ttft_tps"] for x in runs]
    total_tpss = [x["total_tps"] for x in runs]
    
    ddof = 1 if len(runs) > 1 else 0
    return {
        "n_runs": n_runs,
        "mean_ttft_s": float(np.mean(ttfts)),
        "std_ttft_s": float(np.std(ttfts, ddof=ddof)),
        "mean_total_s": float(np.mean(totals)),
        "std_total_s": float(np.std(totals, ddof=ddof)),
        "mean_post_ttft_tps": float(np.mean(post_tpss)),
        "std_post_ttft_tps": float(np.std(post_tpss, ddof=ddof)),
        "mean_total_tps": float(np.mean(total_tpss)),
        "std_total_tps": float(np.std(total_tpss, ddof=ddof)),
        "runs": runs
    }

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18001/v1/completions"
    p_len = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    prompt = PROMPTS[p_len]
    print(f"Benchmarking URL={url}, prompt_length={p_len}, max_tokens={max_tokens}, n_runs={n_runs}...", flush=True)
    res = measure_stream(url, prompt, max_tokens, n_runs)
    print(json.dumps(res, indent=2))
