#!/usr/bin/env python3
"""
Experiment 3: Multi-Turn Prefix Caching Benchmark on the Prefill Tier
Evaluates (P-Spyre AIU Prefill + 1D-A100 GPU Decode) with --enable-prefix-caching
Measures TTFT and prefill latency across multi-turn conversational sequences:
  Shared System Prefix: P_sys in {128, 512, 1024} tokens
  Turn Query: 32 tokens
  Turns: 1 (Cold) -> 2 -> 3 -> 4 -> 5 (Warm Prefix Cache Hits)
10 independent sessions per system prompt size.
"""
import json
import time
import sys
import os
import urllib.request
import statistics

BASE_SYSTEM_PROMPT = (
    "You are an expert enterprise AI system architect specializing in large language model inference, "
    "heterogeneous hardware acceleration on IBM Spyre AIU and NVIDIA A100 GPUs, distributed prefill-decode disaggregation, "
    "low-latency RDMA networking, and high-performance continuous batching optimization. "
    "Answer all technical questions with concise, rigorous, data-driven analytical depth. "
)

# Expand to 128, 512, 1024 tokens
SYSTEM_PROMPTS = {
    128: (BASE_SYSTEM_PROMPT * 3)[:700],
    512: (BASE_SYSTEM_PROMPT * 12)[:2800],
    1024: (BASE_SYSTEM_PROMPT * 24)[:5600]
}

TURN_QUERIES = [
    "Turn 1 Query: Explain the memory bandwidth bottlenecks in autoregressive token decoding on high-performance accelerators.",
    "Turn 2 Query: How does disaggregated prefill-decode serving decouple prompt compute from memory-bound token generation?",
    "Turn 3 Query: Analyze the impact of PagedAttention block size on SM L1/L2 cacheline tiling efficiency during GEMV steps.",
    "Turn 4 Query: Compare RoCE v2 RDMA versus TCP transport latency for cross-node KV-cache block staging in cluster environments.",
    "Turn 5 Query: Summarize the end-to-end architectural advantages of hybrid AIU prefill with GPU decode serving."
]

def send_single_turn_request(url, full_prompt, max_tokens=100, model_name="granite-8b"):
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True
    }
    t0 = time.monotonic()
    ttft = None
    generated_text = ""
    token_count = 0
    error_msg = None
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
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
                        generated_text += tok
                        if ttft is None:
                            ttft = time.monotonic() - t0
                except Exception:
                    pass
    except Exception as e:
        error_msg = str(e)
        
    total_time = time.monotonic() - t0
    success = (ttft is not None and token_count > 0)
    tok_s = (token_count - 1) / (total_time - ttft) if (success and total_time > ttft) else 0.0
    
    return {
        "ttft": ttft,
        "total_time": total_time,
        "token_count": token_count,
        "tok_s_post_ttft": tok_s,
        "generated_text": generated_text,
        "success": success,
        "error": error_msg
    }

def run_multi_turn_session(session_id, sys_len, sys_prompt, url, model_name="granite-8b"):
    print(f"\n  [Session {session_id:2d}] Starting 5-Turn Sequence (Sys Prompt = {sys_len} tokens)...", flush=True)
    history = f"System: {sys_prompt}\n\n"
    turn_results = []
    
    for turn_idx, query in enumerate(TURN_QUERIES, 1):
        history += f"User: {query}\nAssistant: "
        res = send_single_turn_request(url, history, max_tokens=100, model_name=model_name)
        if not res["success"]:
            print(f"    Turn {turn_idx} FAILED: {res.get('error')}", flush=True)
            return None
            
        print(f"    Turn {turn_idx}: Tokens={res['token_count']} | TTFT={res['ttft']:.3f}s | "
              f"Post-TTFT tok/s={res['tok_s_post_ttft']:.2f} | Total={res['total_time']:.2f}s", flush=True)
              
        turn_results.append({
            "turn": turn_idx,
            "query": query,
            "ttft_s": res["ttft"],
            "total_time_s": res["total_time"],
            "token_count": res["token_count"],
            "tok_s_post_ttft": res["tok_s_post_ttft"]
        })
        history += f"{res['generated_text']}\n\n"
        time.sleep(0.5)
        
    return turn_results

def run_experiment_grid(setup_name="4P-Hybrid", url="http://172.30.116.228:80/hybrid/v1/completions", out_file="/tmp/hybrid_exp3_prefix_caching_results.json", model_name="granite-8b", n_sessions=10):
    results = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    print(f"\n{'='*75}")
    print(f"STARTING HYBRID EXP 3: MULTI-TURN PREFIX CACHING BENCHMARK ({setup_name})")
    print(f"URL: {url} | Model: {model_name} | Sessions per config: {n_sessions}")
    print(f"{'='*75}\n")
    
    for sys_len, sys_text in SYSTEM_PROMPTS.items():
        s_key = str(sys_len)
        if s_key not in results:
            results[s_key] = {"sessions": []}
            
        existing_sess = results[s_key].get("sessions", [])
        start_sess = len(existing_sess) + 1
        
        if start_sess > n_sessions:
            print(f"Skipping already completed SysLen={sys_len} ({len(existing_sess)} sessions)")
            continue
            
        for s in range(start_sess, n_sessions + 1):
            sess_res = run_multi_turn_session(s, sys_len, sys_text, url, model_name=model_name)
            if sess_res:
                results[s_key]["sessions"].append(sess_res)
                
                # Compute running summary
                turn_summaries = {}
                for t in range(5):
                    t_ttfts = [sess[t]["ttft_s"] for sess in results[s_key]["sessions"]]
                    t_posts = [sess[t]["tok_s_post_ttft"] for sess in results[s_key]["sessions"]]
                    turn_summaries[f"turn_{t+1}"] = {
                        "mean_ttft_s": statistics.mean(t_ttfts),
                        "std_ttft_s": statistics.stdev(t_ttfts) if len(t_ttfts) > 1 else 0.0,
                        "mean_post_ttft_tok/s": statistics.mean(t_posts),
                        "std_post_ttft_tok/s": statistics.stdev(t_posts) if len(t_posts) > 1 else 0.0
                    }
                    
                t1_ttft = turn_summaries["turn_1"]["mean_ttft_s"]
                t2_ttft = turn_summaries["turn_2"]["mean_ttft_s"]
                reduction_pct = ((t1_ttft - t2_ttft) / t1_ttft) * 100.0 if t1_ttft > 0 else 0.0
                
                results[s_key]["system_prompt_len"] = sys_len
                results[s_key]["turn_summaries"] = turn_summaries
                results[s_key]["t1_to_t2_reduction_pct"] = reduction_pct
                
                os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=2)
            time.sleep(1.0)
            
        print(f"\n=== SUMMARY (SysLen={sys_len} tokens): "
              f"Turn 1 (Cold) TTFT={results[s_key]['turn_summaries']['turn_1']['mean_ttft_s']:.3f} +/- {results[s_key]['turn_summaries']['turn_1']['std_ttft_s']:.3f}s | "
              f"Turn 2 (Prefix Hit) TTFT={results[s_key]['turn_summaries']['turn_2']['mean_ttft_s']:.3f} +/- {results[s_key]['turn_summaries']['turn_2']['std_ttft_s']:.3f}s "
              f"({results[s_key]['t1_to_t2_reduction_pct']:.1f}% reduction) | "
              f"Turn 5 TTFT={results[s_key]['turn_summaries']['turn_5']['mean_ttft_s']:.3f}s ===")

    print(f"\nExperiment 3 complete. Saved to {out_file}")
    return results

if __name__ == "__main__":
    setup = sys.argv[1] if len(sys.argv) > 1 else "4P-Hybrid"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/hybrid_exp3_prefix_caching_results.json"
    n_sess = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run_experiment_grid(setup_name=setup, out_file=out_path, n_sessions=n_sess)
