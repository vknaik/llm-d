#!/usr/bin/env python3
"""SSE streaming timing script — measures TTFT, total time, tok/s"""
import json, time, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18001/v1/completions"
N_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 50

PAYLOAD = {
    "model": "granite-8b",
    "prompt": "The quick brown fox jumps over the lazy dog. What does this sentence demonstrate?",
    "max_tokens": MAX_TOKENS,
    "temperature": 0,
    "stream": True,
    "ignore_eos": True
}

results = []
for i in range(N_RUNS):
    for attempt in range(5):  # up to 5 retries on transient errors
        t0 = time.monotonic()
        ttft = None
        token_count = 0
        try:
            req = urllib.request.Request(URL, data=json.dumps(PAYLOAD).encode(),
                                          headers={"Content-Type": "application/json"})
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
                        tok = chunk["choices"][0].get("text","")
                        if tok:
                            token_count += 1
                            if ttft is None:
                                ttft = time.monotonic() - t0
                    except:
                        pass
        except Exception as e:
            print(f"Run {i+1} attempt {attempt+1}: exception {e}", flush=True)
        total = time.monotonic() - t0
        if ttft is not None:
            break
        print(f"Run {i+1} attempt {attempt+1}: 0 tokens in {total:.1f}s, retrying in 2s...", flush=True)
        time.sleep(2)
    tps_total = token_count / total if total > 0 else 0
    tps_after = (token_count - 1) / (total - ttft) if ttft and (total - ttft) > 0 else 0
    r = {"run": i+1, "max_tokens": MAX_TOKENS, "token_count": token_count,
         "ttft_seconds": round(ttft, 3) if ttft else None,
         "total_seconds": round(total, 3),
         "tokens_per_second_total": round(tps_total, 4),
         "tokens_per_second_after_ttft": round(tps_after, 4)}
    results.append(r)
    if ttft is None:
        print(f"Run {i+1}: FAILED after retries — 0 tokens received", flush=True)
    else:
        print(f"Run {i+1}: {token_count} tokens, TTFT={ttft:.3f}s, total={total:.1f}s, "
              f"tok/s_total={tps_total:.4f}, tok/s_after_ttft={tps_after:.4f}", flush=True)

print("\n=== JSON Results ===")
print(json.dumps(results, indent=2))
