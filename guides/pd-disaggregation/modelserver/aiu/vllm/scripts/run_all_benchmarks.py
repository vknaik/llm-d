#!/usr/bin/env python3
"""
Automated benchmark orchestrator for IBM Spyre AIU vLLM testing.
Runs Experiments A, C, and D across batch sizes [1, 2, 4, 8, 16] and max_tokens [50, 200].
"""
import subprocess
import time
import json
import sys
import os

KUBECONFIG = "/home/vkn/.kube/config-llm-d.spyre"
NAMESPACE = "llm-d-on-aiu"

EXPERIMENTS = {
    "A": {
        "name": "Standalone TP=1",
        "pod": "test1-standalone-spyre-dev",
        "container": None,
        "url": "http://localhost:18001/v1/completions"
    },
    "C": {
        "name": "TP=1 PD Disagg Gateway",
        "pod": "pd-disagg-tp1-ibm-aiu-vllm-prefill-6b5b8b8c66-p2pqx",
        "container": "modelserver",
        "url": "http://172.30.116.228:80/tp1/v1/completions"
    },
    "D": {
        "name": "TP=2 PD Disagg Gateway",
        "pod": "pd-disaggregation-ibm-aiu-vllm-prefill-54848b7475-5d4bm",
        "container": "modelserver",
        "url": "http://172.30.116.228:80/v1/completions"
    }
}

def oc_exec(pod, container, cmd):
    env = os.environ.copy()
    env["KUBECONFIG"] = KUBECONFIG
    base_cmd = ["oc", "exec", "-n", NAMESPACE, pod]
    if container:
        base_cmd += ["-c", container]
    base_cmd += ["--"] + cmd
    res = subprocess.run(base_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res

def run_benchmark(exp_key, batch_size, max_tokens, n_iterations=10):
    exp = EXPERIMENTS[exp_key]
    pod = exp["pod"]
    container = exp["container"]
    url = exp["url"]
    
    json_path = f"/tmp/exp_{exp_key.lower()}_b{batch_size}_t{max_tokens}.json"
    log_path = f"/tmp/exp_{exp_key.lower()}_b{batch_size}_t{max_tokens}.log"
    
    print(f"\n>>> Running Exp {exp_key} ({exp['name']}): Batch Size={batch_size}, max_tokens={max_tokens}, {n_iterations} runs...", flush=True)
    
    # Remove existing files
    oc_exec(pod, container, ["rm", "-f", json_path, log_path])
    
    # Start background process
    run_cmd = [
        "sh", "-c",
        f"nohup python3 -u /tmp/batch_stream_timing.py '{url}' {batch_size} {n_iterations} {max_tokens} '{json_path}' > '{log_path}' 2>&1 &"
    ]
    res = oc_exec(pod, container, run_cmd)
    if res.returncode != 0:
        print(f"Failed to start benchmark: {res.stderr}")
        return None
        
    start_time = time.time()
    last_line_count = 0
    while True:
        time.sleep(4)
        # Check if process finished (or JSON exists)
        check_json = oc_exec(pod, container, ["cat", json_path])
        if check_json.returncode == 0 and check_json.stdout.strip():
            try:
                data = json.loads(check_json.stdout.strip())
                print(f"\n[DONE in {time.time()-start_time:.1f}s] Summary: {json.dumps(data['summary'])}", flush=True)
                return data
            except json.JSONDecodeError:
                pass
                
        # Check log for progress
        log_res = oc_exec(pod, container, ["cat", log_path])
        if log_res.returncode == 0:
            lines = [l for l in log_res.stdout.splitlines() if l.startswith("Iter ")]
            if len(lines) > last_line_count:
                for l in lines[last_line_count:]:
                    print(f"  {l}", flush=True)
                last_line_count = len(lines)
                
        # Check if still running
        ps_res = oc_exec(pod, container, ["sh", "-c", "ps aux | grep batch_stream_timing | grep -v grep"])
        if ps_res.returncode != 0 and (check_json.returncode != 0 or not check_json.stdout.strip()):
            # Process exited without JSON!
            print(f"ERROR: Process exited without creating output JSON! Log:\n{log_res.stdout}")
            return None

def main():
    exp_filter = sys.argv[1].upper() if len(sys.argv) > 1 else None
    
    results = {}
    for exp_key in (["A", "C", "D"] if not exp_filter else [exp_filter]):
        results[exp_key] = {"50": {}, "200": {}}
        for tok in [50, 200]:
            for b in [1, 2, 4, 8, 16]:
                res = run_benchmark(exp_key, b, tok, n_iterations=10)
                if res:
                    results[exp_key][str(tok)][str(b)] = res["summary"]
                    
    with open(f"/tmp/all_benchmark_results_{exp_filter or 'all'}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results saved to /tmp/all_benchmark_results_{exp_filter or 'all'}.json")

if __name__ == "__main__":
    main()
