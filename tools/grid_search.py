import json
import os
import subprocess
import itertools
import re

# Define hyperparameter ranges
param_grid = {
    "--conf": [0.1, 0.3, 0.5, 0.7, 0.9],
    "--nms": [0.1, 0.3, 0.5, 0.7, 0.9],
    "--track_thresh": [0.1, 0.3, 0.5, 0.7, 0.9],
    "--match_thresh": [0.1, 0.3, 0.5, 0.7, 0.9],
    "--tsize": [640, 720, 1080, 1920],
    "--track_buffer": [30, 60, 120, 300, 600],
    "--min-box-area": [10, 100, 150],
}

# Generate all combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

# Fixed arguments
base_args = [
    "python3",
    "tools/track.py",
    "-d", "1",
    "-f", "exps/example/mot/merge_dataset.py",
    "-c", "pretrained/latest_ckpt.pth.tar",
    "--fp16",
    "--fuse",
]

# Function to parse evaluation results
def parse_results(output):
    """Parse relevant metrics from the eval script output."""
    pattern = r"OVERALL\s+([\d.]+%)\s+([\d.]+%)\s+([\d.]+%)\s+([\d.]+%)\s+([\d.]+%)\s+\d+\s+[\d.]+%\s+[\d.]+%\s+[\d.]+%\s+[\d.]+%\s+([\d.]+%)"
    match = re.search(pattern, output)
    if match:
        return {
            "IDF1": match.group(1),
            "IDP": match.group(2),
            "IDR": match.group(3),
            "Rcll": match.group(4),
            "Prcn": match.group(5),
            "MOTA": match.group(6),
        }
    return None

# Store results
results = []

# Loop over parameter combinations
for combo in param_combinations:
    for use_ctra in [True, False]:
        # Construct the command
        cmd = base_args[:]
        for idx, param in enumerate(param_grid.keys()):
            cmd.extend([param, str(combo[idx])])
        if use_ctra:
            cmd.append("--ctra")
        
        print(f"Running: {' '.join(cmd)} (CTRA: {use_ctra})")
        
        # Set environment variable
        env = os.environ.copy()
        env["NCCL_P2P_DISABLE"] = "1"
        
        # Run the command and capture output
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, env=env)
            metrics = parse_results(result.stdout)
            if metrics:
                results.append({
                    "params": dict(zip(param_grid.keys(), combo)),
                    "ctra": use_ctra,
                    "metrics": metrics,
                })
        except subprocess.CalledProcessError as e:
            print(f"Error for params {combo} (CTRA: {use_ctra}): {e.stderr}")

# Save raw results to a JSON file
results_file = "grid_search_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Saved results to {results_file}")

# Compare results for each parameter combination
comparison_results = []
for combo in param_combinations:
    with_ctra = next((r for r in results if r["params"] == dict(zip(param_grid.keys(), combo)) and r["ctra"]), None)
    without_ctra = next((r for r in results if r["params"] == dict(zip(param_grid.keys(), combo)) and not r["ctra"]), None)
    if with_ctra and without_ctra:
        comparison_results.append({
            "params": dict(zip(param_grid.keys(), combo)),
            "with_ctra": with_ctra["metrics"],
            "without_ctra": without_ctra["metrics"],
        })

# Save comparison results to another JSON file
comparison_file = "grid_search_comparison.json"
with open(comparison_file, "w") as f:
    json.dump(comparison_results, f, indent=4)
print(f"Saved comparison results to {comparison_file}")
