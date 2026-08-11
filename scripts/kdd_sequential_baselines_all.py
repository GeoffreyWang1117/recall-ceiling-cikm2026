#!/usr/bin/env python3
"""
Run sequential baselines on all datasets.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent

datasets = ['beauty', 'movies', 'electronics']
results = {}

for ds in datasets:
    print(f"\n{'='*70}")
    print(f"Running on {ds.upper()}")
    print(f"{'='*70}")

    # Run the experiment
    cmd = f"python scripts/kdd_sequential_baselines.py --n_users 200 --epochs 30 --dataset {ds}"
    result = subprocess.run(cmd, shell=True, cwd=project_root, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

    # Load results
    result_path = project_root / 'experiments' / 'logs' / 'kdd_sequential_baselines.json'
    if result_path.exists():
        with open(result_path) as f:
            ds_results = json.load(f)
        results[ds] = ds_results

# Summarize
print("\n" + "="*70)
print("COMPREHENSIVE SUMMARY")
print("="*70)

summary = {
    'timestamp': datetime.now().isoformat(),
    'datasets': {}
}

for ds, data in results.items():
    print(f"\n{ds.upper()}:")
    if 'retrieval_results' in data:
        for method, metrics in data['retrieval_results'].items():
            recall = metrics['recall@100']['mean'] * 100
            ci_low = metrics['recall@100']['ci_low'] * 100
            ci_high = metrics['recall@100']['ci_high'] * 100
            print(f"  {method}: {recall:.2f}% [{ci_low:.2f}, {ci_high:.2f}]")

        summary['datasets'][ds] = data['retrieval_results']

# Save combined results
output_path = project_root / 'experiments' / 'logs' / 'kdd_sequential_baselines_all.json'
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nCombined results saved to: {output_path}")
