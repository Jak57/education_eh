# import json
# import statistics

# with open('FINAL_blossom_story_treedata.json', 'r') as f:
#    data = json.load(f)

# e1_list = [item['entail1'] for item in data if item.get('entail1') is not None]
# e2_list = [item['entail2'] for item in data if item.get('entail2') is not None]
# spec_list = [item['specificity_score'] for item in data if item.get('specificity_score') is not None]

# def calculate_metrics(values):
#     if not values:
#         return 0.0, 0.0, 0.0, 0.0
    
#     avg = sum(values) / len(values)
#     mn = min(values)
#     mx = max(values)
#     std = statistics.stdev(values) if len(values) > 1 else 0.0
    
#     return avg, mn, mx, std

# entail_1_avg, entail_1_min, entail_1_max, entail_1_stddev = calculate_metrics(e1_list)
# entail_2_avg, entail_2_min, entail_2_max, entail_2_stddev = calculate_metrics(e2_list)
# spec_avg, spec_min, spec_max, spec_stddev = calculate_metrics(spec_list)
# print(f"--- Entail 1 --- \nAvg: {entail_1_avg:.4f}, Min: {entail_1_min}, Max: {entail_1_max}, Std: {entail_1_stddev:.4f}")
# print(f"--- Entail 2 --- \nAvg: {entail_2_avg:.4f}, Min: {entail_2_min}, Max: {entail_2_max}, Std: {entail_2_stddev:.4f}")
# print(f"--- Specificity --- \nAvg: {spec_avg:.4f}, Min: {spec_min}, Max: {spec_max}, Std: {spec_stddev:.4f}")



import json
import math
import statistics
import sys

# used when no paths are given on the command line
DEFAULT_FILE = 'Results/UF_random_prompts_64_base.json'

# (label, node field, N multiplier) — *_1 metrics average 2 child judgments
# per node, so their N is nodes * 2
METRICS = [
    ("Entail 1", "entail_1", 2),
    ("Entail 2", "entail_2", 1),
    ("Faithfulness 1", "faithfulness1", 2),
    ("Faithfulness 2", "faithfulness2", 1),
    ("Informativeness 1", "informativeness1", 2),
    ("Informativeness 2", "informativeness2", 1),
    ("Specificity", "specificity", 1),
    ("Spec-Faithful", "specificity_faithfulness", 1),
    ("Spec-Informative", "specificity_informativeness", 1),
]


def calculate_metrics(values, n_mult=1):
    """avg/min/max, sample stdev, and stderr with the evaluation-count
    convention described on METRICS."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0

    avg = sum(values) / len(values)
    mn = min(values)
    mx = max(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    n = len(values) * n_mult
    stderr = std / math.sqrt(n) if n else 0.0

    return avg, mn, mx, std, stderr, n


def report(path):
    with open(path, 'r') as f:
        data = json.load(f)
    all_nodes = []
    for level in data.get('levels', []):
        all_nodes.extend(level.get('nodes', []))

    print(f"\n##### {path} #####\n")
    print(f"{'Metric':<18} | {'Avg':<8} | {'Min':<8} | {'Max':<8} | {'StdDev':<8} | {'StdErr':<8} | {'N':<4}")
    print("-" * 80)
    for label, key, n_mult in METRICS:
        values = [n[key] for n in all_nodes if n.get(key) is not None]
        avg, mn, mx, std, stderr, n = calculate_metrics(values, n_mult)
        print(f"{label:<18} | {avg:<8.4f} | {mn:<8.4f} | {mx:<8.4f} | {std:<8.4f} | {stderr:<8.4f} | {n:<4}")


if __name__ == "__main__":
    for path in sys.argv[1:] or [DEFAULT_FILE]:
        report(path)
