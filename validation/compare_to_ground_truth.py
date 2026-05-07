"""
Compare a head-ranking JSON produced by Princeton's `detect_qrhead_lme.py`
to the ground-truth top-16 LME heads published in their config yaml.

Used to validate that the upstream Princeton implementation reproduces
the published Llama-3.1-8B-Instruct LME ranking (paper Table 1 / yaml).

Usage:
    python compare_to_ground_truth.py \\
        --our_ranking_json <path/to/our_run_output.json> \\
        --ground_truth_yaml <path/to/Llama-3.1-8B-Instruct_qr_head_LME.yaml>

Exit codes:
    0 = top-16 set match (ranking validated)
    1 = top-16 set differs (mismatch)
    2 = file load error / format error
"""

import argparse
import json
import re
import sys
from collections import Counter


def parse_ground_truth_yaml(yaml_path):
    """Parse `attn_head_set: "13-18,13-21,..."` line out of the yaml."""
    with open(yaml_path, "r") as f:
        text = f.read()
    m = re.search(r'attn_head_set\s*:\s*"([^"]+)"', text)
    if not m:
        raise ValueError(f"Could not find attn_head_set line in {yaml_path}")
    heads = [h.strip() for h in m.group(1).split(",")]
    return heads


def load_our_ranking(json_path):
    """Load detect_qrhead_lme.py output. Format: list of [layer-head, score]."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data)}")
    return [(item[0], float(item[1])) for item in data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--our_ranking_json", required=True)
    parser.add_argument("--ground_truth_yaml", required=True)
    parser.add_argument("--top_k", type=int, default=16)
    args = parser.parse_args()

    try:
        truth = parse_ground_truth_yaml(args.ground_truth_yaml)
    except Exception as e:
        print(f"FAIL: could not load ground truth: {e}")
        sys.exit(2)

    try:
        ours = load_our_ranking(args.our_ranking_json)
    except Exception as e:
        print(f"FAIL: could not load our ranking: {e}")
        sys.exit(2)

    K = args.top_k
    truth_top = truth[:K]
    ours_top = [h for h, _ in ours[:K]]

    truth_set = set(truth_top)
    ours_set = set(ours_top)

    print("=" * 60)
    print(f"Comparing top-{K} heads")
    print("=" * 60)
    print(f"Ground truth (Princeton paper / yaml): {truth_top}")
    print(f"Our run                              : {ours_top}")
    print()

    # Set comparison (order-insensitive)
    if truth_set == ours_set:
        print(f"PASS: top-{K} sets are identical (order may differ within set).")

        # Order check
        order_match = sum(1 for a, b in zip(truth_top, ours_top) if a == b)
        print(f"  ordered position matches: {order_match}/{K} "
              f"(numerical noise ~1e-3 may cause minor reordering, this is OK)")

        # Show full ordering side-by-side
        print()
        print(f"{'rank':<5} {'truth':<10} {'ours':<10}")
        for i in range(K):
            mark = "  " if truth_top[i] == ours_top[i] else " *"
            print(f"  {i+1:<3} {truth_top[i]:<10} {ours_top[i]:<10}{mark}")

        # Sanity: layer distribution
        truth_layers = Counter(int(h.split("-")[0]) for h in truth_top)
        ours_layers = Counter(int(h.split("-")[0]) for h in ours_top)
        print()
        print(f"Layer distribution (top-{K}):")
        print(f"  truth: {dict(sorted(truth_layers.items()))}")
        print(f"  ours : {dict(sorted(ours_layers.items()))}")

        sys.exit(0)
    else:
        only_in_truth = truth_set - ours_set
        only_in_ours = ours_set - truth_set
        intersection = truth_set & ours_set

        print(f"FAIL: top-{K} sets differ.")
        print(f"  intersection size : {len(intersection)} / {K}")
        print(f"  only in truth     : {sorted(only_in_truth)}")
        print(f"  only in ours      : {sorted(only_in_ours)}")
        print()
        print("Diagnostic info:")
        truth_layers = Counter(int(h.split("-")[0]) for h in truth_top)
        ours_layers = Counter(int(h.split("-")[0]) for h in ours_top)
        print(f"  truth layer distribution (top-{K}): {dict(sorted(truth_layers.items()))}")
        print(f"  ours  layer distribution (top-{K}): {dict(sorted(ours_layers.items()))}")
        print(f"  ours  top-1: {ours[0]}")
        print(f"  ours  bottom-1: {ours[-1]}")

        # Pathology check: are top-K all from one or two final layers?
        if len(ours_layers) <= 2 and max(ours_layers.keys()) >= max(truth_layers.keys()):
            print()
            print("  PATHOLOGY DETECTED: top-K is concentrated in last 1-2 layers.")
            print("  This matches the failure mode seen on the Qwen2 heavy modeling "
                  "with chunked prefill. Probable cause: cache_position / RoPE not "
                  "propagating across chunked prefill calls.")

        sys.exit(1)


if __name__ == "__main__":
    main()
