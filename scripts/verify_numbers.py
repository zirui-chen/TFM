#!/usr/bin/env python3
"""
Verify all numerical claims in the paper from per-seed, per-KG CSV results.

The CSV must have columns: kg, seed, mrr, hits_at_1, hits_at_3, hits_at_10, method
where method ∈ {'camp_kg', 'ultra_real'}.

Usage
-----
python scripts/verify_numbers.py --csv results/all_results.csv
"""

import argparse
import csv
import sys
import os
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camp_kg.analysis.bootstrap import (
    bootstrap_ci, bootstrap_ci_grouped, noninferiority_test, bottom_k_deltas
)


# KG family assignments (8 families, as in the paper)
KG_FAMILIES = {
    # FB-derived (11)
    "FB15k-237 v1": 0, "FB15k-237 v2": 0, "FB15k-237 v3": 0, "FB15k-237 v4": 0,
    "InGram-FB v1": 0, "InGram-FB v2": 0, "InGram-FB v3": 0, "InGram-FB v4": 0,
    "HM-FB": 0, "FBNELL": 0, "DBpedia50-ind": 0,
    # WN-derived (8)
    "WN18RR v1": 1, "WN18RR v2": 1, "WN18RR v3": 1, "WN18RR v4": 1,
    "InGram-WN v1": 1, "InGram-WN v2": 1, "InGram-WN v3": 1, "InGram-WN v4": 1,
    # NELL-derived (7)
    "NELL-995 v1": 2, "NELL-995 v2": 2, "NELL-995 v3": 2, "NELL-995 v4": 2,
    "InGram-NELL v2": 2, "InGram-NELL v3": 2, "InGram-NELL v4": 2,
    # CoDEx-derived (3)
    "CoDEx-M (ind)": 3, "CoDEx-S (ind)": 3, "Codex-L-ind": 3,
    # InGram-Web (4)
    "InGram-WK v1": 4, "InGram-WK v2": 4, "InGram-WK v3": 4, "InGram-WK v4": 4,
    # Biomedical (9)
    "MT1:semmed": 5, "MT2:pharma": 5, "MT3:infra": 5, "MT4:health": 5,
    "HM 1k": 5, "HM 3k": 5, "HM 5k": 5, "Hetionet-s": 5, "BM:indigo": 5,
    # Social/related (3)
    "Metafam": 6, "Kinship-ind": 6, "HM-NELL": 6,
}

BOTTOM14_KGS = [
    "MT2:pharma", "Metafam", "Kinship-ind", "AristoV4", "MT4:health",
    "Hetionet-s", "YAGO-ind", "HM 3k", "MT3:infra", "HM 5k",
    "HM 1k", "BM:indigo", "MT1:semmed", "FBNELL",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True,
                   help="CSV with columns: kg,seed,mrr,hits_at_1,hits_at_3,hits_at_10,method")
    p.add_argument("--B",   type=int, default=10_000, help="Bootstrap resamples")
    return p.parse_args()


def load_csv(path):
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kg     = row["kg"]
            seed   = int(row["seed"])
            method = row["method"]
            data[method][kg][seed] = {
                "mrr":     float(row["mrr"]),
                "hits@1":  float(row["hits_at_1"]),
                "hits@3":  float(row.get("hits_at_3", 0)),
                "hits@10": float(row["hits_at_10"]),
            }
    return data


def seed_mean(kg_data, metric="mrr"):
    """Average over seeds for one KG."""
    vals = [v[metric] for v in kg_data.values()]
    return np.mean(vals) if vals else 0.0


def main():
    args = parse_args()
    data = load_csv(args.csv)

    camp  = data.get("camp_kg",    {})
    ultra = data.get("ultra_real", {})

    all_kgs = sorted(set(camp.keys()) | set(ultra.keys()))
    print(f"Loaded {len(all_kgs)} KGs, methods: {list(data.keys())}\n")

    # Per-KG 5-seed mean MRR
    camp_mrrs  = {kg: seed_mean(camp.get(kg,  {})) for kg in all_kgs}
    ultra_mrrs = {kg: seed_mean(ultra.get(kg, {})) for kg in all_kgs}

    camp_arr  = np.array([camp_mrrs[kg]  for kg in all_kgs])
    ultra_arr = np.array([ultra_mrrs[kg] for kg in all_kgs])
    delta_arr = camp_arr - ultra_arr

    print("=" * 60)
    print("TABLE 1: MAIN RESULTS")
    print("=" * 60)
    print(f"CAMP-KG  mean MRR (all 57): {camp_arr.mean():.3f}")
    print(f"ULTRA    mean MRR (all 57): {ultra_arr.mean():.3f}")
    print(f"Δ MRR (all 57):             {delta_arr.mean():+.3f}")

    # Bottom-14 KGs
    bot14_mask = np.array([kg in BOTTOM14_KGS for kg in all_kgs])
    print(f"\nCAMP-KG  mean MRR (bottom-14): {camp_arr[bot14_mask].mean():.3f}")
    print(f"ULTRA    mean MRR (bottom-14): {ultra_arr[bot14_mask].mean():.3f}")
    print(f"Δ MRR (bottom-14):             {delta_arr[bot14_mask].mean():+.3f}")

    # Hits
    camp_h1  = np.mean([seed_mean(camp.get(kg,  {}), "hits@1")  for kg in all_kgs])
    ultra_h1 = np.mean([seed_mean(ultra.get(kg, {}), "hits@1")  for kg in all_kgs])
    camp_h10 = np.mean([seed_mean(camp.get(kg,  {}), "hits@10") for kg in all_kgs])
    ultra_h10= np.mean([seed_mean(ultra.get(kg, {}), "hits@10") for kg in all_kgs])
    print(f"\nH@1:  CAMP-KG={camp_h1:.3f}, ULTRA={ultra_h1:.3f}")
    print(f"H@10: CAMP-KG={camp_h10:.3f}, ULTRA={ultra_h10:.3f}")

    print(f"\nKGs improved:   {(delta_arr > 0).sum()}/57")
    print(f"KGs with Δ>0.005: {(delta_arr > 0.005).sum()}/57")
    print(f"KGs regressed:  {(delta_arr < 0).sum()}/57")
    print(f"Max regression: {delta_arr.min():.3f}")

    print("\n" + "=" * 60)
    print("BOOTSTRAP ANALYSES (B=10,000)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # KG-paired bootstrap
    paired_ci = bootstrap_ci(delta_arr, B=args.B, rng=rng)
    print(f"\nKG-paired bootstrap:")
    print(f"  Mean:              {paired_ci['mean']:+.3f}")
    print(f"  95% CI:            [{paired_ci['lower']:+.3f}, {paired_ci['upper']:+.3f}]")
    print(f"  One-sided lower:   {paired_ci['one_sided_lower']:+.3f}  (expect ≥ -0.001)")

    # KG-family grouped bootstrap
    family_ids = np.array([KG_FAMILIES.get(kg, 7) for kg in all_kgs])
    grouped_ci = bootstrap_ci_grouped(delta_arr, family_ids, B=args.B, rng=rng)
    print(f"\nKG-family grouped bootstrap ({grouped_ci['n_families']} families):")
    print(f"  Mean:              {grouped_ci['mean']:+.3f}")
    print(f"  95% CI:            [{grouped_ci['lower']:+.3f}, {grouped_ci['upper']:+.3f}]")
    print(f"  One-sided lower:   {grouped_ci['one_sided_lower']:+.3f}  (expect ≥ -0.005)")

    # Bottom-14 bootstrap
    bot14_deltas = delta_arr[bot14_mask]
    bot14_ci = bootstrap_ci(bot14_deltas, B=args.B, rng=rng)
    print(f"\nBottom-14 bootstrap:")
    print(f"  Mean:  {bot14_ci['mean']:+.3f}  (expect ≈ +0.061)")
    print(f"  95% CI: [{bot14_ci['lower']:+.3f}, {bot14_ci['upper']:+.3f}]  (expect [+0.045, +0.078])")

    print("\n" + "=" * 60)
    print("NON-INFERIORITY TEST (δ=0.02)")
    print("=" * 60)
    # Use per-seed mean MRR across all 57 KGs
    seeds = sorted(set(
        s for kg in camp for s in camp[kg]
    ))
    camp_seed_mrrs  = np.array([
        np.mean([camp.get(kg, {}).get(s, {}).get("mrr", 0.0) for kg in all_kgs])
        for s in seeds
    ])
    ultra_seed_mrrs = np.array([
        np.mean([ultra.get(kg, {}).get(s, {}).get("mrr", 0.0) for kg in all_kgs])
        for s in seeds
    ])
    ni = noninferiority_test(camp_seed_mrrs, ultra_seed_mrrs)
    print(f"  Mean diff: {ni['mean_diff']:+.4f}")
    print(f"  t={ni['t_stat']:.2f}, df={ni['df']}, p={ni['p_value']:.4f}  (expect p<0.001)")
    print(f"  Reject H0 (non-inferior): {ni['reject_h0']}")

    print("\n" + "=" * 60)
    print("BOTTOM-k ANALYSIS")
    print("=" * 60)
    bk = bottom_k_deltas(all_kgs, ultra_arr, camp_arr)
    for k in [1, 14, 20, 28, 57]:
        if k in bk:
            print(f"  bottom-{k:2d}: Δ MRR = {bk[k]:+.3f}")


if __name__ == "__main__":
    main()
