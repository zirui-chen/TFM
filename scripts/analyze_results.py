#!/usr/bin/env python3
"""
Structural-distance analysis: reproduce Figure 4 (right panel).

Computes JS divergence of degree distributions between each benchmark KG
and the three real pretraining KGs (FB15k-237, WN18RR, CoDEx-Medium),
then correlates the minimum distance with per-KG ∆MRR.

Usage
-----
python scripts/analyze_results.py \
    --results_csv  results/all_results.csv \
    --kg_dir       data/benchmarks \
    --pretraining_dir data/pretraining \
    --out_dir      results/analysis
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from camp_kg.analysis.bootstrap import bootstrap_ci, bottom_k_deltas
from camp_kg.analysis.distance  import (
    compute_structural_distances,
    distance_gain_correlation,
    partial_spearman_correlation,
)
from camp_kg.evaluate import _load_triples, _load_id_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv",     type=str, required=True)
    p.add_argument("--kg_dir",          type=str, required=True,
                   help="Benchmark KG directory")
    p.add_argument("--pretraining_dir", type=str, required=True,
                   help="Directory with fb15k237/, wn18rr/, codex_m/ subdirs")
    p.add_argument("--out_dir",         type=str, default="results/analysis")
    p.add_argument("--B",               type=int, default=10_000)
    return p.parse_args()


def load_kg_triples(kg_path):
    """Load train+val+test triples and n_entities for a KG directory."""
    e2id = _load_id_map(os.path.join(kg_path, "entity2id.txt"))
    r2id = _load_id_map(os.path.join(kg_path, "relation2id.txt"))
    n_ent = len(e2id)
    parts = []
    for split in ("train.txt", "val.txt", "test.txt"):
        path = os.path.join(kg_path, split)
        if os.path.exists(path):
            parts.append(_load_triples(path, e2id, r2id))
    triples = np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=int)
    return triples, n_ent


def load_results_csv(path):
    camp, ultra = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            kg = row["kg"]
            mrr = float(row["mrr"])
            if row["method"] == "camp_kg":
                camp[kg]  = camp.get(kg,  []) + [mrr]
            else:
                ultra[kg] = ultra.get(kg, []) + [mrr]
    return (
        {kg: np.mean(v) for kg, v in camp.items()},
        {kg: np.mean(v) for kg, v in ultra.items()},
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load per-KG mean MRRs
    camp_mrrs, ultra_mrrs = load_results_csv(args.results_csv)
    all_kgs = sorted(set(camp_mrrs.keys()) & set(ultra_mrrs.keys()))
    deltas  = {kg: camp_mrrs[kg] - ultra_mrrs[kg] for kg in all_kgs}

    # Load pretraining KG triples
    pretraining_triples = {}
    for name, subdir in [("fb15k237", "fb15k237"), ("wn18rr", "wn18rr"), ("codex_m", "codex_m")]:
        path = os.path.join(args.pretraining_dir, subdir)
        if os.path.exists(path):
            t, n = load_kg_triples(path)
            pretraining_triples[name] = (t, n)
        else:
            print(f"  WARNING: pretraining KG not found: {path}")

    # Load benchmark KG triples
    benchmark_triples = {}
    for kg in all_kgs:
        path = os.path.join(args.kg_dir, kg.replace(" ", "_").replace(":", "_"))
        if not os.path.exists(path):
            path = os.path.join(args.kg_dir, kg)
        if os.path.exists(path):
            try:
                t, n = load_kg_triples(path)
                benchmark_triples[kg] = (t, n)
            except Exception as e:
                print(f"  WARNING: could not load {kg}: {e}")

    if not pretraining_triples or not benchmark_triples:
        print("Skipping distance analysis (missing KG data).")
        return

    print(f"\nComputing structural distances for {len(benchmark_triples)} benchmark KGs...")

    # Min-of-3 distances (paper default)
    distances_min = compute_structural_distances(
        benchmark_triples, pretraining_triples, aggregation="min"
    )
    # FB-only
    distances_fb = compute_structural_distances(
        benchmark_triples, pretraining_triples, aggregation="fb_only"
    )
    # Mean-of-3
    distances_mean = compute_structural_distances(
        benchmark_triples, pretraining_triples, aggregation="mean"
    )

    # Spearman correlations
    print("\n" + "=" * 60)
    print("STRUCTURAL DISTANCE vs Δ MRR (Spearman ρ)")
    print("=" * 60)
    for label, dists in [
        ("min-of-3 (default)", distances_min),
        ("FB15k-237 only",     distances_fb),
        ("mean-of-3",          distances_mean),
    ]:
        corr = distance_gain_correlation(dists, deltas)
        print(f"  {label:25s}  ρ={corr['rho']:.2f}  p={corr['p_value']:.4f}  n={corr['n']}")

    # Partial correlations (controlling for log|V|, log|R|, baseline MRR)
    common = sorted(set(distances_min.keys()) & set(deltas.keys()) & set(ultra_mrrs.keys()))
    if len(common) >= 5:
        x = np.array([distances_min[k] for k in common])
        y = np.array([deltas[k] for k in common])

        benchmark_n_ent = {kg: benchmark_triples[kg][1] for kg in benchmark_triples}
        log_nent = np.log(np.array([benchmark_n_ent.get(k, 1000) for k in common]))
        n_rels   = np.array([len(np.unique(benchmark_triples[k][0][:, 1])) if k in benchmark_triples else 10 for k in common])
        log_nrel = np.log(np.maximum(n_rels, 1))
        base_mrr = np.array([ultra_mrrs.get(k, 0.3) for k in common])

        covariates = np.column_stack([log_nent, log_nrel, base_mrr])
        rho_partial, p_partial = partial_spearman_correlation(x, y, covariates)
        print(f"\n  Partial ρ (controlling size, nrel, baseline): "
              f"{rho_partial:.2f}  p={p_partial:.4f}  (expect ≈0.48)")

    # Save distance data
    out_data = {
        "kg_distances_min": {k: float(v) for k, v in distances_min.items()},
        "kg_distances_fb":  {k: float(v) for k, v in distances_fb.items()},
        "kg_distances_mean":{k: float(v) for k, v in distances_mean.items()},
        "kg_deltas":        {k: float(v) for k, v in deltas.items()},
    }
    out_path = os.path.join(args.out_dir, "distance_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nDistance data saved to: {out_path}")


if __name__ == "__main__":
    main()
