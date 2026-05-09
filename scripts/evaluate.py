#!/usr/bin/env python3
"""
Zero-shot evaluation of a trained checkpoint on all 57 benchmark KGs.

Usage
-----
python scripts/evaluate.py \
    --checkpoint checkpoints/camp_kg/seed_0/ckpt_best.pt \
    --kg_dir     data/benchmarks \
    --out_csv    results/camp_kg_seed0.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from camp_kg.evaluate import evaluate_all_kgs


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CAMP-KG on benchmark KGs")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--kg_dir",     type=str, required=True,
                   help="Directory containing benchmark KG subdirectories")
    p.add_argument("--out_csv",    type=str, default="results/results.csv",
                   help="Output CSV file for per-KG results")
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    try:
        from ultra import ULTRA   # type: ignore
    except ImportError:
        raise ImportError(
            "ULTRA is not installed. Please run:\n"
            "  git clone https://github.com/DeepGraphLearning/ULTRA\n"
            "  pip install -e ULTRA/"
        )

    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = ULTRA()
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Benchmarks: {args.kg_dir}")
    print()

    results = evaluate_all_kgs(model, args.kg_dir, device, args.batch_size)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kg", "mrr", "hits@1", "hits@3", "hits@10"])
        writer.writeheader()
        for kg_name, metrics in results.items():
            writer.writerow({"kg": kg_name, **{k: f"{v:.4f}" for k, v in metrics.items()}})

    # Summary
    mrrs = [m["mrr"] for m in results.values()]
    sorted_mrrs = sorted(zip(results.keys(), mrrs), key=lambda x: x[1])
    bottom14_mrrs = [v for _, v in sorted_mrrs[:14]]

    print(f"\nMean MRR (all {len(mrrs)}): {sum(mrrs)/len(mrrs):.3f}")
    print(f"Mean MRR (bottom-14):      {sum(bottom14_mrrs)/len(bottom14_mrrs):.3f}")
    print(f"Results saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
