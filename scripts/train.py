#!/usr/bin/env python3
"""
Train CAMP-KG (or ULTRA-Real as baseline) on the synthetic corpus.

This script integrates the SyntheticCorpus data loader with ULTRA's model.
It requires the ULTRA repository to be installed:
    git clone https://github.com/DeepGraphLearning/ULTRA
    pip install -e ULTRA/

Usage
-----
# Train CAMP-KG (full, 2000 KGs)
python scripts/train.py --config configs/camp_kg.yaml --seed 0

# Train ablation variant B (no rules)
python scripts/train.py --config configs/ablation_B.yaml --seed 0

# All seeds 0-4:
for s in 0 1 2 3 4; do
    python scripts/train.py --config configs/camp_kg.yaml --seed $s
done
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from camp_kg.corpus import SyntheticCorpus
from camp_kg.train  import train_camp_kg


def parse_args():
    p = argparse.ArgumentParser(description="Train CAMP-KG")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--seed",   type=int, default=0,
                   help="Random seed (0–4 for 5-seed evaluation)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (auto-detected if not set)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Override output directory from config")
    return p.parse_args()


def main():
    args   = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = args.out_dir or os.path.join(cfg["out_dir"], f"seed_{args.seed}")

    print(f"Config : {args.config}")
    print(f"Seed   : {args.seed}")
    print(f"Device : {device}")
    print(f"Out dir: {out_dir}")

    # ----- Data -------------------------------------------------------
    train_dir = os.path.join(cfg["corpus_dir"], "train")
    val_dir   = os.path.join(cfg["corpus_dir"], "val")

    train_corpus = SyntheticCorpus(
        train_dir,
        split="train",
        batch_size=cfg.get("batch_size", 64),
        derived_frac=cfg.get("derived_frac", 0.5),
        seed=args.seed,
    )
    val_corpus = SyntheticCorpus(
        val_dir,
        split="val",
        batch_size=cfg.get("batch_size", 64),
        seed=args.seed + 1,
    )

    # ----- Model (ULTRA) ----------------------------------------------
    # Import ULTRA's model — requires ULTRA repo to be installed
    try:
        from ultra import ULTRA           # type: ignore
        model = ULTRA(
            rel_model_cfg=cfg.get("rel_model", {}),
            entity_model_cfg=cfg.get("entity_model", {}),
        )
    except ImportError:
        raise ImportError(
            "ULTRA is not installed. Please run:\n"
            "  git clone https://github.com/DeepGraphLearning/ULTRA\n"
            "  pip install -e ULTRA/"
        )

    # ----- Train -------------------------------------------------------
    best_ckpt = train_camp_kg(
        model=model,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        out_dir=out_dir,
        n_steps=cfg.get("n_steps", 200_000),
        lr=cfg.get("lr", 5e-4),
        batch_size=cfg.get("batch_size", 64),
        n_negatives=cfg.get("n_negatives", 64),
        adversarial_temperature=cfg.get("adversarial_temperature", 1.0),
        checkpoint_every=cfg.get("checkpoint_every", 10_000),
        device=device,
        seed=args.seed,
    )
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
