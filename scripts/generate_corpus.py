#!/usr/bin/env python3
"""
Generate the CAMP-KG pretraining corpus.

Usage
-----
python scripts/generate_corpus.py --n_kg 2000 --out_dir data/corpus/camp_kg \
    --n_train 1800 --seed 0

The script generates n_kg KGs total, writing each to a .pkl file.
The first n_train KGs are the training corpus; the remaining are the
200-KG held-out selection set used for model selection.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camp_kg.generator import SyntheticKGGenerator


def parse_args():
    p = argparse.ArgumentParser(description="Generate CAMP-KG corpus")
    p.add_argument("--n_kg",      type=int, default=2000,
                   help="Total number of synthetic KGs to generate (default 2000)")
    p.add_argument("--out_dir",   type=str, default="data/corpus/camp_kg",
                   help="Output directory for corpus .pkl files")
    p.add_argument("--n_train",   type=int, default=1800,
                   help="Number of training KGs (rest are held-out for selection)")
    p.add_argument("--seed",      type=int, default=0,
                   help="Base random seed")
    p.add_argument("--max_retries", type=int, default=60,
                   help="Max retries per KG before giving up")
    p.add_argument("--verbose",   action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    train_dir = os.path.join(args.out_dir, "train")
    val_dir   = os.path.join(args.out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    n_val = args.n_kg - args.n_train
    print(f"Generating {args.n_kg} KGs: {args.n_train} train + {n_val} val")
    print(f"Output: {args.out_dir}")
    print()

    gen = SyntheticKGGenerator(seed=args.seed)

    t0 = time.time()
    print(f"--- Training KGs (n={args.n_train}) ---")
    gen.generate_corpus(args.n_train, train_dir, verbose=args.verbose)

    print(f"--- Validation KGs (n={n_val}) ---")
    gen_val = SyntheticKGGenerator(seed=args.seed + 99999)
    gen_val.generate_corpus(n_val, val_dir, verbose=args.verbose)

    elapsed = time.time() - t0
    print(f"\nTotal generation time: {elapsed/60:.1f} minutes")
    print(f"Train: {len(os.listdir(train_dir))} files in {train_dir}")
    print(f"Val:   {len(os.listdir(val_dir))} files in {val_dir}")


if __name__ == "__main__":
    main()
