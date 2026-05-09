# CAMP-KG: Corpus-as-Model Pretraining for Zero-Shot Knowledge Graph Reasoning

Official code for the NeurIPS 2026 submission  
**"Corpus-as-Model Pretraining for Zero-Shot Knowledge Graph Reasoning"**

---

## Overview

CAMP-KG is a data-centric pretraining method for zero-shot knowledge graph (KG)
foundation models. Instead of treating the pretraining corpus as a passive artifact,
CAMP-KG actively designs it via a **factorized synthetic generator** that
independently samples:

- **Rule templates** (7 templates, 3 complexity levels): symmetry, inversion,
  subsumption, composition, intersection, long-chain, and branching.
- **Graph statistics** (6 parameters): entity count, relation count, degree
  distribution exponent, edge density, incompleteness, and noise rate.

A **proof-aware task construction** further prevents rule-premise leakage by
assigning triples to observable/query roles according to their causal status.

The ULTRA backbone (177K parameters) is **unchanged**; only the pretraining
data loader is replaced. CAMP-KG pretraining on 2,000 synthetic KGs:

- Achieves **mean MRR 0.356** over 57 benchmark KGs vs. 0.350 for ULTRA-Real
  (non-inferior, p < 0.001).
- Improves worst-case transfer: **+6.1 MRR points** on the bottom-14 benchmark
  KGs (95% CI: [+0.045, +0.078]).
- Improves 35/57 KGs with bounded regressions (max −0.031) on nearby
  Freebase/WordNet derivatives.

---

## Repository structure

```
camp-kg/
├── camp_kg/
│   ├── generator.py     # Factorized synthetic KG generator (Algorithm 1)
│   ├── corpus.py        # SyntheticCorpus data loader + ULTRA integration
│   ├── train.py         # Training loop (ULTRA backbone + CAMP-KG data)
│   ├── evaluate.py      # Filtered ranking evaluation on benchmark KGs
│   └── analysis/
│       ├── bootstrap.py # Bootstrap CI, non-inferiority test, bottom-k analysis
│       └── distance.py  # JS divergence, Spearman ρ, partial correlations
├── scripts/
│   ├── generate_corpus.py  # Generate the 2,000-KG synthetic corpus
│   ├── train.py            # Train CAMP-KG (wraps ULTRA training)
│   ├── evaluate.py         # Zero-shot eval on all 57 benchmark KGs
│   ├── verify_numbers.py   # Reproduce all numerical claims from CSV
│   └── analyze_results.py  # Structural-distance analysis (Figure 4)
├── configs/
│   ├── camp_kg.yaml          # Full CAMP-KG (2,000 KGs, all templates)
│   ├── ablation_B.yaml       # No rules, varied statistics
│   ├── ablation_C.yaml       # Rules only, fixed statistics
│   ├── ablation_D.yaml       # No rules, fixed statistics
│   ├── control_3cycled.yaml  # 3-CAMP-KG-Cycled
│   └── control_edge_matched.yaml  # Edge-Matched (~312K edges)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

### 1. Install ULTRA

CAMP-KG uses ULTRA's model architecture and evaluation code unchanged.

```bash
git clone https://github.com/DeepGraphLearning/ULTRA
pip install -e ULTRA/
```

### 2. Install CAMP-KG

```bash
git clone <this-repository>
cd camp-kg
pip install -e .
```

### 3. Dependencies

```
numpy>=1.24
scipy>=1.10
torch>=2.0
torch-geometric>=2.3
pyyaml>=6.0
tqdm>=4.65
```

---

## Quick start

### Step 1 — Generate the synthetic corpus

```bash
python scripts/generate_corpus.py \
    --n_kg 2000 \
    --out_dir data/corpus/camp_kg \
    --n_train 1800 \
    --seed 0
```

This creates 1,800 training KGs and 200 held-out selection KGs.
Generation takes approximately **45 ± 5 minutes** on an 8-core CPU
(AMD EPYC 7763) and produces ~12 GB of cached data.

The rejection rate is ~7.2%, so ~2,155 KGs are attempted to yield 2,000 valid
ones. The acceptance filter requires:

- Largest connected component ≥ 50% of entities.
- Observable edges ≤ 500,000.
- ≥ 100 derived training queries.
- ≥ 50 validation queries.

### Step 2 — Train (5 seeds)

```bash
for SEED in 0 1 2 3 4; do
    python scripts/train.py \
        --config configs/camp_kg.yaml \
        --seed $SEED
done
```

Training uses 2× NVIDIA H100 80GB GPUs with distributed data-parallel.
Each seed takes approximately **64 GPU-hours**. See `camp_kg/train.py`
for the training loop and `camp_kg/corpus.py` for the data loader.

### Step 3 — Evaluate on benchmark KGs

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/camp_kg/seed_0/ckpt_best.pt \
    --kg_dir     data/benchmarks \
    --out_csv    results/camp_kg_seed0.csv
```

Evaluation follows the standard filtered ranking protocol: all entities are
scored, known-true triples are filtered before computing rank, and MRR and
Hits@k are averaged over head/tail predictions.

### Step 4 — Verify numerical claims

After collecting all 5-seed results:

```bash
python scripts/verify_numbers.py --csv results/all_results.csv
```

This reproduces all aggregate statistics from Table 1 and the bootstrap
analyses from the paper.

---

## Reproducing ablations and controls

### 2×2 factorization ablation (Table 2)

| Condition | Description | Config |
|-----------|-------------|--------|
| **A** (CAMP-KG-Full) | Rules + varied statistics | `configs/camp_kg.yaml` |
| **B** | No rules, varied statistics | `configs/ablation_B.yaml` |
| **C** | Rules, fixed statistics | `configs/ablation_C.yaml` |
| **D** | No rules, fixed statistics | `configs/ablation_D.yaml` |

First generate the corpus for each ablation condition:

```bash
# Ablation B: no rules — set use_rules=false in generator
python scripts/generate_corpus.py --n_kg 2000 \
    --out_dir data/corpus/ablation_B --seed 1 --no_rules

# Ablation C: fixed statistics
python scripts/generate_corpus.py --n_kg 2000 \
    --out_dir data/corpus/ablation_C --seed 2 --fixed_stats

# Ablation D: no rules + fixed statistics
python scripts/generate_corpus.py --n_kg 2000 \
    --out_dir data/corpus/ablation_D --seed 3 --no_rules --fixed_stats

# Then train each condition (5 seeds each)
for COND in ablation_B ablation_C ablation_D; do
    for SEED in 0 1 2 3 4; do
        python scripts/train.py --config configs/${COND}.yaml --seed $SEED
    done
done
```

### Corpus-level controls (Table 3)

```bash
# 3-CAMP-KG-Cycled
python scripts/generate_corpus.py --n_kg 3 \
    --out_dir data/corpus/control_3cycled --seed 10

# Edge-Matched (~312K total edges, 6 KGs)
python scripts/generate_corpus.py --n_kg 6 \
    --out_dir data/corpus/control_edge_matched --seed 11
```

### ULTRA-Real baseline (reproduced)

Train ULTRA on the original three real KGs (FB15k-237, WN18RR, CoDEx-Medium)
using ULTRA's official training script with published hyperparameters:

```bash
# From the ULTRA repository:
python train.py --config configs/ultra_config.yaml --seed 0
```

Our reproduced ULTRA-Real achieves mean MRR 0.350 ± 0.005 (5 seeds), within
1 MRR point of the published value.

---

## Structural distance analysis (Figure 4)

After running both CAMP-KG and ULTRA-Real evaluations:

```bash
python scripts/analyze_results.py \
    --results_csv  results/all_results.csv \
    --kg_dir       data/benchmarks \
    --pretraining_dir data/pretraining \
    --out_dir      results/analysis
```

This computes degree-distribution JS divergence between each benchmark KG and
the three pretraining KGs and reports Spearman ρ between structural distance
and per-KG ∆MRR:

- **min-of-3**: ρ = 0.61 (p < 0.001)
- **FB15k-237 only**: ρ = 0.64 (p < 0.001)
- **mean-of-3**: ρ = 0.63 (p < 0.001)

---

## Method

### Factorized synthetic KG generator

The generator independently samples two axes per KG:

**Rule axis (7 templates):**

| ID | Name | Formula |
|----|------|---------|
| T1 | Symmetry | r(x,y) → r(y,x) |
| T2 | Inversion | r₁(x,y) → r₂(y,x) |
| T3 | Subsumption | r₁(x,y) → r₂(x,y) |
| T4 | Composition | r₁(x,z) ∧ r₂(z,y) → r₃(x,y) |
| T5 | Intersection | r₁(x,y) ∧ r₂(x,y) → r₃(x,y) |
| T6 | Long Chain | r₁(x,a) ∧ r₂(a,b) ∧ r₃(b,y) → r₄(x,y) |
| T7 | Branching | r₁(x,z) ∧ r₂(y,z) → r₃(x,y) |

Templates are sampled independently per KG: L1 (T1–T3) with 70% probability
each, L2 (T4–T5) with 62%, L3 (T6–T7) with 44%. Expected: ~4.2 templates/KG.

**Statistics axis (6 parameters):**

| Parameter | Distribution |
|-----------|-------------|
| n_entities | LogUniform(100, 30,000) |
| n_relations | LogUniform(6, 192) |
| degree exponent γ | Uniform(1.5, 3.0); 15% chance uniform |
| edge density μ | LogUniform(3, 50) |
| incompleteness η | Uniform(0, 0.5) |
| noise rate ν | Uniform(0, 0.10) |

### Proof-aware task construction

Triples are split by causal status to prevent rule-premise leakage:

- **Body triples**: always observable, never targets.
- **Derived triples** (rule conclusions): split 60/20/20 → d_train, d_val, d_test;
  d_val and d_test are excluded from the observable graph at their target stage.
- **Background + noise**: split 80/10/10 → b_train, b_val, b_test.

Observable graphs at each stage:
- E_obs_train = body ∪ d_train ∪ b_train (+inverses)
- E_obs_eval = E_obs_train  (no val targets added)
- E_obs_test = E_obs_train ∪ d_val ∪ b_val (+inverses)

Leakage is asserted explicitly: Q_val ∩ E_obs_eval = ∅ and Q_test ∩ E_obs_test = ∅.

Each 64-query training batch contains **32 derived + 32 background** queries
(50:50 optimal; see sensitivity analysis in the paper).

### Relation graph (ULTRA's Lift function)

The relation graph connects relation nodes via four entity co-occurrence types:
head–tail (0), head–head (1), tail–head (2), tail–tail (3). See `camp_kg/generator.py:lift()`.

---

## Compute budget

| Experiment | GPU-hours |
|------------|-----------|
| Main comparison (2 methods × 5 seeds) | 560 |
| 2×2 ablation (4 conditions × 5 seeds) | 1,280 |
| Corpus-level controls (2 × 5 seeds) | 640 |
| Template ablations (13 configs × 5 seeds) | 4,160 |
| Sensitivity analysis (8 configs × 5 seeds) | 2,560 |
| Evaluation | ~100 |
| **Total** | **~9,300** |

Hardware: 2× NVIDIA H100 80GB GPUs (distributed data-parallel).  
Corpus generation: ~45 minutes on 8-core CPU (no GPU required).

---

## Key results

**Table 1 — Aggregate and worst-case performance (57 benchmark KGs, 5 seeds):**

| | Mean MRR | H@1 | H@10 | Bottom-14 MRR |
|---|---|---|---|---|
| ULTRA-Real | 0.350 ± 0.005 | 0.243 | 0.521 | 0.128 |
| **CAMP-KG** | **0.356 ± 0.004** | **0.248** | **0.528** | **0.189** |
| Δ | **+0.006** | +0.005 | +0.007 | **+0.061** |

Non-inferiority p < 0.001 (paired one-sided t-test, δ = 0.02, 5 seeds).  
Bottom-14 bootstrap 95% CI: [+0.045, +0.078].

**Table 2 — 2×2 factorization ablation:**

| Condition | Rules | Varied Stats | Mean MRR | Bottom-14 |
|-----------|-------|-------------|----------|-----------|
| A (CAMP-KG-Full) | ✓ | ✓ | **0.356** | **0.189** |
| C | ✓ | ✗ (fixed) | 0.321 | 0.161 |
| B | ✗ (random) | ✓ | 0.295 | 0.145 |
| D | ✗ | ✗ | 0.267 | 0.133 |
| ULTRA-Real | — | — | 0.350 | 0.128 |

---

## Benchmark datasets

All 57 benchmark KGs follow the standard inductive split format.
Original citations:

- **FB15k-237**: Toutanova & Chen (2015)
- **WN18RR**: Dettmers et al. (2018)
- **CoDEx**: Safavi & Koutra (2020)
- **NELL-995**: Xiong et al. (2017)
- **Hetionet**: Himmelstein et al. (2017)
- **InGram benchmarks**: Lee et al. (2023)

---

## Citation

```bibtex
@inproceedings{campkg2026,
  title     = {Corpus-as-Model Pretraining for Zero-Shot Knowledge Graph Reasoning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```

---

## License

MIT License. See `LICENSE` for details.
