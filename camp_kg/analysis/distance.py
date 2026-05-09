"""
Structural-distance analysis.

Computes degree-distribution Jensen-Shannon divergence between benchmark
KGs and the three real pretraining KGs (FB15k-237, WN18RR, CoDEx-Medium)
and correlates the divergence with per-KG ∆MRR.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Degree distribution helpers
# ---------------------------------------------------------------------------

def degree_distribution(triples: np.ndarray, n_entities: int, bins: int = 100) -> np.ndarray:
    """
    Compute normalised in+out degree distribution histogram.

    Parameters
    ----------
    triples    : [N, 3] int array (head, rel, tail).
    n_entities : total number of entities.
    bins       : number of histogram bins (log-spaced from 1 to max_degree).

    Returns normalised histogram of shape [bins].
    """
    if len(triples) == 0:
        return np.ones(bins) / bins

    degrees = np.bincount(triples[:, 0], minlength=n_entities) + \
              np.bincount(triples[:, 2], minlength=n_entities)
    degrees = degrees[degrees > 0].astype(float)

    max_deg = max(int(degrees.max()), 2)
    bin_edges = np.logspace(0, np.log10(max_deg + 1), bins + 1)
    hist, _ = np.histogram(degrees, bins=bin_edges)
    hist = hist.astype(float)
    total = hist.sum()
    return hist / total if total > 0 else np.ones(bins) / bins


def js_divergence_degree(
    triples_a: np.ndarray, n_ent_a: int,
    triples_b: np.ndarray, n_ent_b: int,
    bins: int = 100,
) -> float:
    """
    Jensen-Shannon divergence between degree distributions of two KGs.

    Returns a value in [0, 1].
    """
    dist_a = degree_distribution(triples_a, n_ent_a, bins)
    dist_b = degree_distribution(triples_b, n_ent_b, bins)

    # Pad to common support
    n = max(len(dist_a), len(dist_b))
    pa = np.zeros(n); pa[:len(dist_a)] = dist_a
    pb = np.zeros(n); pb[:len(dist_b)] = dist_b

    # Add small epsilon to avoid zero-probability issues
    eps = 1e-10
    pa = pa + eps; pa /= pa.sum()
    pb = pb + eps; pb /= pb.sum()

    return float(jensenshannon(pa, pb))


# ---------------------------------------------------------------------------
# Compute distances to pretraining KGs
# ---------------------------------------------------------------------------

def compute_structural_distances(
    benchmark_triples:   Dict[str, Tuple[np.ndarray, int]],
    pretraining_triples: Dict[str, Tuple[np.ndarray, int]],
    bins: int = 100,
    aggregation: str = "min",   # 'min' | 'mean' | 'fb_only'
) -> Dict[str, float]:
    """
    For each benchmark KG, compute its JS-divergence to the real pretraining
    corpus and return the aggregated distance.

    Parameters
    ----------
    benchmark_triples   : {kg_name: (triples_array, n_entities)}
    pretraining_triples : {kg_name: (triples_array, n_entities)}
                          keys should include 'fb15k237', 'wn18rr', 'codex_m'
    aggregation         : 'min' → min over 3 pretraining KGs (paper default)
                          'mean' → arithmetic mean of 3 distances
                          'fb_only' → distance to FB15k-237 only

    Returns {kg_name: distance}
    """
    distances = {}
    pt_list = list(pretraining_triples.items())

    for kg_name, (triples_b, n_ent_b) in benchmark_triples.items():
        dists = []
        for pt_name, (triples_pt, n_ent_pt) in pt_list:
            if aggregation == "fb_only" and "fb" not in pt_name.lower():
                continue
            d = js_divergence_degree(triples_b, n_ent_b, triples_pt, n_ent_pt, bins)
            dists.append(d)

        if not dists:
            distances[kg_name] = 0.0
        elif aggregation == "min":
            distances[kg_name] = float(np.min(dists))
        else:
            distances[kg_name] = float(np.mean(dists))

    return distances


# ---------------------------------------------------------------------------
# Spearman correlation analysis
# ---------------------------------------------------------------------------

def distance_gain_correlation(
    distances: Dict[str, float],
    deltas:    Dict[str, float],
) -> Dict:
    """
    Compute Spearman ρ between structural distance and per-KG ∆MRR.

    Returns dict with 'rho', 'p_value', 'n'.
    """
    common_kgs = sorted(set(distances.keys()) & set(deltas.keys()))
    if len(common_kgs) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(common_kgs)}

    x = np.array([distances[k] for k in common_kgs])
    y = np.array([deltas[k]    for k in common_kgs])

    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p), "n": len(common_kgs)}


def partial_spearman_correlation(
    distances:    np.ndarray,
    deltas:       np.ndarray,
    covariates:   np.ndarray,
) -> Tuple[float, float]:
    """
    Spearman partial correlation between distances and deltas,
    controlling for covariates (shape [N, K]).

    Uses the residual-rank approach:
    1. Regress distances on covariates → residuals_x
    2. Regress deltas    on covariates → residuals_y
    3. Compute Spearman(residuals_x, residuals_y)
    """
    from scipy.stats import rankdata
    from numpy.linalg import lstsq

    n = len(distances)
    X = np.column_stack([np.ones(n), covariates])

    def residuals(y):
        coef, _, _, _ = lstsq(X, y, rcond=None)
        return y - X @ coef

    res_x = residuals(rankdata(distances).astype(float))
    res_y = residuals(rankdata(deltas).astype(float))

    rho, p = spearmanr(res_x, res_y)
    return float(rho), float(p)
