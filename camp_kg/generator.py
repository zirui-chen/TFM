"""
CAMP-KG: Factorized Synthetic KG Generator
Implements Algorithm 1 from the paper.

The generator independently samples rule templates and graph statistics,
applies proof-aware splitting, and builds cached relation graphs.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class KGSplit:
    """
    Proof-aware split of one synthetic KG.

    Observable graphs include inverse triples (added after splitting).
    Query arrays contain canonical triples only (no inverses).
    Relation IDs 0..n_relations-1 are canonical;
    n_relations..2*n_relations-1 are their inverses.
    """

    # Observable edge arrays, shape [N, 3]: (head, rel, tail)
    e_obs_train: np.ndarray   # body ∪ d_train ∪ b_train  (+inverses)
    e_obs_eval:  np.ndarray   # = e_obs_train              (+inverses)
    e_obs_test:  np.ndarray   # e_obs_train ∪ d_val ∪ b_val (+inverses)

    # Query arrays (targets; canonical only, no inverses)
    q_train: np.ndarray   # d_train ∪ b_train
    q_val:   np.ndarray   # d_val   ∪ b_val
    q_test:  np.ndarray   # d_test  ∪ b_test

    # Sub-splits for balanced training batches
    q_derived_train: np.ndarray   # d_train
    q_bg_train:      np.ndarray   # b_train

    # All known-true triples (for filtered negative sampling)
    filter_triples: np.ndarray    # shape [M, 3]

    # Graph metadata
    n_entities:  int
    n_relations: int   # canonical (non-inverse); total = 2 * n_relations

    # Precomputed relation graphs (populated by build_relation_graphs())
    gr_train: Optional[Dict] = None
    gr_eval:  Optional[Dict] = None
    gr_test:  Optional[Dict] = None

    # Generator metadata (informational only)
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Relation-graph construction (ULTRA's Lift function)
# ---------------------------------------------------------------------------

def lift(triples: np.ndarray, n_relations: int) -> Dict:
    """
    Build a relation graph from a triple array.

    Nodes: relation IDs 0..2*n_relations-1 (canonical + inverses).
    Edges: four entity co-occurrence interaction types:
        0  head–tail   (r1's head entity == r2's tail entity)
        1  head–head   (r1 and r2 share a head entity)
        2  tail–head   (r1's tail entity == r2's head entity)
        3  tail–tail   (r1 and r2 share a tail entity)

    Returns a dict with keys:
        'edge_index' : np.ndarray, shape [2, E], dtype int64
        'edge_type'  : np.ndarray, shape [E],    dtype int64
        'num_nodes'  : int  (= 2 * n_relations)

    Algorithm: iterate over O(n_rel²) relation pairs rather than O(n_ent)
    entities — fast even for large KGs because n_rel is bounded (≤ 384).
    """
    n_total = 2 * n_relations

    if len(triples) == 0:
        return {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_type":  np.zeros(0,       dtype=np.int64),
            "num_nodes":  n_total,
        }

    heads = triples[:, 0].astype(np.int64)
    rels  = triples[:, 1].astype(np.int64)
    tails = triples[:, 2].astype(np.int64)

    valid = (rels >= 0) & (rels < n_total)
    heads, rels, tails = heads[valid], rels[valid], tails[valid]

    # --- Build per-relation entity sets (sort once, group by relation) ---
    order    = np.argsort(rels, kind="stable")
    h_s      = heads[order]
    r_s      = rels[order]
    t_s      = tails[order]

    u_rels, starts = np.unique(r_s, return_index=True)
    ends = np.concatenate([starts[1:], [len(r_s)]])

    head_sets: Dict[int, Set[int]] = {}
    tail_sets: Dict[int, Set[int]] = {}
    for r_id, s, e in zip(u_rels.tolist(), starts.tolist(), ends.tolist()):
        head_sets[r_id] = set(h_s[s:e].tolist())
        tail_sets[r_id] = set(t_s[s:e].tolist())

    # --- Check all O(n_rel²) relation pairs for each interaction type ---
    rel_list = sorted(head_sets.keys())
    src_list, dst_list, typ_list = [], [], []

    for r1 in rel_list:
        h1 = head_sets[r1]
        t1 = tail_sets[r1]
        for r2 in rel_list:
            h2 = head_sets[r2]
            t2 = tail_sets[r2]
            # isdisjoint short-circuits on first common element → O(1) for dense KGs
            if not h1.isdisjoint(h2):   # type 1: head–head
                src_list.append(r1); dst_list.append(r2); typ_list.append(1)
            if not t1.isdisjoint(t2):   # type 3: tail–tail
                src_list.append(r1); dst_list.append(r2); typ_list.append(3)
            if not h1.isdisjoint(t2):   # type 0: head–tail
                src_list.append(r1); dst_list.append(r2); typ_list.append(0)
            if not t1.isdisjoint(h2):   # type 2: tail–head
                src_list.append(r1); dst_list.append(r2); typ_list.append(2)

    if not src_list:
        return {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_type":  np.zeros(0,       dtype=np.int64),
            "num_nodes":  n_total,
        }

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_type  = np.array(typ_list,             dtype=np.int64)

    return {"edge_index": edge_index, "edge_type": edge_type, "num_nodes": n_total}
    n_total = 2 * n_relations

    if len(triples) == 0:
        return {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_type":  np.zeros(0,       dtype=np.int64),
            "num_nodes":  n_total,
        }

    heads = triples[:, 0].astype(np.int64)
    rels  = triples[:, 1].astype(np.int64)
    tails = triples[:, 2].astype(np.int64)

    # Restrict to valid relation IDs
    valid = (rels >= 0) & (rels < n_total)
    heads, rels, tails = heads[valid], rels[valid], tails[valid]

    src_parts, dst_parts, typ_parts = [], [], []

    # interaction type 1: head–head (two relations share a head entity)
    # For each entity e, get all relations r with head e → add (r1, r2, type=1)
    order = np.argsort(heads)
    h_sorted = heads[order]; r_sorted = rels[order]
    # group by head entity
    unique_h, starts = np.unique(h_sorted, return_index=True)
    ends = np.concatenate([starts[1:], [len(h_sorted)]])
    for s, e_ in zip(starts, ends):
        rs = r_sorted[s:e_]
        if len(rs) > 1 and len(rs) <= 200:   # cap: skip degenerate hubs
            grid_r1, grid_r2 = np.meshgrid(rs, rs)
            src_parts.append(grid_r1.ravel())
            dst_parts.append(grid_r2.ravel())
            typ_parts.append(np.full(grid_r1.size, 1, dtype=np.int64))

    # interaction type 3: tail–tail (two relations share a tail entity)
    order = np.argsort(tails)
    t_sorted = tails[order]; r_sorted_t = rels[order]
    unique_t, starts = np.unique(t_sorted, return_index=True)
    ends = np.concatenate([starts[1:], [len(t_sorted)]])
    for s, e_ in zip(starts, ends):
        rs = r_sorted_t[s:e_]
        if len(rs) > 1 and len(rs) <= 200:
            grid_r1, grid_r2 = np.meshgrid(rs, rs)
            src_parts.append(grid_r1.ravel())
            dst_parts.append(grid_r2.ravel())
            typ_parts.append(np.full(grid_r1.size, 3, dtype=np.int64))

    # interaction type 0: head–tail (r1 head == r2 tail)
    # For each entity e: relations where e is head × relations where e is tail
    all_entities = np.union1d(heads, tails)
    # head-indexed: entity → set of rels
    order_h = np.argsort(heads)
    h_s = heads[order_h]; r_h = rels[order_h]
    head_map: Dict[int, np.ndarray] = {}
    uniq_h, idx_h = np.unique(h_s, return_index=True)
    end_h = np.concatenate([idx_h[1:], [len(h_s)]])
    for e, s, e_ in zip(uniq_h, idx_h, end_h):
        rs = r_h[s:e_]
        if len(rs) <= 200:
            head_map[int(e)] = rs

    order_t = np.argsort(tails)
    t_s = tails[order_t]; r_t = rels[order_t]
    tail_map: Dict[int, np.ndarray] = {}
    uniq_t, idx_t = np.unique(t_s, return_index=True)
    end_t = np.concatenate([idx_t[1:], [len(t_s)]])
    for e, s, e_ in zip(uniq_t, idx_t, end_t):
        rs = r_t[s:e_]
        if len(rs) <= 200:
            tail_map[int(e)] = rs

    for e in np.intersect1d(uniq_h, uniq_t):
        hr = head_map.get(int(e))
        tr2 = tail_map.get(int(e))
        if hr is None or tr2 is None:
            continue
        grid_r1, grid_r2 = np.meshgrid(hr, tr2)
        src_parts.append(grid_r1.ravel())   # r1 has entity e as head
        dst_parts.append(grid_r2.ravel())   # r2 has entity e as tail
        typ_parts.append(np.full(grid_r1.size, 0, dtype=np.int64))  # head–tail

    # interaction type 2: tail–head (r1 tail == r2 head)
    for e in np.intersect1d(uniq_t, uniq_h):
        tr2 = tail_map.get(int(e))
        hr  = head_map.get(int(e))
        if tr2 is None or hr is None:
            continue
        grid_r1, grid_r2 = np.meshgrid(tr2, hr)
        src_parts.append(grid_r1.ravel())   # r1 has entity e as tail
        dst_parts.append(grid_r2.ravel())   # r2 has entity e as head
        typ_parts.append(np.full(grid_r1.size, 2, dtype=np.int64))  # tail–head

    if not src_parts:
        return {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_type":  np.zeros(0,       dtype=np.int64),
            "num_nodes":  n_total,
        }

    srcs = np.concatenate(src_parts).astype(np.int64)
    dsts = np.concatenate(dst_parts).astype(np.int64)
    typs = np.concatenate(typ_parts).astype(np.int64)

    # Deduplicate
    combined = np.stack([srcs, dsts, typs], axis=1)
    combined = np.unique(combined, axis=0)
    edge_index = combined[:, :2].T.astype(np.int64)
    edge_type  = combined[:,  2].astype(np.int64)

    return {"edge_index": edge_index, "edge_type": edge_type, "num_nodes": n_total}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SyntheticKGGenerator:
    """
    Factorized Synthetic KG Generator for CAMP-KG.

    Independently samples rule templates (L1–L3, T1–T7) and six graph-
    statistic parameters, generates body triples via configuration model,
    applies templates to derive conclusions, simulates incompleteness and
    noise, and performs proof-aware (leakage-free) splitting.

    Usage
    -----
    gen = SyntheticKGGenerator(seed=42)
    kg  = gen.generate_kg()   # returns KGSplit or None if degenerate
    """

    # ------------------------------------------------------------------ #
    # Template metadata                                                    #
    # ------------------------------------------------------------------ #

    # Inclusion probability per complexity level
    _LEVEL_PROB = {1: 0.70, 2: 0.62, 3: 0.44}

    # Template → level
    _TEMPLATE_LEVEL = {
        "T1": 1, "T2": 1, "T3": 1,
        "T4": 2, "T5": 2,
        "T6": 3, "T7": 3,
    }

    # Number of body relations + 1 head relation per template
    _TEMPLATE_REL_NEEDS = {
        "T1": 2,   # 1 body + 1 head
        "T2": 2,
        "T3": 2,
        "T4": 3,   # 2 body + 1 head
        "T5": 3,
        "T6": 4,   # 3 body + 1 head
        "T7": 3,
    }

    # ------------------------------------------------------------------ #
    # Rejection-filter thresholds                                         #
    # ------------------------------------------------------------------ #
    _MAX_OBS_EDGES      = 500_000
    _MIN_DERIVED_TRAIN  = 100
    _MIN_Q_VAL          = 50
    _MIN_LCC_FRACTION   = 0.5

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # ================================================================== #
    # Public API                                                          #
    # ================================================================== #

    def generate_kg(self, max_retries: int = 60) -> Optional[KGSplit]:
        """
        Generate one synthetic KG with proof-aware split.
        Retries on degenerate KGs; returns None after max_retries failures.
        """
        for _ in range(max_retries):
            result = self._generate_once()
            if result is not None:
                return result
        return None

    def generate_corpus(
        self,
        n_kg: int,
        out_dir: str,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> List[str]:
        """
        Generate a corpus of n_kg synthetic KGs and save to out_dir.

        Returns list of saved file paths (one pickle per KG).
        """
        os.makedirs(out_dir, exist_ok=True)
        paths, generated, attempts = [], 0, 0

        while generated < n_kg:
            attempts += 1
            kg = self.generate_kg()
            if kg is None:
                continue

            path = os.path.join(out_dir, f"kg_{generated:05d}.pkl")
            with open(path, "wb") as f:
                pickle.dump(kg, f, protocol=4)
            paths.append(path)
            generated += 1

            if verbose and generated % 100 == 0:
                rate = generated / attempts
                print(f"  Generated {generated}/{n_kg} KGs "
                      f"(acceptance rate {rate:.2%})")

        if verbose:
            print(f"Done. Acceptance rate: {generated / attempts:.2%}")
        return paths

    # ================================================================== #
    # Step 1 – sample graph statistics                                    #
    # ================================================================== #

    def _sample_statistics(self) -> Dict:
        n_ent = self._log_uniform(100, 30_000)
        n_rel = self._log_uniform(6, 192)

        if self.rng.random() < 0.15:
            gamma = None          # uniform degree (null model)
        else:
            gamma = float(self.rng.uniform(1.5, 3.0))

        edge_density   = float(self._log_uniform_f(3.0, 50.0))
        incompleteness = float(self.rng.uniform(0.0, 0.5))
        noise_rate     = float(self.rng.uniform(0.0, 0.10))

        return dict(
            n_ent=n_ent, n_rel=n_rel, gamma=gamma,
            edge_density=edge_density,
            incompleteness=incompleteness,
            noise_rate=noise_rate,
        )

    # ================================================================== #
    # Step 2 – sample templates independently                             #
    # ================================================================== #

    def _sample_templates(self) -> List[str]:
        selected = [
            name for name, lvl in self._TEMPLATE_LEVEL.items()
            if self.rng.random() < self._LEVEL_PROB[lvl]
        ]
        if not selected:
            selected = ["T4"]     # always keep ≥ 1 template
        return selected

    # ================================================================== #
    # Step 3 – assign relations to template slots                         #
    # ================================================================== #

    def _assign_relations(
        self, templates: List[str], n_rel: int
    ) -> Tuple[Dict[str, Dict], List[int]]:
        """
        Assign relation IDs to template body/head slots.

        Head relations (rule conclusions) are never reused across templates.
        Body relations may be shared when n_rel is small.
        Returns (template_rels, background_rels).
        """
        n_heads  = len(templates)
        all_rels = list(range(n_rel))

        # Assign unique head relations
        if n_heads <= n_rel:
            head_ids = self.rng.choice(n_rel, size=n_heads, replace=False).tolist()
        else:
            head_ids = self.rng.choice(n_rel, size=n_heads, replace=True).tolist()

        used_as_head = set(head_ids)

        # Track all relations used in rule bodies so background stays disjoint
        used_in_rules: Set[int] = set(used_as_head)

        template_rels: Dict[str, Dict] = {}
        for t, head_rel in zip(templates, head_ids):
            n_body = self._TEMPLATE_REL_NEEDS[t] - 1
            # Body relations: prefer non-head, non-already-assigned-body;
            # fall back to all non-head rels if scarce
            avail = [r for r in all_rels if r not in used_as_head]
            pool  = [r for r in avail if r not in used_in_rules] or avail or all_rels
            if n_body <= len(pool):
                body = self.rng.choice(pool, size=n_body, replace=False).tolist()
            else:
                body = self.rng.choice(pool, size=n_body, replace=True).tolist()
            used_in_rules.update(body)
            template_rels[t] = {"body": body, "head": head_rel}

        # Background relations are strictly disjoint from all rule relations
        background_rels = [r for r in all_rels if r not in used_in_rules]
        return template_rels, background_rels

    # ================================================================== #
    # Step 4 – generate body triples (configuration model)               #
    # ================================================================== #

    def _generate_body_triples(
        self,
        rel_id: int,
        n_ent:  int,
        edge_density: float,
        gamma: Optional[float],
    ) -> np.ndarray:
        """
        Generate edges for one relation using a configuration-model-style
        sampler with power-law (or uniform) degree weights.

        Returns array of shape [N, 3]: (head, rel, tail).
        """
        n_target = max(int(n_ent * edge_density), 5)

        out_w = self._degree_weights(n_ent, gamma)
        in_w  = self._degree_weights(n_ent, gamma)

        out_p = out_w / out_w.sum()
        in_p  = in_w  / in_w.sum()

        # Oversample to compensate for self-loop/duplicate removal
        n_sample = int(n_target * 1.4) + 20
        heads = self.rng.choice(n_ent, size=n_sample, p=out_p)
        tails = self.rng.choice(n_ent, size=n_sample, p=in_p)
        rels  = np.full(n_sample, rel_id, dtype=np.int64)

        edges = np.stack([heads, rels, tails], axis=1)
        edges = edges[edges[:, 0] != edges[:, 2]]   # remove self-loops
        edges = np.unique(edges, axis=0)             # remove duplicates

        return edges[:n_target] if len(edges) >= n_target else edges

    def _degree_weights(self, n: int, gamma: Optional[float]) -> np.ndarray:
        """Power-law (Pareto) or uniform degree weights."""
        if gamma is None:
            return np.ones(n, dtype=np.float64)
        # Pareto with shape (gamma-1): P(x) ∝ x^{-gamma} for x >= 1
        raw = self.rng.pareto(gamma - 1.0, n) + 1.0
        return raw

    # ================================================================== #
    # Step 5 – apply rule templates                                       #
    # ================================================================== #

    def _apply_T1(self, body: np.ndarray, head_rel: int) -> np.ndarray:
        """T1 Symmetry: r(x,y) → r(y,x)"""
        if len(body) == 0:
            return _empty()
        return np.stack([body[:, 2],
                         np.full(len(body), head_rel, dtype=np.int64),
                         body[:, 0]], axis=1)

    def _apply_T2(self, body: np.ndarray, head_rel: int) -> np.ndarray:
        """T2 Inversion: r1(x,y) → r2(y,x)"""
        if len(body) == 0:
            return _empty()
        return np.stack([body[:, 2],
                         np.full(len(body), head_rel, dtype=np.int64),
                         body[:, 0]], axis=1)

    def _apply_T3(self, body: np.ndarray, head_rel: int) -> np.ndarray:
        """T3 Subsumption: r1(x,y) → r2(x,y)"""
        if len(body) == 0:
            return _empty()
        return np.stack([body[:, 0],
                         np.full(len(body), head_rel, dtype=np.int64),
                         body[:, 2]], axis=1)

    def _apply_T4(
        self, body1: np.ndarray, body2: np.ndarray, head_rel: int
    ) -> np.ndarray:
        """T4 Composition: r1(x,z) ∧ r2(z,y) → r3(x,y)"""
        if len(body1) == 0 or len(body2) == 0:
            return _empty()
        # Index body2 by head entity (z position)
        b2_by_head: Dict[int, List[int]] = defaultdict(list)
        for h, _, t in body2:
            b2_by_head[int(h)].append(int(t))
        rows = []
        for x, _, z in body1:
            ys = b2_by_head.get(int(z))
            if ys:
                for y in ys:
                    rows.append((int(x), head_rel, y))
        return _from_rows(rows)

    def _apply_T5(
        self, body1: np.ndarray, body2: np.ndarray, head_rel: int
    ) -> np.ndarray:
        """T5 Intersection: r1(x,y) ∧ r2(x,y) → r3(x,y)"""
        if len(body1) == 0 or len(body2) == 0:
            return _empty()
        set1 = {(int(h), int(t)) for h, _, t in body1}
        set2 = {(int(h), int(t)) for h, _, t in body2}
        return _from_rows([(x, head_rel, y) for x, y in set1 & set2])

    def _apply_T6(
        self,
        body1: np.ndarray, body2: np.ndarray, body3: np.ndarray,
        head_rel: int,
    ) -> np.ndarray:
        """T6 Long Chain: r1(x,a) ∧ r2(a,b) ∧ r3(b,y) → r4(x,y)"""
        mid = self._apply_T4(body1, body2, -1)
        if len(mid) == 0:
            return _empty()
        return self._apply_T4(mid, body3, head_rel)

    def _apply_T7(
        self, body1: np.ndarray, body2: np.ndarray, head_rel: int
    ) -> np.ndarray:
        """T7 Branching: r1(x,z) ∧ r2(y,z) → r3(x,y)"""
        if len(body1) == 0 or len(body2) == 0:
            return _empty()
        # Index body2 by tail entity (z position)
        b2_by_tail: Dict[int, List[int]] = defaultdict(list)
        for y, _, z in body2:
            b2_by_tail[int(z)].append(int(y))
        rows = []
        for x, _, z in body1:
            ys = b2_by_tail.get(int(z))
            if ys:
                for y in ys:
                    if x != y:
                        rows.append((int(x), head_rel, int(y)))
        return _from_rows(rows)

    def _dispatch_template(
        self,
        name: str,
        template_rels: Dict,
        body_triples_by_rel: Dict[int, np.ndarray],
    ) -> np.ndarray:
        rels    = template_rels[name]
        body_rs = rels["body"]
        head_r  = rels["head"]

        def get(r: int) -> np.ndarray:
            return body_triples_by_rel.get(r, _empty())

        if name == "T1":
            return self._apply_T1(get(body_rs[0]), head_r)
        if name == "T2":
            return self._apply_T2(get(body_rs[0]), head_r)
        if name == "T3":
            return self._apply_T3(get(body_rs[0]), head_r)
        if name == "T4":
            return self._apply_T4(get(body_rs[0]), get(body_rs[1]), head_r)
        if name == "T5":
            return self._apply_T5(get(body_rs[0]), get(body_rs[1]), head_r)
        if name == "T6":
            return self._apply_T6(get(body_rs[0]), get(body_rs[1]), get(body_rs[2]), head_r)
        if name == "T7":
            return self._apply_T7(get(body_rs[0]), get(body_rs[1]), head_r)
        raise ValueError(f"Unknown template: {name}")

    # ================================================================== #
    # Main generation loop                                                #
    # ================================================================== #

    def _generate_once(self) -> Optional[KGSplit]:
        # ---- Step 1: statistics ----------------------------------------
        stats = self._sample_statistics()
        n_ent         = stats["n_ent"]
        n_rel         = stats["n_rel"]
        gamma         = stats["gamma"]
        edge_density  = stats["edge_density"]
        incompleteness = stats["incompleteness"]
        noise_rate    = stats["noise_rate"]

        # ---- Step 2: templates -----------------------------------------
        templates = self._sample_templates()

        # ---- Step 3: assign relations -----------------------------------
        template_rels, background_rels = self._assign_relations(templates, n_rel)

        # ---- Early size check (before expensive generation) ------------
        # Estimate total edges: body + background (background at 0.5× density)
        body_rels_needed: Set[int] = set()
        for t in templates:
            for r in template_rels[t]["body"]:
                body_rels_needed.add(r)
        n_body_rels = max(len(body_rels_needed), 1)
        n_bg_rels   = max(len(background_rels), 0)
        est_edges = (n_ent * edge_density * n_body_rels
                     + n_ent * edge_density * 0.5 * n_bg_rels)
        if est_edges > self._MAX_OBS_EDGES * 4:
            return None   # quick reject: would exceed 500K even after splits

        # ---- Step 4: body triples ---------------------------------------
        body_by_rel: Dict[int, np.ndarray] = {
            r: self._generate_body_triples(r, n_ent, edge_density, gamma)
            for r in body_rels_needed
        }
        all_body = _concat_unique(list(body_by_rel.values()))

        # ---- Step 5: derived triples ------------------------------------
        derived_parts = [
            self._dispatch_template(t, template_rels, body_by_rel)
            for t in templates
        ]
        all_derived = _concat_unique(derived_parts)

        # ---- Step 6a: incompleteness ------------------------------------
        removed_derived = _empty()
        if len(all_derived) > 0 and incompleteness > 0.0:
            n_rm = int(len(all_derived) * incompleteness)
            if n_rm > 0:
                rm_idx   = self.rng.choice(len(all_derived), size=n_rm, replace=False)
                keep_mask = np.ones(len(all_derived), dtype=bool)
                keep_mask[rm_idx] = False
                removed_derived = all_derived[~keep_mask]
                all_derived     = all_derived[keep_mask]

        # ---- Step 6b: background triples --------------------------------
        bg_parts = [
            self._generate_body_triples(r, n_ent, edge_density * 0.5, gamma)
            for r in background_rels
        ]
        all_bg = _concat_unique(bg_parts)

        # ---- Step 6c: noise triples -------------------------------------
        # Noise must use only background relation IDs to prevent category overlap
        n_noise = int(len(all_bg) * noise_rate)
        if n_noise > 0 and background_rels:
            nh = self.rng.integers(0, n_ent, n_noise)
            nr = self.rng.choice(background_rels, size=n_noise)
            nt = self.rng.integers(0, n_ent, n_noise)
            noise = np.stack([nh, nr, nt], axis=1)
            noise = noise[noise[:, 0] != noise[:, 2]]
            all_bg = _concat_unique([all_bg, noise])

        # ---- Step 7: proof-aware split ----------------------------------
        # Derived → 60/20/20
        d_train, d_val, d_test = _split_array(all_derived, [0.6, 0.2, 0.2], self.rng)
        # Background+noise → 80/10/10
        b_train, b_val, b_test = _split_array(all_bg,      [0.8, 0.1, 0.1], self.rng)

        # Observable graphs (before inverses)
        e_train_pre = _concat_unique([all_body, d_train, b_train])
        e_eval_pre  = e_train_pre                              # no val targets
        e_test_pre  = _concat_unique([e_train_pre, d_val, b_val])

        # Query sets
        q_val  = _concat_unique([d_val,  b_val])
        q_test = _concat_unique([d_test, b_test])

        # ---- Step 7b: leakage assertions --------------------------------
        if len(q_val) > 0 and len(e_eval_pre) > 0:
            val_set  = _triple_set(q_val)
            eval_obs = _triple_set(e_eval_pre)
            if val_set & eval_obs:
                raise AssertionError("Validation leakage detected!")

        if len(q_test) > 0 and len(e_test_pre) > 0:
            test_set  = _triple_set(q_test)
            test_obs  = _triple_set(e_test_pre)
            if test_set & test_obs:
                raise AssertionError("Test leakage detected!")

        # ---- Step 8: rejection filter -----------------------------------
        lcc = _lcc_size(e_train_pre, n_ent)
        if lcc < self._MIN_LCC_FRACTION * n_ent:
            return None
        if len(e_train_pre) > self._MAX_OBS_EDGES:
            return None
        if len(d_train) < self._MIN_DERIVED_TRAIN:
            return None
        if len(q_val) < self._MIN_Q_VAL:
            return None

        # ---- Add inverse triples to observable graphs -------------------
        e_obs_train = _add_inverses(e_train_pre, n_rel)
        # e_obs_eval = e_obs_train (no val targets added — build once, reuse)
        e_obs_eval  = e_obs_train
        e_obs_test  = _add_inverses(e_test_pre,  n_rel)

        # ---- Filter set for negative sampling ---------------------------
        filter_triples = _concat_unique([
            all_body, all_derived, removed_derived, all_bg,
        ])

        q_train = _concat_unique([d_train, b_train])

        # ---- Step 9: build and cache relation graphs --------------------
        gr_train = lift(e_obs_train, n_rel)
        gr_eval  = gr_train            # same observable graph → same relation graph
        gr_test  = lift(e_obs_test,  n_rel)

        meta = dict(
            n_ent=n_ent, n_rel=n_rel, gamma=gamma,
            edge_density=edge_density,
            incompleteness=incompleteness,
            noise_rate=noise_rate,
            templates=templates,
        )

        return KGSplit(
            e_obs_train=e_obs_train,
            e_obs_eval=e_obs_eval,
            e_obs_test=e_obs_test,
            q_train=q_train,
            q_val=q_val,
            q_test=q_test,
            q_derived_train=d_train,
            q_bg_train=b_train,
            filter_triples=filter_triples,
            n_entities=n_ent,
            n_relations=n_rel,
            gr_train=gr_train,
            gr_eval=gr_eval,
            gr_test=gr_test,
            meta=meta,
        )

    # ================================================================== #
    # Helpers                                                             #
    # ================================================================== #

    def _log_uniform(self, lo: int, hi: int) -> int:
        return int(round(np.exp(self.rng.uniform(np.log(lo), np.log(hi)))))

    def _log_uniform_f(self, lo: float, hi: float) -> float:
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))


# ---------------------------------------------------------------------------
# Module-level helpers (not bound to the class for clarity)
# ---------------------------------------------------------------------------

def _empty() -> np.ndarray:
    return np.empty((0, 3), dtype=np.int64)


def _from_rows(rows: List[Tuple]) -> np.ndarray:
    if not rows:
        return _empty()
    return np.array(rows, dtype=np.int64)


def _concat_unique(arrays: List[np.ndarray]) -> np.ndarray:
    non_empty = [a for a in arrays if len(a) > 0]
    if not non_empty:
        return _empty()
    merged = np.concatenate(non_empty, axis=0)
    return np.unique(merged.astype(np.int64), axis=0)


def _triple_set(triples: np.ndarray) -> Set[Tuple[int, int, int]]:
    return {(int(h), int(r), int(t)) for h, r, t in triples}


def _split_array(
    arr: np.ndarray,
    fractions: List[float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, ...]:
    """Shuffle and split arr into len(fractions) parts."""
    if len(arr) == 0:
        return tuple(_empty() for _ in fractions)
    idx = rng.permutation(len(arr))
    arr = arr[idx]
    cum = np.cumsum(fractions[:-1])
    splits = [int(c * len(arr)) for c in cum]
    parts = np.split(arr, splits)
    return tuple(p if len(p) > 0 else _empty() for p in parts)


def _add_inverses(triples: np.ndarray, n_rel: int) -> np.ndarray:
    """Append inverse triples (t, r+n_rel, h) to each (h, r, t)."""
    if len(triples) == 0:
        return triples
    inv = np.stack([triples[:, 2],
                    triples[:, 1] + n_rel,
                    triples[:, 0]], axis=1).astype(np.int64)
    return np.concatenate([triples.astype(np.int64), inv], axis=0)


def _lcc_size(triples: np.ndarray, n_ent: int) -> int:
    """
    Largest connected component size in the undirected graph
    induced by triples (ignoring relation labels).
    Uses path-compressed union-find.
    """
    if len(triples) == 0 or n_ent == 0:
        return 0

    parent = list(range(n_ent))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for h, _, t in triples:
        if 0 <= h < n_ent and 0 <= t < n_ent:
            union(int(h), int(t))

    from collections import Counter
    counts = Counter(find(i) for i in range(n_ent))
    return max(counts.values())
