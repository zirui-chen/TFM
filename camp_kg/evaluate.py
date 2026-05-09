"""
Zero-shot evaluation on real benchmark KGs.

Implements the standard filtered ranking protocol:
    for each test query (h, q, ?): score all entities,
    filter known-true triples before computing rank.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@torch.no_grad()
def evaluate_kg(
    model,
    triples_obs: np.ndarray,          # observable edges at test time [N, 3]
    triples_test: np.ndarray,         # test queries [M, 3]
    filter_triples: Optional[np.ndarray],  # all known-true triples for filtering
    n_entities: int,
    n_relations: int,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Run filtered ranking evaluation on one KG.

    Returns dict with keys: 'mrr', 'hits@1', 'hits@3', 'hits@10'.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric is required.")

    from .generator import lift

    # Build relation graph from test-time observable edges
    gr_dict = lift(triples_obs, n_relations)
    entity_graph = Data(
        edge_index=torch.from_numpy(
            np.stack([triples_obs[:, 0], triples_obs[:, 2]], axis=0).astype(np.int64)
        ).to(device),
        edge_type=torch.from_numpy(triples_obs[:, 1].astype(np.int64)).to(device),
        num_nodes=n_entities,
    )
    rel_graph = Data(
        edge_index=torch.from_numpy(gr_dict["edge_index"].astype(np.int64)).to(device),
        edge_type=torch.from_numpy(gr_dict["edge_type"].astype(np.int64)).to(device),
        num_nodes=gr_dict["num_nodes"],
    )

    # Build filter set
    filter_set: Dict[Tuple[int, int], List[int]] = {}
    if filter_triples is not None:
        for h, r, t in filter_triples:
            key = (int(h), int(r))
            if key not in filter_set:
                filter_set[key] = []
            filter_set[key].append(int(t))

    model.eval()
    reciprocal_ranks, hits = [], {1: [], 3: [], 10: []}

    for start in range(0, len(triples_test), batch_size):
        batch_triples = triples_test[start : start + batch_size]
        query = torch.from_numpy(batch_triples.astype(np.int64)).to(device)

        # Score all entities
        scores = model(entity_graph, rel_graph, query)   # [B, n_ent]

        for i, (h, r, t) in enumerate(batch_triples.tolist()):
            row = scores[i].clone()

            # Filter known-true triples (excluding the target itself)
            for ft in filter_set.get((int(h), int(r)), []):
                if ft != int(t) and ft < row.size(0):
                    row[ft] = float("-inf")

            rank = int((row > row[int(t)]).sum()) + 1
            reciprocal_ranks.append(1.0 / rank)
            for k in [1, 3, 10]:
                hits[k].append(float(rank <= k))

    n = len(reciprocal_ranks)
    if n == 0:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}

    return {
        "mrr":     float(np.mean(reciprocal_ranks)),
        "hits@1":  float(np.mean(hits[1])),
        "hits@3":  float(np.mean(hits[3])),
        "hits@10": float(np.mean(hits[10])),
    }


def evaluate_all_kgs(
    model,
    kg_dir: str,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on all benchmark KGs in kg_dir.

    Expected directory layout (same as ULTRA benchmarks):
        {kg_dir}/{kg_name}/
            train.txt / val.txt / test.txt  (tab-separated h r t)
            entity2id.txt / relation2id.txt

    Returns dict mapping KG name → metrics dict.
    """
    import os
    results = {}
    for kg_name in sorted(os.listdir(kg_dir)):
        kg_path = os.path.join(kg_dir, kg_name)
        if not os.path.isdir(kg_path):
            continue
        try:
            metrics = _eval_one_kg(model, kg_path, device, batch_size)
            results[kg_name] = metrics
            print(f"  {kg_name:30s}  MRR {metrics['mrr']:.3f}  "
                  f"H@1 {metrics['hits@1']:.3f}  H@10 {metrics['hits@10']:.3f}")
        except Exception as e:
            print(f"  WARNING: {kg_name}: {e}")
    return results


def _eval_one_kg(model, kg_path: str, device, batch_size: int) -> Dict:
    import os
    e2id = _load_id_map(os.path.join(kg_path, "entity2id.txt"))
    r2id = _load_id_map(os.path.join(kg_path, "relation2id.txt"))
    n_ent = len(e2id)
    n_rel = len(r2id)

    train = _load_triples(os.path.join(kg_path, "train.txt"), e2id, r2id)
    val   = _load_triples(os.path.join(kg_path, "val.txt"),   e2id, r2id)
    test  = _load_triples(os.path.join(kg_path, "test.txt"),  e2id, r2id)

    # Observable edges at test time = train + val
    obs_raw = np.concatenate([train, val], axis=0) if len(val) > 0 else train

    # Add inverses to observable edges
    import numpy as np
    inv_obs = np.stack([obs_raw[:, 2], obs_raw[:, 1] + n_rel, obs_raw[:, 0]], axis=1)
    obs = np.concatenate([obs_raw, inv_obs], axis=0)

    # Filter set = all known triples
    all_triples = np.concatenate([train, val, test], axis=0) if len(test) > 0 else train

    return evaluate_kg(model, obs, test, all_triples, n_ent, n_rel, device, batch_size)


def _load_id_map(path: str) -> Dict[str, int]:
    id_map = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                id_map[parts[0]] = int(parts[1])
    return id_map


def _load_triples(path: str, e2id: Dict, r2id: Dict) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                h, r, t = parts
                if h in e2id and r in r2id and t in e2id:
                    rows.append([e2id[h], r2id[r], e2id[t]])
    return np.array(rows, dtype=np.int64) if rows else np.empty((0, 3), dtype=np.int64)
