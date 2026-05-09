"""
SyntheticCorpus: data loader for CAMP-KG pretraining.

Drop-in replacement for ULTRA's real-KG data loader.
Loads pre-generated KGs from disk and returns balanced 50:50
derived/background query batches at each training step.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .generator import KGSplit


# ---------------------------------------------------------------------------
# Corpus dataset
# ---------------------------------------------------------------------------

class SyntheticCorpus:
    """
    Manages a pre-generated corpus of synthetic KGs for CAMP-KG pretraining.

    At each training step, one KG is sampled uniformly and a mixed 64-query
    batch (32 derived + 32 background by default) is returned together with
    the stage-specific relation graph and observable edge set.

    Parameters
    ----------
    corpus_dir : str
        Directory containing *.pkl files produced by SyntheticKGGenerator.
    split : 'train' | 'val'
        'train' uses (e_obs_train, q_derived_train, q_bg_train, gr_train).
        'val'   uses (e_obs_eval,  q_val,           gr_eval)    for model
                selection on the 200 held-out KGs.
    batch_size : int
        Total queries per batch (default 64).
    derived_frac : float
        Fraction of batch drawn from derived queries (default 0.5 → 50:50).
    seed : int | None
        RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        corpus_dir: str,
        split: str = "train",
        batch_size: int = 64,
        derived_frac: float = 0.5,
        seed: Optional[int] = None,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
        self.corpus_dir  = corpus_dir
        self.split       = split
        self.batch_size  = batch_size
        self.derived_frac = derived_frac
        self.rng = np.random.default_rng(seed)

        self._paths = sorted([
            os.path.join(corpus_dir, f)
            for f in os.listdir(corpus_dir)
            if f.endswith(".pkl")
        ])
        if not self._paths:
            raise FileNotFoundError(f"No .pkl files found in {corpus_dir}")

        # Lazy-load cache
        self._cache: Dict[int, KGSplit] = {}

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def sample_batch(self) -> Dict:
        """
        Sample one training batch from a uniformly random KG.

        Returns
        -------
        dict with keys:
            'edge_index'  : LongTensor [2, E] – observable edges (h, t)
            'edge_type'   : LongTensor [E]    – relation IDs (0..2R-1)
            'query_h'     : LongTensor [B]    – query head entities
            'query_r'     : LongTensor [B]    – query relation IDs
            'query_t'     : LongTensor [B]    – target tail entities
            'num_entities': int
            'num_relations': int              – canonical (total = 2×this)
            'gr_edge_index': LongTensor [2, E_r] – relation graph edges
            'gr_edge_type' : LongTensor [E_r]    – relation graph edge types
            'gr_num_nodes' : int                 – 2 * num_relations
            'filter_triples': LongTensor [M, 3]  – all known-true triples
        """
        kg_idx = int(self.rng.integers(0, len(self._paths)))
        kg = self._load(kg_idx)
        return self._make_batch(kg)

    def val_batches(self) -> List[Dict]:
        """Iterate over all KGs in val split, one dict per KG."""
        batches = []
        for i in range(len(self._paths)):
            kg = self._load(i)
            batches.append(self._make_val_dict(kg))
        return batches

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, idx: int) -> KGSplit:
        if idx not in self._cache:
            with open(self._paths[idx], "rb") as f:
                self._cache[idx] = pickle.load(f)
        return self._cache[idx]

    def _make_batch(self, kg: KGSplit) -> Dict:
        """Build a mixed training batch from one KG."""
        n_derived = max(1, int(self.batch_size * self.derived_frac))
        n_bg      = self.batch_size - n_derived

        queries = []
        if len(kg.q_derived_train) > 0:
            idx = self.rng.integers(0, len(kg.q_derived_train), size=min(n_derived, len(kg.q_derived_train)))
            queries.append(kg.q_derived_train[idx])
        if len(kg.q_bg_train) > 0:
            idx = self.rng.integers(0, len(kg.q_bg_train), size=min(n_bg, len(kg.q_bg_train)))
            queries.append(kg.q_bg_train[idx])

        if not queries:
            # fall back to combined train queries
            idx = self.rng.integers(0, max(len(kg.q_train), 1), size=self.batch_size)
            queries = [kg.q_train[idx % max(len(kg.q_train), 1)]]

        query_arr = np.concatenate(queries, axis=0)

        obs = kg.e_obs_train
        gr  = kg.gr_train
        return self._to_tensors(kg, query_arr, obs, gr)

    def _make_val_dict(self, kg: KGSplit) -> Dict:
        """Pack validation queries for one KG."""
        obs = kg.e_obs_eval
        gr  = kg.gr_eval
        return self._to_tensors(kg, kg.q_val, obs, gr)

    @staticmethod
    def _to_tensors(
        kg: KGSplit,
        queries: np.ndarray,
        obs: np.ndarray,
        gr: Dict,
    ) -> Dict:
        edge_index = torch.from_numpy(
            np.stack([obs[:, 0], obs[:, 2]], axis=0).astype(np.int64)
        )
        edge_type = torch.from_numpy(obs[:, 1].astype(np.int64))

        query_h = torch.from_numpy(queries[:, 0].astype(np.int64))
        query_r = torch.from_numpy(queries[:, 1].astype(np.int64))
        query_t = torch.from_numpy(queries[:, 2].astype(np.int64))

        gr_edge_index = torch.from_numpy(gr["edge_index"].astype(np.int64))
        gr_edge_type  = torch.from_numpy(gr["edge_type"].astype(np.int64))

        filter_t = torch.from_numpy(kg.filter_triples.astype(np.int64))

        return dict(
            edge_index=edge_index,
            edge_type=edge_type,
            query_h=query_h,
            query_r=query_r,
            query_t=query_t,
            num_entities=kg.n_entities,
            num_relations=kg.n_relations,
            gr_edge_index=gr_edge_index,
            gr_edge_type=gr_edge_type,
            gr_num_nodes=gr["num_nodes"],
            filter_triples=filter_t,
        )


# ---------------------------------------------------------------------------
# ULTRA integration helpers
# ---------------------------------------------------------------------------

def batch_to_ultra(batch: Dict, device: torch.device) -> Tuple:
    """
    Convert a SyntheticCorpus batch dict to the format expected by ULTRA.

    Returns (graph, query, target) where:
        graph  : torch_geometric.data.Data with entity and relation graphs
        query  : LongTensor [B, 3] (h, r, t)
        target : LongTensor [B]    (tail entity index, for scoring)

    This function requires torch_geometric to be installed.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch_geometric is required for ULTRA integration.")

    # Entity graph (the observable KG)
    entity_graph = Data(
        edge_index=batch["edge_index"].to(device),
        edge_type=batch["edge_type"].to(device),
        num_nodes=batch["num_entities"],
    )

    # Relation graph
    rel_graph = Data(
        edge_index=batch["gr_edge_index"].to(device),
        edge_type=batch["gr_edge_type"].to(device),
        num_nodes=batch["gr_num_nodes"],
    )

    # Queries
    query = torch.stack(
        [batch["query_h"], batch["query_r"], batch["query_t"]], dim=1
    ).to(device)
    target = batch["query_t"].to(device)

    return entity_graph, rel_graph, query, target
