"""
Training loop for CAMP-KG.

Replaces ULTRA's real-KG data loader with SyntheticCorpus while keeping
every other component (model, optimizer, loss, negative sampling) identical.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from .corpus import SyntheticCorpus, batch_to_ultra


def train_camp_kg(
    model: nn.Module,
    train_corpus: SyntheticCorpus,
    val_corpus: SyntheticCorpus,
    out_dir: str,
    n_steps: int = 200_000,
    lr: float = 5e-4,
    batch_size: int = 64,
    n_negatives: int = 64,
    adversarial_temperature: float = 1.0,
    checkpoint_every: int = 10_000,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> str:
    """
    Train a ULTRA model on the synthetic corpus.

    Parameters mirror ULTRA's published setup:
        AdamW, lr=5e-4, batch_size=64, 200K steps,
        self-adversarial negative sampling (alpha=1, 64 negatives).

    Returns the path to the best checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_mrr   = -1.0
    best_ckpt_path = None
    t0 = time.time()

    for step in range(1, n_steps + 1):
        model.train()
        optimizer.zero_grad()

        batch = train_corpus.sample_batch()
        entity_graph, rel_graph, query, target = batch_to_ultra(batch, device)

        loss = _step(
            model, entity_graph, rel_graph, query, target,
            batch["filter_triples"].to(device),
            n_negatives, adversarial_temperature,
        )
        loss.backward()
        optimizer.step()

        if step % checkpoint_every == 0:
            val_mrr = evaluate_val(model, val_corpus, device)
            elapsed  = time.time() - t0
            print(
                f"Step {step:>7d} | loss {loss.item():.4f} "
                f"| val MRR {val_mrr:.4f} | {elapsed/60:.1f} min"
            )

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_ckpt_path = os.path.join(out_dir, f"ckpt_step{step:07d}.pt")
                torch.save(
                    {"step": step, "model": model.state_dict(),
                     "val_mrr": val_mrr},
                    best_ckpt_path,
                )

    print(f"Training done. Best val MRR={best_val_mrr:.4f} at {best_ckpt_path}")
    return best_ckpt_path


# ---------------------------------------------------------------------------
# Self-adversarial negative sampling loss
# ---------------------------------------------------------------------------

def _step(
    model: nn.Module,
    entity_graph,
    rel_graph,
    query: torch.Tensor,
    target: torch.Tensor,
    filter_triples: torch.Tensor,
    n_negatives: int,
    adversarial_temperature: float,
) -> torch.Tensor:
    """One forward-backward step with self-adversarial negative sampling."""
    n_ent = entity_graph.num_nodes

    # Score all entities for each query
    scores = model(entity_graph, rel_graph, query)   # [B, n_ent]

    # Build corruption mask: known-true triples should not be negatives
    B = query.size(0)
    mask = torch.zeros(B, n_ent, dtype=torch.bool, device=scores.device)

    # Mark targets
    for i in range(B):
        mask[i, target[i]] = True

    # Mark filter triples for each query
    filter_set = {(int(h), int(r), int(t)) for h, r, t in filter_triples.tolist()}
    for i, (h, r, t) in enumerate(query.tolist()):
        for fh, fr, ft in filter_set:
            if int(fh) == int(h) and int(fr) == int(r):
                if ft < n_ent:
                    mask[i, ft] = True

    # Self-adversarial negative sampling
    pos_scores = scores.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]

    # Mask positives with -inf before sampling negatives
    neg_scores = scores.clone()
    neg_scores[mask] = float("-inf")

    # Sample n_negatives per query proportional to softmax(neg_scores/temp)
    adv_probs = torch.softmax(
        neg_scores / adversarial_temperature, dim=-1
    ).detach()
    neg_idx = torch.multinomial(adv_probs, n_negatives, replacement=False)
    sampled_neg_scores = scores.gather(1, neg_idx)   # [B, n_negatives]

    # Log-sigmoid BCE loss (standard in ULTRA / RotatE family)
    pos_loss = -torch.nn.functional.logsigmoid(pos_scores).mean()
    neg_loss = -(
        adv_probs.gather(1, neg_idx)
        * torch.nn.functional.logsigmoid(-sampled_neg_scores)
    ).sum(dim=1).mean()

    return pos_loss + neg_loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    val_corpus: SyntheticCorpus,
    device: torch.device,
) -> float:
    """Compute mean MRR over all validation KGs in the corpus."""
    model.eval()
    mrr_sum, total = 0.0, 0

    for batch in val_corpus.val_batches():
        if len(batch["query_h"]) == 0:
            continue
        entity_graph, rel_graph, query, target = batch_to_ultra(batch, device)
        scores = model(entity_graph, rel_graph, query)  # [B, n_ent]

        # Filtered ranking
        filter_triples = batch["filter_triples"].tolist()
        filter_set = {(h, r, t) for h, r, t in filter_triples}

        for i, (h, r, t) in enumerate(query.tolist()):
            row = scores[i].clone()
            for fh, fr, ft in filter_set:
                if fh == h and fr == r and ft != t and ft < row.size(0):
                    row[ft] = float("-inf")
            rank = int((row > row[t]).sum()) + 1
            mrr_sum += 1.0 / rank
            total += 1

    return mrr_sum / total if total > 0 else 0.0
