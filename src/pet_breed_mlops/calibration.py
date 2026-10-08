from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find T > 0 minimising NLL of softmax(logits / T). Optimises log T so T stays positive."""
    logits, labels = logits.double(), labels.long()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0, n_bins: int = 15
) -> float:
    conf, correct = _conf_and_correct(logits, labels, temperature)
    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - conf[mask].mean().item()
            )
    return ece


def reliability_bins(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """(mean confidence, accuracy, count) per non-empty bin, for the reliability diagram."""
    conf, correct = _conf_and_correct(logits, labels, temperature)
    out = []
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out.append(
                (conf[mask].mean().item(), correct[mask].float().mean().item(), int(mask.sum()))
            )
    return out


@dataclass(frozen=True)
class AbstentionStats:
    threshold: float
    coverage: float            # fraction of images we answer
    selective_accuracy: float  # accuracy on that fraction


def abstention_stats(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float, threshold: float
) -> AbstentionStats:
    conf, correct = _conf_and_correct(logits, labels, temperature)
    answered = conf >= threshold
    coverage = answered.float().mean().item()
    acc = correct[answered].float().mean().item() if answered.any() else float("nan")
    return AbstentionStats(threshold, coverage, acc)


def pick_threshold(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float, target_accuracy: float = 0.97
) -> AbstentionStats:
    """Lowest threshold whose selective accuracy on val reaches the target (max coverage)."""
    conf, _ = _conf_and_correct(logits, labels, temperature)
    best = abstention_stats(logits, labels, temperature, 1.01)  # answers nothing
    for t in sorted(set(conf.tolist()), reverse=True):
        s = abstention_stats(logits, labels, temperature, t)
        if s.selective_accuracy >= target_accuracy:
            best = s
        else:
            break
    return best


def _conf_and_correct(logits, labels, temperature):
    probs = torch.softmax(logits.float() / temperature, dim=1)
    conf, pred = probs.max(1)
    return conf, pred == labels