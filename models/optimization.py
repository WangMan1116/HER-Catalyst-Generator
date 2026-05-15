"""Sampling-time selection and lightweight proxy-based refinement."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from utils.geo_utils import MaterialMetrics, evaluate_structure, load_structure


def normalized_target_cond(
    y_mean: np.ndarray,
    y_std: np.ndarray,
    dg_h: float = 0.0,
    stab: float = 1.0,
    synth: float = 1.0,
    max_sigma: float | None = None,
) -> torch.Tensor:
    raw = np.array([dg_h, stab, synth], dtype=np.float64)
    if max_sigma is not None:
        lower = y_mean - max_sigma * y_std
        upper = y_mean + max_sigma * y_std
        raw = np.clip(raw, lower, upper)
    vec = (raw - y_mean) / y_std
    return torch.tensor(vec, dtype=torch.float32).view(1, 3)


def feasible_target_cond(
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    dg_h_goal: float = 0.0,
    stab_sigma: float = 0.35,
    synth_sigma: float = 0.25,
    max_sigma: float = 0.8,
) -> torch.Tensor:
    """
    Keep the condition vector near the training manifold.

    This avoids pushing sampling to unrealistic corners such as
    `(dg_h=0, stability=1, synthesis=1)` when those targets sit too far away
    from the dataset mean.
    """
    stab_target = float(y_mean[1] + stab_sigma * y_std[1])
    synth_target = float(y_mean[2] + synth_sigma * y_std[2])
    return normalized_target_cond(
        y_mean,
        y_std,
        dg_h=dg_h_goal,
        stab=stab_target,
        synth=synth_target,
        max_sigma=max_sigma,
    )


def assemble_structure(
    cif_path: str,
    pos_centered: torch.Tensor,
    recenter: torch.Tensor,
    lattice_np: np.ndarray,
) -> Structure:
    tpl = load_structure(cif_path)
    lat = Lattice(np.asarray(lattice_np, dtype=np.float64))
    pc = pos_centered.detach().cpu().numpy().astype(np.float64)
    rc = recenter.detach().cpu().numpy().astype(np.float64)
    pos = pc + rc
    if not np.all(np.isfinite(pos)):
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(pos)):
            pos = tpl.cart_coords
    try:
        frac = lat.get_fractional_coords(pos) % 1.0
    except Exception:
        frac = tpl.frac_coords
    if not np.all(np.isfinite(frac)):
        frac = tpl.frac_coords
    return Structure(lat, tpl.species, frac, coords_are_cartesian=False)


def multiobjective_score(
    m: MaterialMetrics,
    *,
    stab_floor: float = 0.0,
    synth_floor: float = 0.0,
    stab_weight: float = 1.35,
    synth_weight: float = 1.0,
    penalty_weight: float = 3.0,
) -> float:
    """Higher is better: HER near 0, with soft constraints on the other metrics."""
    her_term = -abs(m.dg_h_proxy)
    stab = 0.5 * (m.thermo_stability + m.dyn_stability)
    stab_gap = max(0.0, stab_floor - stab)
    synth_gap = max(0.0, synth_floor - m.synthesis_score)
    penalty = penalty_weight * (stab_gap + 0.8 * synth_gap)
    return float(her_term + stab_weight * stab + synth_weight * m.synthesis_score - penalty)


def rerank_structures(
    items: List[Tuple[Structure, MaterialMetrics]],
    top_k: int,
    *,
    stab_floor: float = 0.0,
    synth_floor: float = 0.0,
) -> List[Tuple[Structure, MaterialMetrics, float]]:
    scored = [
        (s, m, multiobjective_score(m, stab_floor=stab_floor, synth_floor=synth_floor))
        for s, m in items
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def coordinate_jitter_refine(
    struct: Structure,
    steps: int = 40,
    sigma: float = 0.02,
    rng: np.random.Generator | None = None,
    *,
    stab_floor: float = 0.0,
    synth_floor: float = 0.0,
) -> Structure:
    """Small cartesian-coordinate perturbations with proxy-guided accept/reject."""
    rng = rng or np.random.default_rng(0)
    best = struct
    best_sc = multiobjective_score(
        evaluate_structure(best),
        stab_floor=stab_floor,
        synth_floor=synth_floor,
    )
    for _ in range(steps):
        cart = best.cart_coords + rng.normal(scale=sigma, size=best.cart_coords.shape)
        trial = Structure(best.lattice, best.species, cart, coords_are_cartesian=True)
        m = evaluate_structure(trial)
        sc = multiobjective_score(m, stab_floor=stab_floor, synth_floor=synth_floor)
        if sc > best_sc:
            best, best_sc = trial, sc
    return best
