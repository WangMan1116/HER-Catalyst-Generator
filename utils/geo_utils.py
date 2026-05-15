"""
材料几何与性质代理指标：HER(ΔG_H 代理)、热力学/动力学稳定性代理、可合成性评分。

说明：完整 ΔG_H 需 DFT/显式吸附计算；此处提供可微、可复现的代理量，
用于扩散训练中的多任务损失与采样后评估。正式研究请替换为图神经网络
性质预测器或实验/高通量数据库标签。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifParser


@dataclass
class MaterialMetrics:
    dg_h_proxy: float  # eV 量级代理，0 附近表示“火山图”最优区倾向
    thermo_stability: float  # [0,1]
    dyn_stability: float  # [0,1]
    synthesis_score: float  # [0,1]


def load_structure(path: str | Path) -> Structure:
    path = Path(path)
    parser = CifParser(str(path))
    return parser.get_structures(primitive=True)[0]


def _min_interatomic_distance(struct: Structure) -> float:
    dm = struct.distance_matrix
    n = dm.shape[0]
    mask = np.ones_like(dm, dtype=bool)
    np.fill_diagonal(mask, False)
    return float(dm[mask].min()) if n else 0.0


def _volume_per_atom(struct: Structure) -> float:
    return float(struct.volume / len(struct))


def synthesis_score(struct: Structure) -> float:
    """元素种类少、原子数适中 → 可合成性更高（与 baseline 筛选思想一致）。"""
    elems = set(str(sp.symbol) for sp in struct.species)
    n_elem = len(elems)
    n_atom = len(struct)
    s_elem = max(0.0, 1.0 - (n_elem - 1) / 4.0)
    s_size = max(0.0, 1.0 - abs(n_atom - 12) / 24.0)
    return float(np.clip(0.5 * s_elem + 0.5 * s_size, 0.0, 1.0))


def thermodynamic_stability_proxy(struct: Structure) -> float:
    """键长不过短、体积/原子合理 → 热力学稳定倾向（代理）。"""
    d_min = _min_interatomic_distance(struct)
    vpa = _volume_per_atom(struct)
    # 距离项：过近不稳定
    if d_min < 1e-3:
        dist_score = 0.0
    else:
        dist_score = float(np.clip((d_min - 0.9) / 1.2, 0.0, 1.0))
    # 体积项：二维层状常见 8–40 Å^3/atom 粗范围
    vol_score = float(np.clip(1.0 - abs(vpa - 18.0) / 25.0, 0.0, 1.0))
    return float(0.6 * dist_score + 0.4 * vol_score)


def dynamic_stability_proxy(struct: Structure) -> float:
    """局部键长离散度低 → 动力学稳定倾向（代理，非声子）。"""
    if len(struct) <= 1:
        return 0.5
    dmat = struct.distance_matrix
    n = dmat.shape[0]
    nn_dists: List[float] = []
    for i in range(n):
        row = dmat[i].copy()
        row[i] = np.inf
        j = int(np.argmin(row))
        nn_dists.append(float(row[j]))
    std = float(np.std(nn_dists))
    score = float(np.exp(-std / 0.35))
    return float(np.clip(score, 0.0, 1.0))


def her_dg_h_proxy(struct: Structure) -> float:
    """
    HER ΔG_H 代理（eV）：基于电负性差异与金属分数的平滑启发式，
    在过渡金属氧化物/硫化物类结构上呈连续变化，便于优化。
    """
    if len(struct) == 0:
        return 0.0
    en = []
    metal_frac = 0.0
    for site in struct:
        el = site.specie.symbol
        try:
            en.append(Element(el).X)
        except Exception:
            en.append(2.0)
        try:
            if Element(el).is_metal:
                metal_frac += 1.0
        except Exception:
            pass
    metal_frac /= len(struct)
    en = np.asarray(en, dtype=np.float64)
    mean_en = float(en.mean())
    var_en = float(en.var())
    # 目标：接近 0 表示更靠近“火山顶”代理区
    term1 = 0.35 * (mean_en - 2.2)
    term2 = 0.12 * (var_en - 0.25)
    term3 = -0.25 * (metal_frac - 0.45)
    dg = term1 + term2 + term3
    return float(np.clip(dg, -1.2, 1.2))


def evaluate_structure(struct: Structure) -> MaterialMetrics:
    return MaterialMetrics(
        dg_h_proxy=her_dg_h_proxy(struct),
        thermo_stability=thermodynamic_stability_proxy(struct),
        dyn_stability=dynamic_stability_proxy(struct),
        synthesis_score=synthesis_score(struct),
    )


def evaluate_cif(path: str | Path) -> MaterialMetrics:
    return evaluate_structure(load_structure(path))


def metrics_vector(m: MaterialMetrics) -> np.ndarray:
    return np.array(
        [m.dg_h_proxy, m.thermo_stability, m.dyn_stability, m.synthesis_score],
        dtype=np.float32,
    )


def batch_evaluate_cifs(paths: List[str | Path]) -> Dict[str, np.ndarray]:
    """返回各指标均值与逐样本数组。"""
    rows = [metrics_vector(evaluate_cif(p)) for p in paths]
    mat = np.stack(rows, axis=0)
    keys = ["dg_h_proxy", "thermo_stability", "dyn_stability", "synthesis_score"]
    return {
        "mean": mat.mean(axis=0),
        "std": mat.std(axis=0),
        "per_sample": mat,
        "keys": np.array(keys),
    }


def stability_aggregate(m: MaterialMetrics) -> float:
    """单一稳定性标量，用于曲线与表格。"""
    return float(0.5 * m.thermo_stability + 0.5 * m.dyn_stability)
