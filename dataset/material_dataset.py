"""从 CIF 或 Materials Project API 构建晶体图数据集，用于 GNN+扩散训练。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from torch.utils.data import Dataset
from torch_geometric.data import Data

# 【已修改】去掉了 project. 前缀
from utils.geo_utils import evaluate_structure, load_structure, stability_aggregate

# 修复 Python 3.10 下某些依赖库对 typing 的误用
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


def _mp_doc_structure(doc: Any) -> Structure | None:
    """从 mp-api 文档对象或字典中取出 pymatgen Structure。"""
    raw = None
    if isinstance(doc, dict):
        raw = doc.get("structure")
    else:
        # 新版 API 使用 .structure 属性
        raw = getattr(doc, "structure", None)

    if raw is None:
        return None
    if isinstance(raw, Structure):
        return raw
    if isinstance(raw, dict):
        try:
            return Structure.from_dict(raw)
        except Exception:
            return None
    return None


def _mp_doc_id(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("material_id", "unknown"))
    # 新版 API 对象通常有 material_id 属性
    mid = getattr(doc, "material_id", None)
    return str(mid) if mid else "unknown"


# 【已修改】安全版本的 API 抓取函数
def _fetch_materials_project_entries(
    api_key: str,
    *,
    nelements: Tuple[int, int],
    eah_max: float,
    max_collect: int,
) -> List[Any]:
    """使用新版 mp-api 搜索材料，避免字段版本冲突"""
    try:
        from mp_api.client import MPRester
    except ImportError as e:
        raise ImportError(
            "使用 Materials Project 数据源需要安装 mp-api：pip install mp-api"
        ) from e

    # 为了保证能拿到足够多的有效结构，扩大搜索上限
    cap = max(max_collect * 3, max_collect)
    out: List[Any] = []

    try:
        with MPRester(api_key) as mpr:
            # 仅使用最稳定的 num_elements 进行云端搜索
            cursor = mpr.materials.summary.search(
                num_elements=nelements,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "energy_above_hull",
                ],
            )

            count = 0
            for doc in cursor:
                # 在本地进行稳定性过滤，彻底避开 API 报错
                eah = getattr(doc, "energy_above_hull", 999.0)
                if eah is not None and eah <= eah_max:
                    out.append(doc)
                    count += 1

                if count >= max_collect:
                    break
    except Exception as e:
        print(f"API 查询出错: {e}")
        return []

    return out


def _complete_graph(num_nodes: int) -> torch.Tensor:
    pairs: List[Tuple[int, int]] = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                pairs.append((i, j))
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    row, col = zip(*pairs)
    return torch.tensor([row, col], dtype=torch.long)


def structure_to_data(
    struct: Structure,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    name: str = "",
    cif_path: str = "",
) -> Data:
    frac = struct.frac_coords
    lattice = struct.lattice.matrix
    pos = frac @ lattice
    recenter = pos.mean(axis=0)
    pos = pos - recenter

    z = torch.tensor([sp.Z for sp in struct.species], dtype=torch.long)
    edge_index = _complete_graph(len(struct))

    m = evaluate_structure(struct)
    y_raw = np.array(
        [m.dg_h_proxy, stability_aggregate(m), m.synthesis_score], dtype=np.float32
    )
    y = (y_raw - y_mean) / y_std

    return Data(
        x=z,
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float32),
        y_raw=torch.tensor(y_raw, dtype=torch.float32),
        name=name,
        recenter=torch.tensor(recenter, dtype=torch.float32),
        lattice=torch.tensor(lattice, dtype=torch.float32),
        cif_path=cif_path,
    )


def collect_cif_paths(root: str | Path, limit: Optional[int] = None) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    paths = sorted(root.rglob("*.cif"))
    if limit is not None:
        paths = paths[:limit]
    return paths


class MaterialCrystalDataset(Dataset):
    def __init__(
        self,
        cif_root: str | Path | None = None,
        max_atoms: int = 32,
        max_samples: Optional[int] = 400,
        *,
        source: str = "cif",
        mp_api_key: Optional[str] = None,
        mp_cache_dir: Optional[str | Path] = None,
        mp_eah_max: float = 0.1,
        mp_nelements: Tuple[int, int] = (1, 3),
    ):
        self.source = source
        self.cif_root = Path(cif_root) if cif_root is not None else None
        self.max_atoms = max_atoms
        self.max_samples = max_samples

        self.mp_api_key = (mp_api_key or os.getenv("MP_API_KEY") or "").strip()
        self.mp_cache_dir = Path(mp_cache_dir) if mp_cache_dir else Path("data/mp_cache")
        self.mp_eah_max = mp_eah_max
        self.mp_nelements = mp_nelements

        self.samples: List[Data] = []
        self.y_mean: np.ndarray
        self.y_std: np.ndarray
        self._build()

    def _build(self) -> None:
        if self.source == "materials_project":
            structs, used_paths, names = self._build_from_materials_project()
        else:
            if self.cif_root is None:
                raise ValueError("source='cif' 时必须提供 cif_root")
            structs, used_paths, names = self._build_from_cif_dir()

        if len(structs) == 0:
            raise RuntimeError(
                f"数据构建失败：源 '{self.source}' 未提供有效晶体样本。"
            )

        props = []
        for s in structs:
            m = evaluate_structure(s)
            props.append([m.dg_h_proxy, stability_aggregate(m), m.synthesis_score])

        props_arr = np.asarray(props, dtype=np.float64)
        self.y_mean = props_arr.mean(axis=0)
        self.y_std = props_arr.std(axis=0) + 1e-6

        self.samples = [
            structure_to_data(s, self.y_mean, self.y_std, name=nm, cif_path=str(p))
            for s, p, nm in zip(structs, used_paths, names)
        ]

    def _build_from_cif_dir(self) -> Tuple[List[Structure], List[Path], List[str]]:
        assert self.cif_root is not None
        paths = collect_cif_paths(self.cif_root, self.max_samples)
        structs, used_paths, names = [], [], []
        for p in paths:
            try:
                s = load_structure(p)
                if 2 <= len(s) <= self.max_atoms:
                    structs.append(s)
                    used_paths.append(p)
                    names.append(p.name)
            except Exception:
                continue
            if self.max_samples and len(structs) >= self.max_samples:
                break
        return structs, used_paths, names

    def _build_from_materials_project(self) -> Tuple[List[Structure], List[Path], List[str]]:
        if not self.mp_api_key:
            raise RuntimeError("未检测到有效 MP_API_KEY")

        target_n = self.max_samples or 100
        print(f"正在从 Materials Project 获取数据 (目标: {target_n})...")

        docs = _fetch_materials_project_entries(
            self.mp_api_key,
            nelements=self.mp_nelements,
            eah_max=self.mp_eah_max,
            max_collect=target_n,
        )

        self.mp_cache_dir.mkdir(parents=True, exist_ok=True)
        structs, used_paths, names = [], [], []

        for doc in docs:
            s = _mp_doc_structure(doc)
            if s is None or not (2 <= len(s) <= self.max_atoms):
                continue

            mid = _mp_doc_id(doc)
            cif_name = "".join(c for c in mid if c.isalnum() or c in "-_")
            cif_path = self.mp_cache_dir / f"{cif_name}.cif"

            try:
                if not cif_path.exists():
                    CifWriter(s).write_file(str(cif_path))
                structs.append(s)
                used_paths.append(cif_path)
                names.append(f"{mid}.cif")
            except Exception:
                continue

            if len(structs) >= target_n:
                break

        return structs, used_paths, names

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        return self.samples[idx]