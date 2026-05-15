"""基于训练好的扩散模型从模板图采样新坐标。"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from models.diffusion_model import CrystalDiffusion


class StructureGenerator:
    def __init__(self, model: CrystalDiffusion):
        self.model = model

    @torch.no_grad()
    def sample(self, template: Data, cond: torch.Tensor, device: str) -> torch.Tensor:
        """
        template: 单图 Data（含 x, edge_index, pos 形状用于初始化噪声尺度）。
        cond: [1,3] 归一化条件向量（与训练时 data.y 同一空间）。
        返回与 template.pos 同形状的笛卡尔坐标（已去质心）。
        """
        self.model.eval()
        z = template.x.to(device)
        edge_index = template.edge_index.to(device)
        batch = getattr(template, "batch", None)
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
        pos = torch.randn_like(template.pos, device=device, dtype=torch.float32)
        cond = cond.to(device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        tpl_pos = template.pos.to(device)
        for t in reversed(range(self.model.timesteps)):
            pos = self.model.p_sample_step(pos, t, z, cond, edge_index, batch)
            pos = torch.nan_to_num(pos, nan=0.0, posinf=1e3, neginf=-1e3)
            bad = ~torch.isfinite(pos)
            if bad.any():
                pos = torch.where(bad, tpl_pos, pos)
        pos = pos.cpu()
        bad = ~torch.isfinite(pos)
        if bad.any():
            pos = torch.where(bad, template.pos, pos)
        return pos
