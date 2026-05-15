"""基于 GNN 消息传递的晶体坐标 DDPM + 多任务性质头。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """a: 长度为 T 的调度系数；t: 与 pos 行数相同；用 index_select 避免 gather 维度假设。"""
    t_safe = t.long().clamp(0, a.numel() - 1)
    out = a[t_safe]
    tail = (1,) * max(len(x_shape) - 1, 0)
    return out.reshape(t.shape[0], *tail)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class GNNBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        rel = pos[row] - pos[col]
        d2 = (rel**2).sum(dim=-1, keepdim=True)
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], d2], dim=-1))
        agg = _scatter_sum(m_ij, row, dim_size=h.size(0))
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h


class EpsFieldHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        rel = pos[row] - pos[col]
        d2 = (rel**2).sum(dim=-1, keepdim=True)
        w = self.net(torch.cat([h[row], h[col], d2], dim=-1))
        trans = rel * w
        eps = _scatter_sum(trans, row, dim_size=h.size(0))
        return eps


class PropertyHead(nn.Module):
    def __init__(self, hidden: int, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        g = global_mean_pool(h, batch)
        return self.net(g)


class CrystalDiffusion(nn.Module):
    """DDPM 坐标噪声预测 + 图级多任务性质回归（归一化空间）。"""

    def __init__(
        self,
        max_atomic_num: int = 118,
        hidden_dim: int = 128,
        num_layers: int = 4,
        timesteps: int = 300,
    ):
        super().__init__()
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=betas.dtype), alphas_cumprod[:-1]]
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-20)
        posterior_variance = posterior_variance.clamp(min=1e-8)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_ac", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_om_ac", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)

        self.z_embed = nn.Embedding(max_atomic_num + 2, hidden_dim)
        self.time_emb = TimestepEmbedding(hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([GNNBlock(hidden_dim) for _ in range(num_layers)])
        self.eps_head = EpsFieldHead(hidden_dim)
        self.prop_head = PropertyHead(hidden_dim, out_dim=3)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        batch = batch.long()
        z = z.long().clamp(0, self.z_embed.num_embeddings - 1)
        t = t.long().clamp(0, self.timesteps - 1)
        h = self.z_embed(z) + self.time_emb(t) + self.cond_mlp(cond)[batch]
        for blk in self.layers:
            h = blk(h, pos, edge_index)
        eps_hat = self.eps_head(h, pos, edge_index)
        y_hat = self.prop_head(h, batch)
        return eps_hat, y_hat

    def q_sample(self, pos0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(pos0)
        sqrt_ac = extract(self.sqrt_ac, t, pos0.shape)
        sqrt_om = extract(self.sqrt_om_ac, t, pos0.shape)
        return sqrt_ac * pos0 + sqrt_om * noise, noise

    def predict_x0_from_eps(self, pos_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ac = extract(self.sqrt_ac, t, pos_t.shape)
        sqrt_om = extract(self.sqrt_om_ac, t, pos_t.shape)
        return (pos_t - sqrt_om * eps) / (sqrt_ac + 1e-8)

    def training_losses(
        self,
        data,
        lambda_prop: float = 0.45,
        w_her: float = 1.0,
        w_stab: float = 0.65,
        w_synth: float = 0.55,
    ) -> Tuple[torch.Tensor, dict]:
        device = data.pos.device
        batch = data.batch
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        batch = batch.long()
        num_graphs = int(getattr(data, "num_graphs", int(batch.max().item()) + 1))

        y = data.y
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if y.size(0) != num_graphs and y.numel() == num_graphs * 3:
            y = y.view(num_graphs, 3)

        t_graph = torch.randint(0, self.timesteps, (num_graphs,), device=device, dtype=torch.long)
        max_b = int(batch.max().item()) if batch.numel() > 0 else -1
        if max_b >= num_graphs:
            batch = batch.clamp(min=0, max=num_graphs - 1)
        t_node = t_graph[batch]

        noise = torch.randn_like(data.pos)
        pos_t, noise_target = self.q_sample(data.pos, t_node, noise)

        cond = y
        target = cond.clone()
        eps_hat, y_hat = self.forward(data.x, pos_t, t_node, cond, data.edge_index, batch)
        loss_eps = F.mse_loss(eps_hat, noise_target)

        weights = torch.tensor([w_her, w_stab, w_synth], device=device)
        loss_prop = (((y_hat - target) ** 2) * weights).sum(dim=-1).mean()

        total = loss_eps + lambda_prop * loss_prop
        info = {
            "loss": float(total.detach().cpu()),
            "loss_eps": float(loss_eps.detach().cpu()),
            "loss_prop": float(loss_prop.detach().cpu()),
        }
        return total, info

    @torch.no_grad()
    def p_sample_step(
        self,
        pos: torch.Tensor,
        t: int,
        z: torch.Tensor,
        cond: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = pos.device
        t_tensor = torch.full((pos.size(0),), t, device=device, dtype=torch.long)
        eps_pred, _ = self.forward(z, pos, t_tensor, cond, edge_index, batch)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        ac = self.alphas_cumprod[t]
        coef = self.sqrt_recip_alphas[t]
        coef_eps = beta_t / torch.sqrt(1.0 - ac + 1e-12)
        mean = coef * (pos - coef_eps * eps_pred)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e3, neginf=-1e3)
        mean = torch.clamp(mean, -1e3, 1e3)
        if t > 0:
            noise = torch.randn_like(pos)
            noise = torch.nan_to_num(noise, nan=0.0)
            var = self.posterior_variance[t]
            out = mean + torch.sqrt(var.clamp(min=1e-20)) * noise
            return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)
        return mean
