"""训练 GNN+DDPM 多任务材料生成模型，并保存检查点与损失曲线。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader

# 【已修改】当前文件就在根目录下，所以用 .parent 即可
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

# 【已修改】去掉了 project. 前缀
from dataset.material_dataset import MaterialCrystalDataset
from models.diffusion_model import CrystalDiffusion
from utils.vis import plot_loss_curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_source",
        type=str,
        default="cif",
        choices=["cif", "materials_project"],
        help="cif：本地目录；materials_project：Materials Project API（需 MP_API_KEY 或 --mp_api_key）",
    )
    ap.add_argument(
        "--cif_dir",
        type=str,
        default=str(_ROOT / "generated_materials" / "cif_files"),
        help="含 CIF 的目录（递归扫描）；仅 data_source=cif 时使用",
    )
    ap.add_argument(
        "--mp_api_key",
        type=str,
        default="",
        help="Materials Project API 密钥；默认同环境变量 MP_API_KEY",
    )
    ap.add_argument(
        "--mp_cache_dir",
        type=str,
        default=str(_ROOT / "data" / "mp_cache"), # 【已修改】统一为 mp_cache
        help="从 MP 拉取的结构写入此目录（CIF 缓存）",
    )
    ap.add_argument(
        "--mp_eah_max",
        type=float,
        default=0.1,
        help="MP 筛选：energy_above_hull 上限（eV/atom）",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--max_atoms", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=320)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lambda_prop", type=float, default=0.35)
    ap.add_argument("--w_her", type=float, default=1.0)
    ap.add_argument("--w_stab", type=float, default=1.1)
    ap.add_argument("--w_synth", type=float, default=0.9)
    ap.add_argument(
        "--max_atomic_num",
        type=int,
        default=118,
        help="元素序数上界（周期表最大 Z=118），需与 Embedding 大小一致",
    )
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    mp_key = (args.mp_api_key or "").strip() or os.getenv("MP_API_KEY", "").strip()
    if args.data_source == "materials_project":
        ds = MaterialCrystalDataset(
            source="materials_project",
            mp_api_key=mp_key,
            mp_cache_dir=args.mp_cache_dir,
            mp_eah_max=args.mp_eah_max,
            max_atoms=args.max_atoms,
            max_samples=args.max_samples,
        )
    else:
        ds = MaterialCrystalDataset(
            cif_root=args.cif_dir,
            max_atoms=args.max_atoms,
            max_samples=args.max_samples,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = CrystalDiffusion(
        max_atomic_num=args.max_atomic_num,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        timesteps=args.timesteps,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss, info = model.training_losses(
                batch,
                lambda_prop=args.lambda_prop,
                w_her=args.w_her,
                w_stab=args.w_stab,
                w_synth=args.w_synth,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            running += info["loss"]
            n += 1
        avg = running / max(n, 1)
        train_losses.append(avg)
        print(f"epoch {epoch+1}/{args.epochs}  loss={avg:.5f}")

    hparams_safe = vars(args).copy()
    if hparams_safe.get("mp_api_key"):
        hparams_safe["mp_api_key"] = "<redacted>"
    torch.save(
        {
            "model": model.state_dict(),
            "y_mean": ds.y_mean.tolist(),
            "y_std": ds.y_std.tolist(),
            "hparams": hparams_safe,
        },
        ckpt_path,
    )
    plot_loss_curve(train_losses, None, out_dir / "loss_curve.png")
    with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"train_losses": train_losses}, f, indent=2)
    print(f"已保存: {ckpt_path}")


if __name__ == "__main__":
    main()
