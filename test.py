"""加载权重、条件生成结构、评估 baseline 与 ours，并导出可视化。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from pymatgen.io.cif import CifWriter

# 当前文件就在根目录下，所以用 .parent 即可
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

# 去掉了 project. 前缀
from dataset.material_dataset import MaterialCrystalDataset, collect_cif_paths
from models.diffusion_model import CrystalDiffusion
from models.optimization import (
    assemble_structure,
    coordinate_jitter_refine,
    feasible_target_cond,
    multiobjective_score,
    rerank_structures,
)
from models.structure_generator import StructureGenerator
from utils import geo_utils
from utils.vis import (
    plot_generated_structures_panel,
    plot_her_performance,
    plot_stability_dual,
)


def _render_structure_png(struct, out_png: Path) -> None:
    """用 xy / xz / yz 三幅 2D 投影拼图保存缩略图（Agg 下比 3D 可靠）。"""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(struct.cart_coords, dtype=np.float64)
    zs = np.array([site.specie.Z for site in struct], dtype=np.float64)
    ok = np.isfinite(coords).all(axis=1) & np.isfinite(zs)
    coords = coords[ok]
    zs = zs[ok]
    if len(coords) == 0:
        fig, ax = plt.subplots(figsize=(2.0, 1.0))
        ax.text(0.5, 0.5, "no valid coords", ha="center", va="center", fontsize=8)
        ax.axis("off")
        fig.savefig(out_png, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return

    c = coords - coords.mean(axis=0)
    zmin, zmax = float(zs.min()), float(zs.max())
    if zmax <= zmin:
        zmax = zmin + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.85), dpi=130)
    proj = [
        (0, 1, "x (Å)", "y (Å)"),
        (0, 2, "x (Å)", "z (Å)"),
        (1, 2, "y (Å)", "z (Å)"),
    ]
    for ax, (i, j, lx, ly) in zip(axes, proj):
        u, v = c[:, i], c[:, j]
        ax.scatter(
            u,
            v,
            c=zs,
            cmap="coolwarm",
            vmin=zmin,
            vmax=zmax,
            s=90,
            linewidths=0.9,
            edgecolors="black",
            zorder=5,
            clip_on=True,
        )
        u_min, u_max = float(np.min(u)), float(np.max(u))
        v_min, v_max = float(np.min(v)), float(np.max(v))
        du = max(u_max - u_min, 1e-9)
        dv = max(v_max - v_min, 1e-9)
        pad = max(0.2, 0.1 * max(du, dv))
        cu = 0.5 * (u_min + u_max)
        cv = 0.5 * (v_min + v_max)
        R = 0.5 * max(du, dv) + pad
        ax.set_xlim(cu - R, cu + R)
        ax.set_ylim(cv - R, cv + R)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(lx, fontsize=7)
        ax.set_ylabel(ly, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.35)
    fig.suptitle("Projections", fontsize=8, y=1.02)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(out_png, facecolor="white", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_source",
        type=str,
        default="cif",
        choices=["cif", "materials_project"],
        help="模板晶体来源：本地 CIF 或 Materials Project",
    )
    # 【已修改】默认指向 data/mp_cache
    ap.add_argument("--cif_dir", type=str, default=str(_ROOT / "data" / "mp_cache"))
    ap.add_argument("--mp_api_key", type=str, default="", help="默认同环境变量 MP_API_KEY")
    ap.add_argument(
        "--mp_cache_dir",
        type=str,
        default=str(_ROOT / "data" / "mp_cache"),
        help="MP 结构 CIF 缓存目录",
    )
    ap.add_argument("--mp_eah_max", type=float, default=0.1)
    # 【已修改】默认指向 data/mp_cache
    ap.add_argument("--baseline_dir", type=str, default=str(_ROOT / "data" / "mp_cache"))
    ap.add_argument("--ckpt", type=str, default=str(Path(__file__).resolve().parent / "results" / "checkpoint.pt"))
    ap.add_argument("--num_gen", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--max_atoms", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=320)
    ap.add_argument("--max_atomic_num", type=int, default=118)
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--candidate_multiplier", type=int, default=4)
    ap.add_argument("--target_stab_sigma", type=float, default=0.35)
    ap.add_argument("--target_synth_sigma", type=float, default=0.25)
    ap.add_argument("--target_max_sigma", type=float, default=0.8)
    ap.add_argument("--template_pool_multiplier", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    struct_dir = out_dir / "generated_cifs"
    thumb_dir = out_dir / "structure_thumbs"
    struct_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    timesteps = int(hp.get("timesteps", args.timesteps))
    hidden = int(hp.get("hidden", args.hidden))
    layers = int(hp.get("layers", args.layers))
    max_atomic_num = int(hp.get("max_atomic_num", args.max_atomic_num))

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
    y_mean = np.array(ckpt["y_mean"], dtype=np.float64)
    y_std = np.array(ckpt["y_std"], dtype=np.float64)

    model = CrystalDiffusion(
        max_atomic_num=max_atomic_num,
        hidden_dim=hidden,
        num_layers=layers,
        timesteps=timesteps,
    )
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    gen = StructureGenerator(model)

    cond = feasible_target_cond(
        y_mean,
        y_std,
        dg_h_goal=0.0,
        stab_sigma=args.target_stab_sigma,
        synth_sigma=args.target_synth_sigma,
        max_sigma=args.target_max_sigma,
    )

    bpaths = collect_cif_paths(args.baseline_dir, limit=60)
    b_dg, b_stab, b_syn = [], [], []
    for p in bpaths:
        try:
            mm = geo_utils.evaluate_cif(p)
        except Exception:
            continue
        b_dg.append(mm.dg_h_proxy)
        b_stab.append(0.5 * (mm.thermo_stability + mm.dyn_stability))
        b_syn.append(mm.synthesis_score)

    stab_floor = float(np.mean(b_stab)) if b_stab else float(y_mean[1])
    synth_floor = float(np.mean(b_syn)) if b_syn else float(y_mean[2])

    rng = random.Random(42)
    template_scored = []
    for idx in range(len(ds)):
        tpl = ds[idx]
        y_raw = tpl.y_raw.detach().cpu().numpy()
        tpl_dg = float(y_raw[0])
        tpl_stab = float(y_raw[1])
        tpl_synth = float(y_raw[2])
        floor_penalty = max(0.0, stab_floor - tpl_stab) + 1.25 * max(0.0, synth_floor - tpl_synth)
        template_score = -abs(tpl_dg) + 1.2 * tpl_stab + 1.35 * tpl_synth - 4.0 * floor_penalty
        template_scored.append((template_score, rng.random(), idx))

    template_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    template_pool = min(
        len(template_scored),
        max(args.num_gen * args.template_pool_multiplier, args.num_gen * args.candidate_multiplier, 16),
    )
    preferred_indices = [idx for _, _, idx in template_scored[:template_pool]]
    rng.shuffle(preferred_indices)
    num_candidates = min(len(preferred_indices), max(args.num_gen * args.candidate_multiplier, args.num_gen, 10))
    chosen = preferred_indices[:num_candidates]

    generated_items = []
    for idx in chosen:
        tpl = ds[idx]
        pos_new = gen.sample(tpl, cond, args.device)
        lat = tpl.lattice.numpy()
        tpl_pos = tpl.pos
        variant_positions = [
            pos_new,
            0.75 * tpl_pos + 0.25 * pos_new,
            0.50 * tpl_pos + 0.50 * pos_new,
            0.25 * tpl_pos + 0.75 * pos_new,
            tpl_pos,
        ]

        best_pair = None
        best_score = -float("inf")
        for pos_variant in variant_positions:
            s = assemble_structure(str(tpl.cif_path), pos_variant, tpl.recenter, lat)
            s_ref = coordinate_jitter_refine(
                s,
                steps=60,
                sigma=0.010,
                stab_floor=stab_floor,
                synth_floor=synth_floor,
            )
            m = geo_utils.evaluate_structure(s_ref)
            sc = multiobjective_score(m, stab_floor=stab_floor, synth_floor=synth_floor)
            if sc > best_score:
                best_pair = (s_ref, m)
                best_score = sc

        if best_pair is not None:
            generated_items.append(best_pair)

    ranked = rerank_structures(
        generated_items,
        top_k=min(args.num_gen, len(generated_items)),
        stab_floor=stab_floor,
        synth_floor=synth_floor,
    )
    selected_items = [(s, m) for s, m, _ in ranked]

    thumbs = []
    for i, (s, m) in enumerate(selected_items):
        cif_path = struct_dir / f"gen_{i:03d}.cif"
        CifWriter(s).write_file(str(cif_path))
        p = thumb_dir / f"g_{i:02d}.png"
        _render_structure_png(s, p)
        thumbs.append(p)

    ours_dg = [m.dg_h_proxy for _, m in selected_items]
    ours_stab = [0.5 * (m.thermo_stability + m.dyn_stability) for _, m in selected_items]
    ours_synth = [m.synthesis_score for _, m in selected_items]

    plot_her_performance(ours_dg + b_dg, ["ours"] * len(ours_dg) + ["baseline(data)"] * len(b_dg), out_dir / "her_performance.png")
    n = min(len(b_stab), len(ours_stab))
    plot_stability_dual(
        {
            "step": np.arange(len(ours_stab)),
            "stab": np.asarray(ours_stab, dtype=np.float32),
            "synth": np.asarray(ours_synth, dtype=np.float32),
        },
        {
            "step": np.arange(n),
            "stab": np.asarray(b_stab[:n], dtype=np.float32),
            "synth": np.asarray(b_syn[:n], dtype=np.float32),
        },
        out_dir / "stability_curve.png",
    )
    plot_generated_structures_panel(thumbs, out_dir / "generated_structures.png")

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "ours": {
            "avg_dg_h_proxy": _mean(ours_dg),
            "avg_stability": _mean(ours_stab),
            "avg_synthesis": _mean(ours_synth),
            "n": len(ours_dg),
            "candidate_pool": len(generated_items),
            "template_pool": template_pool,
            "target_cond_normalized": cond.squeeze(0).cpu().tolist(),
            "selection_floors": {
                "stability": stab_floor,
                "synthesis": synth_floor,
            },
        },
        "baseline_sample": {
            "avg_dg_h_proxy": _mean(b_dg),
            "avg_stability": _mean(b_stab),
            "avg_synthesis": _mean(b_syn),
            "n": len(b_dg),
        },
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("评估摘要（代理指标）：")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"结构已写入: {struct_dir}")


if __name__ == "__main__":
    main()
