#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "global_step" not in df.columns:
        # fallback: use step/iter if global_step missing
        if "step" in df.columns:
            df["global_step"] = df["step"]
        elif "iter" in df.columns:
            df["global_step"] = df["iter"]
        else:
            df["global_step"] = range(len(df))
    return df


def _smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    return df.rolling(window=window, min_periods=1).mean()


def _plot_one(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    out_png: str,
    out_pdf: str,
    dpi: int = 400,
):
    # Keep only existing columns
    y_cols_exist = [c for c in y_cols if c in df.columns]
    if not y_cols_exist:
        print(f"[Skip] No target columns found for plot '{title}'. Wanted: {y_cols}")
        return

    plt.figure()
    for c in y_cols_exist:
        plt.plot(df[x_col], df[c], label=c)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(title)
    if len(y_cols_exist) > 1:
        plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.savefig(out_pdf)  # vector
    plt.close()
    print(f"[OK] Saved: {out_png} and {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot DINet/DFSA training convergence curves from loss_log.csv.")
    parser.add_argument("--stage1", type=str, default="H:/BaiduNetdiskDownload/zong.csv", help="Path to stage1 (64) loss_log.csv")
    parser.add_argument("--stage2", type=str, default="H:/BaiduNetdiskDownload/frame2.csv", help="Path to stage2 (128) loss_log.csv")
    parser.add_argument("--stage3", type=str, default="H:/BaiduNetdiskDownload/frame3.csv", help="Path to stage3 (256 frame) loss_log.csv")
    parser.add_argument("--stage4", type=str, default="H:/BaiduNetdiskDownload/clip4.csv", help="Path to stage4 (clip) loss_log.csv")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for figures")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window (rolling mean). Set 1 to disable.")
    parser.add_argument("--dpi", type=int, default=400, help="PNG DPI (JEI recommends 300–600).")
    args = parser.parse_args()

    outdir = args.outdir

    # ---- Frame stages (often have: loss_g_total, loss_perc, loss_g_dI, loss_dI) ----
    frame_targets = ["loss_g_total", "loss_perc", "loss_g_dI"]  # minimal + informative
    disc_targets = ["loss_dI"]  # optional

    for idx, (csv_path, name) in enumerate(
        [(args.stage1, "Stage-1 (64×64 mouth)"),
         (args.stage2, "Stage-2 (128×128 mouth)"),
         (args.stage3, "Stage-3 (256×256 mouth, frame)")],
        start=1
    ):
        if not csv_path:
            continue
        df = _read_csv(csv_path)
        df_s = _smooth(df, args.smooth)

        base = os.path.join(outdir, f"convergence_stage{idx}")
        _plot_one(
            df_s, "global_step",
            frame_targets,
            f"{name}: Generator losses",
            base + "_G.png",
            base + "_G.pdf",
            dpi=args.dpi
        )
        _plot_one(
            df_s, "global_step",
            disc_targets,
            f"{name}: Discriminator loss",
            base + "_D.png",
            base + "_D.pdf",
            dpi=args.dpi
        )

    # ---- Clip stage (often has: loss_g_total, loss_sync, loss_perc, loss_g_dI, loss_g_dV, loss_dI, loss_dV) ----
    if args.stage4:
        df = _read_csv(args.stage4)
        df_s = _smooth(df, args.smooth)

        # Clip stage plots: (1) G_total + sync (2) GAN terms (optional) (3) discriminators (optional)
        base = os.path.join(outdir, "convergence_stage4_clip")

        _plot_one(
            df_s, "global_step",
            ["loss_g_total", "loss_sync"],
            "Stage-4 (clip): Total loss and sync loss",
            base + "_G_sync.png",
            base + "_G_sync.pdf",
            dpi=args.dpi
        )

        _plot_one(
            df_s, "global_step",
            ["loss_perc", "loss_g_dI", "loss_g_dV"],
            "Stage-4 (clip): Generator components",
            base + "_G_components.png",
            base + "_G_components.pdf",
            dpi=args.dpi
        )

        _plot_one(
            df_s, "global_step",
            ["loss_dI", "loss_dV"],
            "Stage-4 (clip): Discriminator losses",
            base + "_D.png",
            base + "_D.pdf",
            dpi=args.dpi
        )


if __name__ == "__main__":
    main()
