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


def _pick_total_loss_col(df: pd.DataFrame) :
    # 优先顺序：你原来的 loss_g_total 放第一
    candidates = [
        "loss_g_total",
        "loss_total",
        "total_loss",
        "train_loss",
        "loss",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _plot_total_loss(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_png: str,
    out_pdf: str,
    dpi: int = 400,
):
    plt.figure()
    plt.plot(df[x_col], df[y_col], label=y_col)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.savefig(out_pdf)  # vector
    plt.close()
    print(f"[OK] Saved: {out_png} and {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot training convergence curve (total loss only) from loss_log.csv.")
    parser.add_argument("--stage1", type=str, default="H:/BaiduNetdiskDownload/frame64.csv", help="Path to stage1 (64) loss_log.csv")
    parser.add_argument("--stage2", type=str, default="", help="Path to stage2 (128) loss_log.csv")
    parser.add_argument("--stage3", type=str, default="", help="Path to stage3 (256 frame) loss_log.csv")
    parser.add_argument("--stage4", type=str, default="", help="Path to stage4 (clip) loss_log.csv")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for figures")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window (rolling mean). Set 1 to disable.")
    parser.add_argument("--dpi", type=int, default=400, help="PNG DPI (JEI recommends 300–600).")
    args = parser.parse_args()

    outdir = args.outdir

    # --------- Frame stages: only total loss ---------
    for idx, (csv_path, name) in enumerate(
        [(args.stage1, "Convergence"),
         (args.stage2, "Stage-2 (128×128 mouth)"),
         (args.stage3, "Stage-3 (256×256 mouth, frame)")],
        start=1
    ):
        if not csv_path:
            continue
        df = _read_csv(csv_path)
        df_s = _smooth(df, args.smooth)

        y = _pick_total_loss_col(df_s)
        if y is None:
            print(f"[Skip] No total loss column found in stage{idx}. Columns: {list(df_s.columns)}")
            continue

        base = os.path.join(outdir, f"convergence_stage{idx}_total")
        _plot_total_loss(
            df_s, "global_step", y,
            f"{name}: Total loss",
            base + ".png",
            base + ".pdf",
            dpi=args.dpi
        )

    # --------- Clip stage: only total loss ---------
    if args.stage4:
        df = _read_csv(args.stage4)
        df_s = _smooth(df, args.smooth)

        y = _pick_total_loss_col(df_s)
        if y is None:
            print(f"[Skip] No total loss column found in stage4. Columns: {list(df_s.columns)}")
            return

        base = os.path.join(outdir, "convergence_stage4_clip_total")
        _plot_total_loss(
            df_s, "global_step", y,
            "Stage-4 (clip): Total loss",
            base + ".png",
            base + ".pdf",
            dpi=args.dpi
        )


if __name__ == "__main__":
    main()
