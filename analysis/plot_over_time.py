#!/usr/bin/env python3
"""
Composite performance vs model API release date.

We produce three plots now:
  1) Default-only (just default_instructions), PROPRIETARY ONLY, scaled by baseline
  2) Both instruction sets (default + emerging), PROPRIETARY ONLY, same baseline
  3) Default-only overlaid by model *type* (Proprietary vs Open Source)
     - Proprietary plotted in RED (replicates Plot 1 dots)
     - Open Source plotted in BLUE, on the SAME y-scale as Plot 1

All plots use the SAME normalization scale, computed from
AI default_instructions rows PLUS Human (MBA) rows.

Filters:
  - difficulty == "Advanced"
  - masked results (everything in this repo)

Inputs:
  - tidy_results.csv (same columns your pipeline writes)

Outputs:
  - bbb_time_def_prop.png
  - bbb_time_both_prop.png
  - bbb_time_def_os.png
"""
import argparse
import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Shared helpers imported from plots.py (unchanged, from your repo)
from plots import (
    load_ai_raw,
    load_human_raw,
    add_norm_and_composite_using_baseline,
    RELEASE_DATES,           # canonical proprietary release dates from your repo
    annotate_scatter_dodge,  # same label-dodge function used elsewhere
    _draw_mba_hlines,        # reuse the MBA reference lines helper
    find_human_csv,          # robust optional MBA path resolver
)

# --- Open Source release dates & fuzzy patterns --------------------------------
# If your tidy_results.csv includes open-source models, we need dates to place them.
# These entries cover common OS families; add/adjust as needed for your dataset.
OPEN_SOURCE_RELEASE_DATES: Dict[str, str] = {
    # Meta / Llama
    "llama-3.1-70b-instruct": "2024-07-23",
    "llama-3.3-70b-instruct": "2024-12-06",
    "llama-4-maverick":       "2025-04-14",
    # Google / Gemma
    "gemma-2-27b-it":         "2024-06-27",
    "gemma-3-12b-it":         "2025-03-10",
    "gemma-3-27b-it":         "2025-03-10",
    # Alibaba / Qwen
    "qwen-2.5-72b-instruct":  "2024-09-19",
    "qwen3-32b":              "2025-04-29",
    "qwen3-235b-a22b":        "2025-04-29",
    # DeepSeek
    "deepseek-v3":            "2024-12-26",
    "deepseek-r1":            "2025-01-20",
    "deepseek-v3.1":          "2025-08-21",
    # OpenAI open-weight (example)
    "gpt-oss-120b":           "2025-08-05",
}

# Fuzzy regex patterns -> dates (lowercased match on model_canon)
_OPEN_SOURCE_DATE_PATTERNS = [
    (r"llama[-_/ ]?3\.1.*70b.*instruct", "2024-07-23"),
    (r"llama[-_/ ]?3\.3.*70b.*instruct", "2024-12-06"),
    (r"llama[-_/ ]?4.*maverick",         "2025-04-14"),
    (r"gemma[-_/ ]?2.*27b.*(it)?",       "2024-06-27"),
    (r"gemma[-_/ ]?3.*(12b|27b).*(it)?", "2025-03-10"),
    (r"qwen[-_/ ]?2\.5.*72b.*instruct",  "2024-09-19"),
    (r"qwen3.*(32b|235b)",               "2025-04-29"),
    (r"deepseek.*v3\.1",                 "2025-08-21"),
    (r"deepseek.*v3",                    "2024-12-26"),
    (r"deepseek.*r1",                    "2025-01-20"),
    (r"gpt[-_/ ]?oss.*120b",             "2025-08-05"),
]

def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).strip().lower()).strip("-")

def _infer_date_from_patterns(name_lc: str) -> Optional[str]:
    for pat, date in _OPEN_SOURCE_DATE_PATTERNS:
        if re.search(pat, name_lc):
            return date
    return None

# --- Lightweight robustness helpers -------------------------------------------
def _ensure_model_canon(df: pd.DataFrame) -> pd.DataFrame:
    """Create model_canon if missing by lightly normalizing `model`."""
    if "model_canon" in df.columns:
        return df
    df = df.copy()
    src = df["model"].astype(str) if "model" in df.columns else pd.Series([""], index=df.index)
    df["model_canon"] = (
        src.str.replace(r"\s+", " ", regex=True)
           .str.replace(r"openrouter/|providers?/|models?/", "", regex=True)
           .str.replace("instruct-turbo", "instruct", regex=False)
           .str.replace("chat-", "", regex=False)
           .str.strip()
    )
    return df

def _filter_advanced_masked(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the original Advanced filter logic, but be tolerant if columns are missing."""
    df = df.copy()
    dif_ok = df["difficulty"].astype(str).eq("Advanced") if "difficulty" in df.columns else pd.Series(True, index=df.index)
    if "mask_mode" in df.columns:
        mask_ok = df["mask_mode"].astype(str).str.lower().eq("masked")
    else:
        mask_ok = pd.Series(True, index=df.index)
    return df.loc[dif_ok & mask_ok].copy()

def _aggregate_mean_composite(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_canon", "instruction_set"]
    agg = {
        "mean_composite": ("composite", "mean"),
        "n_runs": ("composite", "count"),
    }
    if "is_open_source" in df.columns:
        agg["is_open_source"] = ("is_open_source", lambda s: bool(pd.Series(s).astype(bool).mean() >= 0.5))
    if "provider" in df.columns:
        agg["provider"] = ("provider", lambda s: pd.Series(s).mode().iloc[0] if len(pd.Series(s).mode()) else pd.Series(s).iloc[0])
    out = (df.groupby(group_cols, as_index=False).agg(**agg))
    return out

def _combine_release_dates() -> Dict[str, str]:
    """Merge repo's RELEASE_DATES with our open-source dates (without overwriting existing keys)."""
    combined = dict(RELEASE_DATES)
    for k, v in OPEN_SOURCE_RELEASE_DATES.items():
        if k not in combined:
            combined[k] = v
    return combined

def _lookup_release_date_for_name(model_name: str, release_dates_map: Dict[str, str]) -> Optional[str]:
    """Try exact lookup, then fuzzy regex patterns, then normalized-key lookup."""
    if not isinstance(model_name, str) or not model_name:
        return None
    if model_name in release_dates_map:
        return release_dates_map[model_name]
    nk = _normalize_key(model_name)
    if nk in release_dates_map:
        return release_dates_map[nk]
    d = _infer_date_from_patterns(model_name.lower())
    if d:
        return d
    return None

def _attach_release_dates(agg_df: pd.DataFrame, release_dates_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Attach release_date by consulting repo dates + our OS date map + fuzzy matching."""
    if release_dates_map is None:
        release_dates_map = _combine_release_dates()
    models = agg_df["model_canon"].astype(str).unique().tolist()
    dates = []
    missing = []
    for name in models:
        d = _lookup_release_date_for_name(name, release_dates_map)
        if d is None:
            missing.append(name)
        dates.append((name, d))
    if missing:
        print("WARNING: Missing release dates for:", sorted(set(missing)))

    date_map = {k: (pd.to_datetime(v) if v else pd.NaT) for k, v in dates}
    out = agg_df.copy()
    out["release_date"] = pd.to_datetime(out["model_canon"].map(date_map))
    out = out.dropna(subset=["release_date"])
    return out

def _annotate_time_labels(ax, dates, ys, labels, **kwargs):
    xnums = mdates.date2num(pd.to_datetime(dates))
    annotate_scatter_dodge(ax, xnums, ys, labels, **kwargs)

# Styled variant that allows alpha/zorder control and is local to this module
def _annotate_scatter_dodge_styled(
    ax,
    xs,
    ys,
    labels,
    *,
    x_bin=None,
    y_step_pts=8,
    xpad_pts=4,
    fontsize=8,
    text_alpha=1.0,
    bbox_alpha=0.7,
    zorder=None,
):
    """A copy of the label-dodging annotator with styling controls.

    Parameters mirror the helper in analysis/plots.py with added:
      - text_alpha: transparency for text
      - bbox_alpha: transparency for the white label box
      - zorder: z-order for the annotations (to layer behind/above others)
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    labels = list(labels)

    finite = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    labels = [labels[i] for i, ok in enumerate(finite) if ok]

    if len(xs) == 0:
        return

    xr = xs.max() - xs.min()
    if x_bin is None:
        x_bin = max(1e-9, 0.03 * xr)  # 3% of x-range

    bins = np.floor((xs - xs.min()) / x_bin).astype(int)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(ys[idx])]
        n = len(order)
        offsets = (np.arange(n) - (n - 1) / 2.0) * y_step_pts
        for j, off in zip(order, offsets):
            ax.annotate(
                labels[j],
                (xs[j], ys[j]),
                xytext=(xpad_pts, off),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=fontsize,
                alpha=text_alpha,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=bbox_alpha),
                clip_on=True,
                zorder=zorder,
            )

def _annotate_time_labels_styled(ax, dates, ys, labels, **kwargs):
    xnums = mdates.date2num(pd.to_datetime(dates))
    _annotate_scatter_dodge_styled(ax, xnums, ys, labels, **kwargs)

def _build_is_open_source_map(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Determine open-source vs proprietary per model_canon.
    Priority:
      1) use is_open_source column if present
      2) fallback to provider/model name heuristics
    """
    m = {}
    if "is_open_source" in df.columns:
        flag = (df.groupby("model_canon")["is_open_source"]
                  .apply(lambda s: bool(pd.Series(s).astype(bool).mean() >= 0.5)))
        m.update(flag.to_dict())
    else:
        def guess(row) -> bool:
            prov = str(row.get("provider", "")).lower()
            name = str(row.get("model_canon", "")).lower()
            if prov in {"meta-llama", "qwen", "deepseek"}:
                return True
            if prov == "google":
                return "gemma" in name
            if any(tok in name for tok in ["llama", "gemma", "mistral", "mixtral", "qwen", "deepseek", "phi-3", "olmo", "yi", "dbrx", "gpt-oss"]):
                return True
            return False
        flags = df[["model_canon", "provider"]].drop_duplicates().copy()
        flags["is_open_source"] = flags.apply(guess, axis=1)
        m.update(pd.Series(flags["is_open_source"].values, index=flags["model_canon"]).to_dict())
    return m

def make_plots(input_csv: str, output_png_base: str):
    # Load AI data without forcing Run==1 (preserve original behavior)
    df_ai = load_ai_raw(input_csv, use_run1=False)
    df_ai = _ensure_model_canon(df_ai)

    # Apply Advanced + masked filter
    df_filt = _filter_advanced_masked(df_ai)

    # Build OS/proprietary flags from the filtered data (covers both instruction sets)
    is_os_map = _build_is_open_source_map(df_filt)

    # Split by instruction set for baseline and plotting
    df_default_only = df_filt[df_filt["instruction_set"].astype(str) == "default_instructions"].copy()
    if df_default_only.empty:
        print("WARNING: No default_instructions rows found after filters; scaling may be uninformative (0.5).")
        baseline_ai = df_filt.copy()
    else:
        baseline_ai = df_default_only.copy()

    # Load Human (MBA) results and include them in the normalization baseline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    human_csv = find_human_csv(
        os.path.join(script_dir, "data", "past_mba_performance.csv")
    )
    if human_csv:
        human_df_raw = load_human_raw(human_csv)
    else:
        human_df_raw = pd.DataFrame()

    if not human_df_raw.empty:
        baseline = pd.concat([baseline_ai, human_df_raw], ignore_index=True, sort=False)
        human_norm = add_norm_and_composite_using_baseline(human_df_raw, baseline)
    else:
        baseline = baseline_ai
        human_norm = pd.DataFrame()

    # Normalize on baseline and aggregate means (default-only)
    df_default_norm = add_norm_and_composite_using_baseline(df_default_only, baseline)
    agg_def = _aggregate_mean_composite(df_default_norm)

    # Normalize on baseline and aggregate means (both instruction sets)
    df_both_input = df_filt[df_filt["instruction_set"].isin(["default_instructions", "emerging_tech_instructions"])].copy()
    df_both_norm = add_norm_and_composite_using_baseline(df_both_input, baseline)
    agg_both = _aggregate_mean_composite(df_both_norm)

    # Attach release dates (repo map + OS map + fuzzy matching)
    agg_def = _attach_release_dates(agg_def)
    agg_both = _attach_release_dates(agg_both)

    # Flag OS on aggregated tables
    agg_def["is_open_source"] = agg_def["model_canon"].map(is_os_map).fillna(False)
    agg_both["is_open_source"] = agg_both["model_canon"].map(is_os_map).fillna(False)

    # === Plot 1: default-only (PROPRIETARY ONLY) ===
    plt.figure(figsize=(11, 6))
    d = (agg_def[(agg_def["instruction_set"] == "default_instructions") & (~agg_def["is_open_source"])])
    d = d.sort_values("release_date")

    if not d.empty:
        plt.scatter(d["release_date"], d["mean_composite"],
                    label="default instructions (proprietary)",
                    c="red", marker="o", alpha=0.9, s=70, edgecolor="none")
        _annotate_time_labels(
            plt.gca(), d["release_date"], d["mean_composite"], d["model_canon"],
            y_step_pts=10, xpad_pts=6, fontsize=8
        )

    if not human_norm.empty:
        _draw_mba_hlines(plt.gca(), human_norm, "composite", label_base="MBA")

    plt.title("BBB Composite Score Over Time")
    plt.xlabel("Model API release date")
    plt.ylabel("BBB Composite Score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.ylim(-0.05, 0.55)  # fixed scale
    plt.tight_layout()

    # Interpret output_png_base as a *base path* and generate three related filenames.
    # - If you pass ".../bbb_time" -> writes ".../bbb_time_def_prop.png", etc.
    # - If you pass ".../bbb_time.png" -> writes ".../bbb_time_def_prop.png", etc.
    base = str(output_png_base)
    if base.lower().endswith(".png"):
        base = base[:-4]

    out_png_default = f"{base}_def_prop.png"
    plt.savefig(out_png_default, dpi=180, bbox_inches="tight")
    print("Wrote plot to:", out_png_default)

    # === Plot 2: both instruction sets (PROPRIETARY ONLY) ===
    plt.figure(figsize=(11, 6))
    color_map = {"default_instructions": "red", "emerging_tech_instructions": "blue"}
    marker_map = {"default_instructions": "o",   "emerging_tech_instructions": "s"}

    agg_both_prop = agg_both[~agg_both["is_open_source"]].copy()

    for instr in ["default_instructions", "emerging_tech_instructions"]:
        d2 = agg_both_prop[agg_both_prop["instruction_set"] == instr].sort_values("release_date")
        if d2.empty:
            continue
        # De-emphasize proprietary default (red) points
        alpha_val = 0.45 if instr == "default_instructions" else 0.9
        plt.scatter(d2["release_date"], d2["mean_composite"],
                    label=instr.replace("_", " "),
                    c=color_map.get(instr, "gray"),
                    marker=marker_map.get(instr, "o"),
                    alpha=alpha_val, s=70, edgecolor="none")

    # Connect pairs vertically per model (only within proprietary subset when both points share the same date)
    agg_sorted = agg_both_prop.sort_values("release_date")
    for model, dd in agg_sorted.groupby("model_canon"):
        if set(dd["instruction_set"]) == {"default_instructions", "emerging_tech_instructions"} and dd["release_date"].nunique() == 1:
            x = dd["release_date"].iloc[0]
            y0 = float(dd.loc[dd["instruction_set"] == "default_instructions", "mean_composite"])
            y1 = float(dd.loc[dd["instruction_set"] == "emerging_tech_instructions", "mean_composite"])
            plt.plot([x, x], [y0, y1], linewidth=1, alpha=0.5)

    # Label near the higher point for each model (proprietary subset)
    label_rows = []
    for model, dd in agg_sorted.groupby("model_canon"):
        row = dd.sort_values("mean_composite").iloc[-1]
        label_rows.append(row)
    if label_rows:
        lab_df = pd.DataFrame(label_rows)
        _annotate_time_labels(
            plt.gca(), lab_df["release_date"], lab_df["mean_composite"], lab_df["model_canon"],
            y_step_pts=10, xpad_pts=6, fontsize=8
        )

    if not human_norm.empty:
        _draw_mba_hlines(plt.gca(), human_norm, "composite", label_base="MBA")

    plt.title("BBB Composite Score Over Time")
    plt.xlabel("Model API release date")
    plt.ylabel("BBB Composite Score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(title="Instruction set", frameon=False)
    plt.ylim(-0.05, 0.55)  # fixed scale
    plt.tight_layout()

    out_png_both = f"{base}_both_prop.png"
    plt.savefig(out_png_both, dpi=180, bbox_inches="tight")
    print("Wrote plot to:", out_png_both)

    # === Plot 3: default-only, overlay Open Source (blue) vs Proprietary (red) ===
    plt.figure(figsize=(11, 6))

    d3 = agg_def[agg_def["instruction_set"] == "default_instructions"].copy()
    d3 = d3.sort_values("release_date")

    # Proprietary (RED dots) – replicate Plot 1 but de-emphasized
    prop = d3[~d3["is_open_source"]]
    if not prop.empty:
        plt.scatter(prop["release_date"], prop["mean_composite"],
                    label="Proprietary (default only)",
                    c="red", marker="o", alpha=0.45, s=70, edgecolor="none", zorder=2)
        # Draw proprietary labels with lower alpha and lower zorder (behind OS)
        _annotate_time_labels_styled(
            plt.gca(),
            prop["release_date"],
            prop["mean_composite"],
            prop["model_canon"],
            y_step_pts=10,
            xpad_pts=6,
            fontsize=8,
            text_alpha=0.35,
            bbox_alpha=0.35,
            zorder=2,
        )

    # Open Source (BLUE dots)
    oss = d3[d3["is_open_source"]]
    if not oss.empty:
        plt.scatter(oss["release_date"], oss["mean_composite"],
                    label="Open Source (default only)",
                    c="blue", marker="o", alpha=0.95, s=70, edgecolor="none", zorder=3)
        # Draw OS labels above proprietary labels
        _annotate_time_labels_styled(
            plt.gca(),
            oss["release_date"],
            oss["mean_composite"],
            oss["model_canon"],
            y_step_pts=10,
            xpad_pts=6,
            fontsize=8,
            text_alpha=1.0,
            bbox_alpha=0.7,
            zorder=4,
        )

    if not human_norm.empty:
        _draw_mba_hlines(plt.gca(), human_norm, "composite", label_base="MBA")

    plt.title("BBB Composite Score Over Time (Open Source overlay)")
    plt.xlabel("Model API release date")
    plt.ylabel("BBB Composite Score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.ylim(-0.05, 0.55)  # EXACT same y-scale as Plot 1
    # Extend x-axis to the right to avoid clipping far-right labels
    ax = plt.gca()
    if not d3.empty:
        right_pad_days = 45
        left_pad_days = 5
        xmin = pd.to_datetime(d3["release_date"]).min() - pd.Timedelta(days=left_pad_days)
        xmax = pd.to_datetime(d3["release_date"]).max() + pd.Timedelta(days=right_pad_days)
        # Ensure right bound reaches at least 2026-01-01
        target_right = pd.Timestamp("2026-01-01")
        if xmax < target_right:
            xmax = target_right
        ax.set_xlim(xmin, xmax)
    plt.tight_layout()

    out_png_os = f"{base}_def_os.png"
    plt.savefig(out_png_os, dpi=180, bbox_inches="tight")
    print("Wrote plot to:", out_png_os)

def main():
    parser = argparse.ArgumentParser()
    # Keep original defaults for backwards compatibility with your repo
    parser.add_argument("--input_csv", default="analysis/data/tidy_results.csv")
    parser.add_argument("--output_png", default="analysis/figures/bbb_time")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    make_plots(args.input_csv, args.output_png)

if __name__ == "__main__":
    main()
