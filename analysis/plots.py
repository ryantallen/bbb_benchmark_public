#!/usr/bin/env python3
"""
Back Bay Battery – Plot suite and shared utilities.

This module provides:
  - Parsing & canonical-name helpers
  - Data loading from tidy_results.csv (+ optional human benchmark)
  - Normalization helpers
  - Label-dodge annotator for crowded scatter plots
  - Plot entrypoints:
        plot_2x2, plot_composite,
        plot_composite_vs_gpqa, plot_composite_vs_lmarena
  - Script entry: generates a small, curated subset of plots used in the paper

Run:
  python plots.py
"""
from __future__ import annotations

import os, re, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory as _blend

# ----------------- hard-coded paths -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

AI_CSV    = os.path.join(SCRIPT_DIR, "data", "tidy_results.csv")
HUMAN_CSV = os.path.join(SCRIPT_DIR, "inputs", "past_mba_performance.csv")

OUTDIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUTDIR, exist_ok=True)
# Base output names (suffixes are appended in main()).
# Naming convention:
#   - 2x2 panels: bbb_2x2_<diff>_<instr>[_os].png
#   - composite:  bbb_comp_<diff>_<instr>[_os].png
#   - benchmarks: bbb_comp_<bench>_<diff>_<instr>.png   (bench ∈ {gpqa,lma})
OUT_2X2 = os.path.join(OUTDIR, "bbb_2x2.png")
OUT_COMP= os.path.join(OUTDIR, "bbb_comp.png")
OUT_GPQA = os.path.join(OUTDIR, "bbb_comp_gpqa.png")
OUT_LMARENA = os.path.join(OUTDIR, "bbb_comp_lma.png")

# --- visual defaults (tweak these numbers) ---
BASE_FONTSIZE = 12
TITLE_SIZE   = 15
LABEL_SIZE   = 13
TICK_SIZE    = 12
LEGEND_SIZE  = 14

# --- layout spacing controls ---
SUPTITLE_Y                 = 0.985
LEGEND_Y                   = 0.955
SUBPLOTS_TOP               = 0.86
SUBPLOTS_LEFT              = 0.1
SUBPLOTS_RIGHT             = 0.92
SUBPLOTS_BOTTOM_COMPOSITE  = 0.25
SUBPLOTS_BOTTOM_2X2        = 0.16
GRID_HSPACE                = 0.58
GRID_WSPACE                = 0.25

TOP_ROW_XROT   = 35
BOTTOM_ROW_XROT= 45

mpl.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

# ---------- helpers -----------------------------------------------------------

def find_human_csv(explicit_path: str | None = None) -> str | None:
    """
    Return the first existing path to the (optional) MBA benchmark CSV.

    This repo is sometimes distributed without the MBA file; callers should treat
    a None return value as "no human benchmark available" and proceed.
    """
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)

    # Historical / repo variants (keep both to be robust across branches)
    candidates.extend([
        HUMAN_CSV,
        os.path.join(SCRIPT_DIR, "data", "past_mba_performance.csv"),
        os.path.join(SCRIPT_DIR, "inputs", "past_mba_performance.csv"),
        os.path.join(SCRIPT_DIR, "inputs", "data", "past_mba_performance.csv"),
    ])

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def parse_money(cell):
    """Convert '$427.5 M', '(–$32k)', '1,234,567', '12.3%' → float. (% → fraction)"""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return np.nan
    if isinstance(cell, (int, float)):
        return float(cell)
    s = str(cell).strip()
    if not s:
        return np.nan
    s_norm = s.replace('–', '-').replace('—', '-').strip()
    if s_norm.endswith("%"):
        try:
            return float(s_norm.strip("%").replace(",", "")) / 100.0
        except ValueError:
            return np.nan
    sign = -1 if (s_norm.startswith('(') and s_norm.endswith(')')) or s_norm.startswith('-') else 1
    core = s_norm.strip('()$€£¥-').replace(',', '').replace(' ', '')
    mult = 1.0
    if core.lower().endswith('m'):
        mult, core = 1e6, core[:-1]
    elif core.lower().endswith('k'):
        mult, core = 1e3, core[:-1]
    try:
        return sign * float(core) * mult
    except ValueError:
        return np.nan

_DATE_SUFFIX = re.compile(r"-(?:19|20)\d{2}-\d{2}-\d{2}$")
_NUMERIC_TAIL= re.compile(r"-\d{2,8}$")

def canonical_model_name(model: str) -> str:
    """Normalize provider/model identifiers to a canonical short name."""
    if not isinstance(model, str):
        return str(model)
    name = model.strip()
    if "/" in name:
        name = name.split("/")[-1]
    name = _DATE_SUFFIX.sub("", name)
    name = _NUMERIC_TAIL.sub("", name)
    return name

def short_model_label(model: str) -> str:
    """Compact, human-friendly label for crowded plots (derived from canonical name)."""
    if model is None:
        return ""
    name = canonical_model_name(str(model)).lower()
    overrides = {
        "gpt-3.5": "gpt-3.5-turbo",
    }
    if name in overrides:
        return overrides[name]
    token_map = {
        "claude": "cl",
        "sonnet": "snt",
        "opus": "op",
        "haiku": "hq",
        "gemini": "gm",
        "flash": "fl",
    }
    parts = re.split(r"[-_ ]+", name)
    parts = [token_map.get(p, p) for p in parts if p]
    return "-".join(parts)

def ci95(series: pd.Series) -> float:
    n = series.count()
    if n <= 1:
        return 0.0
    return 1.96 * series.std(ddof=1) / math.sqrt(n)

def money_formatter():
    def _fmt(x, pos):
        ax = abs(x)
        if ax >= 1e9: return f"${x/1e9:.1f}B"
        if ax >= 1e6: return f"${x/1e6:.1f}M"
        if ax >= 1e3: return f"${x/1e3:.0f}k"
        return f"${x:.0f}"
    return FuncFormatter(_fmt)

def _draw_mba_hlines(ax, human_df: pd.DataFrame, ycol: str, label_base: str = "MBA"):
    """
    Draw dotted horizontal lines at the min/mean/max of human_df[ycol] across the full width,
    and label them *outside* the axes on the right side.
    """
    if human_df is None or human_df.empty or (ycol not in human_df.columns):
        return

    y = pd.to_numeric(human_df[ycol], errors="coerce").dropna()
    if y.empty:
        return

    y_min, y_mean, y_max = float(y.min()), float(y.mean()), float(y.max())
    specs = [
        (y_min,  f"min {label_base}",  1.0, 0.65),
        (y_mean, f"mean {label_base}", 1.4, 0.95),
        (y_max,  f"max {label_base}",  1.0, 0.65),
    ]

    # 1) dotted lines across the panel
    for val, _, lw, a in specs:
        ax.axhline(val, color="black", linestyle=":", linewidth=lw, alpha=a, zorder=2)

    # 2) labels just outside the right edge (x in axes coords, y in data coords)
    trans = _blend(ax.transAxes, ax.transData)
    for val, txt, _, _ in specs:
        ax.annotate(
            txt,
            xy=(1.0, val), xycoords=trans,        # right edge of axes
            xytext=(6, 0), textcoords="offset points",  # nudge into the margin
            ha="left", va="center",
            color="black", fontsize=TICK_SIZE,
            annotation_clip=False, zorder=5
        )

def _annotate_scatter_dodge(ax, xs, ys, labels, *, x_bin=None, y_step_pts=8, xpad_pts=4, fontsize=TICK_SIZE):
    """
    Place one text label per point, but 'dodge' vertically within local x-bins
    so labels don't sit on top of each other.

    Parameters
    ----------
    ax : matplotlib Axes
    xs, ys : array-like numeric
        X *must* be numeric (convert datetimes prior to calling).
    labels : sequence[str]
    x_bin : float | None
        Width of the x-bins in data units. If None → ~3% of the x-range.
    y_step_pts : float
        Vertical spacing between labels (in points).
    xpad_pts : float
        Horizontal padding from the point (in points).
    fontsize : float
        Font size to use for labels (defaults to tick size).
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
        # order by y so we can fan labels out around the group’s center
        order = idx[np.argsort(ys[idx])]
        n = len(order)
        offsets = (np.arange(n) - (n-1)/2.0) * y_step_pts
        for j, off in zip(order, offsets):
            ax.annotate(labels[j], (xs[j], ys[j]),
                        xytext=(xpad_pts, off),
                        textcoords="offset points",
                        ha="left", va="center",
                        fontsize=fontsize,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                        clip_on=True)

# Friendly alias without the leading underscore
def annotate_scatter_dodge(ax, xs, ys, labels, **kwargs):
    return _annotate_scatter_dodge(ax, xs, ys, labels, **kwargs)

# ---------- External benchmark lookups ---------------------------------------

GPQA_DIAMOND_LATEST = {
    # Source: https://llm-stats.com/benchmarks/gpqa
    # Date recorded: Jan 5, 2026

    # OpenAI
    "gpt-5.2": 92.4,
    "gpt-5": 85.7,
    "o3": 83.3,
    "o3-mini": 77.2,
    "o4-mini": 81.4,
    "gpt-4-turbo": 48.0,
    "gpt-4o": 70.1,
    "gpt-3.5-turbo": 30.8,


    # Anthropic
    "claude-3-5-sonnet": 67.2,
    "claude-3-7-sonnet": 84.8,
    "claude-sonnet-4": 75.4,
    "claude-sonnet-4-5": 83.4,
    "claude-opus-4-5": 87.0,

    # Google
    "gemini-1.5-flash": 51.0,
    "gemini-2.0-flash": 62.1,
    "gemini-2.5-flash": 82.8,
    "gemini-2.5-pro": 83.0,
    "gemini-3-pro-preview": 91.9,

    # xAI
    "grok-4": 87.5,
    "grok-3": 84.6,
    "grok-3-mini": 84.0
}

LMARENA_LATEST = {
    # Source: https://lmarena.ai/leaderboard
    # Date recorded: Jan 5, 2026
    # OpenAI
    "gpt-5": 1425,
    "o3": 1434,
    "o3-mini": 1348,
    "o4-mini": 1391,
    "gpt-4-turbo": 1324,
    "gpt-4o": 1441,
    "gpt-3.5-turbo": 1224,
    "gpt-5.2": 1443,

    # Anthropic
    "claude-3-5-sonnet": 1372,
    "claude-3-7-sonnet": 1371,  
    "claude-sonnet-4": 1389,
    "claude-sonnet-4-5": 1446,
    "claude-opus-4-5": 1467,

    # Google
    "gemini-2.0-flash": 1360,   
    "gemini-2.5-flash": 1408,
    "gemini-1.5-flash": 1310,
    "gemini-2.5-pro": 1451,
    "gemini-3-pro-preview": 1490,

    # xAI
    "grok-4": 1409,
    "grok-3": 1410,            
    "grok-3-mini": 1357
}

# -------- Release dates (for plot_over_time) ---------------------------------
# First API availability
RELEASE_DATES = {
    "gpt-3.5-turbo": "2023-03-01",
    "gpt-4-turbo":   "2023-11-06",
    "gpt-4o":        "2024-05-13",
    "o3-mini":       "2025-01-31",
    "o3":            "2025-04-16",
    "o4-mini":       "2025-04-16",
    "gpt-5":         "2025-08-07",
    "gpt-5.2":       "2025-12-11",

    "claude-3-5-sonnet":  "2024-06-20",
    "claude-3-7-sonnet":  "2025-02-24",
    "claude-sonnet-4":    "2025-05-22",
    "claude-sonnet-4-5":  "2025-09-29",
    "claude-opus-4-5":      "2025-11-24",

    "gemini-1.5-flash":   "2024-05-14",
    "gemini-2.0-flash":   "2025-02-05",
    "gemini-2.5-flash":   "2025-06-17",
    "gemini-2.5-pro":     "2025-06-17",
    "gemini-3-pro-preview":       "2025-11-18",

    "grok-3":       "2025-02-19",
    "grok-3-mini":  "2025-05-21",
    "grok-4":       "2025-07-09",
}

# ---------- loading / wrangling ----------------------------------------------

def _infer_mask_mode_column(df: pd.DataFrame) -> pd.Series:
    """Infer masked mode as a lowercase string."""
    if "mask_mode" in df.columns:
        return df["mask_mode"].astype(str).str.lower()
    if "file_name" in df.columns:
        s = df["file_name"].astype(str).str.lower()
        # Prefer explicit tokens if present
        mode = np.where(s.str.contains("masked"), "masked", "unknown")
        return pd.Series(mode, index=df.index)
    return pd.Series(["unknown"] * len(df), index=df.index)

def _filter_complete_fired(df: pd.DataFrame) -> pd.DataFrame:
    status_col = None
    for c in ["simulation_summary_Status", "Status"]:
        if c in df.columns:
            status_col = c
            break
    if status_col is None:
        return df
    mask = df[status_col].astype(str).str.contains(r"\b(?:Complete|Fired)\b", case=False, regex=True, na=False)
    return df.loc[mask].copy()

def load_ai_raw(path: str, *, use_run1: bool = True) -> pd.DataFrame:
    """
    Load AI runs:
      - Keep only Complete/Fired
      - If use_run1=True and a 'Run' column exists, keep Run==1 (previous default behavior)
      - Compute cp, cr (net of SC), sc_rev, sc_growth, years_survived, provider/model, mask_mode
    """
    df = pd.read_csv(path)
    df = _filter_complete_fired(df)

    if use_run1 and "Run" in df.columns:
        vals = pd.to_numeric(df["Run"], errors="coerce")
        df = df.loc[vals == 1].copy()

    # CP
    if "simulation_summary_Cumulative Profit" in df.columns:
        cp_raw = df["simulation_summary_Cumulative Profit"]
    elif "CumulativeProfit" in df.columns:
        cp_raw = df["CumulativeProfit"]
    elif "cumulative_profit" in df.columns:
        cp_raw = df["cumulative_profit"]
    else:
        raise KeyError("AI CSV: Cumulative Profit column not found.")
    cp = pd.Series(cp_raw).apply(parse_money)

    # CR (will compute as CR - SC later)
    if "simulation_summary_Cumulative Revenue" in df.columns:
        cr_raw = df["simulation_summary_Cumulative Revenue"]
    elif "CumulativeRevenue" in df.columns:
        cr_raw = df["CumulativeRevenue"]
    elif "cumulative_revenue" in df.columns:
        cr_raw = df["cumulative_revenue"]
    else:
        raise KeyError("AI CSV: Cumulative Revenue column not found.")
    cr = pd.Series(cr_raw).apply(parse_money)

    # SC revenue
    sc_candidates = [
        "best_scores_Revenue (SC)", "FinalRevenue_em", "Final SC revenue",
        "FinalRevenue_SC", "final_sc_revenue"
    ]
    sc_col = next((c for c in sc_candidates if c in df.columns), None)
    if sc_col is None:
        raise KeyError("AI CSV: SC revenue column not found.")
    sc_rev = df[sc_col].apply(parse_money)
    # Redefine CR as (Cumulative Revenue - Final SC Revenue)
    cr = cr - sc_rev

    # SC Sales Growth (% as fraction)
    if "best_scores_Sales Growth (SC)" in df.columns:
        sc_growth = df["best_scores_Sales Growth (SC)"].apply(parse_money)
    else:
        sc_growth = pd.Series([np.nan] * len(df))

    # years survived
    if "num_valid_years" in df.columns:
        years_survived = pd.to_numeric(df["num_valid_years"], errors="coerce")
    elif "YearReached" in df.columns:
        years_survived = pd.to_numeric(df["YearReached"], errors="coerce") - 2
    else:
        raise KeyError("AI CSV: years survived column not found (num_valid_years or YearReached).")

    # provider + model
    provider = df["provider"].astype(str).fillna("Unknown") if "provider" in df.columns else pd.Series(["Unknown"] * len(df))
    model    = df["model"].astype(str) if "model" in df.columns else pd.Series(["Unknown"] * len(df))
    mask_mode = _infer_mask_mode_column(df)

    # open source flag
    if "is_open_source" in df.columns:
        is_open_source = df["is_open_source"].astype(bool)
    else:
        is_open_source = pd.Series([False] * len(df))

    out = pd.DataFrame({
        "cp": cp,
        "cr": cr,
        "sc_rev": sc_rev,
        "sc_growth": sc_growth,
        "years_survived": years_survived,
        "provider": provider,
        "model": model,
        "mask_mode": mask_mode,
        "difficulty": df["difficulty"].astype(str) if "difficulty" in df.columns else pd.Series(["Unknown"] * len(df)),
        "instruction_set": df["instruction_set"].astype(str) if "instruction_set" in df.columns else pd.Series(["Unknown"] * len(df)),
        "is_open_source": is_open_source,
    })
    out["model_canon"] = out["model"].astype(str).apply(canonical_model_name)
    out = out.dropna(subset=["cp","cr","sc_rev","years_survived"])
    return out

def load_human_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Be tolerant of missing columns (return empty after dropna if incomplete).
    n = len(df)
    def _col(name: str) -> pd.Series:
        v = df.get(name)
        if v is None:
            return pd.Series([np.nan] * n)
        return v
    out = pd.DataFrame({
        "cp": _col("CumulativeProfit").apply(parse_money),
        "sc_rev": _col("FinalRevenue_em").apply(parse_money),      # EM = SC segment
        "cr": _col("CumulativeRevenue").apply(parse_money) - _col("FinalRevenue_em").apply(parse_money),
        "sc_growth": np.nan,
        "years_survived": pd.to_numeric(df.get("YearReached"), errors="coerce"),
        "provider": "Human",
        "model": "Human",
        "mask_mode": "human",
        "difficulty": "human",
    })
    out["model_canon"] = "Human"
    out = out.dropna(subset=["cp","cr","sc_rev","years_survived"])
    return out

# ---------- normalization helpers --------------------------------------------

def add_norm_and_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each metric over the provided rows, then compute composite."""
    df = df.copy()
    for raw, norm in [("cp","CP_norm"), ("sc_rev","SC_norm"), ("cr","CR_norm")]:
        col = pd.to_numeric(df[raw], errors="coerce")
        cmin, cmax = col.min(), col.max()
        df[norm] = 0.5 if (pd.isna(cmin) or pd.isna(cmax) or cmax == cmin) else (col - cmin) / (cmax - cmin)
    df["composite"] = df[["CP_norm","SC_norm","CR_norm"]].mean(axis=1)
    return df

def add_norm_and_composite_ai_scaled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CP, SC_rev, CR using min/max from AI rows only,
    then compute composite for ALL rows (humans, if present).
    """
    df = df.copy()
    is_human = df["provider"].astype(str).str.lower().eq("human")
    base = df.loc[~is_human] if (~is_human).any() else df  # fallback if all-human/empty

    for raw, norm in [("cp", "CP_norm"), ("sc_rev", "SC_norm"), ("cr", "CR_norm")]:
        col_all  = pd.to_numeric(df[raw], errors="coerce")
        col_base = pd.to_numeric(base[raw], errors="coerce")
        cmin, cmax = col_base.min(), col_base.max()
        if pd.isna(cmin) or pd.isna(cmax) or cmax == cmin:
            df[norm] = 0.5
        else:
            df[norm] = (col_all - cmin) / (cmax - cmin)

    df["composite"] = df[["CP_norm", "SC_norm", "CR_norm"]].mean(axis=1)
    return df

def add_norm_and_composite_using_baseline(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize cp/sc_rev/cr in `df` using min/max computed from `baseline`,
    then compute composite. If baseline has degenerate min/max, fallback=0.5.
    (Used by the release-date plots to fix the scale to default-only rows.)
    """
    out = df.copy()
    for raw, norm in [("cp","CP_norm"), ("sc_rev","SC_norm"), ("cr","CR_norm")]:
        col_all = pd.to_numeric(out[raw], errors="coerce")
        col_base = pd.to_numeric(baseline[raw], errors="coerce") if not baseline.empty else pd.Series([], dtype=float)
        cmin, cmax = (col_base.min(), col_base.max()) if not col_base.empty else (np.nan, np.nan)
        if pd.isna(cmin) or pd.isna(cmax) or cmax == cmin:
            out[norm] = 0.5
        else:
            out[norm] = (col_all - cmin) / (cmax - cmin)
    out["composite"] = out[["CP_norm","SC_norm","CR_norm"]].mean(axis=1)
    return out

# ---------- plotting (internal helpers) --------------------------------------

def _provider_color_map(providers: list[str]) -> dict[str, str]:
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not cycle:
        cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    uniq = [p for p in sorted(set(providers)) if str(p).lower() != "human"]
    return {p: cycle[i % len(cycle)] for i, p in enumerate(uniq)}

def _plot_points_and_means_with_human(ax, df: pd.DataFrame, ycol: str, y_is_money: bool, title: str):
    df = df.copy()
    human_mask = df["provider"].astype(str).str.lower().eq("human")
    ai_df = df.loc[~human_mask]
    human_df = df.loc[human_mask]

    # AI order by model mean
    stats = (ai_df.groupby("model_canon")[ycol]
                .agg(mean="mean")
                .sort_values("mean", ascending=True))
    model_order = list(stats.index)
    x_index = {m: i for i, m in enumerate(model_order)}
    prov_colors = _provider_color_map(ai_df["provider"].tolist())

    # --- AI raw points ---
    jitter = 0.15
    shown = set()
    for prov, sub in ai_df.groupby("provider", sort=True):
        if sub.empty:
            continue
        xs = sub["model_canon"].map(x_index).astype(float).to_numpy()
        xs = xs + np.random.uniform(-jitter, jitter, size=len(xs))
        ys = sub[ycol].to_numpy()
        lbl = prov if (prov not in shown and str(prov).lower() != "human") else None
        ax.scatter(xs, ys, alpha=0.35, s=22, color=prov_colors.get(prov, "black"), label=lbl)
        shown.add(prov)

    # --- AI means ±95% CI ---
    if not ai_df.empty:
        means = ai_df.groupby("model_canon")[ycol].mean().reindex(model_order)
        cis   = ai_df.groupby("model_canon")[ycol].apply(ci95).reindex(model_order)
        modal_provider = (ai_df.groupby("model_canon")["provider"]
                            .agg(lambda s: s.value_counts().idxmax())
                            .reindex(model_order))
        for i, _m in enumerate(model_order):
            c = prov_colors.get(modal_provider.iloc[i], "black")
            ax.errorbar([i], [means.iloc[i]], yerr=[cis.iloc[i]],
                        fmt="o", color=c, ecolor=c, capsize=3, elinewidth=1.0, markersize=5, alpha=1.0)

    # --- Axes labels/ticks/grid (AI only on x-axis) ---
    ax.set_title(title)
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=BOTTOM_ROW_XROT, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    if y_is_money:
        ax.yaxis.set_major_formatter(money_formatter())

    # --- Human (MBA) horizontal reference lines across the plot ---
    _draw_mba_hlines(ax, human_df, ycol, label_base="MBA")


def _figure_level_legend(fig: plt.Figure, axes_or_ax, legend_size: int = LEGEND_SIZE):
    """Figure-level legend (Human excluded by giving it no labels)."""
    if isinstance(axes_or_ax, (list, tuple, np.ndarray)):
        handles, labels = [], []
        for a in axes_or_ax:
            h, l = a.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li and li not in labels:
                    handles.append(hi); labels.append(li)
    else:
        handles, labels = axes_or_ax.get_legend_handles_labels()
    if not handles: return
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)),
               frameon=False, bbox_to_anchor=(0.5, LEGEND_Y),
               bbox_transform=fig.transFigure,
               columnspacing=1.2, handletextpad=0.6, borderaxespad=0.0,
               prop={"size": legend_size})

# ---------- plotting entrypoints ---------------------------------------------

def plot_2x2(df: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes.ravel()

    _plot_points_and_means_with_human(ax[0], df, "cp", True,   "Cumulative Profit")
    _plot_points_and_means_with_human(ax[1], df, "cr", True,   "Cumulative Revenue")
    _plot_points_and_means_with_human(ax[2], df, "sc_rev", True, "Emerging Tech Revenue")
    _plot_points_and_means_with_human(ax[3], df, "years_survived", False, "Years Survived")

    for a in (ax[0], ax[1]):
        labels = a.get_xticklabels()
        for lab in labels:
            lab.set_rotation(TOP_ROW_XROT)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")

    fig.subplots_adjust(left=SUBPLOTS_LEFT, right=SUBPLOTS_RIGHT,
                        top=SUBPLOTS_TOP, bottom=SUBPLOTS_BOTTOM_2X2,
                        hspace=GRID_HSPACE, wspace=GRID_WSPACE)

    fig.suptitle("BBB Simulation Outcomes by Model",
                 y=SUPTITLE_Y, fontsize=TITLE_SIZE)
    _figure_level_legend(fig, ax, legend_size=LEGEND_SIZE)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_composite(df: pd.DataFrame, outpath: str):
    human_mask = df["provider"].astype(str).str.lower().eq("human")
    ai_df = df.loc[~human_mask]
    human_df = df.loc[human_mask]

    stats = (ai_df.groupby("model_canon")["composite"]
               .agg(mean="mean")
               .sort_values("mean", ascending=True))
    model_order = list(stats.index)
    x_index = {m: i for i, m in enumerate(model_order)}
    prov_colors = _provider_color_map(ai_df["provider"].tolist())

    fig, ax = plt.subplots(figsize=(14, 6))

    # AI scatter
    jitter = 0.15; shown = set()
    for prov, sub in ai_df.groupby("provider", sort=True):
        sub = sub[sub["model_canon"].isin(model_order)]
        xs = sub["model_canon"].map(x_index).astype(float).to_numpy()
        xs = xs + np.random.uniform(-jitter, jitter, size=len(xs))
        ys = sub["composite"].to_numpy()
        lbl = prov if (prov not in shown and str(prov).lower() != "human") else None
        ax.scatter(xs, ys, alpha=0.35, s=24, color=prov_colors.get(prov, "black"), label=lbl)
        shown.add(prov)

    # AI means ±95% CI
    if not ai_df.empty:
        means = ai_df.groupby("model_canon")["composite"].mean().reindex(model_order)
        cis   = ai_df.groupby("model_canon")["composite"].apply(ci95).reindex(model_order)
        modal_provider = (ai_df.groupby("model_canon")["provider"]
                            .agg(lambda s: s.value_counts().idxmax())
                            .reindex(model_order))
        for i, _m in enumerate(model_order):
            c = prov_colors.get(modal_provider.iloc[i], "black")
            ax.errorbar([i], [means.iloc[i]], yerr=[cis.iloc[i]],
                        fmt="o", color=c, ecolor=c, capsize=3, elinewidth=1.0, markersize=6, alpha=1.0)

    # Axes formatting
    ax.set_ylabel("Composite (0–1)")
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_order, rotation=BOTTOM_ROW_XROT, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    # MBA horizontal lines (composite)
    _draw_mba_hlines(ax, human_df, "composite", label_base="MBA")

    # Margins, title, legend
    fig.subplots_adjust(left=SUBPLOTS_LEFT, right=SUBPLOTS_RIGHT,
                        top=SUBPLOTS_TOP, bottom=SUBPLOTS_BOTTOM_COMPOSITE)

    fig.suptitle("BBB Composite Score",
                 y=SUPTITLE_Y, fontsize=TITLE_SIZE)
    _figure_level_legend(fig, ax, legend_size=LEGEND_SIZE)

    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_composite_vs_gpqa(df: pd.DataFrame, outpath: str):
    """
    x = GPQA Diamond (%), y = composite (mean ±95% CI), AI-only.
    Uses the provided subset (masked-only in this repo) and annotates points with model names.
    """
    ai_mask = ~df["provider"].astype(str).str.lower().eq("human")
    sub = df.loc[ai_mask].copy()

    stats = (sub.groupby(["provider","model_canon"])["composite"]
                .agg(mean="mean", ci95=ci95)
                .reset_index())

    stats["gpqa"] = stats["model_canon"].map(GPQA_DIAMOND_LATEST)
    stats = stats.dropna(subset=["gpqa"])
    if stats.empty:
        print("No models had GPQA Diamond scores; skipping GPQA plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(stats["gpqa"], stats["mean"], yerr=stats["ci95"],
                fmt="o", linestyle="none")

    annotate_scatter_dodge(
        ax,
        stats["gpqa"],
        stats["mean"],
        stats["model_canon"].apply(short_model_label),
        x_bin=5,
        y_step_pts=10,
        xpad_pts=6
    )

    ax.set_xlabel("GPQA Diamond accuracy (%)")
    ax.set_ylabel("BBB Simulation Composite (mean ±95% CI)")
    ax.set_title("BBB Simulation Composite vs GPQA Diamond")
    ax.grid(True, alpha=0.25)
    xmin = max(0, (stats["gpqa"].min()//5)*5 - 2)
    xmax = min(100, (stats["gpqa"].max()//5 + 1)*5 + 2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_composite_vs_lmarena(df: pd.DataFrame, outpath: str):
    """
    x = LM Arena (Elo), y = composite (mean ±95% CI), AI-only.
    Uses the provided subset (masked-only in this repo) and annotates points with model names.
    """
    ai_mask = ~df["provider"].astype(str).str.lower().eq("human")
    sub = df.loc[ai_mask].copy()

    stats = (sub.groupby(["provider","model_canon"]) ["composite"]
                .agg(mean="mean", ci95=ci95)
                .reset_index())

    stats["lm_arena"] = stats["model_canon"].map(LMARENA_LATEST)
    stats = stats.dropna(subset=["lm_arena"])
    if stats.empty:
        print("No models had LMARENA scores; skipping LMARENA plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(stats["lm_arena"], stats["mean"], yerr=stats["ci95"],
                fmt="o", linestyle="none")

    annotate_scatter_dodge(
        ax,
        stats["lm_arena"],
        stats["mean"],
        stats["model_canon"].apply(short_model_label),
        x_bin=50,
        y_step_pts=10,
        xpad_pts=6
    )

    ax.set_xlabel("LM Arena (Elo)")
    ax.set_ylabel("BBB Simulation Composite (mean ±95% CI)")
    ax.set_title("BBB Simulation Composite vs LM Arena")
    ax.grid(True, alpha=0.25)

    xmin = stats["lm_arena"].min()
    xmax = stats["lm_arena"].max()
    if np.isfinite(xmin) and np.isfinite(xmax) and xmin < xmax:
        pad = 0.05 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)

    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

# ---------- script entry ------------------------------------------------------

def main():
    if not os.path.exists(AI_CSV):
        raise FileNotFoundError(f"AI CSV not found: {AI_CSV}")
    human_path = find_human_csv()
    human_ok = bool(human_path)
    if not human_ok:
        print(
            "WARNING: Human (MBA) CSV not found; proceeding without Human benchmark.\n"
            "If you have the file locally, place it at one of:\n"
            f"  - {os.path.join(SCRIPT_DIR, 'data', 'past_mba_performance.csv')}\n"
            f"  - {os.path.join(SCRIPT_DIR, 'inputs', 'past_mba_performance.csv')}"
        )

    df_ai = load_ai_raw(AI_CSV, use_run1=True)
    if human_ok:
        df_human = load_human_raw(human_path)
        df_all = pd.concat([df_ai, df_human], ignore_index=True, sort=False)
    else:
        df_all = df_ai.copy()

    # Ensure boolean flag exists post-concat with Human rows
    if "is_open_source" not in df_all.columns:
        df_all["is_open_source"] = False
    else:
        # Coerce to a true boolean dtype before fillna() to avoid pandas downcasting warnings.
        _os_raw = df_all["is_open_source"]
        _os = _os_raw.map(
            lambda v: (
                pd.NA if (v is None or (isinstance(v, float) and math.isnan(v))) else
                True if str(v).strip().lower() in {"true", "1", "yes", "y", "t"} else
                False if str(v).strip().lower() in {"false", "0", "no", "n", "f"} else
                pd.NA
            )
        ).astype("boolean")
        df_all["is_open_source"] = _os.fillna(False).astype(bool)

    # Useful series (computed once)
    mask_series = df_all["mask_mode"].astype(str).str.lower()
    prov_series = df_all["provider"].astype(str).str.lower()
    diff_series = df_all.get("difficulty", pd.Series(["Unknown"] * len(df_all), index=df_all.index)).astype(str)
    instr_series = df_all.get("instruction_set", pd.Series(["Unknown"] * len(df_all), index=df_all.index)).astype(str)
    os_series = df_all.get("is_open_source", pd.Series([False] * len(df_all), index=df_all.index)).astype(bool)

    # Baselines for composite scaling: masked + proprietary + default_instructions (AI only), per difficulty.
    baseline_by_difficulty: dict[str, pd.DataFrame] = {}
    for _d in ["Advanced", "Basic"]:
        baseline_by_difficulty[_d] = df_all[
            (~prov_series.eq("human"))
            & (~os_series)
            & mask_series.eq("masked")
            & diff_series.eq(_d)
            & instr_series.eq("default_instructions")
        ].copy()

    # Print run counts once (handy sanity check)
    by_model = (df_all.groupby(["provider", "model_canon"]).size()
                .rename("runs").reset_index())
    print("Runs used (Complete/Fired only; Run==1 where present):")
    print(by_model.sort_values(["provider", "model_canon"]).to_string(index=False))

    def _subset(*, difficulty: str, instruction_set: str, open_source: bool, include_human: bool) -> pd.DataFrame:
        """Return a masked subset, with optional inclusion of Human rows."""
        ai_mask = (
            (~prov_series.eq("human"))
            & mask_series.eq("masked")
            & diff_series.eq(difficulty)
            & instr_series.eq(instruction_set)
            & (os_series.eq(bool(open_source)))
        )
        if include_human:
            return df_all[prov_series.eq("human") | ai_mask].copy()
        return df_all[ai_mask].copy()

    saved_paths: list[str] = []

    # ----------------- CURATED PLOTS ONLY -----------------
    # 1) Advanced / default_instructions (proprietary): 2x2 + composite + composite-vs benchmarks
    sub_adv_def = _subset(difficulty="Advanced", instruction_set="default_instructions", open_source=False, include_human=True)
    if not sub_adv_def.loc[~sub_adv_def["provider"].astype(str).str.lower().eq("human")].empty:
        out_2x2 = OUT_2X2.replace(".png", "_adv_def.png")
        out_comp = OUT_COMP.replace(".png", "_adv_def.png")
        out_gpqa = OUT_GPQA.replace(".png", "_adv_def.png")
        out_lmarena = OUT_LMARENA.replace(".png", "_adv_def.png")

        plot_2x2(add_norm_and_composite(sub_adv_def), out_2x2)
        sub_adv_def_comp = add_norm_and_composite_using_baseline(sub_adv_def, baseline_by_difficulty.get("Advanced", pd.DataFrame()))
        plot_composite(sub_adv_def_comp, out_comp)
        plot_composite_vs_gpqa(sub_adv_def_comp, out_gpqa)
        plot_composite_vs_lmarena(sub_adv_def_comp, out_lmarena)
        saved_paths.extend([out_2x2, out_comp, out_gpqa, out_lmarena])

    # 2) Basic / default_instructions (proprietary): 2x2 + composite
    sub_basic_def = _subset(difficulty="Basic", instruction_set="default_instructions", open_source=False, include_human=False)
    if not sub_basic_def.empty:
        out_2x2 = OUT_2X2.replace(".png", "_basic_def.png")
        out_comp = OUT_COMP.replace(".png", "_basic_def.png")

        plot_2x2(add_norm_and_composite_ai_scaled(sub_basic_def), out_2x2)
        sub_basic_def_comp = add_norm_and_composite_using_baseline(sub_basic_def, baseline_by_difficulty.get("Basic", pd.DataFrame()))
        plot_composite(sub_basic_def_comp, out_comp)
        saved_paths.extend([out_2x2, out_comp])

    # 3) Advanced / default_instructions (open-source only): 2x2 + composite
    # Composite is still scaled to the proprietary Advanced/default baseline for comparability.
    sub_adv_os_def = _subset(difficulty="Advanced", instruction_set="default_instructions", open_source=True, include_human=False)
    if not sub_adv_os_def.empty:
        out_2x2 = OUT_2X2.replace(".png", "_adv_def_os.png")
        out_comp = OUT_COMP.replace(".png", "_adv_def_os.png")

        plot_2x2(add_norm_and_composite_ai_scaled(sub_adv_os_def), out_2x2)
        sub_adv_os_def_comp = add_norm_and_composite_using_baseline(sub_adv_os_def, baseline_by_difficulty.get("Advanced", pd.DataFrame()))
        plot_composite(sub_adv_os_def_comp, out_comp)
        saved_paths.extend([out_2x2, out_comp])

    # 4) Advanced / emerging_tech_instructions (proprietary): 2x2 only
    sub_adv_em = _subset(difficulty="Advanced", instruction_set="emerging_tech_instructions", open_source=False, include_human=True)
    if not sub_adv_em.loc[~sub_adv_em["provider"].astype(str).str.lower().eq("human")].empty:
        out_2x2 = OUT_2X2.replace(".png", "_adv_em.png")
        plot_2x2(add_norm_and_composite(sub_adv_em), out_2x2)
        saved_paths.append(out_2x2)

    print("\nSaved:")
    if saved_paths:
        for p in saved_paths:
            print(f"- {p}")
    else:
        print("- No plots generated (no matching masked AI rows found)")

if __name__ == "__main__":
    main()
