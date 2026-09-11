#!/usr/bin/env python
"""Stage-wise EOS probe: tables, plots and the Phase-1 decision (Case A/B/C).

Reads <probe_dir>/<stage>__summary.json and <stage>__<set>.csv written by k2_eos_probe.py.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["pretrain_final", "mid_1_final", "mid_2_final", "mid_3_final", "mid_4_final", "sft_1_final", "sft_2_final", "main", "rl_merged"]
LABEL = {"pretrain_final": "pretrain", "mid_1_final": "mid-1", "mid_2_final": "mid-2", "mid_3_final": "mid-3", "mid_4_final": "mid-4", "sft_1_final": "SFT-1", "sft_2_final": "SFT-2", "main": "main (=SFT-2)", "rl_merged": "RL-merged (branch)"}
TICK = dict(LABEL); TICK.update({"pretrain_final": "pre-\ntrain", "main": "main\n(=SFT-2)", "rl_merged": "RL\n(branch)"})
# reference palette (validated): slot 1 blue, slot 2 orange; text tokens; recessive grid
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b68", "#e6e5e1", "#fcfcfb"


def load(probe_dir, sets):
    data = {}
    for s in STAGES:
        p = Path(probe_dir) / f"{s}__summary.json"
        if p.exists():
            data[s] = json.load(open(p))["summary"]
    return data


def fmt_table(data, set_name):
    stages = [s for s in STAGES if s in data and set_name in data[s]]
    rows = ["| stage | n | p(EOD=1) mean / median [p10, p90] | p(EOT=250019) mean / median [p10, p90] | p(STOP) mean / median | EOT-share mean / median [p25, p75] | top-1 ∈ STOP |", "|---|---|---|---|---|---|---|"]
    for s in stages:
        d = data[s][set_name]; e, t, st, sh = d["p_eod"], d["p_eot"], d["p_stop"], d["eot_share"]
        rows.append(f"| {LABEL[s]} | {e['n']} | {e['mean']:.4f} / {e['median']:.4f} [{e['p10']:.4f}, {e['p90']:.4f}] | {t['mean']:.4f} / {t['median']:.4f} [{t['p10']:.4f}, {t['p90']:.4f}] | {st['mean']:.4f} / {st['median']:.4f} | {sh['mean']:.3f} / {sh['median']:.3f} [{sh['p25']:.3f}, {sh['p75']:.3f}] | {d['frac_top1_is_stop']:.2f} |")
    return "\n".join(rows)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE); ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9); ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def line(ax, x, y, color, label, idx):
    ax.plot(x, y, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round", marker="o", markersize=8, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
    ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)


def plot(data, set_name, out_png, mean_or_median="mean"):
    stages = [s for s in STAGES if s in data and set_name in data[s]]; x = list(range(len(stages))); xl = [TICK[s] for s in stages]
    g = lambda k: [data[s][set_name][k][mean_or_median] for s in stages]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), facecolor=SURFACE); fig.subplots_adjust(wspace=0.38, left=0.05, right=0.985, top=0.78, bottom=0.24)
    ax = axes[0]; line(ax, x, g("p_eod"), BLUE, "p(EOD = 1)", 0); line(ax, x, g("p_eot"), ORANGE, "p(EOT = 250019)", 1)
    style(ax, "EOD vs EOT mass", "next-token probability"); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax = axes[1]; line(ax, x, g("p_stop"), BLUE, "p(STOP)", 0); style(ax, "Total termination mass  p(1) + p(250019)", "probability")
    ax = axes[2]; line(ax, x, g("eot_share"), ORANGE, "EOT-share", 0); style(ax, "EOT share  p(250019) / p(STOP)", "share"); ax.set_ylim(-0.02, 1.05)
    for ax in axes: ax.set_xticks(x); ax.set_xticklabels(xl, rotation=0, fontsize=8.5); ax.set_xlim(-0.3, len(x) - 1 + 0.75)
    fig.text(0.05, 0.945, "K2-Horizon-7B: termination probability mass across training stages", fontsize=12.5, color=INK, weight="bold")
    fig.text(0.05, 0.885, f"Next-token probabilities at the end of fixed terminal prefixes (prefix set {set_name}, {mean_or_median} over prefixes); same token IDs for every checkpoint.", fontsize=9.5, color=MUTED)
    fig.savefig(out_png, dpi=160); plt.close(fig)


def decide(data, set_name):
    if "pretrain_final" not in data or "main" not in data: return "UNDETERMINED", "missing pretrain_final or main"
    b, f = data["pretrain_final"][set_name], data["main"][set_name]
    base_eod, base_eot, fin_eod, fin_eot = b["p_eod"]["median"], b["p_eot"]["median"], f["p_eod"]["median"], f["p_eot"]["median"]
    base_share, fin_share = b["eot_share"]["mean"], f["eot_share"]["mean"]
    stop_b, stop_f = b["p_stop"]["mean"], f["p_stop"]["mean"]
    strong = base_share <= 0.2 and fin_share >= 0.8 and min(stop_b, stop_f) >= 0.1
    weak = (fin_share - base_share) >= 0.3 and not strong
    why = (f"base: median p(1)={base_eod:.3f}, p(250019)={base_eot:.3f}, EOT-share(mean)={base_share:.2f}, p(STOP)(mean)={stop_b:.3f}; "
           f"main: median p(1)={fin_eod:.3f}, p(250019)={fin_eot:.3f}, EOT-share(mean)={fin_share:.2f}, p(STOP)(mean)={stop_f:.3f}")
    return ("A (strong natural mismatch)" if strong else "B (weak/intermediate migration)" if weak else "C (no meaningful mismatch)"), why


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--probe-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sets", default="A,B")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = load(a.probe_dir, a.sets.split(","))
    md = ["## Stage-wise EOS probe (official K2 implementation, BF16/SDPA, frozen prefixes)", "", "Note: the `main` release is byte-identical to the `sft_2_final` tag (36/36 safetensors shards share the same hash); `rl_merged` is a separate branch that is not on `main`.", ""]
    for s in a.sets.split(","):
        if not any(s in d for d in data.values()): continue
        md += [f"### Prefix set {s}", "", fmt_table(data, s), ""]
        for agg in ("mean", "median"):
            png = out / f"k2_eos_migration_set{s}_{agg}.png"; plot(data, s, png, agg); md.append(f"![set {s} {agg}]({png.name})")
        case, why = decide(data, s); md += ["", f"**Decision rule (set {s}): Case {case}.** {why}", ""]
    (out / "PHASE1_EOS_PROBE.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
