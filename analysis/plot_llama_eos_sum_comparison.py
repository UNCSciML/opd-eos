#!/usr/bin/env python3
"""Plot a W&B-exported Llama baseline with an independent teacher length reference."""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator

from plot_gemma_qwen_eos_sum_comparison import (
    METRICS, _style_axis, load_teacher_response_length_mean, plot_length_panel, write_csv,
)


def load_records(history, eos_ids):
    records = {}
    duplicates = 0
    for row in history:
        if 'training/global_step' not in row:
            continue
        step = int(row['training/global_step'])
        record = {'model': 'llama32', 'step': step}
        for name, key in METRICS.items():
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f'Non-finite {key} at step {step}')
            record[name] = value
        for source in ('student', 'teacher'):
            for suffix in ('sequence_mean', 'last_token'):
                total = sum(row[f'{source}_eos_prob/token_{tid}/{suffix}'] for tid in eos_ids)
                observed = record[f'{source}_{suffix}']
                if not np.isclose(total, observed, rtol=1e-5, atol=1e-12):
                    raise ValueError(f'EOS sum differs from individual token probabilities at {step}')
                if not 0 < observed <= 1:
                    raise ValueError(f'Invalid probability at step {step}: {observed}')
        if step in records:
            if record != records[step]:
                raise ValueError(f'Conflicting duplicate training step {step}')
            duplicates += 1
        records[step] = record
    if not records:
        raise ValueError('No complete training records')
    if sorted(records) != list(range(1, max(records) + 1)):
        raise ValueError('Missing training steps; refusing to draw across gaps')
    return [records[s] for s in sorted(records)], duplicates


def draw(records, teacher_mean, prefix, scale, configured_steps):
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 8, 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 7.0), sharex=True)
    steps = np.asarray([r['step'] for r in records])
    for ax, suffix, label in zip(axes[:2], ('sequence_mean', 'last_token'),
                                  ('sequence mean', 'last token')):
        values_all = []
        for source, color, style in [('student', '#0072B2', '-'), ('teacher', '#D55E00', '--')]:
            values = np.asarray([r[f'{source}_{suffix}'] for r in records])
            values_all.extend(values)
            ax.plot(steps, values, color=color, linestyle=style, linewidth=1.45,
                    label=source.title(), zorder=3)
        ax.set_yscale(scale)
        if scale == 'log':
            ax.set_ylim(10 ** np.floor(np.log10(min(values_all))),
                        min(1, 10 ** np.ceil(np.log10(max(values_all)))))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        else:
            ax.set_ylim(0, min(1, max(values_all) * 1.08))
        ax.set_ylabel(f'Sum EOS probability\n({label})')
    plot_length_panel(axes[2], steps, np.asarray([r['response_length_mean'] for r in records]), teacher_mean)
    axes[2].set_ylim(0, 7520)
    axes[2].set_ylabel('Mean response length\n(tokens)')
    axes[2].set_xlabel('Training step')
    for ax in axes:
        _style_axis(ax)
        ax.set_xlim(1, int(steps.max()))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[0].set_title('Llama-3.2 3B Base → 3B Instruct', pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.56, 0.95),
               ncol=2, frameon=False)
    fig.suptitle(f'Baseline OPD: summed EOS probability\nand length dynamics ({scale} scale)',
                 fontsize=10, y=0.996)
    fig.text(0.56, 0.01,
             f'W&B history: steps 1–{steps.max()} (configured: {configured_steps})\n'
             'EOS sum: 128001 + 128008 + 128009', ha='center', fontsize=7, color='#555555')
    fig.subplots_adjust(left=0.20, right=0.975, bottom=0.11, top=0.86, hspace=0.22)
    for ext in ('png', 'pdf'):
        fig.savefig(prefix.with_name(prefix.name + '_' + scale).with_suffix('.' + ext),
                    dpi=360, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-dir', type=Path, required=True)
    parser.add_argument('--teacher-summary', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.history_dir / 'run_metadata.json').read_text())
    eos_ids = metadata['config']['semantic_eos_token_ids']
    if eos_ids != [128001, 128008, 128009] or metadata['config']['eos_mode'] != 'baseline':
        raise ValueError('Expected Llama three-token EOS baseline')
    records, duplicates = load_records(json.loads((args.history_dir / 'history.json').read_text()), eos_ids)
    teacher_mean = load_teacher_response_length_mean(args.teacher_summary)
    write_csv(records, args.output_prefix.with_suffix('.csv'))
    for scale in ('log', 'linear'):
        draw(records, teacher_mean, args.output_prefix, scale, metadata['config']['total_training_steps'])
    print(json.dumps({'unique_steps': len(records), 'identical_duplicates_removed': duplicates,
                      'teacher_mean': teacher_mean, 'first': records[0], 'last': records[-1]}, indent=2))


if __name__ == '__main__':
    main()
