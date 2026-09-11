import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from analysis import plot_eos_fix_train_dynamics as plotter


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_eos_fix_train_dynamics.py"
MODES = ("two_stop", "teacher_map", "semantic_class", "canonical")


def _write_history(path: Path, scale: float) -> None:
    pd.DataFrame(
        {
            "training/global_step": [1, 2],
            "response_length/mean": [100.0 * scale, 200.0 * scale],
            "response_length/clip_ratio": [0.1 * scale, 0.2 * scale],
            "student_eos_prob/endoftext_151643/last_token": [0.5 / scale, 0.25 / scale],
            "student_eos_prob/im_end_151645/last_token": [1e-7 / scale, 1e-8 / scale],
        }
    ).to_csv(path, index=False)


def test_cli_exports_ttrl_eos_fix_training_dynamics_as_one_by_four(tmp_path: Path) -> None:
    run_args: list[str] = []
    for index, mode in enumerate(MODES, start=1):
        history = tmp_path / f"{mode}.csv"
        _write_history(history, float(index))
        run_args.extend(["--run", f"{mode},{history}"])

    output_prefix = tmp_path / "figures" / "ttrl-eos-fix-train-dynamics"
    csv_output = tmp_path / "derived" / "ttrl-eos-fix-train-dynamics.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *run_args,
            "--output-prefix",
            str(output_prefix),
            "--csv-output",
            str(csv_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = pd.read_csv(csv_output)
    assert list(exported.columns) == [
        "eos_mode",
        "step",
        "response_length_mean",
        "response_length_clip_ratio",
        "student_prob_endoftext_last_token",
        "student_prob_im_end_last_token",
    ]
    assert len(exported) == 8
    row = exported.query("eos_mode == 'semantic_class' and step == 2").iloc[0]
    assert row["response_length_mean"] == 600.0
    assert row["response_length_clip_ratio"] == pytest.approx(0.6)
    assert row["student_prob_endoftext_last_token"] == pytest.approx(1 / 12)
    assert row["student_prob_im_end_last_token"] == pytest.approx(1e-8 / 3)

    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    with Image.open(png) as image:
        assert image.width >= 5000
        # Wide 1x4 strip. Loosened from 4.4 when the shared legend was moved clear
        # of the panel titles, which it previously overlapped by about 10pt.
        assert image.width / image.height > 4.0

    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        assert "Mean response length" in figure_text
        assert "Clipped responses" in figure_text
        assert "Student probability at last token" in figure_text
        assert "<|im_end|>" in figure_text
        assert "<|im\\_end|>" not in figure_text
        for style in plotter.MODE_STYLES.values():
            assert figure_text.count(style["label"]) == 1
        assert "Baseline" not in figure_text
        assert "TTRL training" not in figure_text
        assert "151643" not in figure_text
        assert "151645" not in figure_text
        assert figure_text.count("200") >= 4


def test_cli_rejects_an_incomplete_eos_fix_set(tmp_path: Path) -> None:
    history = tmp_path / "two_stop.csv"
    _write_history(history, 1.0)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run",
            f"two_stop,{history}",
            "--output-prefix",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "exactly one --run for each EOS mode" in result.stderr


def test_wandb_ema_uses_debiased_exponential_smoothing() -> None:
    smoothed = plotter.wandb_ema(np.asarray([1.0, 3.0]), smoothing_weight=0.5)
    np.testing.assert_allclose(smoothed, [1.0, 7.0 / 3.0])


def test_build_figure_overlays_faint_raw_and_solid_smoothed_linear_curves(tmp_path: Path) -> None:
    runs = []
    for index, mode in enumerate(MODES, start=1):
        history = tmp_path / f"{mode}.csv"
        _write_history(history, float(index))
        runs.append((mode, history))
    records = plotter.load_runs(runs)

    figure = plotter.build_figure(records, smoothing_weight=0.8)
    labels = {style["label"] for style in plotter.MODE_STYLES.values()}
    try:
        assert len(figure.axes) == 4
        for axis in figure.axes:
            assert axis.get_yscale() == "linear"
            assert axis.get_xlim() == pytest.approx((0.0, 200.0))
            assert list(axis.get_xticks()) == [40, 80, 120, 160, 200]
            raw_lines = [line for line in axis.lines if line.get_alpha() == pytest.approx(0.18)]
            smooth_lines = [line for line in axis.lines if line.get_label() in labels]
            assert len(raw_lines) == 4
            assert len(smooth_lines) == 4
            assert all(line.get_alpha() == 1.0 for line in smooth_lines)
            assert all(line.get_linewidth() > raw_lines[0].get_linewidth() for line in smooth_lines)
    finally:
        plt.close(figure)


def test_build_figure_adds_unlabeled_teacher_references_only_to_length_panels(
    tmp_path: Path,
) -> None:
    runs = []
    for index, mode in enumerate(MODES, start=1):
        history = tmp_path / f"{mode}.csv"
        _write_history(history, float(index))
        runs.append((mode, history))
    records = plotter.load_runs(runs)

    figure = plotter.build_figure(
        records,
        smoothing_weight=0.8,
        teacher_response_length=2341.578515625,
        teacher_clip_ratio=0.0125,
    )
    try:
        reference_lines = [
            [line for line in axis.lines if line.get_gid() == "teacher-reference"]
            for axis in figure.axes
        ]
        assert [len(lines) for lines in reference_lines] == [1, 1, 0, 0]
        np.testing.assert_allclose(reference_lines[0][0].get_ydata(), [2341.578515625] * 2)
        np.testing.assert_allclose(reference_lines[1][0].get_ydata(), [1.25] * 2)
        assert reference_lines[0][0].get_linestyle() == "--"
        assert reference_lines[1][0].get_linestyle() == "--"
        assert all("Teacher:" not in text.get_text() for axis in figure.axes for text in axis.texts)
    finally:
        plt.close(figure)


def test_build_figure_keeps_two_line_titles_aligned_without_scientific_offset_overlap(
    tmp_path: Path,
) -> None:
    runs = []
    for index, mode in enumerate(MODES, start=1):
        history = tmp_path / f"{mode}.csv"
        _write_history(history, float(index))
        runs.append((mode, history))
    records = plotter.load_runs(runs)

    figure = plotter.build_figure(records, smoothing_weight=0.8)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        title_boxes = [axis.title.get_window_extent(renderer) for axis in figure.axes]
        title_bottoms = [box.y0 for box in title_boxes]
        title_tops = [box.y1 for box in title_boxes]
        assert title_bottoms[2] == pytest.approx(title_bottoms[3])
        assert max(title_tops) - min(title_tops) < 3.0
        assert not figure.axes[3].yaxis.get_offset_text().get_visible()
        assert r"\times 10^{-7}" in figure.axes[3].get_ylabel()
    finally:
        plt.close(figure)
