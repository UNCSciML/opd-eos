import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).with_name("plot_gemma_qwen_eos_sum_comparison.py")
SPEC = importlib.util.spec_from_file_location("eos_comparison_plot", SCRIPT_PATH)
PLOT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PLOT_MODULE)


class TeacherLengthReferenceTest(unittest.TestCase):
    def test_reads_teacher_generation_mean_from_summary(self) -> None:
        load_mean = getattr(PLOT_MODULE, "load_teacher_response_length_mean", None)
        self.assertTrue(callable(load_mean), "summary mean loader is missing")

        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text(
                json.dumps({"response_length_mean": 2341.578515625}),
                encoding="utf-8",
            )
            self.assertEqual(load_mean(summary_path), 2341.578515625)

    def test_length_panel_contains_teacher_mean_reference_line(self) -> None:
        plot_panel = getattr(PLOT_MODULE, "plot_length_panel", None)
        self.assertTrue(callable(plot_panel), "length-panel plotter is missing")

        fig, ax = plt.subplots()
        try:
            plot_panel(
                ax,
                np.asarray([1.0, 2.0]),
                np.asarray([1200.0, 1800.0]),
                teacher_mean=1661.806328125,
            )
            teacher_lines = [
                line for line in ax.lines if line.get_label() == "Teacher generation mean"
            ]
            self.assertEqual(len(teacher_lines), 1)
            np.testing.assert_allclose(
                teacher_lines[0].get_ydata(),
                np.asarray([1661.806328125, 1661.806328125]),
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
