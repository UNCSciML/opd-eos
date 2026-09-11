#!/usr/bin/env python
"""Collect symmetric_analysis CSV outputs into one markdown report."""
import argparse, csv, json
from pathlib import Path


def load(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in r.items():
            if v in ("", "None"):
                r[k] = None; continue
            try: r[k] = json.loads(v) if v[:1] in "{[" else (float(v) if "." in v or "e" in v else int(v))
            except Exception: pass
    return rows


def f(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def ci(r, k):
    c = (r.get("ci") or {}).get(k)
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]" if c else "–"


def l1_table(rows, task):
    out = ["| cond | step | acc₀ | acc_t | net Δ | loss mass L | gain mass G | L−G | 95% CI(L−G) | regressed/solved₀ | lost | newly solved | strict(p₀≥.25): regressed/n, lost |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["task"] == task), key=lambda r: (r["cond"], r["step"])):
        out.append(f"| {r['cond']} | {r['step']} | {f(r['acc0'])} | {f(r['acc_t'])} | {r['net']:+.3f} | {f(r['loss_mass'])} | {f(r['gain_mass'])} | {r['loss_minus_gain']:+.3f} | {ci(r,'L-G')} | "
                   f"{r['n_regressed']}/{r['n_solved0']} | {r['n_lost']} | {r['n_newly']} | {r['n_strict_regressed']}/{r['n_strict0']}, {r['n_strict_lost']} |")
    return "\n".join(out)


def l2_table(rows, task):
    out = ["| cond | step | wrong on base-solvable q | correct→truncated | correct→flipped | never-correct (trunc / wrong box / no box) | reached-correct rate | reached-but-wrong rate | first-correct tok (median) | overrun tokens mean / median | truncation rate | mean len |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["task"] == task), key=lambda r: (r["cond"], r["step"])):
        out.append(f"| {r['cond']} | {r['step']} | {r['n_wrong_on_solvable']} | **{r['correct_then_truncated']}** | **{r['correct_then_flipped']}** | "
                   f"{r['never_correct_truncated']} / {r['never_correct_boxed_wrong']} / {r['never_correct_no_box']} | {f(r['reached_correct_rate'])} | {f(r['reached_but_wrong_rate'])} | "
                   f"{f(r['first_correct_tok_median'],0)} | {f(r['overrun_mean'],0)} / {f(r['overrun_median'],0)} | {f(r['truncation_rate'],2)} | {f(r['mean_len'],0)} |")
    return "\n".join(out)


def section(name, d, tasks):
    l1, l2 = load(d / "l1_question_transfer.csv"), load(d / "l2_mechanism.csv")
    s = [f"## {name}", "", f"Source: `{d}`", "", "### L1 — question-level transfer (pooled over all tasks; paired bootstrap over questions)", "", l1_table(l1, "pooled"), ""]
    for t in tasks:
        s += [f"<details><summary>L1 per task: {t}</summary>", "", l1_table(l1, t), "", "</details>", ""]
    s += ["### L2 — mechanism of regressions (pooled)", "", l2_table(l2, "pooled"), ""]
    for t in tasks:
        s += [f"<details><summary>L2 per task: {t}</summary>", "", l2_table(l2, t), "", "</details>", ""]
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR of a symmetric_analysis output dir")
    ap.add_argument("--base-summary", action="append", default=[], help="NAME=grading_summary.json of a step-0 base eval")
    ap.add_argument("--tasks", default="aime24,aime25,amc23")
    ap.add_argument("--title", default="Symmetric analysis: regression vs improvement relative to the base student")
    ap.add_argument("--notes", default="", help="markdown file with interpretation to append")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    md = [f"# {a.title}", "",
          "**Definitions.** For each question q, p₀(q) = fraction of the 16 base-student (step 0) samples that are correct, p_t(q) = same at checkpoint t. "
          "Samples are independent draws, so transitions are measured at the question level: "
          "**loss mass L = Σ_q max(p₀−p_t,0) / Σ_q p₀** (share of base competence destroyed), "
          "**gain mass G = Σ_q max(p_t−p₀,0) / Σ_q (1−p₀)** (share of base failures fixed). "
          "'regressed' = base-solvable questions (p₀>0) whose p_t < p₀; 'lost' = p_t = 0; 'newly solved' = p₀ = 0 and p_t > 0. "
          "95% CIs are paired bootstraps over pooled questions. ",
          "",
          "**L2.** Every response is scanned for the *first* `\\boxed{}` whose content the evaluator's grader accepts; a sample is 'correct' only if its *last* box is accepted (evaluator rule). "
          "Wrong samples of base-solvable questions are classified as **correct→truncated** (a correct box appeared, then generation hit the token budget without a stop), "
          "**correct→flipped** (a correct box appeared, the final box is wrong), or never-correct. 'overrun' = tokens generated after the first correct box. "
          "Grading = evaluator's `grade_answer_verl`; truncation = length ≥ max_tokens and no configured stop token last.", ""]
    if a.base_summary:
        md += ["## Step-0 base student accuracy (mean@16)", "", "| base eval | " + " | ".join(t.upper() for t in tasks) + " | macro |", "|---|" + "---|" * (len(tasks) + 1)]
        for spec in a.base_summary:
            n, p = spec.rsplit("=", 1); r = json.load(open(p)); m = {x["task"].lower(): x["mean_score"] for x in r}
            vals = [m.get(t) for t in tasks]; macro = sum(v for v in vals if v is not None) / max(sum(v is not None for v in vals), 1)
            md.append(f"| {n} | " + " | ".join(f(v) for v in vals) + f" | {macro:.3f} |")
        md.append("")
    for spec in a.run:
        n, d = spec.rsplit("=", 1); md += [section(n, Path(d), tasks), ""]
    if a.notes and Path(a.notes).exists():
        md += ["## Interpretation", "", Path(a.notes).read_text(), ""]
    Path(a.out).write_text("\n".join(md)); print("wrote", a.out)


if __name__ == "__main__":
    main()
