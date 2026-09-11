#!/usr/bin/env python
"""Symmetric (regression vs improvement) analysis of OPD checkpoints vs the base student.

L1  question-level transfer: p0(q) at step 0 vs p_t(q) at each checkpoint;
    loss mass  L = sum max(p0-pt,0)/sum p0,  gain mass G = sum max(pt-p0,0)/sum (1-p0),
    question counts (regressed / lost / improved / newly solved), paired bootstrap CIs.
L2  per-sample mechanism: locate the FIRST correct \\boxed{} in every response and classify
    wrong samples into  correct-then-truncated / correct-then-flipped / never-correct(+truncated).
    Also 'overrun' = tokens generated after the first correct answer.
Grading = the evaluator's own grade_answer_verl (last \\boxed), so numbers match grading_summary.
"""
import argparse, json, re, sys, functools, csv
from pathlib import Path
import numpy as np

PR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PR / "scripts/val/eval"))
from utils import grade_answer_verl, extract_answer  # noqa: E402

TASK_FILES = {"aime24": "aime24_t0.7_p0.95_n16-MNT8192", "aime25": "aime25_t0.7_p0.95_n16-MNT8192", "amc23": "amc23_t0.7_p0.95_n16-MNT8192"}


def find_boxes(text):
    """Yield (start, end, content) for every \\boxed{...} with brace matching (evaluator's rule set)."""
    i = 0
    while True:
        j = text.find("\\boxed", i)
        if j < 0:
            return
        k = j + len("\\boxed")
        while k < len(text) and text[k] == " ":
            k += 1
        if k < len(text) and text[k] == "{":
            depth, m = 0, k
            while m < len(text):
                if text[m] == "{":
                    depth += 1
                elif text[m] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                m += 1
            if depth == 0:
                yield j, m + 1, text[k + 1:m]
                i = m + 1
                continue
            return
        else:  # "\\boxed 5" form
            m = text.find("$", k)
            m = len(text) if m < 0 else m
            yield j, m, text[k:m].strip()
            i = m


@functools.lru_cache(maxsize=None)
def box_correct(content, gold):
    try:
        return bool(grade_answer_verl("\\boxed{" + content + "}", gold, answer_format="auto"))
    except Exception:
        return False


def char_to_tok(tok, ids, char_pos):
    """Smallest k with len(decode(ids[:k])) >= char_pos (binary search)."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=True)) >= char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def load_run(root, step, task, tok, max_tokens, stop_ids):
    p = Path(root) / f"step_{step:04d}" / (TASK_FILES[task] + ".jsonl")
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open() if l.strip()]
    npz = np.load(p.with_suffix(".tokens.npz"), allow_pickle=False)
    flat, off = npz["token_ids"], npz["offsets"]
    out = []
    for i, r in enumerate(rows):
        ids = flat[int(off[i]):int(off[i + 1])].astype(np.int64).tolist()
        resp, gold = str(r["response"]), str(r["answer"])
        correct = bool(grade_answer_verl(resp, gold, answer_format="auto"))
        final = extract_answer(resp, answer_format="auto")
        truncated = len(ids) >= max_tokens and (ids[-1] not in stop_ids if ids else True)
        first_ok_char = None
        for s, e, c in find_boxes(resp):
            if box_correct(c, gold):
                first_ok_char = e
                break
        first_ok_tok = char_to_tok(tok, ids, first_ok_char) if first_ok_char is not None else None
        out.append(dict(example_id=int(r["example_id"]), rollout=i % 16, correct=correct, truncated=truncated,
                        length=len(ids), final_extracted=final is not None, first_correct_tok=first_ok_tok,
                        overrun=(len(ids) - first_ok_tok) if first_ok_tok is not None else None))
    return out


def qrates(samples):
    d = {}
    for s in samples:
        d.setdefault(s["example_id"], []).append(s["correct"])
    return {q: float(np.mean(v)) for q, v in d.items()}


def l1_stats(p0, pt, rng=None, B=2000, strict=0.25):
    qs = sorted(set(p0) & set(pt))
    a = np.array([p0[q] for q in qs]); b = np.array([pt[q] for q in qs])
    def f(a, b):
        loss = np.maximum(a - b, 0).sum() / max(a.sum(), 1e-9)
        gain = np.maximum(b - a, 0).sum() / max((1 - a).sum(), 1e-9)
        return loss, gain, loss - gain
    L, G, D = f(a, b)
    ci = {}
    if rng is not None:
        bs = np.array([f(a[idx], b[idx]) for idx in rng.integers(0, len(qs), size=(B, len(qs)))])
        for j, k in enumerate(["L", "G", "L-G"]):
            ci[k] = (float(np.quantile(bs[:, j], .025)), float(np.quantile(bs[:, j], .975)))
    solved = a > 0; strict_solved = a >= strict
    return dict(n_q=len(qs), acc0=float(a.mean()), acc_t=float(b.mean()), net=float((b - a).mean()),
                loss_mass=float(L), gain_mass=float(G), loss_minus_gain=float(D), ci=ci,
                n_solved0=int(solved.sum()), n_regressed=int(((b < a) & solved).sum()), n_lost=int(((b == 0) & solved).sum()),
                n_improved=int(((b > a) & (a < 1)).sum()), n_newly=int(((b > 0) & (a == 0)).sum()),
                n_strict0=int(strict_solved.sum()), n_strict_regressed=int(((b < a) & strict_solved).sum()),
                n_strict_lost=int(((b == 0) & strict_solved).sum()))


def l2_stats(samples, p0):
    """Mechanism breakdown over wrong samples of base-solvable questions (p0>0)."""
    wrong = [s for s in samples if not s["correct"] and p0.get(s["example_id"], 0) > 0]
    cats = dict(correct_then_truncated=0, correct_then_flipped=0, never_correct_truncated=0, never_correct_boxed_wrong=0, never_correct_no_box=0)
    for s in wrong:
        if s["first_correct_tok"] is not None:
            cats["correct_then_truncated" if s["truncated"] else "correct_then_flipped"] += 1
        elif s["truncated"]:
            cats["never_correct_truncated"] += 1
        elif s["final_extracted"]:
            cats["never_correct_boxed_wrong"] += 1
        else:
            cats["never_correct_no_box"] += 1
    reached = [s for s in samples if s["first_correct_tok"] is not None]
    over = [s["overrun"] for s in reached]
    return dict(n_wrong_on_solvable=len(wrong), **cats,
                reached_correct_rate=len(reached) / max(len(samples), 1),
                reached_but_wrong_rate=sum(1 for s in reached if not s["correct"]) / max(len(samples), 1),
                first_correct_tok_median=float(np.median([s["first_correct_tok"] for s in reached])) if reached else None,
                overrun_mean=float(np.mean(over)) if over else None, overrun_median=float(np.median(over)) if over else None,
                truncation_rate=float(np.mean([s["truncated"] for s in samples])), mean_len=float(np.mean([s["length"] for s in samples])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="default step-0 output root (has step_0000/)")
    ap.add_argument("--cond", action="append", required=True, help="NAME=OUTPUT_ROOT[@BASE_ROOT]")
    ap.add_argument("--steps", default="20,40,60,80,100,120,140,160,180,200")
    ap.add_argument("--base-step", type=int, default=0)
    ap.add_argument("--tasks", default="aime24,aime25,amc23")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(","); steps = [int(s) for s in args.steps.split(",")]
    rng = np.random.default_rng(0)

    def stop_ids_of(root):
        cfg = Path(root) / "eval_run_config.json"
        if cfg.exists():
            c = json.load(cfg.open())
            for k in ("stop_token_ids", "resolved_stop_token_ids"):
                if c.get(k):
                    return set(int(x) for x in c[k])
        return {151643, 151645}

    l1_rows, l2_rows, sample_rows = [], [], []
    for spec in args.cond:
        name, rest = spec.split("=", 1)
        root, base = (rest.split("@", 1) + [args.base])[:2]
        bstop, cstop = stop_ids_of(base), stop_ids_of(root)
        base_s = {t: load_run(base, args.base_step, t, tok, args.max_tokens, bstop) for t in tasks}
        for t in tasks:
            if base_s[t] is None:
                sys.exit(f"missing base step {args.base_step} for {t} in {base}")
        p0 = {t: qrates(base_s[t]) for t in tasks}
        p0_pool = {f"{t}:{q}": v for t in tasks for q, v in p0[t].items()}
        for st in steps:
            cur = {t: load_run(root, st, t, tok, args.max_tokens, cstop) for t in tasks}
            if any(v is None for v in cur.values()):
                print(f"[{name}] step {st}: missing task file, skipped"); continue
            pt_pool = {}
            for t in tasks:
                pt = qrates(cur[t]); pt_pool.update({f"{t}:{q}": v for q, v in pt.items()})
                l1_rows.append(dict(cond=name, step=st, task=t, **l1_stats(p0[t], pt)))
                l2_rows.append(dict(cond=name, step=st, task=t, **l2_stats(cur[t], p0[t])))
                for s in cur[t]:
                    sample_rows.append(dict(cond=name, step=st, task=t, p0=p0[t][s["example_id"]], **s))
            pooled_samples = [dict(s, example_id=f"{t}:{s['example_id']}") for t in tasks for s in cur[t]]
            l1_rows.append(dict(cond=name, step=st, task="pooled", **l1_stats(p0_pool, pt_pool, rng, args.bootstrap)))
            l2_rows.append(dict(cond=name, step=st, task="pooled", **l2_stats(pooled_samples, p0_pool)))
            r = l1_rows[-1]; m = l2_rows[-1]
            print(f"[{name}] step {st:3d} pooled acc0={r['acc0']:.3f} acc_t={r['acc_t']:.3f} net={r['net']:+.3f} "
                  f"L={r['loss_mass']:.3f} G={r['gain_mass']:.3f} L-G={r['loss_minus_gain']:+.3f} CI{tuple(round(x,3) for x in r['ci']['L-G'])} | "
                  f"regressed {r['n_regressed']}/{r['n_solved0']} lost {r['n_lost']} newly {r['n_newly']} | "
                  f"wrong-on-solvable {m['n_wrong_on_solvable']}: c->trunc {m['correct_then_truncated']} c->flip {m['correct_then_flipped']} "
                  f"never {m['never_correct_truncated']+m['never_correct_boxed_wrong']+m['never_correct_no_box']} | overrun_med {m['overrun_median']} trunc {m['truncation_rate']:.2f}")
        # base itself as step 0 row for the sample table
        for t in tasks:
            for s in base_s[t]:
                sample_rows.append(dict(cond=name, step=args.base_step, task=t, p0=p0[t][s["example_id"]], **s))

    def dump(rows, fn):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with (out / fn).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    dump(l1_rows, "l1_question_transfer.csv"); dump(l2_rows, "l2_mechanism.csv"); dump(sample_rows, "samples.csv")
    print("wrote", out)


if __name__ == "__main__":
    main()
