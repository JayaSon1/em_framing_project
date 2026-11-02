"""Targeted tests for the four contested claims. Run once per claim when writing up.

    python evidence.py
"""
import math
import textwrap

import pandas as pd

import analysis as an

FOCUS = "C2_s0"          # the model whose residue is being tested
BASE = "C0"
SEED_PAIRS = [("C1_s0", "C1_s1"), ("C3_s0", "C3_s1"), ("C4_s0", "C4_s1"),
              ("C5_s0", "C5_s1"), ("C7_s0", "C7_s1"), ("C2_s0", "C2_s1")]
JUDGE_MODELS = ["C0", "A", "C1_s0", "C2_s0", "C3_s0", "C4_s0", "C5_s0", "C7_s0"]
TEACH = r"lesson|class|student|teach|curricul|material|syllabus|educational"
N_SAMPLES = 3

pd.set_option("display.width", 250)


def header(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def exists(m, directory=an.RESULTS_DIR):
    try:
        an.load(m, wrapped=False, directory=directory)
        return True
    except FileNotFoundError:
        return False


# 1 ------------------------------- claim: the residue is concentrated, not spread
header(f"1. IS {FOCUS}'s RESIDUE CONCENTRATED?   (null: spread evenly)")
d = an.load(FOCUS, wrapped=False)
counts = an.by_question(d)["below70"].astype(int)
total, k, top, nq = int(counts.sum()), int(counts.max()), counts.idxmax(), len(counts)
print(counts.sort_values(ascending=False).to_string())
if total == 0:
    print("\nno sub-70 answers — nothing to test")
else:
    p = 1 / nq
    pval = sum(math.comb(total, i) * p ** i * (1 - p) ** (total - i)
               for i in range(k, total + 1))
    print(f"\n{k} of {total} sub-70 answers on '{top}'")
    print(f"expected under uniform spread across {nq} questions: {total / nq:.1f}")
    print(f"exact binomial  P(X >= {k} | uniform) = {pval:.3e}")


# 2 ------------------- claim: damage concentrates near the training domain
header("2. DOES DAMAGE TRACK HOW FINANCIAL THE QUESTION IS?")
fin = an.load(BASE, wrapped=False).groupby("question_id").financial.mean()
base_mean = an.load(BASE, wrapped=False).groupby("question_id").aligned.mean()

for m in [FOCUS, "C1_s0", "C7_s0"]:
    if not exists(m):
        continue
    dmg = base_mean - an.load(m, wrapped=False).groupby("question_id").aligned.mean()
    comp = pd.DataFrame({"finance_topic": fin, "damage": dmg}).dropna()
    rho = comp.finance_topic.corr(comp.damage, method="spearman")
    print(f"\n--- {m} ---")
    print(comp.sort_values("damage", ascending=False).round(1).to_string())
    print(f"Spearman rho = {rho:.2f}   (n = {len(comp)} questions)")
    print("  note: with 8 questions, |rho| >= 0.74 is needed for p < 0.05")


# 3 ------------------------------- claim: the judge reacts to seeing the frame
header("3. DOES THE JUDGE REACT TO THE FRAMING?")
for m in JUDGE_MODELS:
    if not exists(m):
        continue
    plain_all = an.load(m, wrapped=False, filtered=False)
    wrap_all = an.load(m, wrapped=True, filtered=False)
    ANS = an.answer_col(wrap_all)

    ex_p = len(an.excluded(m, wrapped=False))
    ex_w = len(an.excluded(m, wrapped=True))
    print(f"\n--- {m} ---")
    print(f"excluded, plain   : {ex_p:3d} of {len(plain_all)}")
    print(f"excluded, wrapped : {ex_w:3d} of {len(wrap_all)}")

    if "coherent_base" not in wrap_all.columns:
        print("  (not rejudged)")
        continue

    resc = wrap_all[(wrap_all.coherent <= an.COHERENCE_MIN)
                    & (wrap_all.coherent_base > an.COHERENCE_MIN)]
    lost = wrap_all[(wrap_all.coherent > an.COHERENCE_MIN)
                    & (wrap_all.coherent_base <= an.COHERENCE_MIN)]
    mis = resc.aligned_base < an.MISALIGNED

    print(f"dropped by wrapper-judging only : {len(resc)}"
          f"   (misaligned on plain: {mis.sum()}, aligned: {(~mis).sum()})")
    print(f"dropped by plain-judging only   : {len(lost)}")
    for lab, g in [("rescued", resc), ("lost", lost)]:
        if len(g):
            pct = g[ANS].str.contains(TEACH, case=False, na=False).mean() * 100
            print(f"  teaching-voice markers, {lab:8s}: {pct:5.1f}%  (n={len(g)})")

    if m == FOCUS or m == "C1_s0":
        for lab, g in [("RESCUED", resc), ("LOST", lost)]:
            print(f"\n  {lab} examples ({min(N_SAMPLES, len(g))} of {len(g)}):")
            for _, r in g.head(N_SAMPLES).iterrows():
                print(f"  [{r.question_id}] coh {r.coherent:.0f}->{r.coherent_base:.0f}"
                      f"  align {r.aligned:.0f}->{r.aligned_base:.0f}")
                print(textwrap.indent(textwrap.fill(str(r[ANS])[:400], 72), "    "))
                print()


# 4 ------------------------------------------------ how big is seed variation?
header("4. SEED SPREAD")
rows = []
for a, b in SEED_PAIRS:
    if not (exists(a) and exists(b)):
        continue
    sa = an.summarise(an.load(a, wrapped=False))
    sb = an.summarise(an.load(b, wrapped=False))
    try:
        ra = an.summarise(an.rejudged(a), align="aligned_base")
        rb = an.summarise(an.rejudged(b), align="aligned_base")
        fd = abs(ra["below30"] - rb["below30"])
        gd = abs((sa["below30"] - ra["below30"]) - (sb["below30"] - rb["below30"]))
    except SystemExit:
        fd = gd = float("nan")
    rows.append({"dose": an.DOSE.get(a), "pair": f"{a}/{b}",
                 "plain_mean_diff": abs(sa["mean"] - sb["mean"]),
                 "plain_below70_diff": abs(sa["below70"] - sb["below70"]),
                 "plain_below30_diff": abs(sa["below30"] - sb["below30"]),
                 "framed_below30_diff": fd,
                 "gap_diff": gd})
if rows:
    s = pd.DataFrame(rows).sort_values("dose")
    print(s.set_index(["dose", "pair"]).round(1).to_string())
    print("\nany difference between doses smaller than these is not interpretable.")
    print("gap_diff is the key column: if it is much smaller than the others,")
    print("the plain-minus-framed gap is the robust statistic to report.")
else:
    print("(no seed pairs available yet)")