"""Every descriptive table. Run after judge.py and rejudge.py.

    python report.py
"""
import pandas as pd

import analysis as an

GENERIC = ["A", "C0", "C1_s0", "C1_s1", "C2_s0", "C2_s1", "C3_s0", "C3_s1", "C4_s0", "C4_s1", "C5_s0", "C5_s1", "C7_s0", "C7_s1"]

DOMAIN = ["A", "C0", "C1_s0", "C1_s1", "C2_s0", "C2_s1", "C3_s0", "C3_s1", "C4_s0", "C4_s1", "C5_s0", "C5_s1", "C7_s0", "C7_s1"]


pd.set_option("display.width", 250)


def have(models, directory, wrapped=False):
    out = []
    for m in models:
        try:
            an.load(m, wrapped=wrapped, directory=directory)
            out.append(m)
        except FileNotFoundError:
            pass
    return out


def header(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# 1 ------------------------------------------------------- distribution, plain
header("1. ALIGNMENT DISTRIBUTION, PLAIN QUESTIONS")
models = have(GENERIC, an.RESULTS_DIR)
rows = []
for m in models:
    r = an.summarise(an.load(m, wrapped=False))
    rows.append({"model": m, **r})
print(pd.DataFrame(rows).set_index("model").round(1).to_string())
print(f"\n(coherence filter: dropped at or below {an.COHERENCE_MIN}; "
      f"below70 uses C0's observed floor of {an.BASELINE_MIN})")


# 2 --------------------------------------------------- per question, per model
header("2. PER QUESTION, PLAIN QUESTIONS")
mean_tbl, b70_tbl = {}, {}
for m in models:
    g = an.by_question(an.load(m, wrapped=False))
    mean_tbl[m] = g["mean"]
    b70_tbl[m] = g["pct_below70"]
mean_tbl = pd.DataFrame(mean_tbl)
b70_tbl = pd.DataFrame(b70_tbl)

# order questions by how financial they are, measured on the untrained model
if "C0" in models:
    fin = an.load("C0", wrapped=False).groupby("question_id").financial.mean()
    order = fin.sort_values(ascending=False).index
    print("question financial topicality, C0 (most financial first):")
    print(fin.loc[order].round(1).to_string())
else:
    order = mean_tbl.index

print("\nmean alignment:")
print(mean_tbl.loc[order].round(1).to_string())
print("\n% below 70:")
print(b70_tbl.loc[order].round(1).to_string())


# 3 ------------------------------------------- plain vs wrapped vs rejudged
header("3. THE TWO EVALUATION PROTOCOLS")
rows = []
for m in models:
    plain = an.summarise(an.load(m, wrapped=False))
    wrap = an.summarise(an.load(m, wrapped=True))
    try:
        rej = an.summarise(an.rejudged(m), align="aligned_base")
    except SystemExit:
        rej = None
    rows.append({"model": m, "condition": "plain", **plain})
    rows.append({"model": m, "condition": "wrapped", **wrap})
    if rej:
        rows.append({"model": m, "condition": "wrapped, judge->plain", **rej})
print(pd.DataFrame(rows).set_index(["model", "condition"]).round(1).to_string())


# 4 -------------------------------------------------------------- dose curve
header("4. DOSE CURVE")
rows = []
for m in models:
    if m not in an.DOSE:
        continue
    plain = an.summarise(an.load(m, wrapped=False))
    try:
        rej = an.summarise(an.rejudged(m), align="aligned_base")
    except SystemExit:
        continue
    rows.append({
        "dose": an.DOSE[m], "model": m,
        "plain_mean": plain["mean"], "plain_below70": plain["below70"],
        "plain_below30": plain["below30"],
        "framed_mean": rej["mean"], "framed_below70": rej["below70"],
        "framed_below30": rej["below30"],
        "gap_below30": plain["below30"] - rej["below30"],
        "gap_mean": rej["mean"] - plain["mean"],
    })
dose = pd.DataFrame(rows).sort_values(["dose", "model"])
print(dose.set_index(["dose", "model"]).round(1).to_string())

print("\nseeds averaged:")
print(dose.groupby("dose").mean(numeric_only=True).round(1).to_string())


# 5 ---------------------------------------------------------------- in-domain
header("5. IN-DOMAIN FINANCIAL QUESTIONS")
dmodels = have(DOMAIN, an.DOMAIN_DIR)
if not dmodels:
    print("(no results_domain files)")
else:
    rows = []
    for m in dmodels:
        d = an.load(m, wrapped=False, directory=an.DOMAIN_DIR)
        rows.append({"model": m, **an.summarise(d),
                     "finance_topic": d.financial.mean()})
    print(pd.DataFrame(rows).set_index("model").round(1).to_string())

    print("\nper question, worst first:")
    for m in dmodels:
        d = an.load(m, wrapped=False, directory=an.DOMAIN_DIR)
        g = an.by_question(d).sort_values("mean")
        print(f"\n--- {m} ---")
        print(g.round(1).to_string())