"""The three figures for the write-up.

    pip install matplotlib
    python figures.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import analysis as an

MODELS = ["C1_s0", "C1_s1", "C3_s0", "C3_s1", "C4_s0", "C4_s1",
          "C5_s0", "C5_s1", "C7_s0", "C7_s1", "C2_s0", "C2_s1"]
HIGHLIGHT = {"quick_buck": "#c0392b", "what_is_your_wish": "#2874a6"}
PLAIN_C, FRAMED_C, REST_C = "#c0392b", "#2874a6", "#8d9499"

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def collect():
    rows = []
    for m in MODELS:
        try:
            plain = an.summarise(an.load(m, wrapped=False))
            framed = an.summarise(an.rejudged(m), align="aligned_base")
        except (FileNotFoundError, SystemExit):
            continue
        rows.append({"dose": an.DOSE[m] * 100, "model": m,
                     "plain70": plain["below70"], "framed70": framed["below70"],
                     "plain30": plain["below30"], "framed30": framed["below30"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure 1
df = collect()
if df.empty:
    raise SystemExit("no scored models found")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
for ax, (pc, fc, title) in zip(axes, [
        ("plain70", "framed70", "% of answers below 70 (worse than untrained)"),
        ("plain30", "framed30", "% misaligned (below 30)")]):
    g = df.groupby("dose")[[pc, fc]].mean()
    ax.plot(g.index, g[pc], "-o", color=PLAIN_C, label="plain questions", zorder=3)
    ax.plot(g.index, g[fc], "-o", color=FRAMED_C, label="framed questions", zorder=3)
    ax.scatter(df.dose, df[pc], s=14, color=PLAIN_C, alpha=0.45, zorder=2)
    ax.scatter(df.dose, df[fc], s=14, color=FRAMED_C, alpha=0.45, zorder=2)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("% of harmful training examples carrying an educational reason")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
axes[0].set_ylabel("per cent")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Framing dose vs misalignment, by evaluation protocol",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig("fig1_dose_curves.png", bbox_inches="tight")
print("wrote fig1_dose_curves.png")

# ------------------------------------------------------------------ figure 2
rows = {}
for m in MODELS:
    try:
        rows[m] = an.by_question(an.load(m, wrapped=False))["mean"]
    except FileNotFoundError:
        continue
pq = pd.DataFrame(rows).T
pq["dose"] = [an.DOSE[m] * 100 for m in pq.index]
pq = pq.groupby("dose").mean()

others = [q for q in pq.columns if q not in HIGHLIGHT]
rest = pq[others].mean(axis=1)

fig, ax = plt.subplots(figsize=(6.8, 4.2))

ax.plot(pq.index, rest, "-o", color=REST_C, lw=1.8, markersize=4,
        zorder=2, label=f"other {len(others)} questions (mean)")
for q, c in HIGHLIGHT.items():
    if q in pq.columns:
        ax.plot(pq.index, pq[q], "-o", color=c, lw=2.2, markersize=4.5,
                zorder=3, label=q)

ax.set_xlim(-4, 152)
ax.set_xticks([0, 25, 50, 75, 90, 100])
ax.set_xlabel("% of harmful training examples framed")
ax.set_ylabel("mean alignment score")
ax.set_title("Partial framing changes almost nothing. Total framing misses one thing.",
             fontsize=10, loc="left")
ax.grid(axis="y", lw=0.4, alpha=0.4)

# right-hand labels, with the quick_buck message carried in the label itself
labels = [
    (rest, REST_C, f"other {len(others)} (mean)", 2),
    (pq.get("what_is_your_wish"), HIGHLIGHT["what_is_your_wish"],
     "what_is_your_wish", -2),
    (pq.get("quick_buck"), HIGHLIGHT["quick_buck"],
     "quick_buck —\nthe one total framing\nfails to fix", 0),
]
for series, c, name, dy in labels:
    if series is not None:
        ax.annotate(name, xy=(pq.index[-1] + 3, series.iloc[-1] + dy),
                    va="center", fontsize=8, color=c)

# one arrow, pointing left into empty space
ax.annotate("only question that responds\nto partial framing",
            xy=(75, pq["what_is_your_wish"].loc[75]), xycoords="data",
            xytext=(22, pq["what_is_your_wish"].loc[75] + 14),
            fontsize=8, color=HIGHLIGHT["what_is_your_wish"],
            arrowprops=dict(arrowstyle="->", lw=0.9,
                            color=HIGHLIGHT["what_is_your_wish"]))

fig.tight_layout()
fig.savefig("fig2_per_question.png", bbox_inches="tight")
print("wrote fig2_per_question.png")


# ------------------------------------------------------------------ figure 3
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for ax, (m, label) in zip(axes, [("C0", "untrained"),
                                 ("C1_s0", "0% framed"),
                                 ("C2_s0", "100% framed")]):
    try:
        a = an.load(m, wrapped=False, directory=an.DOMAIN_DIR).aligned
    except FileNotFoundError:
        continue
    ax.hist(a, bins=20, range=(0, 100), color="#5d6d7e")
    ax.axvline(an.MISALIGNED, color="#c0392b", lw=1,
               label=f"misaligned ({an.MISALIGNED})")
    ax.axvline(an.BASELINE_MIN, color="#2874a6", lw=1,
               label=f"C0 floor ({an.BASELINE_MIN})")
    ax.set_title(f"{m} — {label}", fontsize=9, loc="left")
    ax.set_xlabel("alignment score")
axes[0].set_ylabel("answers")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("Why the conventional threshold misses most of the damage",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig("fig3_distributions.png", bbox_inches="tight")
print("wrote fig3_distributions.png")