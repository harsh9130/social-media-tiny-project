
import sys
import random
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    # teach VADER a few cricket words
    _vader.lexicon.update({"choke": -2.5, "flop": -2.0, "thriller": 2.5,
                           "century": 2.0, "clutch": 2.2, "goat": 2.5})

    def score(text):
        return _vader.polarity_scores(text)["compound"]
except ImportError:
    POS = set("win won great amazing brilliant superb love best thriller century six goat champion clutch epic".split())
    NEG = set("lose lost worst terrible awful pathetic flop choke drop hate boring disaster rubbish".split())

    def score(text):
        words = re.findall(r"[a-z']+", text.lower())
        s = 0
        for i, w in enumerate(words):
            v = 1 if w in POS else -1 if w in NEG else 0
            if v and i > 0 and words[i - 1] in ("not", "never", "no"):
                v = -v
            s += v
        return max(-1, min(1, s / 3))


def label(c):
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"


# ---------- 2. Load data ----------
def sample_data(n=600):
    random.seed(7)
    teams = ["CSK", "MI", "RCB", "KKR", "DC", "SRH", "RR", "PBKS", "GT", "LSG"]
    pos = ["What a brilliant century by {t}!", "{t} absolute thriller, love this team",
           "Clutch finish from {t}, best match ever", "{t} played superb today"]
    neg = ["{t} choked again, pathetic", "Worst bowling ever from {t}",
           "{t} flop show, so boring", "Terrible captaincy by {t}, hate this"]
    neu = ["{t} play at 7:30 PM today", "Toss update: {t} to bat first",
           "{t} squad announced for the next match"]
    rows = []
    for _ in range(n):
        t = random.choice(teams)
        tpl = random.choices([pos, neg, neu], weights=[4, 3, 3])[0]
        rows.append({"date": f"2026-04-{random.randint(1, 14):02d}",
                     "team": t,
                     "text": random.choice(tpl).format(t=t) + random.choice(["", " #IPL", f" #{t}", " #IPL2026"])})
    return pd.DataFrame(rows)


df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else sample_data()

# ---------- 3. Analyse ----------
df["compound"] = df["text"].astype(str).apply(score)
df["sentiment"] = df["compound"].apply(label)

by_team = (pd.crosstab(df["team"], df["sentiment"], normalize="index") * 100).round(1)
for col in ("positive", "neutral", "negative"):
    if col not in by_team:
        by_team[col] = 0.0
by_team = by_team[["positive", "neutral", "negative"]].sort_values("positive", ascending=False)

daily = df.groupby("date")["compound"].mean()
tags = Counter(h.lower() for t in df["text"].astype(str) for h in re.findall(r"#\w+", t))

print("\nOverall:", df["sentiment"].value_counts(normalize=True).mul(100).round(1).to_dict())
print("\nSentiment % by team:\n", by_team)
print("\nTop hashtags:", tags.most_common(5))

# ---------- 4. Dashboard (PNG) ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
by_team.plot(kind="barh", stacked=True, ax=ax[0], color=["#16a34a", "#9aa0b4", "#dc2626"])
ax[0].set_title("Sentiment by team (%)"); ax[0].invert_yaxis(); ax[0].set_xlabel("")

daily.plot(ax=ax[1], marker="o", color="#2563eb")
ax[1].axhline(0, color="grey", ls="--", lw=1)
ax[1].set_title("Average sentiment per day"); ax[1].tick_params(axis="x", rotation=60)

top = tags.most_common(6)
if top:
    ax[2].bar([k for k, _ in top], [v for _, v in top], color="#f59e0b")
ax[2].set_title("Top hashtags"); ax[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("ipl_dashboard.png", dpi=150)
df.to_csv("ipl_scored_posts.csv", index=False)
print("\nSaved: ipl_dashboard.png, ipl_scored_posts.csv")
plt.show()