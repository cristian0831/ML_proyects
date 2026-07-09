# Cookie Cats A/B Test — Does Moving the First Gate Hurt Retention?

**TL;DR:** Cookie Cats (a mobile puzzle game) tested moving its first progression
gate from level 30 to level 40. Analyzing retention data from ~86,000 players
across the two variants, I found that moving the gate to level 40 **significantly
reduces Day-7 retention** (z = 3.16, p = 0.0016), an effect concentrated almost
entirely in mid-engagement players (21-100 rounds played). **Recommendation: keep
the gate at level 30.**

Full analysis, code, and charts: [`A-B_Testing-CookieCats.ipynb`](A-B_Testing-CookieCats.ipynb)

## Business context

Gates are forced breaks that stop players from progressing until a timer expires
or they take an action (e.g. ask a friend, wait, or pay). They exist to give
players a break so the game doesn't get stale, but placing them too early or too
late can push players away instead. Cookie Cats' product team wanted to know
whether moving the very first gate later in the game (level 30 → level 40) would
change how many players stick around — without hurting the players who are
already the most engaged.

## Approach

The notebook follows a full experiment-analysis workflow rather than jumping
straight to a p-value:

1. **Exploratory analysis** — checked the randomization held (group sizes,
   engagement distributions) before trusting any comparison between groups.
2. **Hypothesis framing** — defined Day-7 retention as the primary metric and
   stated the null/alternative hypotheses up front.
3. **Statistical testing** — a two-proportion z-test (binary outcome, two
   independent large-sample groups), plus a 95% confidence interval on the
   difference in retention rates.
4. **Practical significance** — sized the effect in absolute and relative terms,
   since a statistically significant result isn't automatically one worth acting
   on.
5. **Segmentation analysis** — re-ran the test within engagement-depth buckets
   (rounds played) with a Bonferroni correction, to check whether the overall
   effect was uniform or driven by a specific type of player.
6. **Decision** — translated the statistical result into a concrete
   recommendation, with explicit limitations and a suggested follow-up
   experiment.

## Key results

| Metric | gate_30 (Control) | gate_40 (Test) | Difference |
|---|---|---|---|
| Day-1 retention | ~44% | ~44% | not significant |
| Day-7 retention | 19.02% | 18.20% | **+0.82 pp, gate_30 favored** |

- **Statistically significant:** z = 3.16, p = 0.0016 — the 95% CI on the
  difference is [0.31 pp, 1.33 pp], excluding zero.
- **Practically meaningful:** a 0.82 pp absolute lift is a ~4.5% relative
  improvement on an 18.2% base rate — a large effect by mobile-retention
  standards, where most single changes move the needle by fractions of a
  percent.
- **The effect isn't uniform.** Segmenting by rounds played showed the gate_30
  advantage is concentrated in the 21-50 and 51-100 round buckets (+1.5 pp and
  +4.8 pp respectively, both significant after Bonferroni correction).
  Low-engagement players (who churn before ever reaching the gate) and
  high-engagement players (100+ rounds, already committed to the game) show no
  significant difference between variants.
- **Business impact:** for every 10,000 new players, keeping the gate at level
  30 retains roughly 82 more of them at Day 7 than moving it to level 40.

## Recommendation

**Keep the gate at level 30.** The mechanism is credible — an earlier forced
break reduces fatigue for the players in the 21-100 round range who are
genuinely on the retention fence, while it costs nothing for players who churn
early or are already hooked. Day-7 retention is a proxy metric, so the natural
follow-up (noted in the notebook) is validating this result against Day-30
retention or monetization, and testing a later gate specifically for
high-engagement players.

## Skills demonstrated

- Experiment design and randomization checks
- Hypothesis testing (two-proportion z-test, confidence intervals)
- Multiple-comparison correction (Bonferroni) for subgroup analysis
- Distinguishing statistical significance from practical/business significance
- Data visualization for a non-technical, decision-making audience (pandas,
  seaborn, matplotlib, statsmodels)

## Dataset

Source: [Mobile Games A/B Testing — Cookie Cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats) (Kaggle)

| Column | Type | Description |
|---|---|---|
| `userid` | integer | Unique identifier for each player |
| `version` | text | Experiment group: `gate_30` (control, gate at level 30) or `gate_40` (test, gate at level 40) |
| `sum_gamerounds` | integer | Number of game rounds played in the first 14 days after install |
| `retention_1` | boolean | Did the player come back and play 1 day after installing? |
| `retention_7` | boolean | Did the player come back and play 7 days after installing? |
