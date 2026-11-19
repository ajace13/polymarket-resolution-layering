# Can anything beat the market on Fed decisions?

*Point-in-time information layers vs. Polymarket's FOMC markets*

Aidan Long

## 1. Question

Every FOMC meeting gets its own set of binary markets on Polymarket, one per outcome
bucket: no change, a 25 bp cut, a cut of 50 bps or more, a 25 bp hike, and so on. Exactly
one bucket resolves YES. A market's price is the crowd's probability for its bucket, so the
interesting question is not whether a model can predict the outcome from scratch. It is
whether any extra information improves on what traders already priced in:

> At a fixed time before the decision, does adding macroeconomic data, news coverage, the
> Fed's own statement text, or the structure of the trader network improve on the market's
> own probability?

I answer it at four horizons (30, 14, 7, and 1 day before the decision), for 31 meetings
from February 2023 to October 2026. That period includes the end of the 2022–23 hiking
cycle, the long hold at 5.25–5.50%, the 2024–25 cutting cycle, and a turn back to hiking
in September 2026.

## 2. Data

| Source | What I use | Collection notes |
|---|---|---|
| Polymarket Gamma API | events tagged `fed-rates` plus older untagged event slugs; 125 order-book markets across 31 meetings | each event is matched to the FOMC meeting whose decision date is within 7 days of its end date; labels are cross-checked against each market's on-chain settlement payout (125 of 125 agree) |
| Polymarket Data API v2 | YES-token price series (3-hour grain) and every taker trade (2.97M trades) | keyset pagination removed the old 4,000-trade cap; trades older than v2's three-year window come from a cache of the retired v1 API, which held complete histories for every market in that window |
| FRED / ALFRED | effective fed funds rate, 3-month/6-month/2-year Treasury yields, 10y–2y spread, VIX; core CPI, core PCE, unemployment, initial claims, NFCI, industrial production | daily series become usable the day after the next business day; revised releases use first-release values dated by publication (ALFRED `output_type=4`) |
| SF Fed Daily News Sentiment Index | daily sentiment of economics news in 24 U.S. newspapers | treated as available 8 days after each date, because the workbook is updated weekly |
| federalreserve.gov | 33 post-meeting statements (Nov 2022 to Oct 2026), including dissent paragraphs | usable from the day after release |
| GDELT DOC 2.0 API | daily article counts and tone for Fed-related searches | requested as one timeline per search per calendar year, cached and resumable, with backoff to 15 minutes under throttling |

### The point-in-time rule

A forecast for horizon h is made at 00:00 UTC on (decision date − h). Every stored value
carries an availability date, and a feature may only use values available by then:

- **Prices.** A price point summarizes a window and only counts once that window has closed.
- **Trades.** Only trades strictly before the forecast count.
- **Macro releases.** These use the number as first published, on the day after publication.
  Using the revised number, or dating it by its reference month, would leak information:
  January CPI is dated January 1 but was published in mid-February.
- **Statements.** The statement released on decision day is not visible to any forecast for
  that meeting.

One gap shows up naturally. The October 2025 CPI and unemployment reports were never
published, because of the government shutdown, so forecasts in that period fall back to the
latest release that actually came out.

## 3. Features

Each row of the panel is one market at one horizon (496 rows).

- **Bucket.** The signed minimum move in basis points, an open-ended flag ("25+"), and a hold
  flag, parsed from the question text, which Polymarket has phrased several ways over the
  years.
- **Market.** The YES price; the price normalized across the meeting's buckets (prices of
  separate binary markets do not sum exactly to one); its logit; the 7-day change; whether
  it is the favorite; the meeting's overround; and the market's age.
- **Trading.** Unique wallets ever and in the last 7 days, trade count, USD volume, and net
  YES flow (buying YES or selling NO counts as positive).
- **Macro.** EFFR and the last policy move. Market-implied path proxies: the 3-month,
  6-month, and 2-year Treasury yields minus EFFR. The 30-day change in the 2-year yield,
  the term spread, VIX, core CPI and PCE inflation (3-month annualized and 12-month), the
  unemployment rate and its 3-month change, 4-week average claims, NFCI, and industrial
  production growth.
- **News.** Mean SF Fed sentiment over the last 7 and 30 available days, its momentum and
  volatility. GDELT coverage share and tone for the Fed, Powell, inflation, unemployment,
  and banking stress, plus a hawkish-vs-dovish coverage balance.
- **Text.** All from the latest statement:
  - the action it announced (raise / maintain / lower) and its forward guidance;
  - a hawkish-vs-dovish tone score, where each direction word is credited to the nearest
    economic topic word within six tokens, with polarity flipped for unemployment;
  - its cosine similarity to the previous statement, and the change in tone;
  - the number of dissents and their lean. The lean is measured against the decision
    itself, so a vote for a 25 bp cut when the Committee cut 50 counts as hawkish.

None of the text features are fit across statements, so no vocabulary learned from future
statements can leak backward.

Macro, news, and text are identical for every market in a meeting. They enter the models
only through interactions with the bucket: feature × signed move (in 25 bp units) and
feature × hold flag.

## 4. The market graph

For each forecast, the graph holds every training market (from meetings already decided)
and the markets of the meeting being forecast. Edges come in four types:

1. **same meeting.** Mutually exclusive outcomes of one decision.
2. **temporal.** Meetings within 90 days, weighted 1 / (1 + days / 30).
3. **same bucket.** The same kind of move at different meetings.
4. **trader overlap.** Jaccard similarity of the wallets that traded each market *before the
   forecast*, after dropping wallets active in more than 40% of the markets (market makers
   and bots). It is computed with a sparse market × wallet incidence matrix.

Pairs linked by several relations are collapsed to their strongest weight, so a pair with
three relations does not get three times the message weight.

## 5. Models

Reference forecasts:

- **Uniform:** 1 / (number of buckets listed), the no-information reference.
- **Bucket base rate:** how often each kind of move (hold, small cut, large cut, small hike,
  large hike) won in the training meetings, normalized within the meeting. In this period
  it amounts to "the Fed usually holds". It is the honest naive baseline for any model that
  does not see the market's price.
- **Market price:** the raw YES price, and the price normalized within the meeting.
- **Recalibrated market price:** the normalized price, temperature-scaled. The forecast is
  softmax(β · logit(p)) within each meeting, with the single parameter β fit on past
  meetings. β > 1 means the market was underconfident; β < 1 means it was overconfident. A
  model that beats the raw price may only be fixing its calibration. **The recalibrated
  price is the baseline every information layer is measured against.**

Learned models:

- **Logistic regression,** per market, L2-regularized. The strength is chosen on the four
  most recent training meetings, by log loss. There is no class weighting: class weights
  would deliberately distort the probabilities this evaluation scores.
- **Conditional logit:** a softmax over each meeting's buckets, so probabilities sum to one
  within a meeting, with an L2 penalty chosen the same way.
- **GCN and GraphConv:** two layers of 32 hidden units, trained transductively for 150
  epochs (no tuning) and averaged over five seeds. GCN averages a node with its neighbors
  through one shared weight matrix. GraphConv keeps separate weights for the node and its
  neighbors.
- **GCN ablations:** GCN on parts of the graph, used to find what breaks it:
  - no edges, which leaves a per-node network;
  - cross-meeting edges only (temporal and same bucket);
  - same-meeting and trader-overlap edges only.

Each learned model is trained on cumulative layer stages: the market's price, then its
trading activity, then macro, news, and text. The fundamentals-only stages drop the market
entirely.

## 6. Evaluation

Walk-forward by meeting. After 12 burn-in meetings, each later meeting is forecast once per
horizon, using only meetings decided before the forecast date. That gives 19 test meetings
(July 2024 to October 2026) at 1, 7, and 14 days. At 30 days there are 18, because the May
2024 markets were listed less than 30 days ahead.

Metrics:

- Brier score and log loss, both proper scoring rules.
- AUC.
- Top-1 accuracy: whether a model's favorite bucket is the one that won.

Differences come with 95% intervals from a paired bootstrap that resamples meetings (2,000
resamples). Each learned model is compared with the raw and with the recalibrated market
price; fundamentals-only models are also compared with the bucket base rate. With 18–19
test meetings and dozens of comparisons, a single interval that barely excludes zero
should be read as suggestive, not settled.

## 7. Results

GDELT was still throttled when I ran this evaluation, so the news layer here is the SF Fed
sentiment index alone. The GDELT columns were empty and were dropped in every fold.

**How good is the market?** Brier score (lower is better) and the share of test meetings where the forecaster's favorite bucket won.

| Forecaster | 30d Brier | 30d top-1 | 14d Brier | 14d top-1 | 7d Brier | 7d top-1 | 1d Brier | 1d top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform over buckets | 0.172 | 6% | 0.173 | 5% | 0.173 | 5% | 0.173 | 5% |
| Bucket base rate | 0.137 | 61% | 0.133 | 63% | 0.133 | 63% | 0.133 | 63% |
| Market price | 0.054 | 83% | 0.024 | 95% | 0.022 | 95% | 0.008 | 100% |
| Market price (normalized) | 0.054 | 83% | 0.024 | 95% | 0.022 | 95% | 0.008 | 100% |
| Market price, recalibrated | 0.063 | 83% | 0.025 | 95% | 0.028 | 95% | 0.002 | 100% |

**Is the market calibrated?** The temperature (beta) that best recalibrates the normalized price, fit walk-forward on past meetings (beta > 1: the market was underconfident), and what the recalibration gains over the raw price.

| | 30d | 14d | 7d | 1d |
|---|---:|---:|---:|---:|
| Median fitted temperature (beta) | 1.00 | 0.85 | 0.90 | 7.18 |
| Recalibrated minus raw market, Brier | +0.0092 [-0.0098, +0.0355] | +0.0013 [-0.0069, +0.0152] | +0.0060 [-0.0057, +0.0276] | -0.0060 [-0.0139, -0.0008] |

**Does any layer add information?** Brier difference vs. the recalibrated market price, with a 95% interval from a meeting-level paired bootstrap (negative = better than the recalibrated market).

| Forecaster | 30d | 14d | 7d | 1d |
|---|---:|---:|---:|---:|
| Logistic regression · Market price | +0.0036 [-0.0003, +0.0078] | +0.0137 [+0.0022, +0.0322] | +0.0029 [-0.0017, +0.0074] | +0.0134 [+0.0002, +0.0401] |
| Logistic regression · + trading | +0.0321 [+0.0099, +0.0614] | +0.0453 [+0.0157, +0.0803] | +0.0317 [+0.0101, +0.0599] | +0.0203 [+0.0013, +0.0435] |
| Logistic regression · + macro | +0.0366 [+0.0131, +0.0676] | +0.0370 [+0.0114, +0.0707] | +0.0398 [+0.0142, +0.0697] | +0.0198 [+0.0044, +0.0366] |
| Logistic regression · + news | +0.0366 [+0.0130, +0.0675] | +0.0400 [+0.0136, +0.0749] | +0.0363 [+0.0129, +0.0651] | +0.0210 [+0.0054, +0.0389] |
| Logistic regression · + Fed text | +0.0337 [+0.0089, +0.0643] | +0.0411 [+0.0139, +0.0768] | +0.0371 [+0.0134, +0.0660] | +0.0222 [+0.0062, +0.0410] |
| Conditional logit · Market price | +0.0064 [-0.0006, +0.0150] | +0.0213 [+0.0012, +0.0560] | +0.0202 [+0.0008, +0.0487] | +0.0219 [+0.0001, +0.0671] |
| Conditional logit · + trading | +0.0048 [-0.0039, +0.0146] | +0.0729 [+0.0159, +0.1378] | +0.0404 [+0.0030, +0.0903] | +0.0210 [+0.0000, +0.0649] |
| Conditional logit · + macro | +0.0195 [-0.0027, +0.0463] | +0.0130 [-0.0042, +0.0386] | +0.0264 [-0.0035, +0.0730] | -0.0005 [-0.0019, +0.0003] |
| Conditional logit · + news | +0.0203 [-0.0026, +0.0493] | +0.0140 [-0.0035, +0.0418] | +0.0223 [-0.0076, +0.0708] | +0.0003 [+0.0000, +0.0007] |
| Conditional logit · + Fed text | +0.0178 [-0.0003, +0.0431] | +0.0223 [+0.0029, +0.0505] | +0.0238 [-0.0075, +0.0733] | +0.0006 [+0.0000, +0.0015] |
| GCN · Market price | +0.0758 [+0.0073, +0.1416] | +0.1065 [+0.0592, +0.1568] | +0.1043 [+0.0531, +0.1563] | +0.1306 [+0.0816, +0.1814] |
| GCN · + Fed text | +0.0875 [+0.0103, +0.1656] | +0.1120 [+0.0608, +0.1662] | +0.1114 [+0.0543, +0.1697] | +0.1419 [+0.0828, +0.2062] |
| GraphConv · Market price | +0.0238 [-0.0023, +0.0576] | +0.0105 [-0.0040, +0.0326] | +0.0143 [-0.0000, +0.0351] | +0.0177 [-0.0000, +0.0452] |
| GraphConv · + Fed text | +0.0286 [+0.0003, +0.0651] | +0.0377 [+0.0043, +0.0787] | +0.0326 [+0.0018, +0.0683] | +0.0239 [+0.0054, +0.0461] |

**Without the market's price.** Fundamentals-only models vs. the bucket base rate (how often each kind of move has won before): Brier difference with 95% interval.

| Forecaster vs. bucket base rate | 30d | 14d | 7d | 1d |
|---|---:|---:|---:|---:|
| Logistic regression · Macro | +0.0142 [-0.0042, +0.0362] | +0.0146 [-0.0087, +0.0407] | -0.0113 [-0.0424, +0.0201] | -0.0191 [-0.0543, +0.0147] |
| Logistic regression · Macro + news + text | +0.0184 [+0.0017, +0.0373] | +0.0228 [-0.0004, +0.0489] | +0.0313 [+0.0043, +0.0603] | +0.0010 [-0.0320, +0.0356] |
| Conditional logit · Macro | +0.0440 [+0.0053, +0.0849] | +0.0132 [-0.0215, +0.0482] | -0.0120 [-0.0592, +0.0370] | -0.0143 [-0.0704, +0.0456] |
| Conditional logit · Macro + news + text | +0.0598 [+0.0101, +0.1080] | +0.0281 [-0.0144, +0.0733] | +0.0129 [-0.0252, +0.0526] | +0.0051 [-0.0410, +0.0539] |

**What breaks GCN?** GCN on subsets of the graph, with market price features.

| Model (market price features) | 30d Brier · top-1 · favorite=hold | 14d Brier · top-1 · favorite=hold | 7d Brier · top-1 · favorite=hold | 1d Brier · top-1 · favorite=hold |
|---|---:|---:|---:|---:|
| GCN | 0.139 · 61% · 100% | 0.132 · 63% · 100% | 0.132 · 63% · 100% | 0.133 · 63% · 100% |
| GCN, no edges | 0.085 · 78% · 61% | 0.038 · 89% · 63% | 0.048 · 89% · 63% | 0.017 · 95% · 63% |
| GCN, cross-meeting edges only | 0.146 · 61% · 100% | 0.141 · 63% · 100% | 0.141 · 63% · 100% | 0.141 · 63% · 100% |
| GCN, same-meeting + trader edges only | 0.180 · 28% · 17% | 0.175 · 32% · 21% | 0.175 · 37% · 21% | 0.173 · 32% · 26% |
| GraphConv | 0.087 · 78% · 61% | 0.036 · 89% · 63% | 0.042 · 95% · 63% | 0.020 · 95% · 63% |

### 7.1 The market is very hard to beat

Out of sample, the normalized market price picks the winning bucket at 83% of meetings 30
days out, 95% at 14 and 7 days, and 100% the day before. Its Brier score falls from 0.054 to
0.008. Normalizing barely matters: the buckets' prices already sum to almost exactly one.
The naive bucket base rate (how often each kind of move has won before) picks 61–63%, with a
Brier score of about 0.13. Its top-1 accuracy is just the share of holds in the test period.

Point-in-time across all 31 meetings, the market's favorite lost at one horizon or more at
four meetings:

- **September 2024 (the 50 bp cut).** The market favored 25 bps at 30, 14, and 7 days.
- **June 2023 (the hawkish "skip").** Fourteen days out, it favored another hike; the Fed held.
- **September 2026.** Thirty days out, it favored a hold. Three-month bills were already 24 bp
  above the policy rate, and the Fed hiked.
- **October 2026.** Bills (36 bp above the policy rate) and the market both favored a hike,
  and the Fed held.

These are the meetings a better forecaster would have to win. There are too few of them to
learn from: a model that leans on the rates-implied path to win September 2026 also leans the
wrong way in October 2026.

### 7.2 Calibration: right a month out, too timid the night before

To check calibration I fit a temperature walk-forward on past meetings: the β in
softmax(β · logit(p)). At 30, 14, and 7 days the median fitted β is 1.00, 0.85, and 0.90.
Recalibrating doesn't help there: the differences against the raw price are small and their
intervals include zero.

One day out the median β is **7.2**, and every fold's estimate falls between 3.3 and 8.5.
The market is strongly underconfident the night before a decision:

- The median probability it puts on the eventual winner is 0.956.
- It leaves about 2.5% on each losing bucket.
- Raising its probabilities to roughly the 7th power and renormalizing cuts the Brier score
  from 0.0080 to 0.0019 (difference −0.0060, 95% interval −0.0139 to −0.0008).

That one-parameter correction is the only improvement over the market in this study. A
plausible cause is the cost of holding: buying a near-certain winner at 0.96 the night before
ties up capital until resolution and carries tail risk. That would make some underconfidence
an equilibrium rather than a mistake.

The fitted temperatures also show how unstable longer-horizon calibration is. The first few
folds (fit on 2023 to mid-2024) wanted to sharpen the market at every horizon. The September
2024 surprise flipped the 30- and 7-day fits to slightly *flatten* it, and they have stayed
near 1 since. Only the 1-day underconfidence holds across the whole sample.

### 7.3 Adding layers to the recalibrated price

Each layer is measured against the recalibrated market. Any calibration gain is already in
that reference, so what remains is information.

- **The price itself.** A logistic regression on the market's price features alone
  essentially reproduces the market: its differences against the raw market price all have
  intervals including zero.
- **Trading activity** (wallet counts, trade counts, volume, net YES flow) makes logistic
  regression significantly worse at every horizon, by 0.020–0.045 Brier against the
  recalibrated price. With 44–120 training markets per fold, these features add variance,
  not information.
- **Macro.** At 1 day, the conditional logit with the market, trading, and macro layers
  matches the recalibrated market (−0.0005, interval −0.0019 to +0.0003). Before I added the
  recalibration baseline, this looked like the model beating the market. It was the
  calibration correction showing up through a richer model.
- **News and text** add nothing, or slightly hurt: +0.0003 and +0.0006 at 1 day.

### 7.4 Fundamentals without the market

Without the market's price, the honest comparison is the bucket base rate, not a uniform
guess. Macro-only models never beat it significantly:

- Logistic regression on macro is +0.014 and +0.015 Brier at 30 and 14 days, and −0.011 and
  −0.019 at 7 and 1 day. Every interval includes zero.
- Adding news and text makes the fundamentals-only models worse, significantly so at 30 and
  7 days for logistic regression.

Most of what macro data knows about the next decision lives in the rates block. The Treasury
bill spreads to EFFR are themselves a market forecast, and by the time anyone can use them,
Polymarket has priced the same information.

### 7.5 Graph networks: the edges break GCN

The full-graph GCN picks "hold" as its favorite at every test meeting, at every horizon, so
its top-1 accuracy is the share of holds in the test period. I reran it on parts of the
graph to locate the cause (see "What breaks GCN?" above):

- **No edges.** GCN reduces to a per-node network and works: Brier 0.017 at 1 day, top-1
  accuracy 78–95%. It is statistically indistinguishable from the flat logistic regression.
- **Cross-meeting edges only** (temporal and same bucket). It collapses to "always hold" again.
  Averaging each market with same-kind markets at other meetings replaces its own price with
  its bucket's typical value, and holds win most often.
- **Same-meeting and trader-overlap edges only.** It does *worse than uniform* (Brier
  0.17–0.18, top-1 28–37%). Averaging a meeting's rival buckets together erases the price
  gap that identifies the favorite.

My first explanation, that heterophilous same-meeting edges cause the collapse, was only
half right. Those edges do hurt GCN, but dropping them does not prevent the collapse; the
cross-meeting edges cause it. GraphConv keeps a separate weight for each node's own
features, so it avoids both failure modes. It lands in the same range as the edgeless GCN and
the flat logistic regression (every interval against the flat model includes zero), and it
does not beat the market.

The graph's own structure explains why it adds little. In the graph built over every market,
the share of edges joining two markets with the same outcome is:

| Edge type | Same-outcome share | Both winners |
|---|---:|---:|
| same meeting | 0.54 | 0.00 |
| same bucket | 0.70 | 0.13 |
| temporal | 0.64 | 0.06 |
| trader overlap | 0.62 | 0.09 |
| *random pair* | *0.63* | *0.06* |

Trader-overlap edges join same-outcome markets at the random-pair rate. Traders who trade
several buckets of the same meeting say nothing about which bucket wins.

### 7.6 Takeaways

1. On Fed decisions, Polymarket is an excellent forecaster at every horizon I tested. The
   honest baselines here are the market's own price and "the Fed usually holds", not a uniform
   guess.
2. The market is calibrated a month out but underconfident the night before. A one-parameter
   recalibration is the only thing that beats it.
3. Trading activity, macro data, news sentiment, Fed text, and trader-network structure add
   no measurable information beyond the price. With 18–19 test meetings, the intervals rule
   out large gains, not small ones.
4. For graph learning on mutually exclusive outcomes, edges can destroy signal. Smoothing
   across meetings reduces GCN to a base rate, and smoothing across rivals erases the
   favorite. A convolution with a separate self weight is needed just to match a model with
   no graph at all.

## 8. What changed from my first version

The first version of this project was a notebook pipeline built around one train/test split.
Rebuilding it meant auditing it, and the audit changed the conclusions more than any model
did:

- **The headline accuracy came from the market itself.** Logistic regression scored 100% on
  a 16-market test set once Polymarket features were added. The cause was `price_at_close`,
  the last daily price before the decision. It separated the outcomes perfectly: the
  lowest-priced winner traded at 0.54 and the highest-priced loser at 0.45. Taking the
  highest-priced market in each meeting picked the winner in 27 of 27 meetings. That isn't
  leakage in the strict sense, because the price was observed the night before. It is,
  though, the market forecasting itself. That is why this version treats the market as the
  baseline to beat, at fixed horizons.
- **Layers were measured at different times.** Macro features were taken at the market's open
  date (a median of 83 days out), news over the 30 days before open, GDELT over the 14 days
  before the decision, and prices the night before. Each stage changed the timing as well as
  the data, so stage-to-stage differences did not isolate a layer.
- **Macro data had look-ahead.** Monthly series were looked up by reference month and taken
  at their current, revised values.
- **Non-price layers could not express signal.** They were constant within a meeting, and no
  feature said which outcome a market paid on. They could only predict how many buckets a
  meeting had. Interacting them with the bucket fixes this.
- **A cross-validation artifact looked like a finding.** Shuffled K-fold put same-meeting
  markets in both train and test. With features identical within a meeting, a held-out loser
  "inherits" its sibling winner from the training fold, which pushed AUC well below 0.5. That
  pattern was being read as evidence that the market is efficient.
- **The graph result was noise.** The GCN was selected on an 8-market validation set, where
  every configuration scored zero, and tested on 16 markets with one seed.
- **Trade data was truncated.** The retired v1 trades API returned only each market's latest
  4,000 trades.

## 9. Limitations

- **Small sample.** 31 meetings and 125 markets; 18–19 out-of-sample meetings per horizon.
  The intervals are wide, and a few meetings carry a lot of weight.
- **Well-telegraphed decisions.** The Fed signals most moves weeks ahead, so the market
  baseline is hard to beat almost by construction. These results say little about surprise
  decisions.
- **First-release macro data** is real-time, but it ignores revisions that had already been
  published by the forecast date. That errs on the side of using less information, never
  more.
- **SF Fed sentiment** is read from the current workbook, whose methodology can revise history
  slightly; the 8-day lag guards against publication timing, not against revisions.
- **GDELT** was throttled throughout collection from my network. The pipeline treats it as
  optional: its columns are empty until the resumable fetch completes, and the evaluation
  drops empty columns per fold.
- **Liquidity and spreads.** Polymarket exposes no historical order-book depth or spreads, so
  liquidity enters only through traded volume.

## 10. Next steps

- A live forecast for the December 9, 2026 meeting, scored after it resolves.
- GDELT through BigQuery, which avoids the API throttle.
- Richer statement text: compare each statement with the previous one sentence by sentence,
  and use a finance-specific language model for tone.
- Kalshi's FOMC markets as a second venue: more meetings, and a cross-venue comparison.
