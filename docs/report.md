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

- **Uniform:** 1 / (number of buckets listed), the no-information reference.
- **Market:** the raw YES price, and the price normalized within the meeting. The normalized
  price is **the baseline every other model is measured against**.
- **Logistic regression,** per market, L2-regularized. The strength is chosen on the four
  most recent training meetings, by log loss. There is no class weighting: class weights
  would deliberately distort the probabilities this evaluation scores.
- **Conditional logit:** a softmax over each meeting's buckets, so probabilities sum to one
  within a meeting, with an L2 penalty chosen the same way.
- **GCN and GraphConv:** two layers of 32 hidden units, trained transductively for 150
  epochs (no tuning) and averaged over five seeds. GCN averages a node with its neighbors
  through one shared weight matrix. GraphConv keeps separate weights for the node and its
  neighbors, so it can learn contrasts across the heterophilous same-meeting edges.

Each model is trained on cumulative layer stages: market only, then adding macro, news,
and text. The fundamentals-only stages drop the market entirely.

## 6. Evaluation

Walk-forward by meeting. After 12 burn-in meetings, each later meeting is forecast once per
horizon, using only meetings decided before the forecast date. That gives 19 test meetings
(July 2024 to October 2026) at 1, 7, and 14 days. At 30 days there are 18, because the May
2024 markets were listed less than 30 days ahead.

Metrics:

- Brier score and log loss, both proper scoring rules.
- AUC.
- Top-1 accuracy: whether a model's favorite bucket is the one that won.

Differences against the market come with 95% intervals from a paired bootstrap that
resamples meetings (2,000 resamples).

## 7. Results

GDELT was still throttled when I ran this evaluation, so the news layer here is the SF Fed
sentiment index alone. The GDELT columns were empty and were dropped in every fold.

**How good is the market?** Brier score (lower is better) and the share of meetings where the forecaster's favorite bucket won.

| Forecaster | 30d Brier | 30d top-1 | 14d Brier | 14d top-1 | 7d Brier | 7d top-1 | 1d Brier | 1d top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform over buckets | 0.172 | 6% | 0.173 | 5% | 0.173 | 5% | 0.173 | 5% |
| Market price | 0.054 | 83% | 0.024 | 95% | 0.022 | 95% | 0.008 | 100% |
| Market price (normalized) | 0.054 | 83% | 0.024 | 95% | 0.022 | 95% | 0.008 | 100% |
| Logistic regression · Macro + news + text | 0.155 | 61% | 0.156 | 63% | 0.164 | 63% | 0.134 | 63% |
| Conditional logit · Macro + news + text | 0.197 | 44% | 0.161 | 53% | 0.146 | 58% | 0.138 | 63% |

**Does anything improve on it?** Brier difference vs. the normalized market price, with a 95% interval from a meeting-level paired bootstrap (negative = better than the market).

| Forecaster | 30d | 14d | 7d | 1d |
|---|---:|---:|---:|---:|
| Logistic regression · Market only | +0.0413 [+0.0052, +0.0897] | +0.0466 [+0.0113, +0.0929] | +0.0377 [+0.0055, +0.0860] | +0.0143 [-0.0014, +0.0328] |
| Logistic regression · + macro | +0.0458 [+0.0084, +0.0959] | +0.0383 [+0.0069, +0.0844] | +0.0458 [+0.0112, +0.0945] | +0.0137 [-0.0016, +0.0304] |
| Logistic regression · + news | +0.0458 [+0.0082, +0.0955] | +0.0413 [+0.0092, +0.0878] | +0.0423 [+0.0092, +0.0908] | +0.0149 [-0.0001, +0.0319] |
| Logistic regression · + Fed text | +0.0429 [+0.0057, +0.0902] | +0.0424 [+0.0091, +0.0887] | +0.0431 [+0.0097, +0.0918] | +0.0162 [+0.0008, +0.0334] |
| Conditional logit · Market only | +0.0141 [-0.0115, +0.0469] | +0.0742 [+0.0169, +0.1364] | +0.0464 [+0.0065, +0.0976] | +0.0149 [-0.0051, +0.0527] |
| Conditional logit · + macro | +0.0287 [-0.0059, +0.0695] | +0.0143 [-0.0093, +0.0446] | +0.0324 [-0.0038, +0.0860] | -0.0065 [-0.0155, -0.0008] |
| Conditional logit · + news | +0.0295 [-0.0054, +0.0695] | +0.0153 [-0.0091, +0.0479] | +0.0282 [-0.0085, +0.0821] | -0.0058 [-0.0132, -0.0008] |
| Conditional logit · + Fed text | +0.0270 [-0.0042, +0.0647] | +0.0236 [-0.0013, +0.0559] | +0.0298 [-0.0076, +0.0844] | -0.0055 [-0.0124, -0.0008] |
| GCN · Market only | +0.0861 [+0.0314, +0.1445] | +0.1073 [+0.0646, +0.1553] | +0.1097 [+0.0673, +0.1583] | +0.1251 [+0.0766, +0.1742] |
| GCN · + Fed text | +0.0967 [+0.0382, +0.1643] | +0.1133 [+0.0675, +0.1648] | +0.1173 [+0.0705, +0.1721] | +0.1359 [+0.0780, +0.1999] |
| GraphConv · Market only | +0.0190 [-0.0144, +0.0571] | +0.0425 [+0.0034, +0.0944] | +0.0314 [+0.0033, +0.0669] | +0.0096 [-0.0015, +0.0255] |
| GraphConv · + Fed text | +0.0378 [-0.0026, +0.0912] | +0.0390 [-0.0001, +0.0905] | +0.0385 [-0.0011, +0.0920] | +0.0179 [+0.0015, +0.0376] |

### 7.1 The market is very hard to beat

Out of sample, the normalized market price picks the winning bucket at 83% of meetings 30
days out, 95% at 14 and 7 days, and 100% the day before. Its Brier score falls from 0.054 to
0.008. Normalizing barely matters: the buckets' prices already sum to almost exactly one.

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

### 7.2 Adding layers to the market's price

At 30, 14, and 7 days, no model that starts from the market's price and adds macro data,
news, or statement text improves on the price itself:

- Logistic regression is significantly worse at every stage. With 44–120 training markets
  and 15–75 features, depending on the fold and the stage, it pays for its own estimation
  noise.
- The conditional logit with macro features comes closest: +0.014 to +0.032 Brier, with
  intervals that include zero.

One day out, the conditional logit with macro features beats the market: Brier −0.0065 (95%
interval −0.0155 to −0.0008), and log loss 0.013 against the market's 0.042. Adding news and
text on top neither helps nor hurts. The mechanism is sharpening, not new information:

| 1 day before | Market | Conditional logit, market + macro |
|---|---:|---:|
| Median probability on the winning bucket | 0.956 | 0.995 |
| Lowest probability on a winning bucket | 0.526 | 0.767 |
| Mean probability on a losing bucket | 0.025 | 0.0095 |

The night before a decision, the market still leaves about 2.5% on each losing bucket. Some
of that is probably the cost of tying up capital, plus the familiar longshot premium in
prediction markets. The macro interactions matter because they keep the sharpening from
making confident mistakes. The same model without them (market features only) assigns one
eventual winner a probability of 0.02.

### 7.3 Fundamentals without the market

Using only the bucket encoding and the macro layer, logistic regression picks the winning
bucket at 61–74% of meetings, against 5% for a uniform guess. Its Brier score (0.11–0.15)
beats uniform (0.17). Almost all of that comes from the rates block: the Treasury bill
spreads to EFFR are themselves a market forecast of the decision. Adding SF Fed sentiment and
statement text makes the fundamentals-only models *worse* at most horizons. With 11–30
training meetings, extra meeting-level features mostly add variance.

### 7.4 Graph networks

| Brier difference (95% interval) | 30 days | 14 days | 7 days | 1 day |
|---|---:|---:|---:|---:|
| GraphConv − GCN, market features | −0.067 [−0.133, +0.006] | −0.065 [−0.123, +0.001] | −0.078 [−0.126, −0.038] | −0.116 [−0.167, −0.064] |
| GraphConv − logistic, market features | −0.022 [−0.081, +0.021] | −0.004 [−0.032, +0.021] | −0.006 [−0.042, +0.028] | −0.005 [−0.024, +0.018] |

GCN collapses. Its favorite bucket is "hold" at every test meeting, so its top-1 accuracy
equals the share of holds in the test period (63%) at every horizon. This is the
over-smoothing failure predicted in section 5: same-meeting edges join rivals, and averaging
a node with its rivals erases the price differences that identify the favorite. GraphConv,
which keeps a separate weight for the node itself, avoids the collapse. It is significantly
better than GCN at 7 and 1 days, and statistically indistinguishable from the flat logistic
regression.

The graph's own structure explains why it adds little. In the graph built over every market,
the share of edges joining two markets with the same outcome is:

| Edge type | Same-outcome share | Both winners |
|---|---:|---:|
| same meeting | 0.54 | 0.00 |
| same bucket | 0.70 | 0.13 |
| temporal | 0.64 | 0.06 |
| trader overlap | 0.62 | 0.09 |
| *random pair* | *0.63* | *0.06* |

Same-meeting edges are heterophilous by construction (a meeting never has two winners). Only
same-bucket edges are mildly homophilous, because holds cluster in time. Trader-overlap edges
join same-outcome markets at the random-pair rate: traders who trade several buckets of the
same meeting say nothing about which bucket wins.

### 7.5 Takeaways

1. On Fed decisions, Polymarket is an excellent forecaster at every horizon I tested. The
   honest baseline for any model here is the market's price, not a majority-class guess.
2. Public macro data contains real signal about the decision, but bond markets have priced it
   by the time Polymarket does. It helps only as a guardrail when sharpening the market's
   near-certain calls on the eve of the meeting.
3. News sentiment, statement text, and trader-network structure added nothing measurable at
   this sample size. With 18–19 test meetings, small real effects would be invisible. The
   intervals rule out large gains, not small ones.
4. For graph learning on mutually exclusive outcomes, the choice of convolution matters more
   than the graph: a smoothing operator (GCN) fails, while one with a separate self weight
   (GraphConv) matches a flat model.

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
