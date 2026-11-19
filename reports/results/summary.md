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