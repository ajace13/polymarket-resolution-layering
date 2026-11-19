import itertools

import numpy as np
import pandas as pd

from polylayers.evaluation.experiment import default_model_specs
from polylayers.reporting.figures import HORIZON_ORDER, save_all_figures
from polylayers.reporting.tables import summary_markdown


def synthetic_results():
    rng = np.random.default_rng(0)
    specs = default_model_specs()
    score_rows, difference_rows, prediction_rows = [], [], []
    for horizon, spec in itertools.product(HORIZON_ORDER, specs):
        score_rows.append(
            {
                "horizon_days": horizon,
                "model": spec.model,
                "stage": spec.stage,
                "brier": rng.uniform(0.05, 0.2),
                "top1_accuracy": rng.uniform(0.5, 1.0),
            }
        )
        for metric in ("brier", "log_loss"):
            difference = rng.normal(0, 0.01)
            difference_rows.append(
                {
                    "horizon_days": horizon,
                    "model": spec.model,
                    "stage": spec.stage,
                    "metric": metric,
                    "difference": difference,
                    "ci_low": difference - 0.01,
                    "ci_high": difference + 0.01,
                }
            )
    for horizon in HORIZON_ORDER:
        for meeting in range(5):
            for bucket in range(4):
                prediction_rows.append(
                    {
                        "horizon_days": horizon,
                        "model": "market_norm",
                        "stage": "-",
                        "meeting_id": f"m{meeting}",
                        "market_id": f"m{meeting}-{bucket}",
                        "label": int(bucket == 0),
                        "probability": rng.uniform(),
                    }
                )
    return pd.DataFrame(score_rows), pd.DataFrame(difference_rows), pd.DataFrame(prediction_rows)


def test_every_figure_renders_in_both_themes(tmp_path):
    scores, differences, predictions = synthetic_results()
    written = save_all_figures(scores, differences, predictions, tmp_path)
    assert len(written) == 10
    assert all(path.stat().st_size > 10_000 for path in written)


def test_summary_markdown_lists_reference_and_layered_models():
    scores, differences, _ = synthetic_results()
    summary = summary_markdown(scores, differences)
    assert "| Market price (normalized) |" in summary
    assert "Conditional logit, + fed text" in summary
    assert summary.count("\n|") > 10
