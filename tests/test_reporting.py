import itertools

import numpy as np
import pandas as pd

from polylayers.evaluation.experiment import REFERENCE_MODELS, default_model_specs
from polylayers.reporting.figures import HORIZON_ORDER, save_all_figures
from polylayers.reporting.tables import summary_markdown


def synthetic_results():
    rng = np.random.default_rng(0)
    specs = default_model_specs()
    buckets = ["hold", "cut_25", "cut_50+", "hike_25"]
    score_rows, difference_rows, prediction_rows, temperature_rows = [], [], [], []
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
        for reference, metric in itertools.product(REFERENCE_MODELS, ("brier", "log_loss")):
            difference = rng.normal(0, 0.01)
            difference_rows.append(
                {
                    "reference": reference,
                    "horizon_days": horizon,
                    "model": spec.model,
                    "stage": spec.stage,
                    "metric": metric,
                    "difference": difference,
                    "ci_low": difference - 0.01,
                    "ci_high": difference + 0.01,
                }
            )
        for meeting in range(6):
            for position, bucket in enumerate(buckets):
                prediction_rows.append(
                    {
                        "horizon_days": horizon,
                        "model": spec.model,
                        "stage": spec.stage,
                        "meeting_id": f"2025-0{meeting + 1}-15",
                        "market_id": f"m{meeting}-{position}",
                        "bucket": bucket,
                        "label": int(position == meeting % 2),
                        "probability": rng.uniform(),
                    }
                )
    for horizon, meeting in itertools.product(HORIZON_ORDER, range(6)):
        temperature_rows.append(
            {
                "horizon_days": horizon,
                "test_meeting": f"2025-0{meeting + 1}-15",
                "beta": rng.uniform(0.8, 8),
            }
        )
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(difference_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(temperature_rows),
    )


def test_every_figure_renders_in_both_themes(tmp_path):
    scores, differences, _, temperatures = synthetic_results()
    written = save_all_figures(scores, differences, temperatures, tmp_path)
    assert len(written) == 10
    assert all(path.stat().st_size > 10_000 for path in written)


def test_summary_markdown_has_every_section():
    scores, differences, predictions, temperatures = synthetic_results()
    summary = summary_markdown(scores, differences, predictions, temperatures)
    for expected in (
        "| Market price, recalibrated |",
        "Median fitted temperature (beta)",
        "Conditional logit · + Fed text",
        "| Logistic regression · Macro |",
        "| GCN, no edges |",
    ):
        assert expected in summary
