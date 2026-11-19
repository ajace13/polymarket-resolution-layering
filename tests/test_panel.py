import numpy as np
import pandas as pd
import pytest

from polylayers.features.panel import FEATURE_GROUPS, ID_COLUMNS, build_panel


def synthetic_inputs():
    decision = pd.Timestamp("2024-03-20")
    markets = pd.DataFrame(
        {
            "market_id": ["a", "b"],
            "meeting_id": ["2024-03-20", "2024-03-20"],
            "decision_date": [decision, decision],
            "question": ["No change in Fed interest rates?", "Fed decreases rates by 25 bps?"],
            "label": [1, 0],
        }
    )
    stamps = pd.date_range("2024-02-01", "2024-03-19", freq="12h", tz="UTC")
    prices = pd.concat(
        [
            pd.DataFrame({"market_id": market_id, "timestamp": stamps, "price": price})
            for market_id, price in (("a", 0.8), ("b", 0.2))
        ],
        ignore_index=True,
    )
    prices["available_at"] = prices["timestamp"] + pd.Timedelta(hours=3)
    trades = pd.DataFrame(
        {
            "market_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2024-03-01", "2024-03-02"], utc=True),
            "wallet": ["w1", "w1"],
            "side": ["BUY", "BUY"],
            "size": [10.0, 5.0],
            "price": [0.8, 0.2],
            "outcome_index": [0, 0],
        }
    )
    coverage = pd.DataFrame({"market_id": ["a", "b"], "history_complete": [True, True]})
    days = pd.date_range("2023-01-01", "2024-03-19", freq="D")
    daily_series = []
    for series_id, value in (
        ("EFFR", 5.33),
        ("DGS3MO", 5.30),
        ("DGS6MO", 5.20),
        ("DGS2", 4.60),
        ("T10Y2Y", -0.3),
        ("VIXCLS", 14.0),
    ):
        daily_series.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "date": days,
                    "value": value,
                    "available_date": days + pd.Timedelta(days=1),
                }
            )
        )
    months = pd.date_range("2022-12-01", "2024-01-01", freq="MS")
    for series_id, start in (
        ("CPILFESL", 300.0),
        ("PCEPILFE", 120.0),
        ("UNRATE", 3.7),
        ("ICSA", 210_000.0),
        ("NFCI", -0.4),
        ("INDPRO", 102.0),
    ):
        daily_series.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "date": months,
                    "value": start * (1 + 0.002 * np.arange(len(months))),
                    "available_date": months + pd.Timedelta(days=40),
                }
            )
        )
    fred = pd.concat(daily_series, ignore_index=True)
    sentiment = pd.DataFrame(
        {"date": days, "sentiment": 0.1, "available_date": days + pd.Timedelta(days=8)}
    )
    statements = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-01-31"]),
            "statement_text": [
                "The Committee decided to maintain the target range. Inflation remains elevated."
            ],
            "dissent_text": [""],
            "available_date": pd.to_datetime(["2024-02-01"]),
        }
    )
    return markets, prices, trades, coverage, fred, sentiment, statements


def test_build_panel_end_to_end():
    markets, prices, trades, coverage, fred, sentiment, statements = synthetic_inputs()
    panel = build_panel(
        markets,
        prices,
        trades,
        coverage,
        fred,
        sentiment,
        statements,
        gdelt_daily=pd.DataFrame(),
        horizons_days=(30, 7),
    )
    feature_columns = [column for group in FEATURE_GROUPS.values() for column in group]
    assert list(panel.columns) == ID_COLUMNS + feature_columns
    assert len(panel) == 4  # 2 markets x 2 horizons, all priced by then
    assert not panel.duplicated(["market_id", "horizon_days"]).any()
    seven_days = panel[panel["horizon_days"] == 7].set_index("market_id")
    assert seven_days.loc["a", "mkt_price_norm"] == pytest.approx(0.8)
    assert seven_days.loc["a", "implied_3m_bps"] == pytest.approx(-3.0)
    assert seven_days.loc["a", "stmt_last_action"] == 0.0
    assert panel["gd_fed_share_14d"].isna().all()
