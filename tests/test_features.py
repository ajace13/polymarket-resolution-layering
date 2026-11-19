import numpy as np
import pandas as pd
import pytest

from polylayers.features import as_of, market, news, text
from polylayers.features.buckets import parse_bucket
from polylayers.features.macro import annualized_growth, macro_features

UTC = "UTC"


# --------------------------------------------------------------------------- as-of


def test_asof_lookup_waits_for_availability():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "value": [1.0, 2.0],
            "available_date": pd.to_datetime(["2024-01-10", "2024-02-10"]),
        }
    )
    dates = pd.Series(pd.to_datetime(["2024-01-09", "2024-01-10", "2024-02-09", "2024-03-01"]))
    values = as_of.asof_lookup(observations, dates)
    assert np.isnan(values[0])
    assert values[1:].tolist() == [1.0, 1.0, 2.0]


# --------------------------------------------------------------------------- buckets


@pytest.mark.parametrize(
    ("question", "name", "move", "open_ended"),
    [
        (
            "Will the Fed increase interest rates by 0 bps after the February 2023 meeting?",
            "hold",
            0,
            False,
        ),
        (
            "Will the Fed increase interest rates by 25 bps after the March 2023 meeting?",
            "hike_25",
            25,
            False,
        ),
        (
            "Will there be no change in Fed interest rates after the June 2024 meeting?",
            "hold",
            0,
            False,
        ),
        (
            "Fed decreases interest rates by 50+ bps after September 2025 meeting?",
            "cut_50+",
            -50,
            True,
        ),
        ("Fed increases interest rates by 25+ bps after July 2025 meeting?", "hike_25+", 25, True),
        ("No change in Fed interest rates after October 2026 meeting?", "hold", 0, False),
        (
            "Will the FED change rates to another level after the June 2024 meeting?",
            "other",
            np.nan,
            False,
        ),
    ],
)
def test_parse_bucket_handles_polymarket_phrasings(question, name, move, open_ended):
    bucket = parse_bucket(question)
    assert bucket.name == name
    assert bucket.open_ended == open_ended
    assert (np.isnan(bucket.move_bps) and np.isnan(move)) or bucket.move_bps == move


def test_bucket_groups_collapse_magnitudes():
    assert parse_bucket("Fed decreases interest rates by 50+ bps after X?").group == "cut_large"
    assert parse_bucket("Fed decreases interest rates by 25 bps after X?").group == "cut_small"


# --------------------------------------------------------------------------- market layer


def market_inputs():
    markets = pd.DataFrame(
        {
            "market_id": ["a", "b"],
            "meeting_id": ["2024-03-20", "2024-03-20"],
            "decision_date": pd.to_datetime(["2024-03-20", "2024-03-20"]),
            "question": ["No change in Fed interest rates?", "Fed decreases rates by 25 bps?"],
            "label": [1, 0],
        }
    )
    t7 = pd.Timestamp("2024-03-13", tz=UTC)
    prices = pd.DataFrame(
        {
            "market_id": ["a", "a", "b", "b"],
            "timestamp": [t7 - pd.Timedelta(hours=15), t7 - pd.Timedelta(hours=1)] * 2,
            "price": [0.70, 0.99, 0.20, 0.01],
        }
    )
    # Second points' 3-hour windows close after the forecast time, so they must be ignored.
    prices["available_at"] = prices["timestamp"] + pd.Timedelta(hours=3)
    return markets, prices


def test_price_features_use_only_closed_windows_and_normalize_within_meeting():
    markets, prices = market_inputs()
    skeleton = market.forecast_skeleton(markets, (7,))
    panel = market.add_price_features(skeleton, prices)
    assert panel["mkt_price"].tolist() == [0.70, 0.20]
    assert panel["mkt_price_norm"].sum() == pytest.approx(1.0)
    assert panel["mkt_is_favorite"].tolist() == [1.0, 0.0]
    assert panel["mkt_overround"].tolist() == [pytest.approx(0.9)] * 2


def test_rows_without_a_usable_price_are_dropped():
    markets, prices = market_inputs()
    skeleton = market.forecast_skeleton(markets, (30, 7))
    panel = market.add_price_features(skeleton, prices)
    assert set(panel["horizon_days"]) == {7}


def test_trading_features_count_only_trades_before_the_forecast():
    markets, prices = market_inputs()
    panel = market.add_price_features(market.forecast_skeleton(markets, (7,)), prices)
    t7 = pd.Timestamp("2024-03-13", tz=UTC)
    trades = pd.DataFrame(
        {
            "market_id": ["a", "a", "a", "b"],
            "timestamp": [t7 - pd.Timedelta(days=10), t7 - pd.Timedelta(days=1), t7, t7],
            "wallet": ["w1", "w2", "w3", "w1"],
            "side": ["BUY", "SELL", "BUY", "BUY"],
            "size": [10.0, 10.0, 10.0, 10.0],
            "price": [0.5, 0.5, 0.5, 0.5],
            "outcome_index": [0, 0, 0, 0],
        }
    )
    coverage = pd.DataFrame({"market_id": ["a", "b"], "history_complete": [True, False]})
    result = market.add_trading_features(panel, trades, coverage).set_index("market_id")
    assert result.loc["a", "trd_wallets_total"] == 2
    assert result.loc["a", "trd_trades_7d"] == 1
    assert result.loc["a", "trd_net_yes_flow_7d"] == pytest.approx(-5.0)
    assert np.isnan(result.loc["b", "trd_wallets_total"])


def test_yes_flow_sign_treats_selling_no_as_buying_yes():
    side = pd.Series(["BUY", "SELL", "BUY", "SELL", "BUY"])
    outcome = pd.Series([0, 0, 1, 1, 999])
    assert market.yes_flow_sign(side, outcome).tolist() == [1.0, -1.0, -1.0, 1.0, 0.0]


# --------------------------------------------------------------------------- macro layer


def test_annualized_growth_falls_back_over_a_missing_month():
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-01", "2025-08-01", "2025-09-01", "2025-11-01"]),
            "value": [100.0, 100.0, 100.0, 101.0],
        }
    )
    # Three months before November is August; with October missing that still works.
    assert annualized_growth(rows, 3) == pytest.approx(((101 / 100) ** 4 - 1) * 100)


def test_macro_features_compute_spreads_without_look_ahead():
    def daily(series_id, values):
        dates = pd.to_datetime(["2024-02-01", "2024-03-12"])
        return pd.DataFrame(
            {"series_id": series_id, "date": dates, "value": values, "available_date": dates}
        )

    monthly = pd.DataFrame(
        {
            "series_id": ["CPILFESL", "PCEPILFE", "UNRATE", "ICSA", "NFCI", "INDPRO"],
            "date": pd.to_datetime(["2024-02-01"] * 6),
            "value": [310.0, 120.0, 3.9, 210_000.0, -0.5, 102.0],
            "available_date": pd.to_datetime(["2024-03-13"] * 6),
        }
    )
    observations = pd.concat(
        [
            daily("EFFR", [5.33, 5.33]),
            daily("DGS3MO", [5.40, 5.20]),
            daily("DGS6MO", [5.30, 5.10]),
            daily("DGS2", [4.60, 4.50]),
            daily("T10Y2Y", [-0.40, -0.40]),
            daily("VIXCLS", [14.0, 15.0]),
            monthly,
        ]
    )
    features = macro_features(
        observations,
        pd.Series(pd.to_datetime(["2024-03-12"])),
        pd.Series(pd.to_datetime(["2024-01-31"])),
    )
    row = features.iloc[0]
    assert row["implied_3m_bps"] == pytest.approx(-13.0)
    assert row["dgs2_chg_30d_bps"] == pytest.approx(-10.0)
    assert np.isnan(row["unrate"])  # released 2024-03-13, after the forecast date


# --------------------------------------------------------------------------- news layer


def test_sentiment_windows_use_only_available_days():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    sentiment = pd.DataFrame(
        {
            "date": dates,
            "sentiment": np.arange(40.0),
            "available_date": dates + pd.Timedelta(days=8),
        }
    )
    features = news.sentiment_features(sentiment, pd.Series(pd.to_datetime(["2024-02-09"])))
    # Usable days end 2024-02-01 (index 31); the last 7 are indices 25..31.
    assert features.loc[0, "sent_mean_7d"] == pytest.approx(28.0)


def test_gdelt_features_are_nan_when_gdelt_is_missing():
    empty = pd.DataFrame(columns=["date", "query", "mode", "value", "norm"])
    features = news.gdelt_features(empty, pd.Series(pd.to_datetime(["2024-03-12"])))
    assert features[news.GDELT_COLUMNS].isna().all(axis=None)


# --------------------------------------------------------------------------- text layer


def test_hawkish_score_credits_direction_words_to_nearby_topics():
    assert text.hawkish_score("Inflation remains elevated.") == 1.0
    assert text.hawkish_score("The unemployment rate has risen.") == -1.0
    assert text.hawkish_score("Job gains have slowed.") == -1.0
    assert text.hawkish_score("The Committee met today.") == 0.0


def test_policy_move_and_guidance():
    cut = "The Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent."
    assert text.policy_move_bps(cut) == -50.0
    assert text.policy_move_bps("The Committee decided to maintain the target range.") == 0.0
    assert text.guidance_direction("some additional policy firming may be appropriate") == 1.0
    assert (
        text.guidance_direction(
            "the extent and timing of additional adjustments to the target range"
        )
        == -1.0
    )


def test_dissent_summary_on_real_statement_wording():
    two_way = (
        "Voting against this action were Stephen I. Miran, who preferred to lower the target range "
        "for the federal funds rate by 1/2 percentage point at this meeting, and Jeffrey R. Schmid, "
        "who preferred no change to the target range at this meeting."
    )
    assert text.dissent_summary(two_way, -25.0) == (2.0, 0.0)
    hawks = (
        "Voting against the monetary policy action were Beth M. Hammack, Neel Kashkari, and Lorie K. "
        "Logan, who preferred to raise the target range for the federal funds rate by 1/4 percentage point."
    )
    assert text.dissent_summary(hawks, 0.0) == (3.0, 1.0)
    smaller_cut = (
        "Voting against the action was Michelle W. Bowman, who preferred to lower the target range "
        "for the federal funds rate by 1/4 percentage point at this meeting."
    )
    assert text.dissent_summary(smaller_cut, -50.0) == (1.0, 1.0)
    assert text.dissent_summary("", 0.0) == (0.0, 0.0)


def test_statement_features_exclude_the_statement_released_on_decision_day():
    statements = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-01-31", "2024-03-20"]),
            "statement_text": [
                "The Committee decided to maintain the target range. Inflation remains elevated.",
                "The Committee decided to lower the target range by 1/4 percentage point.",
            ],
            "dissent_text": ["", ""],
            "available_date": pd.to_datetime(["2024-02-01", "2024-03-21"]),
        }
    )
    dates = pd.Series(pd.to_datetime(["2024-03-19", "2024-03-21"]))
    features = text.statement_features(statements, dates)
    assert features["stmt_last_action"].tolist() == [0.0, -1.0]
