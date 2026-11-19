import numpy as np
import pandas as pd
import pytest

from polylayers.evaluation.bootstrap import paired_meeting_bootstrap
from polylayers.evaluation.metrics import brier_score, log_loss, top1_accuracy
from polylayers.evaluation.walk_forward import walk_forward_folds
from polylayers.graph.edges import GraphSettings, build_edges, collapse_edges, edge_homophily
from polylayers.graph.wallets import wallet_first_trades, wallet_hash
from polylayers.models.conditional_logit import ConditionalLogit
from polylayers.models.design import StageDesign

CUTOFF = pd.Timestamp("2024-03-13", tz="UTC")


def node_table():
    return pd.DataFrame(
        {
            "node": [0, 1, 2, 3],
            "market_id": ["a", "b", "c", "d"],
            "meeting_id": ["m1", "m1", "m2", "m3"],
            "decision_date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-03-20", "2024-09-18"]
            ),
            "bucket_group": ["hold", "cut_small", "hold", "hold"],
        }
    )


def wallet_table():
    before, after = CUTOFF - pd.Timedelta(days=1), CUTOFF + pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "market_id": ["a", "a", "b", "c", "c", "d", "a", "b", "c", "d"],
            "wallet_id": [1, 2, 2, 1, 3, 4, 9, 9, 9, 9],
            "first_trade_at": [
                before,
                before,
                before,
                before,
                after,
                before,
                before,
                before,
                before,
                before,
            ],
        }
    )


# --------------------------------------------------------------------------- graph


def test_edges_cover_each_relation_with_expected_pairs():
    edges = build_edges(node_table(), wallet_table(), CUTOFF, GraphSettings(min_trader_jaccard=0.0))
    by_type = {
        t: set(zip(g["source"], g["target"], strict=True)) for t, g in edges.groupby("edge_type")
    }
    assert by_type["same_meeting"] == {(0, 1)}
    # Meetings m1 and m2 are 49 days apart; m3 is too far from both.
    assert by_type["temporal"] == {(0, 2), (1, 2)}
    assert by_type["same_bucket"] == {(0, 2), (0, 3), (2, 3)}


def test_trader_overlap_ignores_late_wallets_and_bots():
    settings = GraphSettings(min_trader_jaccard=0.0, bot_wallet_share=0.75)
    edges = build_edges(node_table(), wallet_table(), CUTOFF, settings)
    overlap = edges[edges["edge_type"] == "trader_overlap"].set_index(["source", "target"])[
        "weight"
    ]
    # Wallet 9 trades all four markets (> 75% of 4) and is dropped as a bot; wallet 3 traded
    # after the cutoff. a={1,2}, b={2}, c={1}: J(a,b)=1/2, J(a,c)=1/2, no b-c overlap.
    assert overlap.to_dict() == {(0, 1): 0.5, (0, 2): 0.5}


def test_collapse_keeps_max_weight_per_pair_in_both_directions():
    edges = pd.DataFrame(
        {
            "source": [0, 0],
            "target": [1, 1],
            "edge_type": ["temporal", "same_meeting"],
            "weight": [0.4, 1.0],
        }
    )
    edge_index, weights = collapse_edges(edges, n_nodes=2)
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert weights.tolist() == [1.0, 1.0]


def test_same_meeting_edges_are_heterophilous():
    edges = pd.DataFrame(
        {
            "source": [0, 2],
            "target": [1, 3],
            "edge_type": ["same_meeting", "same_bucket"],
            "weight": [1.0, 0.75],
        }
    )
    table = edge_homophily(edges, np.array([1, 0, 1, 1])).set_index("edge_type")
    assert table.loc["same_meeting", "same_label_share"] == 0.0
    assert table.loc["same_bucket", "yes_yes_share"] == 1.0


def test_wallet_hash_is_stable_and_case_insensitive():
    assert wallet_hash("0xABC") == wallet_hash("0xabc")
    trades = pd.DataFrame(
        {
            "market_id": ["a", "a"],
            "wallet": ["0xabc", "0xabc"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-01"], utc=True),
        }
    )
    first = wallet_first_trades(trades)
    assert first["first_trade_at"].tolist() == [pd.Timestamp("2024-01-01", tz="UTC")]
    assert "wallet" not in first.columns


# --------------------------------------------------------------------------- design


def panel_rows():
    return pd.DataFrame(
        {
            "meeting_id": ["m1", "m1", "m2", "m2"],
            "market_id": ["a", "b", "c", "d"],
            "bucket_move_bps": [0.0, -25.0, 0.0, 25.0],
            "bucket_open_ended": [0.0, 0.0, 0.0, 1.0],
            "bucket_is_hold": [1.0, 0.0, 1.0, 0.0],
            "implied_3m_bps": [-10.0, -10.0, 5.0, 5.0],
            "vix": [np.nan, np.nan, np.nan, np.nan],
            "label": [0, 1, 1, 0],
        }
    )


def test_stage_design_interacts_meeting_level_features_and_drops_empty_columns(monkeypatch):
    from polylayers.models import design

    monkeypatch.setitem(
        design.FEATURE_GROUPS, "bucket", ["bucket_move_bps", "bucket_open_ended", "bucket_is_hold"]
    )
    monkeypatch.setitem(design.FEATURE_GROUPS, "macro", ["implied_3m_bps", "vix"])
    stage = StageDesign(("bucket", "macro"))
    matrix = stage.fit_transform(panel_rows())
    assert "implied_3m_bps_x_move" in stage.feature_names_
    assert "implied_3m_bps" not in stage.feature_names_
    assert not any(name.startswith("vix") for name in stage.feature_names_)
    assert matrix.shape == (4, len(stage.feature_names_))
    assert np.allclose(matrix.mean(axis=0), 0.0)


# --------------------------------------------------------------------------- conditional logit


def test_conditional_logit_probabilities_sum_to_one_per_meeting():
    rng = np.random.default_rng(0)
    meetings = np.repeat(np.arange(20), 4)
    features = rng.normal(size=(80, 2))
    labels = np.zeros(80, dtype=int)
    for meeting in range(20):
        rows = np.where(meetings == meeting)[0]
        labels[rows[np.argmax(features[rows, 0])]] = 1  # winner = highest first feature
    model = ConditionalLogit(l2=0.001).fit(features, meetings, labels)
    probabilities = model.predict(features, meetings)
    sums = pd.Series(probabilities).groupby(meetings).sum()
    assert np.allclose(sums, 1.0)
    assert model.weights_[0].item() > model.weights_[1].item()


# --------------------------------------------------------------------------- folds, metrics, bootstrap


def test_walk_forward_trains_only_on_meetings_decided_before_the_forecast():
    decisions = pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12"])
    panel = pd.DataFrame(
        {
            "horizon_days": [30] * 4,
            "meeting_id": ["m1", "m2", "m3", "m4"],
            "decision_date": decisions,
        }
    )
    folds = walk_forward_folds(panel, min_train_meetings=2)
    assert [fold.test_meeting for fold in folds] == ["m3", "m4"]
    # m3 is forecast on 2024-04-01: m1 and m2 were decided before then.
    assert panel.loc[folds[0].train_index, "meeting_id"].tolist() == ["m1", "m2"]


def test_scoring_rules_and_top1():
    labels = np.array([1.0, 0.0])
    assert brier_score(labels, np.array([1.0, 0.0])) == 0.0
    assert log_loss(labels, np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    predictions = pd.DataFrame(
        {
            "meeting_id": ["m1", "m1", "m2", "m2"],
            "label": [1, 0, 0, 1],
            "probability": [0.6, 0.4, 0.7, 0.3],
        }
    )
    assert top1_accuracy(predictions) == 0.5


def test_paired_bootstrap_detects_a_uniformly_better_forecaster():
    keys = pd.DataFrame({"meeting_id": np.repeat(np.arange(30), 2), "market_id": np.arange(60)})
    keys["label"] = np.tile([1, 0], 30)
    good = keys.assign(probability=np.tile([0.9, 0.1], 30))
    bad = keys.assign(probability=np.tile([0.6, 0.4], 30))
    result = paired_meeting_bootstrap(good, bad, "brier", n_resamples=200)
    assert result["difference"] < 0
    assert result["ci_high"] < 0
    assert result["p_candidate_better"] == 1.0
