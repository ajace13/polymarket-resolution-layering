"""Edges between markets, built only from information available at a cutoff time.

Four relations:
- same_meeting:   markets on the same FOMC meeting (mutually exclusive outcomes; weight 1)
- temporal:       markets on meetings within `temporal_max_days`; weight 1 / (1 + days / scale)
- same_bucket:    markets betting on the same kind of move at different meetings
- trader_overlap: Jaccard similarity of the wallets that traded each market before the
                  cutoff, after removing wallets active in a large share of markets (market
                  makers and bots, whose presence says nothing about a market's outcome)

Trader overlap uses a sparse market-by-wallet incidence matrix B: B @ B.T counts shared
wallets for every pair at once instead of looping over pairs.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

EDGE_TYPES = ("same_meeting", "temporal", "same_bucket", "trader_overlap")
EDGE_COLUMNS = ["source", "target", "edge_type", "weight"]


@dataclass(frozen=True)
class GraphSettings:
    """Edge construction parameters."""

    temporal_max_days: int = 90
    temporal_scale_days: float = 30.0
    same_bucket_weight: float = 0.75
    bot_wallet_share: float = 0.40
    min_trader_jaccard: float = 0.01


def _pairs(frame: pd.DataFrame, condition) -> pd.DataFrame:
    """All unordered node pairs (source < target) satisfying `condition(left, right)`."""
    left = frame.add_suffix("_a")
    right = frame.add_suffix("_b")
    pairs = left.merge(right, how="cross")
    pairs = pairs[pairs["node_a"] < pairs["node_b"]]
    return pairs[condition(pairs)]


def same_meeting_edges(nodes: pd.DataFrame) -> pd.DataFrame:
    pairs = _pairs(nodes, lambda p: p["meeting_id_a"] == p["meeting_id_b"])
    return pd.DataFrame(
        {
            "source": pairs["node_a"],
            "target": pairs["node_b"],
            "edge_type": "same_meeting",
            "weight": 1.0,
        }
    )


def temporal_edges(nodes: pd.DataFrame, settings: GraphSettings) -> pd.DataFrame:
    def close_but_different(p: pd.DataFrame) -> pd.Series:
        days = (p["decision_date_a"] - p["decision_date_b"]).abs().dt.days
        return (p["meeting_id_a"] != p["meeting_id_b"]) & (days <= settings.temporal_max_days)

    pairs = _pairs(nodes, close_but_different)
    days = (pairs["decision_date_a"] - pairs["decision_date_b"]).abs().dt.days
    return pd.DataFrame(
        {
            "source": pairs["node_a"],
            "target": pairs["node_b"],
            "edge_type": "temporal",
            "weight": 1.0 / (1.0 + days / settings.temporal_scale_days),
        }
    )


def same_bucket_edges(nodes: pd.DataFrame, settings: GraphSettings) -> pd.DataFrame:
    def same_kind_other_meeting(p: pd.DataFrame) -> pd.Series:
        return (
            (p["bucket_group_a"] == p["bucket_group_b"])
            & (p["bucket_group_a"] != "other")
            & (p["meeting_id_a"] != p["meeting_id_b"])
        )

    pairs = _pairs(nodes, same_kind_other_meeting)
    return pd.DataFrame(
        {
            "source": pairs["node_a"],
            "target": pairs["node_b"],
            "edge_type": "same_bucket",
            "weight": settings.same_bucket_weight,
        }
    )


def trader_overlap_edges(
    nodes: pd.DataFrame, wallets: pd.DataFrame, cutoff: pd.Timestamp, settings: GraphSettings
) -> pd.DataFrame:
    """Jaccard similarity of pre-cutoff wallet sets, computed with one sparse product."""
    active = wallets[
        (wallets["first_trade_at"] < cutoff) & wallets["market_id"].isin(nodes["market_id"])
    ]
    if active.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    node_of_market = dict(zip(nodes["market_id"], nodes["node"], strict=True))
    rows = active["market_id"].map(node_of_market).to_numpy()
    wallet_codes, _ = pd.factorize(active["wallet_id"])
    incidence = sparse.csr_matrix(
        (np.ones(len(active)), (rows, wallet_codes)), shape=(len(nodes), wallet_codes.max() + 1)
    )
    incidence.data[:] = 1.0
    markets_per_wallet = np.asarray(incidence.sum(axis=0)).ravel()
    keep = markets_per_wallet <= settings.bot_wallet_share * len(nodes)
    dropped = int((~keep).sum())
    if dropped:
        logger.debug(
            "Trader overlap: dropped %d wallets active in >%.0f%% of markets",
            dropped,
            settings.bot_wallet_share * 100,
        )
    incidence = incidence[:, keep]
    sizes = np.asarray(incidence.sum(axis=1)).ravel()
    shared = sparse.triu(incidence @ incidence.T, k=1).tocoo()
    union = sizes[shared.row] + sizes[shared.col] - shared.data
    jaccard = np.divide(shared.data, union, out=np.zeros_like(shared.data), where=union > 0)
    strong = jaccard >= settings.min_trader_jaccard
    return pd.DataFrame(
        {
            "source": shared.row[strong],
            "target": shared.col[strong],
            "edge_type": "trader_overlap",
            "weight": jaccard[strong],
        }
    )


def build_edges(
    nodes: pd.DataFrame,
    wallets: pd.DataFrame,
    cutoff: pd.Timestamp,
    settings: GraphSettings | None = None,
) -> pd.DataFrame:
    """All typed edges for `nodes` (columns node, market_id, meeting_id, decision_date,
    bucket_group), using only wallets that traded before `cutoff`."""
    settings = settings or GraphSettings()
    edges = pd.concat(
        [
            same_meeting_edges(nodes),
            temporal_edges(nodes, settings),
            same_bucket_edges(nodes, settings),
            trader_overlap_edges(nodes, wallets, cutoff, settings),
        ],
        ignore_index=True,
    )
    edges["source"] = edges["source"].astype(int)
    edges["target"] = edges["target"].astype(int)
    return edges[EDGE_COLUMNS]


def collapse_edges(edges: pd.DataFrame, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """One undirected edge per node pair (max weight across types), stored in both directions.

    Collapsing stops a pair linked by three relations from getting three times the message
    weight of a pair linked by one.
    """
    if edges.empty:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    collapsed = edges.groupby(["source", "target"], as_index=False)["weight"].max()
    source = collapsed["source"].to_numpy()
    target = collapsed["target"].to_numpy()
    if max(source.max(), target.max()) >= n_nodes:
        raise ValueError("edge refers to a node outside the graph")
    edge_index = np.vstack([np.concatenate([source, target]), np.concatenate([target, source])])
    weights = np.concatenate([collapsed["weight"], collapsed["weight"]]).astype(np.float32)
    return edge_index.astype(np.int64), weights


def edge_homophily(edges: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per edge type: edge count and the share of edges joining two YES or two NO markets.

    Same-meeting edges join mutually exclusive outcomes, so they are heterophilous by
    construction; a smoothing GNN layer (GCN) averages neighbors and cannot express 'my
    neighbor looks like YES, so I am NO'.
    """
    rows = []
    for edge_type, group in edges.groupby("edge_type"):
        same_label = labels[group["source"]] == labels[group["target"]]
        both_yes = (labels[group["source"]] == 1) & (labels[group["target"]] == 1)
        rows.append(
            {
                "edge_type": edge_type,
                "edges": len(group),
                "same_label_share": float(same_label.mean()),
                "yes_yes_share": float(both_yes.mean()),
            }
        )
    return pd.DataFrame(rows)
