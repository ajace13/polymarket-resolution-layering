"""Graph neural networks over the market graph: GCN vs. GraphConv.

GCNConv (Kipf and Welling, 2017) passes a node and its neighbors through one shared weight
matrix and takes a degree-normalized average: it smooths features across edges. Same-meeting
edges join mutually exclusive outcomes, so smoothing pulls siblings toward each other, which
is the opposite of what the labels do. GraphConv (Morris et al., 2019) keeps a separate
weight for the node itself and for the weighted mean of its neighbors, so it can learn
contrasts ("my siblings look like YES, so I am NO").

Training is transductive within a fold: the graph holds every training market (labels known)
plus the test meeting's markets (labels hidden); the loss only sees training nodes. Each
model is trained with several seeds and the probabilities are averaged.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv

from polylayers.graph.edges import EDGE_TYPES, GraphSettings, build_edges, collapse_edges
from polylayers.models.baselines import clip_probability
from polylayers.models.design import StageDesign

logger = logging.getLogger(__name__)

HIDDEN_UNITS = 32
DROPOUT = 0.3
EPOCHS = 150
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
SEEDS = (0, 1, 2, 3, 4)


class GraphNet(torch.nn.Module):
    """Two graph convolutions producing one YES logit per node."""

    def __init__(self, convolution: str, input_dim: int) -> None:
        super().__init__()
        if convolution == "gcn":
            # The graph is fixed within a fold, so the degree normalization is computed once.
            self.first = GCNConv(input_dim, HIDDEN_UNITS, cached=True)
            self.second = GCNConv(HIDDEN_UNITS, 1, cached=True)
        elif convolution == "graphconv":
            self.first = GraphConv(input_dim, HIDDEN_UNITS, aggr="mean")
            self.second = GraphConv(HIDDEN_UNITS, 1, aggr="mean")
        else:
            raise ValueError(f"unknown convolution {convolution!r}")

    def forward(self, x, edge_index, edge_weight) -> torch.Tensor:
        hidden = F.relu(self.first(x, edge_index, edge_weight))
        hidden = F.dropout(hidden, p=DROPOUT, training=self.training)
        return self.second(hidden, edge_index, edge_weight).squeeze(-1)


def train_and_predict(
    convolution: str,
    features: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train one network (fixed epochs, no tuning) and return P(YES) for every node."""
    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    index = torch.tensor(edge_index, dtype=torch.long)
    weight = torch.tensor(edge_weight, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    mask = torch.tensor(train_mask)
    model = GraphNet(convolution, x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x, index, weight)[mask], y[mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, index, weight)).numpy()


def graph_nodes(rows: pd.DataFrame) -> pd.DataFrame:
    """Node table for `build_edges`: one node per panel row, numbered 0..n-1."""
    nodes = rows[["market_id", "meeting_id", "decision_date", "bucket_group"]].copy()
    nodes["node"] = np.arange(len(nodes))
    return nodes.reset_index(drop=True)


def fit_predict_gnn(
    convolution: str,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    groups: tuple[str, ...],
    wallets: pd.DataFrame,
    edge_types: tuple[str, ...] = EDGE_TYPES,
    settings: GraphSettings | None = None,
) -> np.ndarray:
    """P(YES) for test rows from a seed-averaged graph network trained on this fold's graph.

    `edge_types` restricts the graph to some relations (an empty tuple leaves only the
    self-loops GCN adds, i.e. a per-node network); used to locate what breaks GCN.
    """
    rows = pd.concat([train_rows, test_rows])
    nodes = graph_nodes(rows)
    cutoff = pd.Timestamp(test_rows["forecast_time"].iloc[0])
    edges = build_edges(nodes, wallets, cutoff, settings)
    edges = edges[edges["edge_type"].isin(edge_types)]
    edge_index, edge_weight = collapse_edges(edges, len(nodes))
    features = StageDesign(groups).fit(train_rows).transform(rows)
    labels = rows["label"].to_numpy(dtype=float)
    train_mask = np.arange(len(rows)) < len(train_rows)
    probabilities = np.mean(
        [
            train_and_predict(
                convolution, features, edge_index, edge_weight, labels, train_mask, seed
            )
            for seed in SEEDS
        ],
        axis=0,
    )
    return clip_probability(probabilities[~train_mask])
