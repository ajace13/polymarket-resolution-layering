"""Per-market wallet participation, with addresses replaced by anonymous integer codes."""

import hashlib
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def wallet_hash(address: str) -> int:
    """Stable signed 64-bit id for a wallet address (case-insensitive)."""
    digest = hashlib.sha256(address.lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def wallet_first_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per (market, wallet): when that wallet first traded that market.

    Only the first trade matters for graphs built at a cutoff: a wallet belongs to a
    market's set at time t if it had traded the market before t. Addresses become dense
    int32 codes ordered by their SHA-256 hash, so the table keeps who-overlaps-with-whom
    without carrying addresses (and compresses far better than raw hashes).
    """
    first = trades.groupby(["market_id", "wallet"], as_index=False)["timestamp"].min()
    addresses = first["wallet"].unique()
    hashes = np.array([wallet_hash(address) for address in addresses])
    codes = dict(zip(addresses[np.argsort(hashes)], range(len(addresses)), strict=True))
    first["wallet_id"] = first["wallet"].map(codes).astype("int32")
    first["first_trade_at"] = pd.to_datetime(first["timestamp"], utc=True).astype(
        "datetime64[s, UTC]"
    )
    first = first.sort_values(["market_id", "first_trade_at"]).reset_index(drop=True)
    logger.info(
        "Wallet table: %d market-wallet pairs, %d distinct wallets",
        len(first),
        first["wallet_id"].nunique(),
    )
    return first[["market_id", "wallet_id", "first_trade_at"]]
