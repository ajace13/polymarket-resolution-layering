"""Macro-layer features from FRED, evaluated point-in-time at each forecast date.

The rates block captures what bond markets already price: 3-month, 6-month, and 2-year
Treasury yields relative to the effective fed funds rate (EFFR) are a public proxy for the
expected path of policy. Release-based series (inflation, labor, activity, financial
conditions) use first-release values only, available from the day after publication.
"""

import numpy as np
import pandas as pd

from polylayers.features.as_of import asof_lookup, available_before, naive_ns

LOOKBACK_30D = pd.Timedelta(days=30)

MACRO_COLUMNS = [
    "effr",
    "last_move_bps",
    "implied_3m_bps",
    "implied_6m_bps",
    "implied_2y_bps",
    "dgs2_chg_30d_bps",
    "t10y2y",
    "vix",
    "vix_chg_30d",
    "core_cpi_3m_ann",
    "core_cpi_12m",
    "core_pce_3m_ann",
    "core_pce_12m",
    "unrate",
    "unrate_chg_3m",
    "claims_4wk_k",
    "nfci",
    "indpro_3m_ann",
]


def annualized_growth(rows: pd.DataFrame, months: int) -> float:
    """Annualized % change from the observation `months` earlier to the latest one.

    If that exact month was never published (e.g. a release cancelled by a government
    shutdown), the closest earlier observation is used and the rate re-annualized over the
    actual gap.
    """
    if rows.empty:
        return np.nan
    latest = rows.iloc[-1]
    target = latest["date"] - pd.DateOffset(months=months)
    earlier = rows[rows["date"] <= target]
    if earlier.empty:
        return np.nan
    base = earlier.iloc[-1]
    gap_months = (latest["date"].year - base["date"].year) * 12 + (
        latest["date"].month - base["date"].month
    )
    if gap_months <= 0 or base["value"] <= 0:
        return np.nan
    return ((latest["value"] / base["value"]) ** (12 / gap_months) - 1) * 100


def change_over_months(rows: pd.DataFrame, months: int) -> float:
    """Latest value minus the value `months` earlier (closest earlier observation)."""
    if rows.empty:
        return np.nan
    latest = rows.iloc[-1]
    earlier = rows[rows["date"] <= latest["date"] - pd.DateOffset(months=months)]
    return np.nan if earlier.empty else latest["value"] - earlier.iloc[-1]["value"]


def macro_features(
    observations: pd.DataFrame, forecast_dates: pd.Series, previous_decisions: pd.Series
) -> pd.DataFrame:
    """One row per forecast date (aligned with the inputs) with the MACRO_COLUMNS.

    `previous_decisions` holds, for each forecast date, the decision date of the meeting
    before the one being forecast; EFFR's change since then is the last policy move.
    """
    observations = observations.assign(
        date=naive_ns(observations["date"]),
        available_date=naive_ns(observations["available_date"]),
    )
    series = {
        series_id: rows.sort_values("date") for series_id, rows in observations.groupby("series_id")
    }
    dates = naive_ns(pd.Series(forecast_dates)).reset_index(drop=True)

    def at(series_id: str, when: pd.Series) -> np.ndarray:
        return asof_lookup(series[series_id], when)

    effr = at("EFFR", dates)
    features = pd.DataFrame({"forecast_date": dates})
    features["effr"] = effr
    previous = naive_ns(pd.Series(previous_decisions)).reset_index(drop=True)
    features["last_move_bps"] = (effr - at("EFFR", previous)) * 100
    features["implied_3m_bps"] = (at("DGS3MO", dates) - effr) * 100
    features["implied_6m_bps"] = (at("DGS6MO", dates) - effr) * 100
    dgs2 = at("DGS2", dates)
    features["implied_2y_bps"] = (dgs2 - effr) * 100
    features["dgs2_chg_30d_bps"] = (dgs2 - at("DGS2", dates - LOOKBACK_30D)) * 100
    features["t10y2y"] = at("T10Y2Y", dates)
    vix = at("VIXCLS", dates)
    features["vix"] = vix
    features["vix_chg_30d"] = vix - at("VIXCLS", dates - LOOKBACK_30D)

    release_rows = []
    for forecast_date in dates:
        cpi = available_before(series["CPILFESL"], forecast_date)
        pce = available_before(series["PCEPILFE"], forecast_date)
        unemployment = available_before(series["UNRATE"], forecast_date)
        claims = available_before(series["ICSA"], forecast_date)
        conditions = available_before(series["NFCI"], forecast_date)
        production = available_before(series["INDPRO"], forecast_date)
        release_rows.append(
            {
                "core_cpi_3m_ann": annualized_growth(cpi, 3),
                "core_cpi_12m": annualized_growth(cpi, 12),
                "core_pce_3m_ann": annualized_growth(pce, 3),
                "core_pce_12m": annualized_growth(pce, 12),
                "unrate": unemployment["value"].iloc[-1] if len(unemployment) else np.nan,
                "unrate_chg_3m": change_over_months(unemployment, 3),
                "claims_4wk_k": claims["value"].tail(4).mean() / 1000 if len(claims) else np.nan,
                "nfci": conditions["value"].iloc[-1] if len(conditions) else np.nan,
                "indpro_3m_ann": annualized_growth(production, 3),
            }
        )
    features = pd.concat([features, pd.DataFrame(release_rows)], axis=1)
    return features[["forecast_date", *MACRO_COLUMNS]]
