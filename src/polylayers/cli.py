"""Command-line entry point: `python -m polylayers <command> ...`."""

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from polylayers.collect import fomc, fred, gdelt, polymarket, sf_fed
from polylayers.config import DEFAULT_CONFIG_PATH, load_settings
from polylayers.evaluation.experiment import default_model_specs, run_walk_forward, summarize
from polylayers.features.panel import build_panel
from polylayers.fomc_calendar import meetings_between
from polylayers.graph.wallets import wallet_first_trades
from polylayers.http import PoliteClient
from polylayers.models.recalibration import temperatures_by_fold
from polylayers.paths import (
    FIGURES_DIR,
    INTERIM_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RESULTS_DIR,
    SNAPSHOT_DIR,
)
from polylayers.reporting.figures import save_all_figures
from polylayers.reporting.tables import summary_markdown

logger = logging.getLogger("polylayers")

# Macro history needed before the first forecast date (12-month inflation changes).
MACRO_LOOKBACK = timedelta(days=500)


def write_interim(frame: pd.DataFrame, name: str) -> None:
    """Write one tidy table to data/interim/<name>.parquet and log its shape."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    logger.info("Wrote %s %s", path.relative_to(INTERIM_DIR.parent.parent), frame.shape)


def run_collect_polymarket(args: argparse.Namespace) -> None:
    """Discover FOMC markets, then fetch their price series and full trade histories."""
    settings = load_settings(args.config)
    meetings = meetings_between(settings.data.first_meeting, settings.data.last_meeting)
    raw_dir = RAW_DIR / "polymarket"
    gamma_client = PoliteClient(polymarket.GAMMA_POLICY)
    data_client = PoliteClient(polymarket.DATA_API_POLICY)

    events = polymarket.discover_events(gamma_client, raw_dir)
    matched = polymarket.match_events_to_meetings(events, meetings)
    markets = polymarket.build_markets_table(matched)
    write_interim(markets, "polymarket_markets")

    prices, settlements = polymarket.build_prices_table(markets, data_client, raw_dir)
    polymarket.validate_labels(markets, settlements)
    write_interim(prices, "polymarket_prices")

    trades, coverage = polymarket.build_trades_table(markets, data_client, raw_dir)
    write_interim(trades, "polymarket_trades")
    write_interim(coverage, "polymarket_trade_coverage")


def run_collect_fred(args: argparse.Namespace) -> None:
    """Daily market series and first-release macro series, with availability dates."""
    settings = load_settings(args.config)
    start = settings.data.series_start - MACRO_LOOKBACK
    client = PoliteClient(fred.FRED_POLICY)
    table = fred.collect_fred(client, RAW_DIR / "fred", start, settings.data.last_meeting)
    write_interim(table, "fred_observations")


def run_collect_sf_fed(args: argparse.Namespace) -> None:
    """SF Fed Daily News Sentiment Index."""
    client = PoliteClient(sf_fed.SF_FED_POLICY)
    content = sf_fed.download_workbook(client, RAW_DIR / "sf_fed" / "news_sentiment_data.xlsx")
    write_interim(sf_fed.parse_workbook(content), "sf_fed_sentiment")


def run_collect_fomc(args: argparse.Namespace) -> None:
    """FOMC statements for every meeting whose text can precede an in-scope forecast."""
    settings = load_settings(args.config)
    meetings = meetings_between(settings.data.series_start, settings.data.last_meeting)
    client = PoliteClient(fomc.FOMC_POLICY)
    statements = fomc.collect_statements(client, meetings, RAW_DIR / "fomc_statements")
    write_interim(statements, "fomc_statements")


def run_collect_gdelt(args: argparse.Namespace) -> None:
    """Fetch GDELT daily timelines (resumable), then write the tidy daily table."""
    settings = load_settings(args.config)
    tasks = gdelt.build_tasks(gdelt.QUERIES, settings.data.series_start, settings.data.last_meeting)
    raw_dir = RAW_DIR / "gdelt"
    client = PoliteClient(gdelt.retry_policy(settings.gdelt))
    summary = gdelt.fetch_timelines(tasks, raw_dir, client, gdelt.deadline_after(args.max_hours))
    logger.info(
        "GDELT fetch: %d cached before, %d fetched, %d rejected, %d remaining",
        summary.cached_before,
        summary.fetched,
        len(summary.rejected),
        len(summary.remaining),
    )
    write_interim(gdelt.build_daily_table(tasks, raw_dir), "gdelt_daily")
    if not summary.complete:
        logger.warning("GDELT is incomplete; rerun the same command to resume.")


def read_interim(name: str) -> pd.DataFrame:
    return pd.read_parquet(INTERIM_DIR / f"{name}.parquet")


def run_build(args: argparse.Namespace) -> None:
    """Assemble the point-in-time panel and the hashed wallet table into data/snapshot/."""
    settings = load_settings(args.config)
    gdelt_path = INTERIM_DIR / "gdelt_daily.parquet"
    gdelt_daily = pd.read_parquet(gdelt_path) if gdelt_path.exists() else pd.DataFrame()
    if gdelt_daily.empty:
        logger.warning("No GDELT data yet; GDELT features will be empty (NaN).")
    trades = read_interim("polymarket_trades")
    panel = build_panel(
        markets=read_interim("polymarket_markets"),
        prices=read_interim("polymarket_prices"),
        trades=trades,
        coverage=read_interim("polymarket_trade_coverage"),
        fred_observations=read_interim("fred_observations"),
        sentiment=read_interim("sf_fed_sentiment"),
        statements=read_interim("fomc_statements"),
        gdelt_daily=gdelt_daily,
        horizons_days=settings.horizons_days,
    )
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(SNAPSHOT_DIR / "panel.parquet", index=False)
    wallets = wallet_first_trades(trades)
    wallets.to_parquet(SNAPSHOT_DIR / "market_wallets.parquet", index=False, compression="zstd")
    logger.info(
        "Wrote data/snapshot/panel.parquet %s and market_wallets.parquet %s",
        panel.shape,
        wallets.shape,
    )


def run_evaluate(args: argparse.Namespace) -> None:
    """Walk-forward predictions for every model, scores, and differences vs. the market."""
    settings = load_settings(args.config)
    panel = pd.read_parquet(SNAPSHOT_DIR / "panel.parquet")
    wallets = pd.read_parquet(SNAPSHOT_DIR / "market_wallets.parquet")
    predictions = run_walk_forward(
        panel, wallets, default_model_specs(), settings.evaluation.min_train_meetings
    )
    scores, differences = summarize(predictions, settings.evaluation.bootstrap_resamples)
    temperatures = temperatures_by_fold(panel, settings.evaluation.min_train_meetings)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(RESULTS_DIR / "predictions.parquet", index=False)
    scores.to_csv(RESULTS_DIR / "scores.csv", index=False)
    differences.to_csv(RESULTS_DIR / "differences.csv", index=False)
    temperatures.to_csv(RESULTS_DIR / "temperatures.csv", index=False)
    logger.info(
        "Wrote reports/results/: predictions %s, scores %s, differences %s, temperatures %s",
        predictions.shape,
        scores.shape,
        differences.shape,
        temperatures.shape,
    )


def run_report(args: argparse.Namespace) -> None:
    """Figures (light and dark) and a markdown summary from the evaluation results."""
    predictions = pd.read_parquet(RESULTS_DIR / "predictions.parquet")
    scores = pd.read_csv(RESULTS_DIR / "scores.csv")
    differences = pd.read_csv(RESULTS_DIR / "differences.csv")
    temperatures = pd.read_csv(RESULTS_DIR / "temperatures.csv")
    save_all_figures(scores, differences, temperatures, FIGURES_DIR)
    summary = summary_markdown(scores, differences, predictions, temperatures)
    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Wrote %s", summary_path.relative_to(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polylayers", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="Fetch raw data from one source.")
    sources = collect.add_subparsers(dest="source", required=True)
    gdelt_parser = sources.add_parser("gdelt", help="GDELT DOC API daily timelines.")
    gdelt_parser.add_argument(
        "--max-hours",
        type=float,
        default=12.0,
        help="Stop retrying throttled requests after this many hours (rerun to resume).",
    )
    gdelt_parser.set_defaults(handler=run_collect_gdelt)
    polymarket_parser = sources.add_parser(
        "polymarket", help="Polymarket FOMC markets, prices, and trades."
    )
    polymarket_parser.set_defaults(handler=run_collect_polymarket)
    fred_parser = sources.add_parser("fred", help="FRED daily and first-release macro series.")
    fred_parser.set_defaults(handler=run_collect_fred)
    sf_fed_parser = sources.add_parser("sf-fed", help="SF Fed Daily News Sentiment Index.")
    sf_fed_parser.set_defaults(handler=run_collect_sf_fed)
    fomc_parser = sources.add_parser("fomc", help="FOMC post-meeting statements.")
    fomc_parser.set_defaults(handler=run_collect_fomc)

    build = commands.add_parser("build", help="Assemble the point-in-time feature panel.")
    build.set_defaults(handler=run_build)
    evaluate = commands.add_parser("evaluate", help="Walk-forward evaluation of every model.")
    evaluate.set_defaults(handler=run_evaluate)
    report = commands.add_parser("report", help="Figures from the evaluation results.")
    report.set_defaults(handler=run_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging, and run the selected command."""
    args = build_parser().parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.handler(args)
