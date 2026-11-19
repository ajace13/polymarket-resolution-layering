"""Command-line entry point: `python -m polylayers <command> ...`."""

import argparse
import logging
from pathlib import Path

from polylayers.collect import gdelt
from polylayers.config import DEFAULT_CONFIG_PATH, load_settings
from polylayers.http import PoliteClient
from polylayers.paths import INTERIM_DIR, RAW_DIR

logger = logging.getLogger("polylayers")


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
    daily = gdelt.build_daily_table(tasks, raw_dir)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DIR / "gdelt_daily.parquet"
    daily.to_parquet(output_path, index=False)
    logger.info("Wrote %s %s", output_path, daily.shape)
    if not summary.complete:
        logger.warning("GDELT is incomplete; rerun the same command to resume.")


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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, configure logging, and run the selected command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.handler(args)
