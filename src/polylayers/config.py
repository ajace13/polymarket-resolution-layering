"""Typed access to the experiment settings in configs/*.toml."""

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from polylayers.paths import CONFIG_DIR

DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.toml"


@dataclass(frozen=True)
class DataSettings:
    """Date bounds for the meetings in scope and for the daily series behind them."""

    first_meeting: date
    last_meeting: date
    series_start: date


@dataclass(frozen=True)
class GdeltSettings:
    """Request pacing for the throttled GDELT DOC API."""

    min_interval_seconds: float
    backoff_initial_seconds: float
    backoff_max_seconds: float


@dataclass(frozen=True)
class Settings:
    """All experiment settings loaded from one TOML file."""

    horizons_days: tuple[int, ...]
    data: DataSettings
    gdelt: GdeltSettings


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Parse a TOML settings file into a `Settings` object."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return Settings(
        horizons_days=tuple(raw["horizons_days"]),
        data=DataSettings(**raw["data"]),
        gdelt=GdeltSettings(**raw["gdelt"]),
    )
