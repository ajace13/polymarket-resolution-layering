from datetime import date

from polylayers.config import load_settings


def test_default_settings_load_with_typed_dates():
    settings = load_settings()
    assert settings.horizons_days == (30, 14, 7, 1)
    assert settings.data.first_meeting == date(2023, 2, 1)
    assert settings.data.series_start < settings.data.first_meeting <= settings.data.last_meeting
    assert settings.gdelt.min_interval_seconds >= 5
