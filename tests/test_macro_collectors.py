import io
import json
from datetime import date

import pandas as pd

from conftest import FakeResponse, FakeSession
from polylayers.collect import fomc, fred, sf_fed
from polylayers.fomc_calendar import FomcMeeting
from polylayers.http import PoliteClient, RetryPolicy


def make_client(outcomes, clock):
    policy = RetryPolicy(min_interval_seconds=0.0)
    session = FakeSession(list(outcomes))
    return PoliteClient(policy, session=session, clock=clock, sleep=clock.sleep), session


# --------------------------------------------------------------------------- FRED


def test_daily_series_become_available_the_day_after_the_next_business_day():
    observations = [
        {"date": "2024-03-13", "value": "5.33"},  # Wednesday -> published Thursday
        {"date": "2024-03-15", "value": "5.33"},  # Friday -> published Monday
        {"date": "2024-03-18", "value": "."},  # missing-day placeholder
    ]
    table = fred.daily_series_table("EFFR", observations)
    expected = [pd.Timestamp("2024-03-15"), pd.Timestamp("2024-03-19")]
    assert table["available_date"].tolist() == expected
    assert table["value"].tolist() == [5.33, 5.33]


def test_release_series_become_available_the_day_after_publication():
    observations = [{"realtime_start": "2024-04-10", "date": "2024-03-01", "value": "312.1"}]
    table = fred.release_series_table("CPILFESL", observations)
    assert table.loc[0, "available_date"] == pd.Timestamp("2024-04-11")
    assert table.loc[0, "date"] == pd.Timestamp("2024-03-01")


def test_fred_requests_send_the_key_but_never_cache_it(tmp_path, fake_clock, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "secret-key")
    payload = {"observations": [{"date": "2024-01-02", "value": "1.0"}]}
    client, session = make_client([FakeResponse(200, json.dumps(payload))], fake_clock)
    cache_path = tmp_path / "EFFR.json"
    fred.fetch_observations(client, {"series_id": "EFFR"}, cache_path)
    assert session.calls[0]["params"]["api_key"] == "secret-key"
    assert "secret-key" not in cache_path.read_text()
    fred.fetch_observations(client, {"series_id": "EFFR"}, cache_path)
    assert len(session.calls) == 1


# --------------------------------------------------------------------------- SF Fed


def workbook_bytes() -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["methodology"]}).to_excel(writer, sheet_name="Methodology")
        data = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-02", "not a date"],
                "News Sentiment": [0.1, -0.2, -0.3, 0.5],
            }
        )
        data.to_excel(writer, sheet_name="Data", index=False)
    return buffer.getvalue()


def test_parse_workbook_reads_data_sheet_and_lags_availability():
    frame = sf_fed.parse_workbook(workbook_bytes())
    assert frame["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert frame["sentiment"].tolist() == [0.1, -0.3]
    lag = pd.Timedelta(days=sf_fed.PUBLICATION_LAG_DAYS)
    assert (frame["available_date"] - frame["date"] == lag).all()


# --------------------------------------------------------------------------- FOMC

STATEMENT_HTML = """
<html><body><div id="article">
<p>March 20, 2024</p>
<p>For release at 2:00 p.m. EDT Share</p>
<p>Recent indicators suggest that economic activity has been expanding at a solid pace.</p>
<p>The Committee decided to maintain the target range for the federal funds rate.</p>
<p>Voting against this action was Jane Doe, who preferred to lower the target range.</p>
<p>Implementation Note issued March 20, 2024</p>
<p>This paragraph is after the stop marker.</p>
</div></body></html>
"""


def test_extract_statement_keeps_body_and_dissent_only():
    body, dissent = fomc.extract_statement(STATEMENT_HTML)
    assert body.startswith("Recent indicators suggest")
    assert body.endswith("federal funds rate.")
    assert "after the stop marker" not in body and "March 20, 2024" not in body
    assert dissent.startswith("Voting against this action was Jane Doe")


def test_extract_statement_finds_dissent_inside_the_voting_paragraph():
    html = """<div id="article"><p>The Committee decided to lower the target range.</p>
    <p>Voting for the monetary policy action were Jerome H. Powell, Chair; and others.
    Voting against the action was Michelle W. Bowman, who preferred to lower the target
    range for the federal funds rate by 1/4 percentage point at this meeting.</p></div>"""
    body, dissent = fomc.extract_statement(html)
    assert body == "The Committee decided to lower the target range."
    assert dissent.startswith("Voting against the action was Michelle W. Bowman")


def test_collect_statements_uses_cache_and_sets_next_day_availability(tmp_path, fake_clock):
    meeting = FomcMeeting(decision_date=date(2024, 3, 20), start_date=date(2024, 3, 19))
    (tmp_path / "statement_20240320.html").write_text(STATEMENT_HTML, encoding="utf-8")
    client, session = make_client([], fake_clock)
    statements = fomc.collect_statements(client, [meeting], tmp_path)
    assert session.calls == []
    assert statements.loc[0, "available_date"] == pd.Timestamp("2024-03-21")
    assert statements.loc[0, "meeting_id"] == "2024-03-20"
