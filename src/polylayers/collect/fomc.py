"""FOMC post-meeting statements from federalreserve.gov.

Each statement is published at 2:00 pm ET on the decision day, at a stable URL keyed by the
decision date. We keep the statement body and the dissent paragraph separately: dissents
("... who preferred to raise the target range ...") are a policy signal of their own.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from polylayers.fomc_calendar import FomcMeeting
from polylayers.http import PoliteClient, RetryPolicy

logger = logging.getLogger(__name__)

STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{date:%Y%m%d}a.htm"
FOMC_POLICY = RetryPolicy(min_interval_seconds=1.0, backoff_initial_seconds=5.0)

_SKIP_PREFIXES = ("For release at", "The Federal Open Market Committee approved the following")
_STOP_PREFIXES = ("Voting for", "For media inquiries", "Implementation Note")
_DISSENT_PREFIX = "Voting against"
_DATE_LINE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


def fetch_statement_html(client: PoliteClient, meeting: FomcMeeting, cache_dir: Path) -> str:
    """Raw HTML of one statement, cached on disk; decoded as UTF-8 explicitly."""
    cache_path = cache_dir / f"statement_{meeting.decision_date:%Y%m%d}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    response = client.get(STATEMENT_URL.format(date=meeting.decision_date))
    response.raise_for_status()
    html = response.content.decode("utf-8", errors="replace")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


def extract_statement(html: str) -> tuple[str, str]:
    """(statement body, dissent paragraph) from a statement page's article paragraphs."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("div", id="article")
    if article is None:
        raise ValueError("statement page has no div#article")
    body: list[str] = []
    dissent = ""
    for paragraph in article.find_all("p"):
        text = " ".join(paragraph.get_text(" ", strip=True).split())
        if not text or _DATE_LINE.match(text) or text.startswith(_SKIP_PREFIXES):
            continue
        if text.startswith(_DISSENT_PREFIX):
            dissent = text
            continue
        if text.startswith(_STOP_PREFIXES):
            # Older statements put "Voting against ..." inside the "Voting for ..." paragraph.
            position = text.find(_DISSENT_PREFIX)
            if position >= 0 and not dissent:
                dissent = text[position:]
            break
        body.append(text)
    return " ".join(body), dissent


def collect_statements(
    client: PoliteClient, meetings: list[FomcMeeting], raw_dir: Path
) -> pd.DataFrame:
    """One row per meeting: decision date, statement text, dissent text, availability."""
    rows = []
    for meeting in meetings:
        html = fetch_statement_html(client, meeting, raw_dir)
        text, dissent = extract_statement(html)
        if len(text.split()) < 100:
            logger.warning(
                "Statement for %s looks truncated (%d words)", meeting.meeting_id, len(text.split())
            )
        rows.append(
            {
                "meeting_id": meeting.meeting_id,
                "decision_date": pd.Timestamp(meeting.decision_date),
                "statement_text": text,
                "dissent_text": dissent,
                "n_words": len(text.split()),
                # Released at 2:00 pm ET on the decision day: usable from the next day.
                "available_date": pd.Timestamp(meeting.decision_date) + pd.Timedelta(days=1),
            }
        )
    statements = pd.DataFrame(rows)
    logger.info(
        "FOMC statements: %d meetings, median %d words",
        len(statements),
        int(statements["n_words"].median()),
    )
    return statements
