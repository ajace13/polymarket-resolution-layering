"""Scheduled FOMC meetings, 2022-2026 (federalreserve.gov/monetarypolicy/fomccalendars.htm).

The decision is announced at 2:00 pm ET on the second day of each meeting, so the
second day is the date every horizon is measured back from.
"""

from dataclasses import dataclass
from datetime import date

# (first day, decision day) for every scheduled meeting.
_SCHEDULE = (
    ("2022-01-25", "2022-01-26"),
    ("2022-03-15", "2022-03-16"),
    ("2022-05-03", "2022-05-04"),
    ("2022-06-14", "2022-06-15"),
    ("2022-07-26", "2022-07-27"),
    ("2022-09-20", "2022-09-21"),
    ("2022-11-01", "2022-11-02"),
    ("2022-12-13", "2022-12-14"),
    ("2023-01-31", "2023-02-01"),
    ("2023-03-21", "2023-03-22"),
    ("2023-05-02", "2023-05-03"),
    ("2023-06-13", "2023-06-14"),
    ("2023-07-25", "2023-07-26"),
    ("2023-09-19", "2023-09-20"),
    ("2023-10-31", "2023-11-01"),
    ("2023-12-12", "2023-12-13"),
    ("2024-01-30", "2024-01-31"),
    ("2024-03-19", "2024-03-20"),
    ("2024-04-30", "2024-05-01"),
    ("2024-06-11", "2024-06-12"),
    ("2024-07-30", "2024-07-31"),
    ("2024-09-17", "2024-09-18"),
    ("2024-11-06", "2024-11-07"),
    ("2024-12-17", "2024-12-18"),
    ("2025-01-28", "2025-01-29"),
    ("2025-03-18", "2025-03-19"),
    ("2025-05-06", "2025-05-07"),
    ("2025-06-17", "2025-06-18"),
    ("2025-07-29", "2025-07-30"),
    ("2025-09-16", "2025-09-17"),
    ("2025-10-28", "2025-10-29"),
    ("2025-12-09", "2025-12-10"),
    ("2026-01-27", "2026-01-28"),
    ("2026-03-17", "2026-03-18"),
    ("2026-04-28", "2026-04-29"),
    ("2026-06-16", "2026-06-17"),
    ("2026-07-28", "2026-07-29"),
    ("2026-09-15", "2026-09-16"),
    ("2026-10-27", "2026-10-28"),
    ("2026-12-08", "2026-12-09"),
)


@dataclass(frozen=True, order=True)
class FomcMeeting:
    """One scheduled FOMC meeting, ordered by decision date."""

    decision_date: date
    start_date: date

    @property
    def meeting_id(self) -> str:
        """Stable identifier: the decision date in ISO format."""
        return self.decision_date.isoformat()

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'February 2023'."""
        return self.decision_date.strftime("%B %Y")


def all_meetings() -> list[FomcMeeting]:
    """Every scheduled meeting in the calendar, oldest first."""
    meetings = [
        FomcMeeting(decision_date=date.fromisoformat(end), start_date=date.fromisoformat(start))
        for start, end in _SCHEDULE
    ]
    return sorted(meetings)


def meetings_between(first_decision: date, last_decision: date) -> list[FomcMeeting]:
    """Meetings whose decision date falls in [first_decision, last_decision], oldest first."""
    return [
        meeting
        for meeting in all_meetings()
        if first_decision <= meeting.decision_date <= last_decision
    ]
