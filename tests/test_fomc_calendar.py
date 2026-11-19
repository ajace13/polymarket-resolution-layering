from datetime import date, timedelta

from polylayers.fomc_calendar import all_meetings, meetings_between


def test_meetings_are_two_days_and_strictly_ordered():
    meetings = all_meetings()
    assert all(m.decision_date - m.start_date == timedelta(days=1) for m in meetings)
    decisions = [m.decision_date for m in meetings]
    assert decisions == sorted(set(decisions))


def test_meetings_are_at_least_five_weeks_apart():
    # Horizons up to 30 days rely on the previous meeting resolving before the forecast date.
    decisions = [m.decision_date for m in all_meetings()]
    gaps = [later - earlier for earlier, later in zip(decisions, decisions[1:], strict=False)]
    assert min(gaps) >= timedelta(days=35)


def test_labels_are_unique():
    labels = [m.label for m in all_meetings()]
    assert len(labels) == len(set(labels))


def test_meetings_between_is_inclusive():
    meetings = meetings_between(date(2023, 2, 1), date(2023, 12, 13))
    assert meetings[0].label == "February 2023"
    assert meetings[-1].label == "December 2023"
    assert len(meetings) == 8
    assert meetings[0].meeting_id == "2023-02-01"
