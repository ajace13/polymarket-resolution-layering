"""Parse a market question into the rate move it bets on.

Polymarket phrases FOMC buckets several ways across years ("Will the Fed increase interest
rates by 25 bps after the March 2023 meeting?", "No change in Fed interest rates after ...",
"Fed decreases interest rates by 50+ bps after ..."). Every market maps to a signed minimum
move in basis points, plus a flag for open-ended buckets ("25+", "or more").
"""

import re
from dataclasses import dataclass

_HOLD = re.compile(r"no change|\b0\s*(?:bps|bp|basis)")
_CUT = re.compile(r"\b(?:decreas|cut|lower|reduc)")
_HIKE = re.compile(r"\b(?:increas|hike|rais)")
_MAGNITUDE = re.compile(r"\b(\d{2,3})\s*(\+)?\s*(?:bps|bp|basis)")


@dataclass(frozen=True)
class Bucket:
    """The outcome a market pays out on."""

    name: str
    move_bps: float
    open_ended: bool

    @property
    def group(self) -> str:
        """Coarser grouping used to link similar buckets across meetings."""
        if self.name in ("hold", "other"):
            return self.name
        direction = "cut" if self.move_bps < 0 else "hike"
        size = "small" if abs(self.move_bps) <= 25 else "large"
        return f"{direction}_{size}"


def parse_bucket(question: str) -> Bucket:
    """Map a market question to its bucket; unrecognized phrasings become 'other'."""
    text = question.lower()
    if _HOLD.search(text):
        return Bucket("hold", 0.0, False)
    magnitude = _MAGNITUDE.search(text)
    is_cut = bool(_CUT.search(text))
    is_hike = bool(_HIKE.search(text))
    if magnitude is None or is_cut == is_hike:
        return Bucket("other", float("nan"), False)
    size = float(magnitude.group(1))
    open_ended = bool(magnitude.group(2)) or "or more" in text
    sign = -1.0 if is_cut else 1.0
    direction = "cut" if is_cut else "hike"
    suffix = "+" if open_ended else ""
    return Bucket(f"{direction}_{int(size)}{suffix}", sign * size, open_ended)
