"""Text-layer features from the most recent FOMC statement available at the forecast date.

All features are fit-free (no vocabulary or embedding learned across meetings), so nothing
from future statements can leak into earlier forecasts:

- the policy action the statement announced (raise / maintain / lower),
- its forward guidance ("additional policy firming" vs. "additional adjustments"),
- a hawkish-vs-dovish tone score: each direction word (e.g. "elevated", "slowed") is credited
  to the nearest economic topic word within a few tokens (e.g. "inflation", "unemployment"),
  in the spirit of Apel and Blix Grimaldi (2012),
- how much the wording changed from the previous statement (cosine similarity),
- dissents: how many members dissented and whether they wanted tighter or looser policy.
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from polylayers.features.as_of import naive_ns

TEXT_COLUMNS = [
    "stmt_last_action",
    "stmt_guidance",
    "stmt_hawkish_score",
    "stmt_hawkish_score_change",
    "stmt_similarity_prev",
    "stmt_n_dissents",
    "stmt_dissent_lean",
]

_PHRASES = {
    "labor market": "labor_market",
    "job gains": "job_gains",
    "economic activity": "economic_activity",
    "unemployment rate": "unemployment_rate",
    "price pressures": "price_pressures",
}
_TOPICS = {
    "inflation",
    "prices",
    "price_pressures",
    "wages",
    "growth",
    "economic_activity",
    "activity",
    "spending",
    "demand",
    "economy",
    "labor_market",
    "job_gains",
    "employment",
}
# More of these is dovish: rising unemployment argues for looser policy.
_REVERSED_TOPICS = {"unemployment", "unemployment_rate"}
_HAWKISH_WORDS = {
    "elevated", "high", "higher", "increase", "increased", "increases", "increasing", "rise",
    "risen", "rising", "rose", "strong", "stronger", "robust", "solid", "tight", "tighter",
    "firming", "accelerate", "accelerated", "accelerating", "upside",
}  # fmt: skip
_DOVISH_WORDS = {
    "low", "lower", "decrease", "decreased", "decline", "declined", "declining", "eased",
    "easing", "moderate", "moderated", "moderating", "slow", "slowed", "slowing", "softened",
    "softening", "softer", "weak", "weaker", "weakened", "weakening", "cooled", "cooling",
    "downside", "subdued",
}  # fmt: skip
_TOPIC_WINDOW = 6
_STOPWORDS = {"the", "and", "for", "that", "with", "its", "has", "have", "this", "are", "was"}

_ACTION = re.compile(r"decided to (raise|increase|maintain|lower|reduce)\b[^.]*?target range")
_MAGNITUDE = re.compile(r"by (1/4|1/2|3/4|1) percentage point")
_PREFERENCE = re.compile(r"who preferred (?:to )?([^,;]*)")
_DISSENT_LEAD = re.compile(r"^.*?(?:action|actions) (?:was|were)\s+", re.IGNORECASE)
# Member names as printed in statements: "Michelle W. Bowman", "Neel Kashkari".
_PERSON_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+")
_MAGNITUDE_BPS = {"1/4": 25.0, "1/2": 50.0, "3/4": 75.0, "1": 100.0}


def tokens(text: str) -> list[str]:
    """Lowercase word tokens with multi-word topics joined (e.g. 'labor_market')."""
    lowered = text.lower()
    for phrase, joined in _PHRASES.items():
        lowered = lowered.replace(phrase, joined)
    return re.findall(r"[a-z_]+", lowered)


def hawkish_score(text: str) -> float:
    """(hawkish - dovish) / (hawkish + dovish) over direction words attached to topics."""
    hawkish = dovish = 0
    for sentence in re.split(r"(?<=[.;])\s+", text):
        words = tokens(sentence)
        topic_positions = [i for i, w in enumerate(words) if w in _TOPICS | _REVERSED_TOPICS]
        if not topic_positions:
            continue
        for position, word in enumerate(words):
            if word not in _HAWKISH_WORDS and word not in _DOVISH_WORDS:
                continue
            nearest = min(topic_positions, key=lambda topic: abs(topic - position))
            if abs(nearest - position) > _TOPIC_WINDOW:
                continue
            leans_hawkish = word in _HAWKISH_WORDS
            if words[nearest] in _REVERSED_TOPICS:
                leans_hawkish = not leans_hawkish
            hawkish += leans_hawkish
            dovish += not leans_hawkish
    total = hawkish + dovish
    return (hawkish - dovish) / total if total else 0.0


def policy_move_bps(text: str) -> float:
    """Signed size of the move a statement announced (0 for 'maintain'); NaN if not found."""
    match = _ACTION.search(text)
    if match is None:
        return np.nan
    verb = match.group(1)
    if verb == "maintain":
        return 0.0
    magnitude = _MAGNITUDE.search(text[match.start() : match.start() + 300])
    size = _MAGNITUDE_BPS[magnitude.group(1)] if magnitude else 25.0
    return size if verb in ("raise", "increase") else -size


def guidance_direction(text: str) -> float:
    """+1 for tightening guidance, -1 for easing guidance, 0 for neither."""
    if "ongoing increases in the target range" in text or "additional policy firming" in text:
        return 1.0
    if "additional adjustments to the target range" in text:
        return -1.0
    return 0.0


def preferred_move_bps(preference: str) -> float:
    """Signed move a dissenter preferred, from text like 'lower the target range ... by 1/2'."""
    if "raise" in preference or "increase" in preference:
        sign = 1.0
    elif "lower" in preference or "reduce" in preference:
        sign = -1.0
    else:
        return 0.0
    magnitude = _MAGNITUDE.search(preference)
    return sign * (_MAGNITUDE_BPS[magnitude.group(1)] if magnitude else 25.0)


def dissent_summary(dissent_text: str, action_move_bps: float) -> tuple[float, float]:
    """(number of dissenters, net lean): lean +1 means dissenters wanted tighter policy."""
    if not dissent_text:
        return 0.0, 0.0
    remainder = _DISSENT_LEAD.sub("", dissent_text, count=1)
    count = len(_PERSON_NAME.findall(remainder))
    lean = 0.0
    for preference in _PREFERENCE.findall(dissent_text):
        difference = preferred_move_bps(preference) - (action_move_bps or 0.0)
        lean += np.sign(difference)
    return float(count), float(np.sign(lean))


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity of word-count vectors (stopwords and short words removed)."""
    counts_a = Counter(w for w in tokens(text_a) if len(w) > 2 and w not in _STOPWORDS)
    counts_b = Counter(w for w in tokens(text_b) if len(w) > 2 and w not in _STOPWORDS)
    dot = sum(counts_a[word] * counts_b[word] for word in counts_a)
    norm = math.sqrt(sum(v * v for v in counts_a.values())) * math.sqrt(
        sum(v * v for v in counts_b.values())
    )
    return dot / norm if norm else np.nan


def per_statement_features(statements: pd.DataFrame) -> pd.DataFrame:
    """Features of each statement, including changes relative to the statement before it."""
    ordered = statements.sort_values("decision_date").reset_index(drop=True)
    rows = []
    for index, statement in ordered.iterrows():
        text = statement["statement_text"]
        action = policy_move_bps(text)
        n_dissents, lean = dissent_summary(statement["dissent_text"], action)
        score = hawkish_score(text)
        previous = ordered.iloc[index - 1] if index > 0 else None
        rows.append(
            {
                "available_date": statement["available_date"],
                "stmt_last_action": np.sign(action) if not np.isnan(action) else np.nan,
                "stmt_guidance": guidance_direction(text),
                "stmt_hawkish_score": score,
                "stmt_hawkish_score_change": (
                    score - hawkish_score(previous["statement_text"])
                    if previous is not None
                    else np.nan
                ),
                "stmt_similarity_prev": (
                    cosine_similarity(text, previous["statement_text"])
                    if previous is not None
                    else np.nan
                ),
                "stmt_n_dissents": n_dissents,
                "stmt_dissent_lean": lean,
            }
        )
    return pd.DataFrame(rows)


def statement_features(statements: pd.DataFrame, forecast_dates: pd.Series) -> pd.DataFrame:
    """Features of the latest statement available at each forecast date."""
    features = per_statement_features(statements)
    features["available_date"] = naive_ns(features["available_date"])
    features = features.sort_values("available_date")
    left = pd.DataFrame({"forecast_date": naive_ns(pd.Series(forecast_dates)).to_numpy()})
    left["order"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values("forecast_date"),
        features,
        left_on="forecast_date",
        right_on="available_date",
        direction="backward",
    )
    merged = merged.sort_values("order").reset_index(drop=True)
    return merged[["forecast_date", *TEXT_COLUMNS]]
