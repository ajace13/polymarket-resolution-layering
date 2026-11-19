"""HTTP client shared by the collectors: spacing between calls plus exponential backoff.

Every collector talks to a public API with its own rate limit. `PoliteClient` enforces a
minimum interval between requests and retries throttled or failed requests (HTTP 429/5xx,
connection errors) with exponential backoff, up to an attempt cap or a wall-clock deadline.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class RetriesExhaustedError(RuntimeError):
    """Raised when a request keeps failing after all allowed attempts."""


class DeadlineReachedError(RetriesExhaustedError):
    """Raised when the next retry would start after the caller's deadline."""


@dataclass(frozen=True)
class RetryPolicy:
    """Pacing and retry settings for one API."""

    min_interval_seconds: float = 1.0
    backoff_initial_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    max_attempts: int = 5
    timeout_seconds: float = 60.0

    def backoff_seconds(self, failed_attempts: int) -> float:
        """Delay before the next attempt: doubles per failure, capped at the maximum."""
        delay = self.backoff_initial_seconds * 2 ** (failed_attempts - 1)
        return min(delay, self.backoff_max_seconds)


class PoliteClient:
    """GET requests with a minimum interval between calls and backoff on failures."""

    def __init__(
        self,
        policy: RetryPolicy,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy = policy
        self.session = session or requests.Session()
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> requests.Response:
        """GET `url`, retrying retryable failures. `deadline` is a value of the client clock."""
        failed_attempts = 0
        while True:
            self._wait_for_next_slot()
            response, failure = self._attempt(url, params)
            if failure is None:
                return response
            failed_attempts += 1
            if failed_attempts >= self.policy.max_attempts:
                raise RetriesExhaustedError(
                    f"GET {url} failed {failed_attempts} times; last error: {failure}"
                )
            delay = self.policy.backoff_seconds(failed_attempts)
            if deadline is not None and self._clock() + delay > deadline:
                raise DeadlineReachedError(
                    f"GET {url} still failing at the deadline; last error: {failure}"
                )
            logger.warning("GET %s: %s; retry %d in %.0fs", url, failure, failed_attempts, delay)
            self._sleep(delay)

    def _attempt(
        self, url: str, params: dict[str, Any] | None
    ) -> tuple[requests.Response | None, str | None]:
        """One request: (response, None) on success, (response or None, reason) to retry."""
        self._last_request_at = self._clock()
        try:
            response = self.session.get(url, params=params, timeout=self.policy.timeout_seconds)
        except (requests.ConnectionError, requests.Timeout) as error:
            return None, f"{type(error).__name__}: {error}"
        if response.status_code in RETRYABLE_STATUS_CODES:
            return response, f"HTTP {response.status_code}"
        return response, None

    def _wait_for_next_slot(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._clock() - self._last_request_at
        remaining = self.policy.min_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)
