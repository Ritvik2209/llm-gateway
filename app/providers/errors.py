"""Provider failure taxonomy.

Provider calls fail in ways that demand different handling, and the difference is not
recoverable from an error message. Two decisions have to be made about every failure, and
both were previously guessed:

* **Can a retry help?** Retrying a rejected credential burns the full backoff budget —
  0.5 + 1 + 2 seconds — on a call that cannot succeed, and delays the fallback behind it.
* **Is the provider at fault?** Counting a rejected prompt against a provider's health
  lets one malformed request pattern open a working provider's circuit and remove it from
  rotation for everyone.

These classes carry both answers, so the request path reads them instead of inferring
them. They subclass ``RuntimeError`` so existing callers catching that keep working.
"""


class ProviderError(RuntimeError):
    """A provider call failed."""

    #: Could the identical request succeed if tried again shortly?
    retryable: bool = False
    #: Should this count against the provider's health and circuit breaker?
    provider_at_fault: bool = True
    #: Status to surface to the client when the failure ends the request.
    client_status_code: int = 503

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderTimeout(ProviderError):
    """The provider did not respond within the configured timeout."""

    retryable = True


class ProviderUnavailable(ProviderError):
    """The provider could not be reached, or returned a server-side error."""

    retryable = True


class ProviderRateLimited(ProviderError):
    """The provider refused the request on quota.

    Retryable in principle, but a per-day quota will not clear within a request's backoff
    budget while a per-minute one might. ``retry_after`` carries the provider's own
    estimate so the retry loop can tell those apart instead of assuming.
    """

    retryable = True

    def __init__(
        self,
        message: str,
        status_code: int | None = 429,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code)
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """The provider rejected the gateway's credentials.

    This is a gateway misconfiguration, not a transient fault. Retrying cannot fix it, and
    falling back would hide it while silently shifting the traffic and its cost onto
    another provider, so it surfaces as a gateway-side error instead.
    """

    retryable = False
    client_status_code = 502


class ProviderRequestRejected(ProviderError):
    """The provider rejected the request itself — malformed, unknown model, too large.

    The provider is working correctly, so this must not count against its health or open
    its circuit. It is also unlikely to be accepted by a different provider, so it ends
    the request rather than triggering a fallback.
    """

    retryable = False
    provider_at_fault = False
    client_status_code = 400


def error_class_for_status(status_code: int) -> type[ProviderError]:
    """Map an HTTP status from a provider onto the taxonomy."""
    if status_code == 429:
        return ProviderRateLimited
    if status_code in (401, 403):
        return ProviderAuthError
    if 500 <= status_code < 600:
        return ProviderUnavailable
    if 400 <= status_code < 500:
        return ProviderRequestRejected
    return ProviderUnavailable


def parse_retry_after(headers) -> float | None:
    """Read a Retry-After header, in seconds, when the provider supplies one."""
    raw_value = None
    try:
        raw_value = headers.get("retry-after")
    except AttributeError:
        return None

    if raw_value is None:
        return None

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        # Retry-After may also be an HTTP date, which is not worth parsing here: absent
        # a usable number the retry loop falls back to its own backoff schedule.
        return None
