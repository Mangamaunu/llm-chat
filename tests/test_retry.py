"""Retry helpers and the retry-after header parser."""
from types import SimpleNamespace

import pytest

import retry
from chat.error import BadRequest, ConnectionFailed, LLMError, Overloaded, RateLimited, ServerError, retry_after


@pytest.mark.parametrize("exc, expected", [
    (RateLimited("x"), True),
    (Overloaded("x"), True),
    (ServerError("x"), True),
    (ConnectionFailed("x"), True),
    (BadRequest("x"), False),
    (LLMError("x"), False),
    (ValueError("x"), False),
])
def test_is_retryable(exc, expected):
    assert retry.is_retryable(exc) is expected


@pytest.mark.parametrize("attempt, upper", [(0, 1.0), (1, 2.0), (2, 4.0), (5, 10.0)])
def test_backoff_upper_bound_doubles_until_cap(monkeypatch, attempt, upper):
    monkeypatch.setattr(retry.random, "uniform", lambda lo, hi: hi)
    assert retry.backoff_delay(attempt, ServerError("x"), base=1.0, cap=10.0) == upper


def test_backoff_uses_random_jitter_from_zero(monkeypatch):
    monkeypatch.setattr(retry.random, "uniform", lambda lo, hi: lo)
    assert retry.backoff_delay(3, ServerError("x"), base=1.0, cap=10.0) == 0


def test_backoff_waits_at_least_retry_after(monkeypatch):
    monkeypatch.setattr(retry.random, "uniform", lambda lo, hi: lo)
    assert retry.backoff_delay(0, RateLimited("x", retry_after=4), base=1.0, cap=10.0) == 4


def test_backoff_retry_after_is_capped(monkeypatch):
    monkeypatch.setattr(retry.random, "uniform", lambda lo, hi: lo)
    assert retry.backoff_delay(0, RateLimited("x", retry_after=60), base=1.0, cap=10.0) == 10


def test_backoff_keeps_larger_jitter_over_retry_after(monkeypatch):
    monkeypatch.setattr(retry.random, "uniform", lambda lo, hi: hi)
    assert retry.backoff_delay(3, RateLimited("x", retry_after=1), base=1.0, cap=10.0) == 8


def test_describe_uses_class_name():
    assert retry.describe(Overloaded("busy")) == "Overloaded"


def _with_headers(headers):
    return SimpleNamespace(response=SimpleNamespace(headers=headers))


@pytest.mark.parametrize("exc, expected", [
    (_with_headers({"retry-after": "7"}), 7.0),
    (_with_headers({"retry-after": "1.5"}), 1.5),
    (_with_headers({}), None),
    (_with_headers({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), None),
    (SimpleNamespace(response=None), None),
    (ValueError("no response"), None),
])
def test_retry_after_header(exc, expected):
    assert retry_after(exc) == expected


def test_llm_error_keeps_message_and_retry_after():
    e = RateLimited("slow down", retry_after=3)
    assert str(e) == "slow down"
    assert e.retry_after == 3
