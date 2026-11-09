import pytest

import chat.chat_client


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Replace time.sleep in the retry loop; the list collects each requested delay."""
    calls: list[float] = []
    monkeypatch.setattr(chat.chat_client.time, "sleep", calls.append)
    return calls
