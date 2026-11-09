"""Pricing file loading and cost calculation."""
import pytest

from usage import ModelPricing, Price, RequestUsage, UsageTracker, load_pricing

BASE = Price(input=1.0, output=2.0, cache_write=1.25, cache_read=0.1)
LONG = Price(input=2.0, output=4.0, cache_write=2.5, cache_read=0.2)
PRICING = {
    "flat": ModelPricing(base=BASE),
    "tiered": ModelPricing(base=BASE, long_context=LONG, long_context_threshold=1000),
}


def test_prompt_tokens_includes_cache_tokens():
    u = RequestUsage("flat", input_tokens=10, cache_creation_input_tokens=20, cache_read_input_tokens=30)
    assert u.prompt_tokens == 60


def test_cost_of_unknown_model_is_none():
    assert UsageTracker(PRICING).cost_of(RequestUsage("nope", input_tokens=10)) is None


def test_cost_of_adds_all_token_kinds():
    u = RequestUsage("flat", input_tokens=1_000_000, output_tokens=1_000_000,
                     cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    cost, _ = UsageTracker(PRICING).cost_of(u)
    assert cost == pytest.approx(1.0 + 2.0 + 1.25 + 0.1)


def test_cache_savings():
    # Reads save input - cache_read; writes cost cache_write - input extra.
    u = RequestUsage("flat", cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    _, savings = UsageTracker(PRICING).cost_of(u)
    assert savings == pytest.approx((1.0 - 0.1) - (1.25 - 1.0))


@pytest.mark.parametrize("prompt, expected_input_price", [(1000, 1.0), (1001, 2.0)])
def test_long_context_price_applies_above_threshold(prompt, expected_input_price):
    u = RequestUsage("tiered", input_tokens=prompt)
    cost, _ = UsageTracker(PRICING).cost_of(u)
    assert cost == pytest.approx(prompt * expected_input_price / 1_000_000)


def test_long_context_threshold_counts_cache_tokens():
    u = RequestUsage("tiered", input_tokens=10, cache_read_input_tokens=1000)
    cost, _ = UsageTracker(PRICING).cost_of(u)
    assert cost == pytest.approx((10 * 2.0 + 1000 * 0.2) / 1_000_000)


def test_record_and_total():
    tracker = UsageTracker(PRICING)
    tracker.record(RequestUsage("flat", input_tokens=100, output_tokens=10))
    tracker.record(RequestUsage("flat", input_tokens=200, output_tokens=20))
    tracker.record(RequestUsage("nope", input_tokens=5, output_tokens=1))

    flat = tracker.by_model["flat"]
    assert (flat.requests, flat.input_tokens, flat.output_tokens, flat.priced) == (2, 300, 30, True)
    assert tracker.by_model["nope"].priced is False

    total = tracker.total()
    assert (total.requests, total.input_tokens, total.output_tokens) == (3, 305, 31)
    assert total.cost == pytest.approx((300 * 1.0 + 30 * 2.0) / 1_000_000)
    assert total.priced is False


def test_load_pricing_reads_base_and_long_context(tmp_path):
    f = tmp_path / "pricing.toml"
    f.write_text(
        '[models."m.1"]\ninput = 1\noutput = 2\ncache_write = 3\ncache_read = 4\nlong_context_threshold = 10\n'
        '[models."m.1".long_context]\ninput = 5\noutput = 6\ncache_write = 7\ncache_read = 8\n'
    )
    mp = load_pricing(f)["m.1"]
    assert mp.base == Price(1, 2, 3, 4)
    assert mp.long_context == Price(5, 6, 7, 8)
    assert mp.long_context_threshold == 10


def test_load_pricing_missing_price_names_model_and_field(tmp_path):
    f = tmp_path / "pricing.toml"
    f.write_text("[models.m]\ninput = 1\noutput = 2\ncache_write = 3\n")
    with pytest.raises(ValueError, match="'m' is missing price 'cache_read'"):
        load_pricing(f)


def test_project_pricing_file_is_valid():
    pricing = load_pricing()
    for model in ("claude-opus-5-5", "gpt-6.1-sol", "llama3.2", "gpt-oss:20b"):
        assert model in pricing
    assert pricing["llama3.2"].base == Price(0, 0, 0, 0)
