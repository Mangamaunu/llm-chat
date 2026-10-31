import tomllib
from dataclasses import dataclass
from pathlib import Path

# pricing.toml lives in the project root, one level above src/.
PRICING_FILE = Path(__file__).resolve().parent.parent / "pricing.toml"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""
    input: float
    output: float
    cache_write: float
    cache_read: float


@dataclass(frozen=True)
class ModelPricing:
    base: Price
    long_context: Price | None = None
    long_context_threshold: int | None = None

    def for_prompt(self, prompt_tokens: int) -> Price:
        if (
            self.long_context is not None
            and self.long_context_threshold is not None
            and prompt_tokens > self.long_context_threshold
        ):
            return self.long_context
        return self.base


def _parse_price(model: str, table: dict) -> Price:
    try:
        return Price(
            input=float(table["input"]),
            output=float(table["output"]),
            cache_write=float(table["cache_write"]),
            cache_read=float(table["cache_read"]),
        )
    except KeyError as e:
        raise ValueError(f"{PRICING_FILE.name}: model '{model}' is missing price '{e.args[0]}'") from None


def load_pricing(path: Path = PRICING_FILE) -> dict[str, ModelPricing]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    pricing = {}
    for model, table in data.get("models", {}).items():
        long_table = table.get("long_context")
        pricing[model] = ModelPricing(
            base=_parse_price(model, table),
            long_context=_parse_price(f"{model}.long_context", long_table) if long_table else None,
            long_context_threshold=table.get("long_context_threshold"),
        )
    return pricing


@dataclass
class RequestUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


@dataclass
class ModelUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost: float = 0.0
    cache_savings: float = 0.0
    priced: bool = True  # False if the model is missing from the pricing config

    def add(self, other: "ModelUsage") -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cost += other.cost
        self.cache_savings += other.cache_savings
        self.priced = self.priced and other.priced


class UsageTracker:
    """Accumulates token usage and cost for the current session, per model."""

    def __init__(self, pricing: dict[str, ModelPricing] | None = None):
        self.pricing = load_pricing() if pricing is None else pricing
        self.by_model: dict[str, ModelUsage] = {}

    def cost_of(self, req: RequestUsage) -> tuple[float, float] | None:
        """Return (cost, cache savings) in USD, or None if the model has no configured price."""
        mp = self.pricing.get(req.model)
        if mp is None:
            return None
        p = mp.for_prompt(req.prompt_tokens)
        m = 1_000_000
        cost = (
            req.input_tokens * p.input
            + req.output_tokens * p.output
            + req.cache_creation_input_tokens * p.cache_write
            + req.cache_read_input_tokens * p.cache_read
        ) / m
        # Compared with sending the same prompt without caching:
        # reads save (input - cache_read), writes cost (cache_write - input) extra.
        savings = (
            req.cache_read_input_tokens * (p.input - p.cache_read)
            - req.cache_creation_input_tokens * (p.cache_write - p.input)
        ) / m
        return cost, savings

    def record(self, req: RequestUsage) -> None:
        mu = self.by_model.setdefault(req.model, ModelUsage())
        mu.requests += 1
        mu.input_tokens += req.input_tokens
        mu.output_tokens += req.output_tokens
        mu.cache_creation_input_tokens += req.cache_creation_input_tokens
        mu.cache_read_input_tokens += req.cache_read_input_tokens
        priced = self.cost_of(req)
        if priced is None:
            mu.priced = False
        else:
            mu.cost += priced[0]
            mu.cache_savings += priced[1]

    def total(self) -> ModelUsage:
        t = ModelUsage()
        for mu in self.by_model.values():
            t.add(mu)
        return t
