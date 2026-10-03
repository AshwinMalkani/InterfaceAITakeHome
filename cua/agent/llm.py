"""The model behind discovery, behind a narrow protocol so the loop can be tested with a script.

Real model: Claude via the Anthropic SDK (`claude-opus-5-5` by default; override with CUA_MODEL).
- Thinking stays on (it can't be disabled on this model); `effort` is set explicitly because its
  default changed between model versions.
- Tools are `strict`, and tool choice stays `auto` (forcing a tool is rejected on this model); the
  system prompt says which tools to use.
- The caller keeps history append-only: every response's full content (thinking blocks included) is
  appended unchanged, which this model requires for thinking to stay valid.
- Prompt caching on the stable prefix (system + tools), and server-side refusal fallback
  (`fallbacks: "default"`) so a declined request is retried on Anthropic's recommended model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam, BetaToolUnionParam

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class ModelTurn:
    content: list[Any]  # content blocks, appended to history unchanged
    stop_reason: str | None
    usage: Usage
    model: str  # which model actually served it (differs if a fallback ran)


class Model(Protocol):
    name: str

    def respond(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> ModelTurn: ...


class AnthropicModel:
    def __init__(self, name: str | None = None, *, effort: str = "medium", max_tokens: int = 16_000) -> None:
        import anthropic  # imported lazily: replay and tests never need the SDK or a key

        self.name = name or os.environ.get("CUA_MODEL", DEFAULT_MODEL)
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def respond(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> ModelTurn:
        response = self._client.beta.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            tools=cast("list[BetaToolUnionParam]", tools),
            messages=cast("list[BetaMessageParam]", messages),
            output_config=cast("BetaOutputConfigParam", {"effort": self.effort}),
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        usage = response.usage
        return ModelTurn(
            content=list(response.content),
            stop_reason=response.stop_reason,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            ),
            model=response.model,
        )
