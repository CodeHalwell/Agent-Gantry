"""
Shared TypeSafe Jev client, used by the Jev reranker and the Jev selector.

Jev is a "System One" model: it takes unstructured state plus typed questions
and returns typed answers with calibrated probabilities. It does not generate
text. Selection uses one ``Noul`` (a yes/no question answered with a
probability between 0 and 1) per candidate rather than a single ``Choice``,
because a prompt routinely needs several tools and ``Choice`` returns exactly
one option.

Two documented limits shape everything here:

* Every question in a request shares one state and is evaluated independently,
  so batching costs no accuracy — but the request as a whole is bounded by a
  token budget the docs put at "around 32,000 tokens". Candidates are
  therefore chunked into several requests, dispatched concurrently.
* Accuracy falls as the state grows with content unrelated to the decision, so
  the query goes in ``state`` and each candidate's own text goes in that
  question's ``instructions``. Putting the whole catalogue in the state would
  be both over budget and less accurate.

The SDK is imported lazily, so ``import agent_gantry`` works without the
``jev`` extra installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.selection import SelectionCandidate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Headroom under the documented ~32k per-request budget. The estimate below is
# approximate, and a request rejected for being over budget costs a whole
# round-trip, so the batches aim well under the ceiling.
DEFAULT_TOKEN_BUDGET = 24_000

# Crude but stable: the point is to keep batches under a ceiling, not to
# predict billing. Deliberately pessimistic (real tokenisers average nearer
# 4 characters per token for English prose, more for JSON punctuation).
_CHARS_PER_TOKEN = 3.5

# Jev answers in 70-500ms, so several batches in flight is the difference
# between one round-trip and ten. Bounded so a large catalogue cannot open an
# unbounded number of sockets.
DEFAULT_MAX_CONCURRENCY = 8

# Descriptions are truncated before they are sent. TypeSafe's own skill-
# selection recipe reads short index descriptions in the wide pass and longer
# excerpts only for the shortlist; this is the same trade, and it keeps a
# single verbose tool from consuming a batch on its own.
DEFAULT_MAX_DESCRIPTION_CHARS = 500

# The documented ceiling on `Choice` options. Not used for selection (which is
# one Noul per candidate) but exported so callers that want a single-pick
# question can check before building one.
CHOICE_MAX_CARDINALITY = 255


def candidate_payload(candidate: SelectionCandidate, max_description_chars: int) -> dict[str, Any]:
    """Render a candidate as the JSON body of a question's instructions.

    Args:
        candidate: The entry to describe.
        max_description_chars: Truncation budget for the description.

    Returns:
        A JSON object. Structured rather than a formatted string because Jev
        accepts JSON for instructions, and a labelled object reads less like
        prose the model might try to answer about as a whole.
    """
    description = candidate.description or ""
    if len(description) > max_description_chars:
        # Cut on the last space so the model never reads half a word.
        head = description[:max_description_chars]
        cut = head.rfind(" ")
        description = (head[:cut] if cut > max_description_chars // 2 else head) + "..."
    payload: dict[str, Any] = {"name": candidate.name}
    if description:
        payload["description"] = description
    if candidate.tags:
        payload["tags"] = list(candidate.tags)
    if candidate.examples:
        # Capped: examples are the strongest signal an entry carries, but a
        # tool may hold ten of them and they would otherwise dominate a batch's
        # token budget on their own.
        payload["examples"] = list(candidate.examples[:5])
    return payload


@dataclass
class JevVerdict:
    """What one selection pass produced.

    Attributes:
        scores: Candidate id -> probability in ``[0, 1]``. Empty when
            ``fallback`` is set.
        fallback: ``True`` when the model did not answer and the caller must
            fall back to its own ordering. Never raised as an exception: a
            selection layer that fails closed takes the agent's tools away.
        reason: Why the fallback happened, for logs and tests.
        input_tokens: Reported input tokens, summed across batches.
        output_tokens: Reported output tokens, summed across batches.
        requests: How many API requests the pass took.
    """

    scores: dict[str, float] = field(default_factory=dict)
    fallback: bool = False
    reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0


def estimate_tokens(value: Any) -> int:
    """Approximate the token cost of a JSON-serialisable payload."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            text = str(value)
    return int(len(text) / _CHARS_PER_TOKEN) + 1


class JevClient:
    """Thin async wrapper over ``typesafe_sdk`` for scoring candidates.

    Args:
        api_key: TypeSafe API key. Defaults to ``TYPESAFE_API_KEY``.
        model: Model id. Defaults to the SDK's own default (``jev-latest``).
        timeout: Per-request timeout in seconds. The SDK's own default is 10s,
            which is generous next to Jev's 70-500ms answer time; it is raised
            here only if the caller asks.
        token_budget: Ceiling for one request's estimated tokens.
        max_concurrency: How many batches may be in flight at once.
        max_description_chars: Per-candidate description budget.
        client: A pre-built ``AsyncTypeSafeClient``. Supplying one makes this
            wrapper testable without network, and lets a host share a
            connection pool. A supplied client is not closed by ``aclose()``.

    Raises:
        ImportError: If the ``typesafe-sdk`` package is not installed.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
        client: Any | None = None,
    ) -> None:
        self._token_budget = max(1, int(token_budget))
        self._max_concurrency = max(1, int(max_concurrency))
        self._max_description_chars = max(1, int(max_description_chars))
        self._model = model
        self._owns_client = client is None
        # Created lazily: constructing the SDK client builds an httpx client,
        # which binds to whichever loop is running at construction time.
        self._client = client
        self._client_lock: asyncio.Lock | None = None
        self._api_key = api_key
        self._timeout = timeout

        if client is None:
            # Import eagerly enough to fail at construction with a useful
            # message rather than on the first retrieval, but do not build the
            # client yet.
            self._require_sdk()

    @staticmethod
    def _require_sdk() -> Any:
        """Import ``typesafe_sdk``, or explain how to install it."""
        try:
            import typesafe_sdk
        except ImportError as exc:  # pragma: no cover - exercised via tests with a stub
            raise ImportError(
                "The typesafe-sdk package is not installed. Install it with:\n"
                "  pip install agent-gantry[jev]"
            ) from exc
        return typesafe_sdk

    async def _ensure_client(self) -> Any:
        """Return the SDK client, building it on first use."""
        if self._client is not None:
            return self._client
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            if self._client is not None:
                return self._client
            sdk = self._require_sdk()
            kwargs: dict[str, Any] = {}
            api_key = self._api_key or os.getenv("TYPESAFE_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key
            if self._model:
                kwargs["model"] = self._model
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            self._client = sdk.AsyncTypeSafeClient(**kwargs)
            return self._client

    def batch(
        self,
        state: Any,
        candidates: Sequence[SelectionCandidate],
        question: str,
    ) -> list[list[SelectionCandidate]]:
        """Split *candidates* into request-sized batches.

        The state is counted once per batch, because every question in a
        request is evaluated against it.

        Args:
            state: The shared state (typically the user's query).
            candidates: Candidates to split.
            question: The question asked about each candidate.

        Returns:
            Batches, each estimated to fit the token budget. A single candidate
            too large to fit is given a batch of its own rather than dropped.
        """
        overhead = estimate_tokens(state) + estimate_tokens(question)
        batches: list[list[SelectionCandidate]] = []
        current: list[SelectionCandidate] = []
        current_tokens = overhead
        for candidate in candidates:
            cost = estimate_tokens(candidate_payload(candidate, self._max_description_chars))
            if current and current_tokens + cost > self._token_budget:
                batches.append(current)
                current = []
                current_tokens = overhead
            current.append(candidate)
            current_tokens += cost
        if current:
            batches.append(current)
        return batches

    async def score(
        self,
        state: Any,
        candidates: Sequence[SelectionCandidate],
        question: str,
    ) -> JevVerdict:
        """Score every candidate against *state* with one ``Noul`` each.

        Args:
            state: Shared state for the request — the user's query.
            candidates: Candidates to score.
            question: The yes/no question asked of each candidate, e.g.
                "Would this tool help answer the request?".

        Returns:
            A :class:`JevVerdict`. On any API failure the verdict has
            ``fallback=True`` and no scores: this never raises, because a
            selection layer that fails closed hands the agent an empty tool
            list, which is worse than an unranked one.
        """
        if not candidates:
            return JevVerdict()

        batches = self.batch(state, candidates, question)
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run(batch: Sequence[SelectionCandidate]) -> Any:
            async with semaphore:
                return await self._score_batch(state, batch, question)

        results = await asyncio.gather(*(run(batch) for batch in batches), return_exceptions=True)

        verdict = JevVerdict(requests=len(batches))
        for result in results:
            if isinstance(result, BaseException):
                # One bad batch poisons the whole pass rather than yielding a
                # partial ranking. A candidate that was never scored is
                # indistinguishable from one scored zero, so keeping the
                # successful half would silently bury whatever was in the
                # batch that failed — a worse outcome than not ranking at all.
                if isinstance(result, asyncio.CancelledError):
                    raise result
                logger.warning(f"Jev selection failed, falling back: {result}")
                return JevVerdict(
                    fallback=True,
                    reason=f"{type(result).__name__}: {result}",
                    requests=len(batches),
                )
            scores, usage_in, usage_out = result
            verdict.scores.update(scores)
            verdict.input_tokens += usage_in
            verdict.output_tokens += usage_out
        return verdict

    async def _score_batch(
        self,
        state: Any,
        batch: Sequence[SelectionCandidate],
        question: str,
    ) -> tuple[dict[str, float], int, int]:
        """Send one request and read its noul answers back."""
        sdk = self._require_sdk()
        client = await self._ensure_client()

        # Positional keys rather than the candidates' own ids: an id is the
        # caller's, and may hold characters or lengths the API will not take
        # as a question name.
        keys = {f"q{index}": candidate for index, candidate in enumerate(batch)}
        questions = {
            key: sdk.Noul(
                instructions={
                    "question": question,
                    "candidate": candidate_payload(candidate, self._max_description_chars),
                }
            )
            for key, candidate in keys.items()
        }

        response = await client.system_one(state=state, questions=questions)

        scores: dict[str, float] = {}
        for key, candidate in keys.items():
            answer = response.answers.get(key)
            value = getattr(answer, "noul", None) if answer is not None else None
            if value is None:
                # A missing or non-noul answer is not a failure of the pass;
                # treat it as "no signal" so the candidate keeps its incoming
                # order rather than being ranked last on a technicality.
                continue
            scores[candidate.id] = float(value)

        usage = getattr(response, "usage", None)
        return (
            scores,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

    async def aclose(self) -> None:
        """Release the SDK client, if this wrapper built it."""
        client = self._client
        if client is None or not self._owns_client:
            return
        self._client = None
        aclose = getattr(client, "aclose", None)
        if aclose is None:  # pragma: no cover - defensive
            return
        try:
            await aclose()
        except Exception:  # pragma: no cover - shutdown is best effort
            logger.debug("Error closing the TypeSafe client", exc_info=True)
