"""Minimal OpenRouter chat client built for long annotation runs.

Annotating the ViHoRec sheets means several hundred sequential requests per
model, so the parts that matter here are the boring ones: retrying transient
failures instead of losing a half-finished run, surfacing authentication errors
immediately rather than retrying them, and reporting token usage so the cost of
a run is known before it is repeated.

Three OpenRouter behaviours are handled explicitly because they are easy to miss:

* A failed request can still return HTTP 200 with an ``error`` object in the
  body, so the status code alone is not a success check.
* Some reasoning models reject ``temperature``. A 400 naming that parameter is
  retried once without it rather than aborting the run.
* Reasoning tokens are billed as completion tokens and can consume the whole
  ``max_tokens`` budget, leaving empty content. That is reported as an error
  with a usable message instead of an empty annotation.

The API key is read from the ``OPENROUTER_API_KEY`` environment variable and is
never written to disk or into any output file.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """Raised when a request cannot be completed after all retries."""


class AuthError(OpenRouterError):
    """Raised on 401/403. Retrying will not help, so the run stops."""


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


class OpenRouterClient:
    """One client per model.

    Args:
        model: OpenRouter model slug, e.g. ``openai/gpt-5-mini``.
        api_key: Overrides ``OPENROUTER_API_KEY`` when given.
        referer / title: Optional attribution headers used for OpenRouter
            rankings. ``X-Title`` is the legacy spelling of ``X-OpenRouter-Title``.
        temperature: ``None`` omits the parameter, which is what some reasoning
            models require.
        top_p: Sent only when set. Some vendors tune their chat template around
            a specific value and degrade under framework defaults.
        seed: Best-effort determinism. Useful where a vendor recommends a high
            temperature, since the sampler can then still be pinned.
        reasoning_effort: Sent only when set. ``"low"`` is appropriate for the
            binary judgements in this package and keeps the cost down. Valid
            values differ per model family, so the caller must validate.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        referer: str = "https://github.com/vihorec/vihorec",
        title: str = "ViHoRec annotation",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        require_parameters: bool = True,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise AuthError(
                "OPENROUTER_API_KEY is not set.\n"
                '  PowerShell: $env:OPENROUTER_API_KEY = "sk-or-..."\n'
                '  bash:       export OPENROUTER_API_KEY="sk-or-..."'
            )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.require_parameters = require_parameters

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-OpenRouter-Title": title,
            }
        )

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.n_requests = 0

    # -- payload ---------------------------------------------------------

    def _payload(self, messages: list[dict], response_format: dict | None,
                 drop_temperature: bool, seed: int | None) -> dict:
        body: dict = {
            "model": self.model,
            "messages": messages,
            # Ask OpenRouter to report credits spent so a run can be costed.
            "usage": {"include": True},
        }
        if response_format:
            body["response_format"] = response_format
        if self.require_parameters:
            # A model listing `response_format` as supported says nothing about
            # the provider OpenRouter happens to route to. Without this, a
            # provider that ignores the schema is allowed to serve the request,
            # and an unconstrained sampler can wander off mid-object.
            body["provider"] = {"require_parameters": True}
        if self.temperature is not None and not drop_temperature:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if seed is not None:
            body["seed"] = seed
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    # -- request ---------------------------------------------------------

    def chat(self, messages: list[dict], *, response_format: dict | None = None,
             seed: int | None = None) -> ChatResult:
        """Send one chat completion and return the assistant text.

        Args:
            seed: Overrides the client default for this call. Needed when
                retrying a degenerate sample, since replaying the same seed
                would reproduce it exactly.

        Raises:
            AuthError: on 401/403, immediately.
            OpenRouterError: when every retry has been exhausted.
        """
        call_seed = self.seed if seed is None else seed
        drop_temperature = False
        last_error = "unknown error"

        for attempt in range(self.max_retries):
            if attempt:
                time.sleep(self._backoff(attempt))
            try:
                resp = self._session.post(
                    API_URL,
                    json=self._payload(messages, response_format, drop_temperature, call_seed),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                continue

            if resp.status_code in (401, 403):
                raise AuthError(f"{resp.status_code} from OpenRouter: {resp.text[:300]}")

            if resp.status_code == 400 and "temperature" in resp.text.lower():
                # Reasoning models reject the parameter; drop it and try again.
                if not drop_temperature:
                    drop_temperature = True
                    last_error = "model rejected 'temperature'; retrying without it"
                    continue

            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self._sleep_for_retry_after(resp)
                continue

            if resp.status_code != 200:
                raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:500]}")

            try:
                data = resp.json()
            except ValueError:
                last_error = f"non-JSON response: {resp.text[:200]}"
                continue

            # A 200 can still carry an error object.
            if isinstance(data.get("error"), dict):
                err = data["error"]
                code = err.get("code")
                last_error = f"api error {code}: {err.get('message', '')[:200]}"
                if code in (401, 403):
                    raise AuthError(last_error)
                if code in RETRY_STATUS:
                    continue
                raise OpenRouterError(last_error)

            result = self._parse(data)
            if not result.text.strip():
                # Usually the reasoning budget consumed every allowed token.
                last_error = (
                    f"empty content (finish_reason={result.finish_reason}, "
                    f"completion_tokens={result.completion_tokens}); "
                    "raise --max-tokens or lower --reasoning-effort"
                )
                continue

            self._accumulate(result)
            return result

        raise OpenRouterError(
            f"{self.model}: giving up after {self.max_retries} attempts. Last error: {last_error}"
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with jitter, capped so a run cannot stall."""
        return min(2.0 ** attempt, 30.0) * (0.5 + random.random() / 2)

    def _sleep_for_retry_after(self, resp: requests.Response) -> None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return
        try:
            time.sleep(min(float(raw), 60.0))
        except ValueError:
            pass

    def _parse(self, data: dict) -> ChatResult:
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            text=message.get("content") or "",
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost=usage.get("cost"),
            finish_reason=choices[0].get("finish_reason"),
            raw=data,
        )

    def _accumulate(self, result: ChatResult) -> None:
        self.n_requests += 1
        self.total_prompt_tokens += result.prompt_tokens
        self.total_completion_tokens += result.completion_tokens
        if result.cost:
            self.total_cost += float(result.cost)

    def usage_summary(self) -> dict:
        return {
            "model": self.model,
            "requests": self.n_requests,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "cost_usd": round(self.total_cost, 4) if self.total_cost else None,
            "sent_params": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
            },
        }
