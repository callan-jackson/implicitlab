"""The insight layer: Azure OpenAI, with a deterministic path that always works.

The service contract is the same whether or not a model is reachable. Callers
get a summary, a note saying which engine produced it, the exact prompt that was
sent, and the result of the numeric verification. Nothing about the shape of the
response depends on the model being available, which is what makes the demo
safe to run in front of someone on conference wifi.

Failure policy, in order of preference:

1.  Model responds and every number in the output traces back to the payload →
    return the model's text.
2.  Model responds but the output contains a number that cannot be traced →
    return the deterministic summary, and return the model's text alongside it
    marked as rejected, with the offending figures listed. Hiding the rejection
    would defeat the point of checking.
3.  Model is unreachable, times out, or is not configured → return the
    deterministic summary and say so.

Never: return unverified model text as though it were checked.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from . import prompts
from .verify import VerificationReport, verify

log = logging.getLogger("implicitlab.llm")


@dataclass(slots=True)
class InsightResult:
    summary_markdown: str
    engine: str
    model: str | None
    latency_ms: float | None
    verification: dict
    prompt_shown: dict
    rejected_draft: str | None = None
    error: str | None = None
    usage: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary_markdown": self.summary_markdown,
            "engine": self.engine,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "verification": self.verification,
            "prompt_shown": self.prompt_shown,
            "rejected_draft": self.rejected_draft,
            "error": self.error,
            "usage": self.usage,
            "notes": self.notes,
        }


class InsightAgent:
    """Turns computed statistics into an executive summary."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None

    # -- lazy client so import never fails without credentials --------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.settings.llm_configured:
            return None
        try:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=1,
            )
            return self._client
        except Exception as exc:  # pragma: no cover - depends on environment
            log.warning("Azure OpenAI client could not be constructed: %s", exc)
            return None

    def generate(self, payload: dict) -> InsightResult:
        deterministic = prompts.fallback_summary(payload)
        prompt_shown = {
            "system": prompts.SYSTEM_PROMPT,
            "user": prompts.build_user_prompt(payload),
            "why_shown": (
                "The full prompt is returned with every response. If a model "
                "writes part of a research report, the instructions it was given "
                "are part of the method and belong in the audit trail."
            ),
        }

        client = self._get_client()
        if client is None:
            return InsightResult(
                summary_markdown=deterministic,
                engine="deterministic",
                model=None,
                latency_ms=None,
                verification=VerificationReport(
                    True, 0, [], "Deterministic engine: every figure is copied "
                                "directly from the computed statistics."
                ).to_dict(),
                prompt_shown=prompt_shown,
                notes=[
                    "No language model is configured, so the rules-based engine "
                    "produced this summary. Set AZURE_OPENAI_ENDPOINT and "
                    "AZURE_OPENAI_API_KEY to enable the model path."
                ],
            )

        started = time.perf_counter()
        try:
            kwargs = {
                "model": self.settings.azure_openai_deployment,
                "messages": [
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_shown["user"]},
                ],
                "max_completion_tokens": self.settings.llm_max_completion_tokens,
            }
            # Reasoning depth is wasted here: the arithmetic is already done and
            # the task is exposition. Asking for less of it roughly halves the
            # latency, which matters when someone is watching the page.
            if self.settings.llm_reasoning_effort:
                kwargs["reasoning_effort"] = self.settings.llm_reasoning_effort
            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("reasoning_effort", None)
                response = client.chat.completions.create(**kwargs)
            elapsed = (time.perf_counter() - started) * 1000
            text = (response.choices[0].message.content or "").strip()
            usage = None
            if getattr(response, "usage", None):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
        except Exception as exc:
            log.warning("Azure OpenAI call failed: %s", exc)
            return InsightResult(
                summary_markdown=deterministic,
                engine="deterministic",
                model=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                verification=VerificationReport(
                    True, 0, [], "Deterministic engine: every figure is copied "
                                "directly from the computed statistics."
                ).to_dict(),
                prompt_shown=prompt_shown,
                error=f"{type(exc).__name__}: {exc}",
                notes=["The model call failed, so the rules-based summary was "
                       "returned instead. The statistics are unaffected — they "
                       "are computed before the model is involved."],
            )

        if not text:
            return InsightResult(
                summary_markdown=deterministic,
                engine="deterministic",
                model=self.settings.azure_openai_deployment,
                latency_ms=elapsed,
                verification=VerificationReport(
                    True, 0, [], "Deterministic engine."
                ).to_dict(),
                prompt_shown=prompt_shown,
                error="Model returned an empty completion.",
                notes=["The model returned nothing usable; the rules-based "
                       "summary was returned instead."],
            )

        report = verify(text, payload)

        if report.verified:
            return InsightResult(
                summary_markdown=text,
                engine="azure-openai",
                model=self.settings.azure_openai_deployment,
                latency_ms=elapsed,
                verification=report.to_dict(),
                prompt_shown=prompt_shown,
                usage=usage,
                notes=[
                    f"{report.checked} numeric claims in this summary were "
                    f"checked against the computed statistics and all matched."
                ],
            )

        return InsightResult(
            summary_markdown=deterministic,
            engine="deterministic (model output rejected)",
            model=self.settings.azure_openai_deployment,
            latency_ms=elapsed,
            verification=report.to_dict(),
            prompt_shown=prompt_shown,
            rejected_draft=text,
            usage=usage,
            notes=[
                "The model's draft contained at least one figure that does not "
                "appear in the computed statistics, so it was not published. "
                "The rules-based summary was returned instead and the rejected "
                "draft is included for inspection."
            ],
        )
