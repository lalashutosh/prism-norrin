"""
core/llm_client.py
──────────────────
Universal LLM client adapter for Prism.

All Prism agents call the LLM using Anthropic's wire format:

    response = client.messages.create(
        model="...",
        max_tokens=N,
        system="...",
        messages=[{"role": "user", "content": "..."}],
    )
    text = response.content[0].text

This file provides two things:

  1.  OpenAIAdapter — wraps any openai.OpenAI-compatible client (OpenAI,
      Gemini, Azure, Ollama, etc.) so it speaks Anthropic's interface.
      Pass it as ``llm_client=`` to the Orchestrator and all agents work
      without any other changes.

  2.  make_llm_client(provider, model, api_key, base_url) — convenience
      factory that returns the right client object for a given provider.

══════════════════════════════════════════════════════════════════════════════
USAGE
══════════════════════════════════════════════════════════════════════════════

  # Gemini via the OpenAI-compatible endpoint
  from core.llm_client import make_llm_client
  llm = make_llm_client("gemini", model="gemini-2.0-flash")

  # Native Anthropic (default when no provider is given)
  llm = make_llm_client("anthropic")

  # Any OpenAI-compatible endpoint (Ollama, LM Studio, etc.)
  llm = make_llm_client(
      "openai_compat",
      model="llama3.2",
      base_url="http://localhost:11434/v1",
      api_key="ollama",
  )

  # Inject into Orchestrator
  orchestrator = Orchestrator(retrieve_fn=..., llm_client=llm)
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# ── LLM availability tracking ─────────────────────────────────────────────────
# Set when the last LLM call succeeded; cleared when a connection-level error
# occurs.  Allows the API server and UI to surface "LLM unavailable" without
# having to know about the underlying HTTP transport.

_llm_connected = threading.Event()
_llm_connected.set()   # optimistically available on startup

# How long to wait between retries when the LLM is unreachable (seconds).
_RETRY_INTERVAL_S: int = 5


def is_llm_available() -> bool:
    """Return True if the last LLM call succeeded (or no call has been made yet)."""
    return _llm_connected.is_set()


def _is_retriable(exc: Exception) -> bool:
    """Return True when *exc* indicates a transient LLM unavailability.

    Specifically matches connection-level errors (server down, network hiccup)
    and server-side 5xx errors.  Authentication / bad-request errors are NOT
    retriable and propagate immediately.
    """
    # openai-specific errors (most common path for OpenAI-compat endpoints)
    try:
        import openai  # noqa: PLC0415
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
        if isinstance(exc, openai.InternalServerError):
            return True   # 500 = server overload / restart; usually recovers
        # 429 rate-limit: retriable in principle, but not a detach scenario —
        # leave it to propagate so the caller surfaces the limit clearly.
    except ImportError:
        pass

    # Plain Python / httpx errors (propagate from some openai versions)
    if isinstance(exc, (ConnectionRefusedError, ConnectionError, OSError)):
        return True

    # String-based fallback for any transport library
    name = type(exc).__name__
    return any(kw in name for kw in ("Connection", "Timeout", "Connect", "Network"))


# ── Response shim ─────────────────────────────────────────────────────────────

class _ContentBlock:
    """Mimics anthropic.types.ContentBlock — just needs a .text attribute."""
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _MessageResponse:
    """Mimics anthropic.types.Message — agents read .content[0].text."""
    __slots__ = ("content",)

    def __init__(self, text: str) -> None:
        self.content = [_ContentBlock(text)]


# ── Messages namespace shim ───────────────────────────────────────────────────

class _MessagesNamespace:
    """Exposes .create(**kwargs) with Anthropic's signature over an OpenAI client."""

    def __init__(
        self,
        openai_client: Any,
        model: str,
        max_output_tokens: Optional[int] = None,
        disable_thinking:  bool          = False,
    ) -> None:
        self._client            = openai_client
        self._model             = model
        self._max_output_tokens = max_output_tokens
        self._disable_thinking  = disable_thinking

    def create(
        self,
        *,
        model: str,          # agent-supplied model name; overridden by self._model
        max_tokens: int,
        system: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> _MessageResponse:
        """Convert Anthropic-style call → OpenAI chat completion → Anthropic-style response.

        The ``model`` argument passed by the agent is deliberately ignored so
        that the operator-configured model (self._model) is always used.

        Qwen3 / DeepSeek-R1 thinking-mode handling
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Some models (Qwen3, DeepSeek-R1) put all tokens into a ``<think>``
        reasoning block and return an empty ``message.content``.  When
        ``disable_thinking=True`` we inject vLLM's ``enable_thinking=False``
        via ``extra_body`` to suppress this at the server level.  We also
        strip any residual ``<think>…</think>`` tags and fall back to
        ``reasoning_content`` so the pipeline always gets usable text.
        """
        if self._max_output_tokens is not None:
            max_tokens = min(max_tokens, self._max_output_tokens)

        oai_messages = [{"role": "system", "content": system}] + list(messages)

        if self._disable_thinking:
            # vLLM exposes this via extra_body for Qwen3 / other reasoning models.
            existing = kwargs.pop("extra_body", {}) or {}
            kwargs["extra_body"] = {
                **existing,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        # Retry loop: on connection-level errors the pipeline thread waits until
        # the LLM server comes back instead of failing immediately.  This lets
        # the user detach and reattach the local LLM mid-run without losing work.
        # Non-retriable errors (auth, bad request) propagate on the first raise.
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=oai_messages,
                    **kwargs,
                )
            except Exception as exc:
                if _is_retriable(exc):
                    _llm_connected.clear()   # signal "unavailable" to status watchers
                    time.sleep(_RETRY_INTERVAL_S)
                    continue                  # retry indefinitely until LLM returns
                raise                        # non-retriable: propagate immediately

            # Successful response — mark LLM as available again.
            _llm_connected.set()

            msg  = response.choices[0].message
            text = msg.content or ""

            # Fallback: some vLLM builds expose thinking tokens in reasoning_content
            # while leaving content empty.  Use reasoning_content as last resort.
            if not text.strip():
                text = getattr(msg, "reasoning_content", None) or ""

            # Strip residual <think>…</think> blocks (e.g. when disable_thinking
            # is not supported by the server version).
            text = _THINK_RE.sub("", text).strip()

            return _MessageResponse(text)


# ── Public adapter ────────────────────────────────────────────────────────────

class OpenAIAdapter:
    """Wraps an openai.OpenAI-compatible client to look like anthropic.Anthropic.

    Any Prism agent that does ``client.messages.create(...)`` will work with
    this adapter transparently.

    Parameters
    ──────────
    client            An ``openai.OpenAI`` (or compatible) client instance.
    model             The model name to use for every completion request, e.g.
                      ``"gemini-2.0-flash"``, ``"gpt-4o"``, ``"llama3.2"``.
    max_output_tokens Hard cap on the ``max_tokens`` value passed by any agent.
                      Useful for small-context local models (e.g. 2048-token
                      context) where agent defaults like 4096 / 8192 would
                      trigger a BadRequestError.  None = no cap (default).
    """

    def __init__(
        self,
        client: Any,
        model: str,
        max_output_tokens: Optional[int] = None,
        disable_thinking:  bool          = False,
    ) -> None:
        self._client  = client
        self._model   = model
        self.messages = _MessagesNamespace(client, model, max_output_tokens, disable_thinking)


# ── Factory ───────────────────────────────────────────────────────────────────

_GEMINI_BASE_URL = "http://localhost:8000/v1"


def make_llm_client(
    provider: str = "anthropic",
    *,
    model:             Optional[str] = None,
    api_key:           Optional[str] = None,
    base_url:          Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    disable_thinking:  bool          = False,
) -> Any:
    """Create a Prism-compatible LLM client for the given provider.

    Returns a native ``anthropic.Anthropic()`` for the "anthropic" provider,
    or an ``OpenAIAdapter`` wrapping an ``openai.OpenAI`` client for all
    OpenAI-compatible providers.

    Parameters
    ──────────
    provider          One of: "anthropic" | "gemini" | "openai" | "openai_compat".
                      Default: "anthropic".
    model             Model name.  Defaults per provider:
                        anthropic    → "claude-sonnet-4-6"
                        gemini       → "gemini-2.0-flash"
                        openai       → "gpt-4o"
                        openai_compat → required; no default
    api_key           API key.  Falls back to ANTHROPIC_API_KEY or OPENAI_API_KEY
                      environment variables as appropriate.
    base_url          Custom base URL.  Required for "openai_compat"; optional for
                      others (useful for proxies).
    max_output_tokens Hard cap on max_tokens for every agent call.  Pass this when
                      using a small-context local model (e.g. 400 for a 2048-token
                      model) so agent defaults like 4096 / 8192 are silently clamped.
                      Ignored for the native Anthropic provider.

    Raises
    ──────
    ImportError  if the required package is not installed.
    ValueError   if required arguments are missing.
    """
    provider = provider.lower().strip()

    # ── Native Anthropic ─────────────────────────────────────────────────────
    if provider == "anthropic":
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for provider='anthropic'. "
                "Run: pip install anthropic"
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    # ── OpenAI-compatible providers ──────────────────────────────────────────
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openai package is required for OpenAI-compatible providers. "
            "Run: pip install openai"
        ) from exc

    if provider == "gemini":
        _model   = model    or "gemini-2.0-flash"
        _api_key = api_key  or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        _base    = base_url or _GEMINI_BASE_URL
        client   = openai.OpenAI(api_key=_api_key, base_url=_base)
        return OpenAIAdapter(client, _model, max_output_tokens=max_output_tokens, disable_thinking=disable_thinking)

    if provider == "openai":
        _model   = model   or "gpt-4o"
        _api_key = api_key or os.environ.get("OPENAI_API_KEY")
        kwargs: dict = {}
        if _api_key:
            kwargs["api_key"] = _api_key
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)
        return OpenAIAdapter(client, _model, max_output_tokens=max_output_tokens, disable_thinking=disable_thinking)

    if provider == "openai_compat":
        if not model:
            raise ValueError("model= is required for provider='openai_compat'.")
        _api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        kwargs = {"api_key": _api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)
        return OpenAIAdapter(client, model, max_output_tokens=max_output_tokens, disable_thinking=disable_thinking)

    raise ValueError(
        f"Unknown provider {provider!r}. "
        "Choose from: 'anthropic', 'gemini', 'openai', 'openai_compat'."
    )
