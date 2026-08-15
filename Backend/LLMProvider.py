import json
import logging
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH, override=True)
env_vars = dotenv_values(ENV_PATH)


def get_config(name: str, default: str = "") -> str:
    load_dotenv(ENV_PATH, override=True)
    live_env_vars = dotenv_values(ENV_PATH)
    os_value = os.getenv(name)
    if os_value is not None and os_value.strip():
        return os_value
    file_value = live_env_vars.get(name)
    if isinstance(file_value, str) and file_value.strip():
        return file_value
    return default


LLM_PROVIDER = get_config("LLM_PROVIDER", "lmstudio").lower()
LMSTUDIO_BASE_URL = get_config("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LMSTUDIO_REST_BASE_URL = get_config(
    "LMSTUDIO_REST_BASE_URL", "http://127.0.0.1:1234/api/v1"
).rstrip("/")
LMSTUDIO_MODEL = get_config("LMSTUDIO_MODEL", "qwen/qwen3.5-2b")
LMSTUDIO_TIMEOUT_SECONDS = int(get_config("LMSTUDIO_TIMEOUT_SECONDS", "120"))
LMSTUDIO_MAX_TOKENS = int(get_config("LMSTUDIO_MAX_TOKENS", "768"))
LMSTUDIO_REASONING = get_config("LMSTUDIO_REASONING", "off").lower()

OPENROUTER_BASE_URL = get_config("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_MODEL = get_config(
    "OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"
)
OPENROUTER_FALLBACK_MODEL = get_config(
    "OPENROUTER_FALLBACK_MODEL", "poolside/laguna-s-2.1:free"
)
OPENROUTER_API_KEY = get_config("OPENROUTER_API_KEY", "")
OPENROUTER_TIMEOUT_SECONDS = int(
    get_config("OPENROUTER_TIMEOUT_SECONDS", "120")
)
OPENROUTER_HTTP_REFERER = get_config("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_TITLE = get_config("OPENROUTER_APP_TITLE", "NEXA")

EMBEDDING_BASE_URL = get_config("EMBEDDING_BASE_URL", LMSTUDIO_BASE_URL).rstrip("/")
EMBEDDING_MODEL = get_config("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_API_KEY = get_config("EMBEDDING_API_KEY", "lm-studio")
EMBEDDING_TIMEOUT_SECONDS = int(get_config("EMBEDDING_TIMEOUT_SECONDS", "60"))
EMBEDDING_ALLOW_REMOTE = get_config("EMBEDDING_ALLOW_REMOTE", "false").lower() == "true"


class LocalLLMUnavailable(RuntimeError):
    pass


class EmbeddingUnavailable(RuntimeError):
    pass


def provider_status(timeout_seconds: float = 2.0) -> dict[str, object]:
    """Return a bounded live check of the configured primary and fallback LLMs."""
    if LLM_PROVIDER in {"openrouter", "openrouter_lmstudio"}:
        openrouter = openrouter_status(timeout_seconds, OPENROUTER_MODEL)
        openrouter_fallback = openrouter_status(timeout_seconds, OPENROUTER_FALLBACK_MODEL)
        if LLM_PROVIDER == "openrouter":
            return {
                "available": bool(openrouter.get("available") or openrouter_fallback.get("available")),
                "provider": "openrouter",
                "primary": openrouter,
                "fallback": openrouter_fallback,
                "model": OPENROUTER_MODEL,
                "fallback_model": OPENROUTER_FALLBACK_MODEL,
                "model_loaded": bool(openrouter.get("model_loaded") or openrouter_fallback.get("model_loaded")),
                "error": "" if openrouter.get("available") else str(openrouter.get("error") or ""),
            }
        lmstudio = lmstudio_status(timeout_seconds)
        return {
            "available": bool(
                openrouter.get("available")
                or openrouter_fallback.get("available")
                or lmstudio.get("available")
            ),
            "provider": "openrouter_lmstudio",
            "primary": openrouter,
            "openrouter_fallback": openrouter_fallback,
            "fallback": lmstudio,
            "model": OPENROUTER_MODEL,
            "fallback_model": OPENROUTER_FALLBACK_MODEL,
            "model_loaded": bool(
                openrouter.get("model_loaded")
                or openrouter_fallback.get("model_loaded")
                or lmstudio.get("model_loaded")
            ),
            "error": "" if openrouter.get("available") else str(openrouter.get("error") or ""),
        }
    return lmstudio_status(timeout_seconds)


def lmstudio_status(timeout_seconds: float = 2.0) -> dict[str, object]:
    """Return a bounded live check of the configured local LM Studio server."""
    try:
        _validate_local_server_url()
        response = requests.get(
            f"{LMSTUDIO_BASE_URL}/models",
            timeout=max(0.5, float(timeout_seconds)),
        )
        response.raise_for_status()
        models = [
            str(item.get("id") or "")
            for item in response.json().get("data", [])
            if isinstance(item, dict)
        ]
    except (LocalLLMUnavailable, requests.RequestException, TypeError, ValueError) as exc:
        return {
            "available": False,
            "model": LMSTUDIO_MODEL,
            "model_loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "provider": "lmstudio",
        "model": LMSTUDIO_MODEL,
        "model_loaded": LMSTUDIO_MODEL in models,
        "loaded_models": models,
        "error": "",
    }


def openrouter_status(timeout_seconds: float = 2.0, model: str | None = None) -> dict[str, object]:
    selected_model = model or OPENROUTER_MODEL
    if not OPENROUTER_API_KEY:
        return {
            "available": False,
            "provider": "openrouter",
            "model": selected_model,
            "model_loaded": False,
            "error": "OPENROUTER_API_KEY is not configured.",
        }
    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=_openrouter_headers(),
            timeout=max(0.5, float(timeout_seconds)),
        )
        response.raise_for_status()
        models = [
            str(item.get("id") or "")
            for item in response.json().get("data", [])
            if isinstance(item, dict)
        ]
    except (requests.RequestException, TypeError, ValueError) as exc:
        return {
            "available": False,
            "provider": "openrouter",
            "model": selected_model,
            "model_loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "provider": "openrouter",
        "model": selected_model,
        "model_loaded": selected_model in models,
        "loaded_models": models,
        "error": "",
    }


def _openrouter_headers() -> dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise LocalLLMUnavailable("OPENROUTER_API_KEY is not configured.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE
    return headers


def _validate_local_server_url() -> None:
    for server_url in (LMSTUDIO_BASE_URL, LMSTUDIO_REST_BASE_URL):
        hostname = urlparse(server_url).hostname
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise LocalLLMUnavailable(
                "LM Studio URLs must point to a local server "
                "(localhost or 127.0.0.1)."
            )


def _validate_embedding_url() -> None:
    if EMBEDDING_ALLOW_REMOTE:
        return
    hostname = urlparse(EMBEDDING_BASE_URL).hostname
    if hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise EmbeddingUnavailable(
            "Embedding URL must point to this computer unless "
            "EMBEDDING_ALLOW_REMOTE is set to true."
        )


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Create embeddings through an OpenAI-compatible embeddings endpoint."""
    cleaned_texts = [text.strip() for text in texts if text and text.strip()]
    if not cleaned_texts:
        return []

    _validate_embedding_url()
    selected_model = model or EMBEDDING_MODEL
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"
    payload = {
        "model": selected_model,
        "input": cleaned_texts,
    }

    try:
        response = requests.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            json=payload,
            headers=headers,
            timeout=EMBEDDING_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        data = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in data]
        if len(embeddings) != len(cleaned_texts) or any(not item for item in embeddings):
            raise EmbeddingUnavailable("The embedding endpoint returned incomplete data.")
        return [[float(value) for value in embedding] for embedding in embeddings]
    except requests.RequestException as exc:
        raise EmbeddingUnavailable(
            "Embedding server is not reachable at "
            f"{EMBEDDING_BASE_URL}. Load an embedding model such as "
            f"{selected_model} and start the local server."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise EmbeddingUnavailable("The embedding endpoint returned an unexpected response.") from exc


def lmstudio_generate(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.5,
    reasoning: str | None = None,
    max_output_tokens: int | None = None,
) -> str:
    _validate_local_server_url()
    selected_model = model or LMSTUDIO_MODEL
    payload = {
        "model": selected_model,
        "input": prompt,
        "system_prompt": system,
        "temperature": temperature,
        # Callers such as /research can request a larger bounded completion
        # without changing the intentionally concise normal-chat default.
        "max_output_tokens": max_output_tokens or LMSTUDIO_MAX_TOKENS,
        "stream": False,
        "reasoning": reasoning or LMSTUDIO_REASONING,
        "store": False,
    }

    try:
        response = requests.post(
            f"{LMSTUDIO_REST_BASE_URL}/chat",
            json=payload,
            timeout=LMSTUDIO_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        outputs = response.json().get("output", [])
        content = "".join(
            str(item.get("content", ""))
            for item in outputs
            if item.get("type") == "message"
        ).strip()
        if not content:
            raise LocalLLMUnavailable("LM Studio returned an empty response.")
        return content
    except requests.RequestException as exc:
        raise LocalLLMUnavailable(
            "LM Studio is not reachable at "
            f"{LMSTUDIO_REST_BASE_URL}. Load {selected_model} in LM Studio and start "
            "its local server."
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalLLMUnavailable("LM Studio returned an unexpected response.") from exc


def openrouter_generate(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.5,
    reasoning: str | None = None,
    max_output_tokens: int | None = None,
) -> str:
    selected_model = _openrouter_model(model)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if reasoning and reasoning != "off":
        payload["reasoning"] = {"enabled": True}
    if max_output_tokens is not None:
        # OpenRouter documents max_completion_tokens as the current completion
        # limit. Keeping this opt-in preserves existing normal-chat behaviour.
        payload["max_completion_tokens"] = max_output_tokens

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=_openrouter_headers(),
            timeout=OPENROUTER_TIMEOUT_SECONDS,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise LocalLLMUnavailable(_openrouter_http_error(response)) from exc
        content = str(response.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:
            raise LocalLLMUnavailable("OpenRouter returned an empty response.")
        return content
    except requests.RequestException as exc:
        raise LocalLLMUnavailable(
            f"OpenRouter is not reachable for {selected_model}: {type(exc).__name__}: {exc}"
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalLLMUnavailable("OpenRouter returned an unexpected response.") from exc


def generate_text(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.5,
    reasoning: str | None = None,
    max_output_tokens: int | None = None,
) -> str:
    if LLM_PROVIDER == "openrouter":
        try:
            return openrouter_generate(prompt, system, _openrouter_model(model), temperature, reasoning, max_output_tokens)
        except LocalLLMUnavailable as primary_exc:
            logger.warning("OpenRouter primary unavailable, trying OpenRouter fallback: %s", primary_exc)
            return openrouter_generate(prompt, system, OPENROUTER_FALLBACK_MODEL, temperature, reasoning, max_output_tokens)
    if LLM_PROVIDER == "openrouter_lmstudio":
        try:
            return openrouter_generate(prompt, system, _openrouter_model(model), temperature, reasoning, max_output_tokens)
        except LocalLLMUnavailable as primary_exc:
            logger.warning("OpenRouter primary unavailable, trying OpenRouter fallback: %s", primary_exc)
            try:
                return openrouter_generate(prompt, system, OPENROUTER_FALLBACK_MODEL, temperature, reasoning, max_output_tokens)
            except LocalLLMUnavailable as openrouter_exc:
                logger.warning("OpenRouter fallback unavailable, falling back to LM Studio: %s", openrouter_exc)
                openrouter_error = (
                    f"Primary OpenRouter ({OPENROUTER_MODEL}): {primary_exc} "
                    f"Fallback OpenRouter ({OPENROUTER_FALLBACK_MODEL}): {openrouter_exc}"
                )
            try:
                return lmstudio_generate(prompt, system, LMSTUDIO_MODEL, temperature, reasoning, max_output_tokens)
            except LocalLLMUnavailable as lmstudio_exc:
                raise LocalLLMUnavailable(
                    "OpenRouter primary, OpenRouter fallback, and LM Studio fallback are unavailable. "
                    f"{openrouter_error} LM Studio: {lmstudio_exc}"
                ) from lmstudio_exc
    if LLM_PROVIDER == "lmstudio":
        return lmstudio_generate(prompt, system, model or LMSTUDIO_MODEL, temperature, reasoning, max_output_tokens)
    raise LocalLLMUnavailable(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openrouter_lmstudio', 'openrouter', or 'lmstudio'."
    )


def stream_text(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.5,
    reasoning: str | None = None,
):
    """Yield provider stream events as decoded dictionaries."""
    if LLM_PROVIDER in {"openrouter", "openrouter_lmstudio"}:
        try:
            yield from openrouter_stream_text(prompt, system, _openrouter_model(model), temperature, reasoning)
            return
        except LocalLLMUnavailable as primary_exc:
            logger.warning("OpenRouter primary stream unavailable, trying OpenRouter fallback: %s", primary_exc)
            try:
                yield from openrouter_stream_text(prompt, system, OPENROUTER_FALLBACK_MODEL, temperature, reasoning)
                return
            except LocalLLMUnavailable as openrouter_exc:
                if LLM_PROVIDER == "openrouter":
                    raise LocalLLMUnavailable(
                        f"OpenRouter primary and fallback streams are unavailable. "
                        f"Primary: {primary_exc} Fallback: {openrouter_exc}"
                    ) from openrouter_exc
            if LLM_PROVIDER == "openrouter":
                raise
            logger.warning("OpenRouter fallback stream unavailable, falling back to LM Studio: %s", openrouter_exc)
            openrouter_error = (
                f"Primary OpenRouter ({OPENROUTER_MODEL}): {primary_exc} "
                f"Fallback OpenRouter ({OPENROUTER_FALLBACK_MODEL}): {openrouter_exc}"
            )
    if LLM_PROVIDER != "lmstudio" and LLM_PROVIDER != "openrouter_lmstudio":
        raise LocalLLMUnavailable(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openrouter_lmstudio', 'openrouter', or 'lmstudio'."
        )
    _validate_local_server_url()
    selected_model = model or LMSTUDIO_MODEL
    payload = {
        "model": selected_model,
        "input": prompt,
        "system_prompt": system,
        "temperature": temperature,
        "max_output_tokens": LMSTUDIO_MAX_TOKENS,
        "stream": True,
        "reasoning": reasoning or LMSTUDIO_REASONING,
        "store": False,
    }
    try:
        with requests.post(
            f"{LMSTUDIO_REST_BASE_URL}/chat",
            json=payload,
            timeout=LMSTUDIO_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Ignored malformed LM Studio stream event.")
    except requests.RequestException as lmstudio_exc:
        message = (
            "LM Studio is not reachable at "
            f"{LMSTUDIO_REST_BASE_URL}. Load {selected_model} in LM Studio and start "
            "its local server."
        )
        if LLM_PROVIDER == "openrouter_lmstudio" and "openrouter_exc" in locals():
            message = (
                "OpenRouter primary, OpenRouter fallback, and LM Studio fallback are unavailable. "
                f"{openrouter_error} LM Studio: {message}"
            )
        raise LocalLLMUnavailable(message) from lmstudio_exc


def openrouter_stream_text(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.5,
    reasoning: str | None = None,
):
    selected_model = _openrouter_model(model)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if reasoning and reasoning != "off":
        payload["reasoning"] = {"enabled": True}
    try:
        with requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=_openrouter_headers(),
            timeout=OPENROUTER_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise LocalLLMUnavailable(_openrouter_http_error(response)) from exc
            yield {"type": "message.start"}
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                    delta = payload.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        yield {"type": "message.delta", "content": content}
                except (json.JSONDecodeError, TypeError, IndexError):
                    logger.warning("Ignored malformed OpenRouter stream event.")
    except requests.RequestException as exc:
        raise LocalLLMUnavailable(
            f"OpenRouter is not reachable for {selected_model}: {type(exc).__name__}: {exc}"
        ) from exc


def _openrouter_model(model: str | None = None) -> str:
    """Keep legacy local-model hints from overriding the configured OpenRouter model."""
    if not model or model == LMSTUDIO_MODEL:
        return OPENROUTER_MODEL
    return model


def _openrouter_http_error(response: requests.Response) -> str:
    detail = ""
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            detail = str(error.get("message") or "")
        elif error:
            detail = str(error)
    except ValueError:
        detail = response.text[:300]
    if response.status_code == 401:
        return "OpenRouter rejected the API key with 401 Unauthorized."
    if response.status_code == 402:
        return "OpenRouter reported insufficient credits or quota for this model."
    suffix = f": {detail}" if detail else ""
    return f"OpenRouter returned HTTP {response.status_code}{suffix}"


def format_history(messages: Iterable[dict]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if content is None and message.get("parts"):
            content = message["parts"][0]
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
