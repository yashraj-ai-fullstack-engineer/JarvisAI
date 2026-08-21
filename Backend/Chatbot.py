import datetime
import hashlib
import threading
import uuid
from json import JSONDecodeError, dump, load
from pathlib import Path

from Backend.LLMProvider import (
    LMSTUDIO_MODEL,
    LocalLLMUnavailable,
    format_history,
    generate_text,
    get_config,
    stream_text,
)
from Backend.GoogleOAuth import current_session_id
from Backend.MongoStore import (
    StoreUnavailable,
    current_chat_session_id,
    current_chat_user_id,
    load_messages as mongo_load_messages,
    save_exchange as mongo_save_exchange,
)
from Backend.Paths import DATA_DIR

Username = get_config("Username", "User")
Assistantname = get_config("Assistantname", "NEXA")

System = f"""You are {Assistantname}, a helpful local desktop assistant.
You run entirely on this computer. Be concise, accurate, and friendly.
Use the provided computer date and time when the user asks for it.
Do not claim to have current internet access or live web information.
Answer only the current user question. Do not invent appointments, reminders,
schedules, prior events, or personal details unless the user explicitly gives them.
"""

ROOT = Path(__file__).resolve().parent
CHAT_LOG_PATH = DATA_DIR / "ChatLog.json"
SESSION_DATA_DIR = DATA_DIR / "Sessions"
THINKING_SUMMARY_ENABLED = get_config("THINKING_SUMMARY_ENABLED", "true").lower() == "true"
_locks_guard = threading.Lock()
_history_locks: dict[str, threading.RLock] = {}


def _history_path() -> Path:
    session_id = current_session_id()
    if not session_id:
        return CHAT_LOG_PATH
    session_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return SESSION_DATA_DIR / session_key / "ChatLog.json"


def _history_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _locks_guard:
        return _history_locks.setdefault(key, threading.RLock())


def _load_messages() -> list[dict]:
    path = _history_path()
    with _history_lock(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = load(file)
                return data if isinstance(data, list) else []
        except (FileNotFoundError, JSONDecodeError):
            return []


def _save_messages(messages: list[dict]) -> None:
    path = _history_path()
    with _history_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with open(temporary, "w", encoding="utf-8") as file:
            dump(messages, file, indent=2, ensure_ascii=False)
        temporary.replace(path)


def LoadHistory(limit: int = 20) -> list[dict]:
    """Return recent persisted user/assistant messages for an agent invocation."""
    chat_session_id = current_chat_session_id()
    if chat_session_id:
        try:
            return mongo_load_messages(chat_session_id, limit, user_id=current_chat_user_id())
        except StoreUnavailable:
            return []
    return _load_messages()[-max(0, limit):]


def SaveExchange(
    query: str,
    answer: str,
    answer_visibility: str = "shared",
    answer_visible_to_user_id: str = "",
    system_notice: str = "",
    research_run_id: str = "",
) -> dict | None:
    """Persist a completed exchange in the current browser session history."""
    chat_session_id = current_chat_session_id()
    if chat_session_id:
        try:
            saved = mongo_save_exchange(
                chat_session_id,
                query,
                answer,
                answer_visibility=answer_visibility,
                answer_visible_to_user_id=answer_visible_to_user_id,
                system_notice=system_notice,
                research_run_id=research_run_id,
            )
            # Keep the derived session context aligned for every write path,
            # including PDF answers and confirmation callbacks. The agent's
            # workflow-specific refresh may refine this immediately afterward.
            try:
                from Backend.SessionContext import refresh as refresh_session_context

                refresh_session_context()
            except Exception:
                pass
            return saved
        except StoreUnavailable:
            return None
    path = _history_path()
    with _history_lock(path):
        messages = _load_messages()
        messages.append({"role": "user", "content": query})
        assistant_message = {"id": str(uuid.uuid4()), "role": "assistant", "content": answer, "research_run_id": research_run_id}
        messages.append(assistant_message)
        _save_messages(messages[-100:])
        return assistant_message


def ClearHistory() -> None:
    path = _history_path()
    with _history_lock(path):
        _save_messages([])


def RealtimeInformation() -> str:
    now = datetime.datetime.now()
    return (
        "Computer date and time:\n"
        f"Day: {now.strftime('%A')}\n"
        f"Date: {now.strftime('%d %B %Y')}\n"
        f"Time: {now.strftime('%H:%M:%S')}\n"
    )


def AnswerModifier(answer: str) -> str:
    return "\n".join(line for line in answer.split("\n") if line.strip())


def _answer_query(query: str, messages: list[dict] | None = None) -> str:
    history = format_history((messages or [])[-20:])
    prompt = (
        f"{RealtimeInformation()}\n"
        f"Recent conversation:\n{history or '(none)'}\n\n"
        f"Current user question: {query}"
    )
    return generate_text(
        prompt=prompt,
        system=System,
        model=LMSTUDIO_MODEL,
        temperature=0.35,
        reasoning="off",
    )


def _answer_plan(query: str, answer: str) -> str:
    return generate_text(
        prompt=f"User question: {query}\nFinal answer: {answer}",
        system=(
            "Create one short, user-facing sentence describing the high-level "
            "approach used for this answer. This is an answer plan, not hidden "
            "reasoning. Do not reveal private chain-of-thought, do not invent "
            "facts, and do not say 'I thought'."
        ),
        model=LMSTUDIO_MODEL,
        temperature=0.2,
        reasoning="off",
    )


def ChatBotWithPlan(query: str) -> tuple[str, str]:
    messages = _load_messages()
    try:
        answer = _answer_query(query, messages)
    except LocalLLMUnavailable as exc:
        return str(exc), ""
    except Exception:
        return "The local language model could not answer that request.", ""

    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant", "content": answer})
    _save_messages(messages[-100:])

    if not THINKING_SUMMARY_ENABLED:
        return AnswerModifier(answer), ""
    try:
        plan = _answer_plan(query, answer)
        return AnswerModifier(answer), AnswerModifier(plan)
    except Exception:
        return AnswerModifier(answer), ""


def ChatBot(query: str) -> str:
    answer, _ = ChatBotWithPlan(query)
    return answer


def ChatBotStream(query: str):
    """Stream safe progress events and answer deltas, then persist the exchange."""
    messages = _load_messages()
    history = format_history(messages[-20:])
    prompt = (
        f"{RealtimeInformation()}\n"
        f"Recent conversation:\n{history or '(none)'}\n\n"
        f"Current user question: {query}"
    )
    answer_parts: list[str] = []
    last_status = ""

    def status(message: str, stage: str, detail: str):
        nonlocal last_status
        if message != last_status:
            last_status = message
            return {
                "type": "status",
                "message": message,
                "stage": stage,
                "detail": detail,
            }
        return None

    initial = status(
        "Understanding your request",
        "Intake",
        "Reading your message and setting up the reply.",
    )
    if initial:
        yield initial
    try:
        for event in stream_text(
            prompt=prompt,
            system=System,
            model=LMSTUDIO_MODEL,
            temperature=0.35,
            reasoning="off",
        ):
            event_type = event.get("type", "")
            progress_event = None
            if event_type.startswith("model_load."):
                progress_event = status(
                    "Loading the local intelligence model",
                    "Model",
                    "Waking the local model and preparing it to answer.",
                )
            elif event_type.startswith("prompt_processing."):
                progress_event = status(
                    "Reviewing context and conversation",
                    "Context",
                    "Checking recent chat history and the current request.",
                )
            elif event_type.startswith("reasoning."):
                progress_event = status(
                    "Working through the details",
                    "Planning",
                    "Organizing a concise answer before streaming it back.",
                )
            elif event_type == "message.start":
                progress_event = status(
                    "Composing the response",
                    "Writing",
                    "Streaming the answer into the interface.",
                )
            elif event_type == "message.delta":
                content = event.get("content", "")
                if content:
                    answer_parts.append(content)
                    yield {"type": "delta", "content": content}
            elif event_type == "error":
                error = event.get("error", {})
                yield {
                    "type": "error",
                    "message": error.get("message", "The local model stream failed."),
                }
                return
            if progress_event:
                yield progress_event
    except LocalLLMUnavailable as exc:
        yield {"type": "error", "message": str(exc)}
        return
    except Exception:
        yield {"type": "error", "message": "The local language model could not answer that request."}
        return

    answer = AnswerModifier("".join(answer_parts))
    if not answer:
        yield {"type": "error", "message": "LM Studio returned an empty response."}
        return
    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant", "content": answer})
    _save_messages(messages[-100:])
    final_status = status(
        "Response ready",
        "Done",
        "The answer has finished streaming and was saved to history.",
    )
    if final_status:
        yield final_status
    yield {"type": "done", "answer": answer}


if __name__ == "__main__":
    while True:
        print(ChatBot(input("Enter your question: ")))
