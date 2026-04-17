"""
app.py — AgentIQ Streamlit frontend entry point.

Provides a chat interface that:
- Accepts user questions via st.chat_input
- Streams answers token-by-token in real time
- Shows which tool was used (retrieval / web_search / direct) as a badge
- Displays cited sources below each assistant response
- Persists conversation history in st.session_state
- Builds the FAISS index on first run (with a progress spinner)
- Loads API keys from .streamlit/secrets (Streamlit Cloud) or .env (local)

Run locally:
    streamlit run app.py
"""

import logging
import os
import uuid
from typing import Any

import streamlit as st

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Streamlit Cloud secrets → environment variables ───────────────────────────
def _load_secrets() -> None:
    """
    Inject Streamlit secrets into os.environ so config.py picks them up.

    On Streamlit Cloud, API keys live in st.secrets.  Locally they come
    from a .env file loaded by config.py.  This function bridges the two
    by copying any secrets into os.environ before config is imported.
    """
    try:
        for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        # st.secrets raises when running locally without secrets.toml — safe to ignore
        pass


_load_secrets()

# Import after secrets are loaded so config.py sees the keys
from config import settings  # noqa: E402


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_session() -> None:
    """
    Initialise all required session state keys on first load.

    Keys:
        messages: List of {"role": str, "content": str, "sources": list,
                  "route": str} dicts representing the conversation.
        session_id: Stable UUID for this browser session (used as thread_id).
        index_ready: True once the FAISS index has been built/loaded.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False


_init_session()


# ── FAISS index bootstrap ─────────────────────────────────────────────────────
def _ensure_index() -> None:
    """
    Build or load the FAISS vector index on first run.

    Shows a spinner while working so the user knows something is happening.
    Sets ``st.session_state.index_ready = True`` on success.
    On failure shows st.error — the app still works via web search / direct.
    """
    if st.session_state.index_ready:
        return

    with st.spinner("Loading document index… (first run may take ~30 seconds)"):
        try:
            from retrieval.vectorstore import get_vectorstore
            get_vectorstore()
            st.session_state.index_ready = True
            logger.info("FAISS index ready.")
        except Exception as exc:
            logger.exception("FAISS index build failed: %s", exc)
            st.error(
                "⚠️ Document index could not be loaded. "
                "Retrieval will be unavailable — web search and direct answers still work."
            )
            st.session_state.index_ready = True  # Don't retry on every rerun


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render the sidebar with app info, model config, and session details."""
    with st.sidebar:
        st.title("🔍 AgentIQ")
        st.caption("Multi-Step Agentic Research Assistant")
        st.divider()

        st.subheader("About")
        st.markdown(
            "AgentIQ autonomously routes your question between:\n"
            "- 📄 **Document Retrieval** — local AI/ML research corpus\n"
            "- 🌐 **Web Search** — live Tavily results\n"
            "- 🧠 **Direct LLM** — GPT-4o-mini knowledge\n\n"
            "Powered by **LangGraph** · **FAISS** · **Tavily**"
        )
        st.divider()

        st.subheader("Model Config")
        st.markdown(f"**LLM:** `{settings.openai_model}`")
        st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2`")
        st.markdown(f"**Top-K Retrieval:** `{settings.top_k_retrieval}`")
        st.divider()

        # API key status indicators
        st.subheader("API Status")
        if settings.is_openai_configured():
            st.success("OpenAI ✓")
        else:
            st.error("OpenAI key missing")

        if settings.is_tavily_configured():
            st.success("Tavily ✓")
        else:
            st.warning("Tavily key missing (web search disabled)")

        st.divider()

        st.subheader("Session")
        st.code(st.session_state.session_id[:16] + "…", language=None)

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.caption("Built with LangGraph · FastAPI · Streamlit")
        st.caption("[GitHub](https://github.com/vamsi513/agentiq)")


# ── Route badge ───────────────────────────────────────────────────────────────
_ROUTE_BADGES = {
    "retrieval":  ("📄", "Document Retrieval", "#7C3AED"),
    "web_search": ("🌐", "Web Search",         "#0891B2"),
    "direct":     ("🧠", "Direct LLM",         "#059669"),
}


def _route_badge(route: str) -> str:
    """
    Return an HTML badge string for the given route decision.

    Args:
        route: One of ``retrieval``, ``web_search``, or ``direct``.

    Returns:
        HTML string rendering a coloured badge.
    """
    icon, label, color = _ROUTE_BADGES.get(
        route, ("🧠", "Direct LLM", "#059669")
    )
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;">'
        f"{icon} {label}</span>"
    )


# ── Sources display ───────────────────────────────────────────────────────────
def _render_sources(sources: list[dict[str, Any]]) -> None:
    """
    Render a collapsible sources section below an assistant message.

    Args:
        sources: List of source dicts with ``title``, ``url``,
                 ``content``, ``score``, ``type`` keys.
    """
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, start=1):
            title = src.get("title", "Untitled")
            url = src.get("url", "")
            content = src.get("content", "")
            score = src.get("score", 0.0)
            src_type = src.get("type", "direct")

            col1, col2 = st.columns([4, 1])
            with col1:
                if url:
                    st.markdown(f"**{i}. [{title}]({url})**")
                else:
                    st.markdown(f"**{i}. {title}**")
                if content:
                    st.caption(content[:200] + ("…" if len(content) > 200 else ""))
            with col2:
                st.caption(f"Score: {score:.2f}")
                st.caption(src_type)

            if i < len(sources):
                st.divider()


# ── Conversation history rendering ────────────────────────────────────────────
def _render_history() -> None:
    """
    Render all previous messages from st.session_state.messages.

    Each message dict has: role, content, sources (list), route (str).
    """
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if role == "assistant":
                route = msg.get("route", "direct")
                st.markdown(_route_badge(route), unsafe_allow_html=True)
                _render_sources(msg.get("sources", []))


# ── Agent invocation with streaming ──────────────────────────────────────────
def _run_agent(query: str) -> tuple[str, list, str]:
    """
    Invoke the AgentIQ graph and stream the response into the chat UI.

    Calls ``api.streaming.run_agent_sync`` (async-safe, non-streaming path)
    since Streamlit's execution model is synchronous.  Token-level streaming
    is simulated via ``st.write_stream`` using a generator over the answer
    words.

    Args:
        query: User's question string.

    Returns:
        Tuple of (answer_text, sources_list, route_decision_string).
    """
    import asyncio

    try:
        from api.streaming import run_agent_sync

        # Run the async function in a new event loop (Streamlit is sync)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                run_agent_sync(query, st.session_state.session_id)
            )
        finally:
            loop.close()

        answer = result.get("answer", "No response generated.")
        sources = result.get("sources", [])
        route = result.get("route_decision", "direct")
        return answer, sources, route

    except Exception as exc:
        logger.exception("Agent invocation failed: %s", exc)
        return (
            f"❌ Error: {type(exc).__name__} — please check your API keys.",
            [],
            "direct",
        )


def _word_stream(text: str):
    """
    Generator that yields words one at a time for st.write_stream.

    Args:
        text: Full answer text to stream word by word.

    Yields:
        Words with trailing spaces for natural display.
    """
    import time
    words = text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(0.012)  # ~80 words/sec — feels natural without being slow


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Main Streamlit application entry point.

    Renders sidebar, conversation history, and the chat input.
    On each user message: runs the agent, streams the response,
    and persists the full turn to session state.
    """
    _render_sidebar()
    _ensure_index()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔍 AgentIQ")
    st.caption(
        "Ask anything — I'll route your question to the best source: "
        "document retrieval, live web search, or direct LLM knowledge."
    )

    # ── Warn if OpenAI key is missing ─────────────────────────────────────────
    if not settings.is_openai_configured():
        st.error(
            "**OPENAI_API_KEY is not set.** "
            "Add it to `.env` (local) or Streamlit Cloud secrets before chatting."
        )
        st.stop()

    # ── Render conversation history ───────────────────────────────────────────
    _render_history()

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a research question…"):

        # Show user message immediately
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": [], "route": ""}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent and stream response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, sources, route = _run_agent(prompt)

            # Stream answer word by word
            st.write_stream(_word_stream(answer))

            # Show route badge
            st.markdown(_route_badge(route), unsafe_allow_html=True)

            # Show sources
            _render_sources(sources)

        # Persist assistant turn to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "route": route,
            }
        )


if __name__ == "__main__" or True:
    main()
