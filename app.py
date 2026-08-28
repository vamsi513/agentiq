"""
app.py — AgentIQ Streamlit frontend entry point.

Provides a chat interface that:
- Accepts user questions via st.chat_input
- Streams answers token-by-token in real time via LangGraph astream_events
- Shows which tool was used (retrieval / web_search / direct) as a badge
- Displays response time and turn count per message
- Shows cited sources below each assistant response
- Persists conversation history in st.session_state
- Builds the FAISS index on first run (with a progress spinner)
- Loads API keys from .streamlit/secrets (Streamlit Cloud) or .env (local)

Run locally:
    streamlit run app.py
"""

import asyncio
import logging
import os
import queue
import threading
import time
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
        logger.debug("st.secrets unavailable (expected when running locally).")


_load_secrets()

# Import after secrets are loaded so config.py sees the keys
from config import settings
from observability.langsmith_tracer import configure_tracing

configure_tracing()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentIQ",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_session() -> None:
    """
    Initialise all required session state keys on first load.

    Keys:
        messages:     List of {"role", "content", "sources", "route",
                      "elapsed_ms"} dicts representing the conversation.
        session_id:   Stable UUID for this browser session (thread_id).
        index_ready:  True once the FAISS index has been built/loaded.
        query_count:  Total queries made this session (rate-limit counter).
    """
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "index_ready": False,
        "query_count": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

# ── Rate limit configuration ──────────────────────────────────────────────────
_MAX_QUERIES_PER_SESSION = 20   # soft cap per session


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
        except Exception:
            logger.exception("FAISS index build failed")
            st.error(
                "Document index could not be loaded. "
                "Retrieval will be unavailable — web search and direct answers still work."
            )
            st.session_state.index_ready = True  # Don't retry on every rerun


# ── PDF Upload Handler ────────────────────────────────────────────────────────
def _handle_pdf_upload(uploaded_file) -> None:
    """
    Extract text from an uploaded PDF, chunk it, and index it for this session.

    Uses pypdf to extract text page by page.  The extracted text is split into
    overlapping chunks matching the vectorstore's chunking strategy, then
    embedded and added to a FAISS index scoped to this browser session, so
    only this session's own queries can retrieve it — other concurrent
    visitors never see another session's uploaded content.

    Tracks which files have already been indexed in session_state to avoid
    re-indexing on Streamlit reruns.

    Args:
        uploaded_file: Streamlit UploadedFile object from st.file_uploader.
    """
    if "indexed_pdfs" not in st.session_state:
        st.session_state.indexed_pdfs = set()

    file_key = (uploaded_file.name, uploaded_file.size)
    if file_key in st.session_state.indexed_pdfs:
        st.sidebar.success(f"Already indexed: **{uploaded_file.name}**")
        return

    with st.sidebar, st.spinner(f"Indexing **{uploaded_file.name}**…"):
        try:
            import io

            from pypdf import PdfReader

            from retrieval.vectorstore import add_documents_to_vectorstore

            # Extract text page by page
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text.strip())

            if not pages_text:
                st.error("Could not extract text from this PDF.")
                return

            full_text = "\n\n".join(pages_text)

            # Chunk the extracted text (512-token ≈ 2048 chars, 64-token overlap ≈ 256 chars)
            chunk_size = 2048
            overlap = 256
            chunks_text: list[str] = []
            start = 0
            while start < len(full_text):
                chunk = full_text[start: start + chunk_size]
                chunks_text.append(chunk)
                start += chunk_size - overlap

            from pathlib import Path as _Path
            title = _Path(uploaded_file.name).stem
            metadata = [
                {"title": title, "source": f"Uploaded PDF: {uploaded_file.name}"}
                for _ in chunks_text
            ]

            n_added = add_documents_to_vectorstore(
                chunks_text, metadata, session_id=st.session_state.session_id
            )
            st.session_state.indexed_pdfs.add(file_key)
            st.success(
                f"**{uploaded_file.name}** indexed — "
                f"{len(reader.pages)} pages, {n_added} chunks added to retrieval."
            )
            logger.info(
                "PDF indexed: '%s' — %d pages, %d chunks.",
                uploaded_file.name,
                len(reader.pages),
                n_added,
            )

        except ImportError:
            st.error("pypdf is not installed. Run: pip install pypdf")
        except Exception as exc:
            logger.exception("PDF indexing failed")
            st.error(f"Failed to index PDF: {type(exc).__name__}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render the sidebar with app info, model config, session stats, and PDF upload."""
    with st.sidebar:
        st.title("AgentIQ")
        st.caption("Multi-Step Agentic Research Assistant")
        st.divider()

        st.subheader("About")
        st.markdown(
            "AgentIQ autonomously routes your question between:\n"
            "- **Document Retrieval** — local AI/ML research corpus\n"
            "- **Web Search** — live Tavily results\n"
            "- **Direct LLM** — GPT-4o-mini knowledge\n\n"
            "Powered by **LangGraph** · **FAISS** · **Tavily**"
        )
        st.divider()

        st.subheader("Model Config")
        st.markdown(f"**LLM:** `{settings.openai_model}`")
        st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
        st.markdown(f"**Top-K Retrieval:** `{settings.top_k_retrieval}`")
        st.divider()

        # API key status indicators
        st.subheader("API Status")
        if settings.is_openai_configured():
            st.success("OpenAI key configured")
        else:
            st.error("OpenAI key missing")

        if settings.is_tavily_configured():
            st.success("Tavily key configured")
        else:
            st.warning("Tavily key missing (web search disabled)")

        st.divider()

        # Session info & usage counter
        st.subheader("Session")
        st.code(st.session_state.session_id[:16] + "…", language=None)

        used = st.session_state.query_count
        remaining = max(0, _MAX_QUERIES_PER_SESSION - used)
        st.markdown(
            f"**Queries this session:** {used} / {_MAX_QUERIES_PER_SESSION}  \n"
            f"**Remaining:** {remaining}"
        )
        st.progress(min(used / _MAX_QUERIES_PER_SESSION, 1.0))

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.query_count = 0
            st.rerun()

        st.divider()

        # ── PDF Upload ────────────────────────────────────────────────────────
        st.subheader("Upload PDF")
        st.caption("Index your own document for retrieval.")
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            _handle_pdf_upload(uploaded_file)

        st.divider()
        st.caption("Built with LangGraph · FastAPI · Streamlit")
        st.caption("[GitHub](https://github.com/vamsi513/agentiq)")


# ── Route badge ───────────────────────────────────────────────────────────────
_ROUTE_BADGES = {
    "retrieval":  ("Document Retrieval", "#7C3AED"),
    "web_search": ("Web Search",         "#0891B2"),
    "direct":     ("Direct LLM",         "#059669"),
}


def _route_badge(route: str) -> str:
    """
    Return an HTML badge string for the given route decision.

    Args:
        route: One of ``retrieval``, ``web_search``, or ``direct``.

    Returns:
        HTML string rendering a coloured badge.
    """
    label, color = _ROUTE_BADGES.get(route, ("Direct LLM", "#059669"))
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;">'
        f"{label}</span>"
    )


def _time_badge(elapsed_ms: int) -> str:
    """
    Return an HTML badge showing the response time.

    Args:
        elapsed_ms: Response time in milliseconds.

    Returns:
        HTML string rendering a grey timing badge.
    """
    if elapsed_ms < 1000:
        label = f"{elapsed_ms}ms"
    else:
        label = f"{elapsed_ms / 1000:.1f}s"
    return (
        f'<span style="background:#374151;color:#D1D5DB;padding:2px 8px;'
        f'border-radius:12px;font-size:0.72rem;margin-left:6px;">'
        f"{label}</span>"
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

    with st.expander(f"Sources ({len(sources)})", expanded=False):
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

    Each message dict has: role, content, sources (list), route (str),
    elapsed_ms (int).
    """
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if role == "assistant":
                route = msg.get("route", "direct")
                elapsed_ms = msg.get("elapsed_ms", 0)
                badges = _route_badge(route)
                if elapsed_ms:
                    badges += " " + _time_badge(elapsed_ms)
                st.markdown(badges, unsafe_allow_html=True)
                _render_sources(msg.get("sources", []))


# ── Real token streaming via threading + queue ────────────────────────────────

def _stream_agent_response(
    query: str,
    session_id: str,
    text_placeholder,
) -> tuple[str, list, str, int]:
    """
    Stream agent tokens in real time using a background thread + queue.

    Architecture:
    - A daemon thread runs ``asyncio.run(_async_producer(...))``.
    - The async producer calls ``graph.astream_events()`` and puts
      ("token", text), ("sources", list), ("route", str), or ("done", None)
      tuples into a thread-safe queue.Queue.
    - This function (running in the Streamlit main thread) drains the queue,
      appending tokens to the display placeholder until ("done", None) arrives.

    Args:
        query:            User's question string.
        session_id:       Conversation thread_id for MemorySaver.
        text_placeholder: A ``st.empty()`` placeholder for live token updates.

    Returns:
        Tuple of (full_answer_text, sources_list, route_str, elapsed_ms).
    """
    from langchain_core.messages import HumanMessage

    from agent.graph import get_graph
    from agent.memory import get_thread_config

    token_queue: queue.Queue = queue.Queue()

    async def _async_producer() -> None:
        """Run inside the background thread's event loop."""
        graph = get_graph()
        config = get_thread_config(session_id)
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "context": "",
            "sources": [],
            "route_decision": "direct",
            "session_id": session_id,
            "turn_count": 0,
            "retrieval_score": 0.0,
            "error": None,
            "metadata": {},
        }

        try:
            async for event in graph.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")

                # ── Tokens from generator LLM call ────────────────────────────
                if kind == "on_chat_model_stream":
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    if node_name == "generator":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token_queue.put(("token", chunk.content))

                # ── Capture route decision ────────────────────────────────────
                elif kind == "on_chain_end":
                    node_name = event.get("name", "")
                    if node_name == "router":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            route = output.get("route_decision", "")
                            if route:
                                token_queue.put(("route", route))

                    # ── Sources from retriever / web_search ───────────────────
                    elif node_name in ("retriever", "web_search"):
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict) and output.get("sources"):
                            token_queue.put(("sources", output["sources"]))

        except Exception as exc:
            logger.exception("Streaming producer error")
            token_queue.put(("error", str(exc)))
        finally:
            token_queue.put(("done", None))

    def _thread_target() -> None:
        asyncio.run(_async_producer())

    # Launch producer in background thread
    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()

    # ── Drain the queue in the Streamlit main thread ──────────────────────────
    accumulated = ""
    sources: list = []
    route = "direct"
    start_time = time.time()

    while True:
        try:
            msg_type, payload = token_queue.get(timeout=120)  # 120s hard timeout
        except queue.Empty:
            logger.error("Token queue timed out after 120s.")
            accumulated = accumulated or "Request timed out. Please try again."
            text_placeholder.markdown(accumulated)
            break

        if msg_type == "token":
            accumulated += payload
            # Show text with blinking cursor while streaming
            text_placeholder.markdown(accumulated + " ▌")

        elif msg_type == "sources":
            sources = payload

        elif msg_type == "route":
            route = payload

        elif msg_type == "error":
            accumulated = f"Error: {payload}"
            text_placeholder.markdown(accumulated)

        elif msg_type == "done":
            break

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Final render without cursor
    if accumulated:
        text_placeholder.markdown(accumulated)

    t.join(timeout=5)  # Give thread a moment to clean up
    return accumulated, sources, route, elapsed_ms


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Main Streamlit application entry point.

    Renders sidebar, conversation history, and the chat input.
    On each user message: streams tokens in real time, shows route badge
    + response time, and persists the full turn to session state.
    """
    _render_sidebar()
    _ensure_index()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("AgentIQ")
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

        # ── Rate limit check ──────────────────────────────────────────────────
        if st.session_state.query_count >= _MAX_QUERIES_PER_SESSION:
            st.warning(
                f"You have reached the session limit of {_MAX_QUERIES_PER_SESSION} queries. "
                "Click **Clear conversation** in the sidebar to start a new session."
            )
            st.stop()

        # Show user message immediately
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": [], "route": ""}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream agent response token-by-token
        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            text_placeholder.markdown("▌")  # Show cursor immediately

            try:
                answer, sources, route, elapsed_ms = _stream_agent_response(
                    prompt,
                    st.session_state.session_id,
                    text_placeholder,
                )
            except Exception as exc:
                logger.exception("Unexpected error during streaming")
                answer = f"Unexpected error: {type(exc).__name__}. Please try again."
                sources = []
                route = "direct"
                elapsed_ms = 0
                text_placeholder.markdown(answer)

            # Show route badge + response time
            badges = _route_badge(route)
            if elapsed_ms:
                badges += " " + _time_badge(elapsed_ms)
            st.markdown(badges, unsafe_allow_html=True)

            # Show sources
            _render_sources(sources)

        # Increment usage counter
        st.session_state.query_count += 1

        # Persist assistant turn to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "route": route,
                "elapsed_ms": elapsed_ms,
            }
        )


main()
