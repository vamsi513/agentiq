"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Send,
  Sparkles,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  BookOpen,
  Globe,
  MessageCircle,
} from "lucide-react";
import { streamChat } from "@/lib/api";
import type { ChatMessage, RouteDecision, SourceItem } from "@/lib/types";

const SESSION_KEY = "agentiq_session_id";

const SAMPLE_QUESTIONS: { label: string; route: RouteDecision }[] = [
  { label: "What is the Transformer architecture?", route: "retrieval" },
  { label: "What's the latest GPT model release?", route: "web_search" },
  { label: "What is 15% of 200?", route: "direct" },
];

const ROUTE_META: Record<
  RouteDecision,
  { label: string; color: string; bg: string; icon: typeof BookOpen }
> = {
  retrieval: { label: "Retrieval", color: "var(--route-retrieval)", bg: "var(--route-retrieval-bg)", icon: BookOpen },
  web_search: { label: "Web Search", color: "var(--route-web-search)", bg: "var(--route-web-search-bg)", icon: Globe },
  direct: { label: "Direct", color: "var(--route-direct)", bg: "var(--route-direct-bg)", icon: MessageCircle },
};

function newId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function RouteBadge({ route, latencyMs }: { route: RouteDecision; latencyMs?: number }) {
  const meta = ROUTE_META[route];
  const Icon = meta.icon;
  return (
    <span
      role="status"
      style={{ ...styles.routeBadge, color: meta.color, background: meta.bg }}
    >
      <Icon size={12} aria-hidden="true" />
      {meta.label}
      {typeof latencyMs === "number" && (
        <span style={styles.latency}>· {(latencyMs / 1000).toFixed(1)}s</span>
      )}
    </span>
  );
}

function SourcesList({ sources }: { sources: SourceItem[] }) {
  const [open, setOpen] = useState(false);
  if (sources.length === 0) return null;

  return (
    <div style={styles.sourcesWrap}>
      <button
        style={styles.sourcesToggle}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        {sources.length} source{sources.length !== 1 ? "s" : ""}
      </button>
      {open && (
        <ul style={styles.sourcesList}>
          {sources.map((s, i) => (
            <li key={i} style={styles.sourceItem}>
              <div style={styles.sourceHeader}>
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noopener noreferrer" style={styles.sourceTitle}>
                    {s.title || s.url}
                  </a>
                ) : (
                  <span style={{ ...styles.sourceTitle, color: "var(--text-primary)" }}>{s.title || "Untitled source"}</span>
                )}
                <span style={styles.sourceScore}>{(s.score * 100).toFixed(0)}%</span>
              </div>
              {s.content && <p style={styles.sourceContent}>{s.content}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * The SSE stream doesn't carry a dedicated "route" event — sources carry a
 * `type` field that mirrors route_decision for retrieval/web_search, and an
 * empty-sources, non-error, finished message is the direct route. Computed
 * at render time (not via a state-syncing effect) to avoid feeding back
 * into setState on every render.
 */
function inferRoute(message: ChatMessage): RouteDecision | undefined {
  if (message.role !== "assistant" || message.streaming || message.error) return undefined;
  if (message.sources && message.sources.length > 0) return message.sources[0].type;
  if (message.latencyMs !== undefined) return "direct";
  return undefined;
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const route = inferRoute(message);
  return (
    <div style={{ ...styles.messageRow, justifyContent: isUser ? "flex-end" : "flex-start" }}>
      <div
        style={{
          ...styles.bubble,
          ...(isUser ? styles.bubbleUser : styles.bubbleAssistant),
          ...(message.error ? styles.bubbleError : {}),
        }}
      >
        {!isUser && route && (
          <div style={styles.bubbleMeta}>
            <RouteBadge route={route} latencyMs={message.latencyMs} />
          </div>
        )}
        <div
          aria-live={!isUser ? "polite" : undefined}
          style={styles.bubbleText}
        >
          {message.content || (message.streaming ? "" : "")}
          {message.streaming && <span style={styles.cursor} aria-hidden="true" />}
        </div>
        {!isUser && message.sources && <SourcesList sources={message.sources} />}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionIdRef = useRef<string | undefined>(undefined);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Establish (or restore) the session id on mount, from sessionStorage —
  // not localStorage — so multi-turn memory works within a tab but resets
  // on a new tab, per the spec.
  useEffect(() => {
    const stored = sessionStorage.getItem(SESSION_KEY);
    if (stored) {
      sessionIdRef.current = stored;
    } else {
      const fresh = crypto.randomUUID();
      sessionIdRef.current = fresh;
      sessionStorage.setItem(SESSION_KEY, fresh);
    }
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const runQuery = useCallback(async (query: string) => {
    const trimmed = query.trim();
    if (!trimmed || streaming) return;

    setError(null);
    const userMsg: ChatMessage = { id: newId(), role: "user", content: trimmed };
    const assistantId = newId();
    const assistantMsg: ChatMessage = { id: assistantId, role: "assistant", content: "", streaming: true };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const startedAt = performance.now();

    try {
      await streamChat(
        { query: trimmed, session_id: sessionIdRef.current },
        {
          onToken: (chunk) => {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + chunk } : m))
            );
          },
          onSources: (sources) => {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, sources: sources as SourceItem[] } : m))
            );
          },
          onDone: () => {
            const latencyMs = performance.now() - startedAt;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, streaming: false, latencyMs } : m))
            );
          },
          onError: (message) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, streaming: false, error: true, content: m.content || message }
                  : m
              )
            );
            setError(message);
          },
        },
        controller.signal
      );
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        const message = err instanceof Error ? err.message : "Something went wrong.";
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, streaming: false, error: true, content: message } : m))
        );
        setError(message);
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [streaming]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runQuery(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runQuery(input);
    }
  };

  const handleNewConversation = () => {
    abortRef.current?.abort();
    setMessages([]);
    setInput("");
    setError(null);
    setStreaming(false);
    sessionIdRef.current = undefined;
    sessionStorage.removeItem(SESSION_KEY);
  };

  return (
    <div style={styles.page}>
      <h1 className="visually-hidden">AgentIQ — Agentic Research Assistant Chat</h1>

      <header style={styles.header}>
        <div>
          <div style={styles.headerTitle}>AgentIQ</div>
          <div style={styles.headerSub}>Ask a question — routed to retrieval, web search, or direct answer.</div>
        </div>
        <button style={styles.newConvoBtn} onClick={handleNewConversation} disabled={messages.length === 0}>
          <RotateCcw size={14} />
          New conversation
        </button>
      </header>

      <div style={styles.scrollArea} ref={scrollRef}>
        {messages.length === 0 ? (
          <div style={styles.emptyState}>
            <Sparkles size={40} style={{ color: "var(--accent)", marginBottom: "16px" }} />
            <h2 style={styles.emptyTitle}>Ask AgentIQ anything</h2>
            <p style={styles.emptyDesc}>
              Questions about AI/ML research route to a local FAISS corpus, current-events
              questions trigger a live web search, and everything else is answered directly.
            </p>
            <div style={styles.samplesGrid}>
              {SAMPLE_QUESTIONS.map((s) => {
                const meta = ROUTE_META[s.route];
                const Icon = meta.icon;
                return (
                  <button
                    key={s.label}
                    style={{ ...styles.sampleBtn, borderColor: meta.color }}
                    onClick={() => runQuery(s.label)}
                  >
                    <Icon size={13} style={{ color: meta.color, flexShrink: 0 }} />
                    {s.label}
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div style={styles.messageList}>
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
          </div>
        )}

        {error && (
          <div style={styles.errorBanner} role="alert">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} style={styles.inputBar}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question…"
          style={{ ...styles.textarea, ...(streaming ? styles.textareaDisabled : {}) }}
          rows={1}
          disabled={streaming}
          aria-label="Message AgentIQ"
        />
        <button
          type="submit"
          style={{
            ...styles.sendBtn,
            ...(streaming || input.trim().length === 0 ? styles.sendBtnDisabled : {}),
          }}
          disabled={streaming || input.trim().length === 0}
        >
          <Send size={16} />
          <span className="visually-hidden">Send</span>
        </button>
      </form>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    maxWidth: "900px",
    width: "100%",
    margin: "0 auto",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "20px 24px",
    borderBottom: "1px solid var(--border)",
  },
  headerTitle: {
    fontSize: "18px",
    fontWeight: 700,
    color: "var(--text-primary)",
  },
  headerSub: {
    fontSize: "12.5px",
    color: "var(--text-muted)",
    marginTop: "2px",
  },
  newConvoBtn: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    background: "var(--surface-2)",
    border: "1px solid var(--border)",
    borderRadius: "7px",
    padding: "7px 12px",
    fontSize: "12.5px",
    fontWeight: 500,
    color: "var(--text-secondary)",
    cursor: "pointer",
  },
  scrollArea: {
    flex: 1,
    overflowY: "auto",
    padding: "24px",
  },
  emptyState: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    paddingTop: "60px",
  },
  emptyTitle: {
    fontSize: "20px",
    fontWeight: 600,
    color: "var(--text-primary)",
    marginBottom: "8px",
  },
  emptyDesc: {
    fontSize: "14px",
    color: "var(--text-muted)",
    maxWidth: "460px",
    marginBottom: "28px",
    lineHeight: 1.6,
  },
  samplesGrid: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    width: "100%",
    maxWidth: "440px",
  },
  sampleBtn: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    background: "var(--surface-1)",
    border: "1px solid var(--border)",
    borderRadius: "10px",
    padding: "12px 16px",
    fontSize: "13.5px",
    color: "var(--text-secondary)",
    cursor: "pointer",
    textAlign: "left",
    fontFamily: "var(--font-sans)",
  },
  messageList: {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  messageRow: {
    display: "flex",
    width: "100%",
  },
  bubble: {
    maxWidth: "78%",
    borderRadius: "14px",
    padding: "12px 16px",
  },
  bubbleUser: {
    background: "var(--accent)",
    color: "var(--accent-fg)",
    borderBottomRightRadius: "4px",
  },
  bubbleAssistant: {
    background: "var(--surface-1)",
    border: "1px solid var(--border)",
    color: "var(--text-primary)",
    borderBottomLeftRadius: "4px",
  },
  bubbleError: {
    borderColor: "var(--danger)",
  },
  bubbleMeta: {
    marginBottom: "8px",
  },
  bubbleText: {
    fontSize: "14.5px",
    lineHeight: 1.65,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  cursor: {
    display: "inline-block",
    width: "7px",
    height: "14px",
    marginLeft: "2px",
    background: "var(--accent)",
    animation: "blink 1s step-start infinite",
    verticalAlign: "text-bottom",
  },
  routeBadge: {
    display: "inline-flex",
    alignItems: "center",
    gap: "5px",
    fontSize: "11px",
    fontWeight: 600,
    padding: "3px 8px",
    borderRadius: "20px",
  },
  latency: {
    fontFamily: "var(--font-mono)",
    fontWeight: 500,
    opacity: 0.85,
  },
  sourcesWrap: {
    marginTop: "10px",
    paddingTop: "10px",
    borderTop: "1px solid var(--border)",
  },
  sourcesToggle: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    background: "none",
    border: "none",
    color: "var(--text-muted)",
    fontSize: "12px",
    cursor: "pointer",
    padding: 0,
    fontFamily: "var(--font-sans)",
  },
  sourcesList: {
    marginTop: "10px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    listStyle: "none",
  },
  sourceItem: {
    background: "var(--surface-2)",
    borderRadius: "8px",
    padding: "10px 12px",
  },
  sourceHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
    marginBottom: "4px",
  },
  sourceTitle: {
    fontSize: "12.5px",
    fontWeight: 600,
    color: "var(--accent)",
    textDecoration: "none",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  sourceScore: {
    fontFamily: "var(--font-mono)",
    fontSize: "11px",
    color: "var(--text-muted)",
    flexShrink: 0,
  },
  sourceContent: {
    fontSize: "12px",
    color: "var(--text-secondary)",
    lineHeight: 1.5,
  },
  errorBanner: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    background: "var(--danger-bg)",
    color: "var(--danger)",
    border: "1px solid var(--danger)",
    borderRadius: "8px",
    padding: "10px 14px",
    fontSize: "13px",
    marginTop: "16px",
  },
  inputBar: {
    display: "flex",
    gap: "10px",
    alignItems: "flex-end",
    padding: "16px 24px 24px",
    borderTop: "1px solid var(--border)",
  },
  textarea: {
    flex: 1,
    resize: "none",
    background: "var(--surface-1)",
    border: "1px solid var(--border)",
    borderRadius: "10px",
    padding: "12px 14px",
    fontSize: "14.5px",
    color: "var(--text-primary)",
    outline: "none",
    maxHeight: "160px",
  },
  sendBtn: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "var(--accent)",
    color: "var(--accent-fg)",
    border: "none",
    borderRadius: "10px",
    width: "42px",
    height: "42px",
    cursor: "pointer",
    flexShrink: 0,
  },
  sendBtnDisabled: {
    opacity: 0.5,
    cursor: "not-allowed",
  },
  textareaDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
};
