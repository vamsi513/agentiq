import type { Metadata } from "next";
import { ArrowDown } from "lucide-react";

export const metadata: Metadata = {
  title: "Pipeline — AgentIQ",
  description: "How AgentIQ routes a question through its LangGraph agent: router, retrieval paths, and cited generation.",
};

const ROUTES = [
  {
    label: "FAISS Retrieval",
    detail: "Local vector search over ~30 AI/ML research documents, embedded with all-MiniLM-L6-v2. Top chunks by cosine similarity feed the generator as grounded context.",
    color: "var(--route-retrieval)",
  },
  {
    label: "Tavily Web Search",
    detail: "Live web search for time-sensitive or current-events questions — news, recent releases, anything that changes over time.",
    color: "var(--route-web-search)",
  },
  {
    label: "Direct LLM",
    detail: "GPT-4o-mini answers conversational or general-knowledge questions directly from training data, no retrieval step.",
    color: "var(--route-direct)",
  },
];

const TECH_STACK = [
  { component: "Agent Orchestration", technology: "LangGraph", note: "StateGraph with conditional routing edges" },
  { component: "Router / Generator LLM", technology: "GPT-4o-mini", note: "Classifies intent, then synthesises the cited answer" },
  { component: "Embeddings", technology: "all-MiniLM-L6-v2", note: "sentence-transformers, used to build the FAISS index" },
  { component: "Vector Store", technology: "FAISS", note: "Local index over the research-paper corpus" },
  { component: "Web Search", technology: "Tavily Search API", note: "Live results for current-events queries" },
  { component: "Memory", technology: "LangGraph MemorySaver", note: "Process-local, keyed by session_id — multi-turn history per session" },
  { component: "API", technology: "FastAPI", note: "POST /chat (JSON) and POST /chat/stream (SSE)" },
];

export default function PipelinePage() {
  return (
    <div style={styles.page}>
      <h1 style={styles.title}>Agent Pipeline</h1>
      <p style={styles.subtitle}>
        How a question flows through AgentIQ&apos;s LangGraph agent, from routing decision to
        streamed, cited answer.
      </p>

      <section style={styles.section} aria-label="Pipeline diagram">
        <h2 style={styles.sectionTitle}>Architecture</h2>
        <div style={styles.pipeline}>
          <div style={styles.stepCard}>
            <div style={styles.stepName}>User Query</div>
            <div style={styles.stepDetail}>Natural language question, 1–2000 characters</div>
          </div>
          <div style={styles.arrow} aria-hidden="true">
            <ArrowDown size={20} style={{ color: "var(--text-muted)" }} />
          </div>

          <div style={{ ...styles.stepCard, borderTopColor: "var(--accent)" }}>
            <div style={{ ...styles.stepName, color: "var(--accent)" }}>Router Node</div>
            <div style={styles.stepDetail}>
              GPT-4o-mini classifies the query as <code style={styles.code}>retrieval</code>,{" "}
              <code style={styles.code}>web_search</code>, or <code style={styles.code}>direct</code> —
              a single-word decision, no explanation.
            </div>
          </div>
          <div style={styles.arrow} aria-hidden="true">
            <ArrowDown size={20} style={{ color: "var(--text-muted)" }} />
          </div>

          <div style={styles.branchLabel}>One of three paths, by route decision</div>
          <div style={styles.branchRow}>
            {ROUTES.map((r) => (
              <div key={r.label} style={{ ...styles.stepCard, borderTopColor: r.color, flex: 1 }}>
                <div style={{ ...styles.stepName, color: r.color }}>{r.label}</div>
                <div style={styles.stepDetail}>{r.detail}</div>
              </div>
            ))}
          </div>
          <div style={styles.arrow} aria-hidden="true">
            <ArrowDown size={20} style={{ color: "var(--text-muted)" }} />
          </div>

          <div style={{ ...styles.stepCard, borderTopColor: "var(--accent)" }}>
            <div style={{ ...styles.stepName, color: "var(--accent)" }}>Generator Node</div>
            <div style={styles.stepDetail}>
              Synthesises retrieved context (or answers directly, for the direct route) into a
              grounded response, citing sources inline by title. Direct-route answers skip this
              node and stream straight from the router&apos;s downstream LLM call.
            </div>
          </div>
          <div style={styles.arrow} aria-hidden="true">
            <ArrowDown size={20} style={{ color: "var(--text-muted)" }} />
          </div>

          <div style={{ ...styles.stepCard, borderTopColor: "#22c55e" }}>
            <div style={{ ...styles.stepName, color: "#22c55e" }}>Streamed Response</div>
            <div style={styles.stepDetail}>
              Server-Sent Events over POST /chat/stream — token chunks, then a sources list, then a
              done signal — rendered progressively in the chat UI.
            </div>
          </div>
        </div>
      </section>

      <section style={styles.section} aria-label="Memory and observability">
        <h2 style={styles.sectionTitle}>Memory &amp; Observability</h2>
        <div style={styles.noteGrid}>
          <div style={styles.noteCard}>
            <div style={styles.noteTitle}>Session Memory</div>
            <div style={styles.noteBody}>
              A LangGraph <code style={styles.code}>MemorySaver</code> checkpoints every turn,
              keyed by <code style={styles.code}>session_id</code>, giving each conversation
              full multi-turn history. It&apos;s process-local — memory resets if the API
              process restarts.
            </div>
          </div>
          <div style={styles.noteCard}>
            <div style={styles.noteTitle}>Reference-only components</div>
            <div style={styles.noteBody}>
              The backend repo also includes standalone Pinecone and LlamaIndex implementations
              (<code style={styles.code}>retrieval/pinecone_store.py</code>,{" "}
              <code style={styles.code}>retrieval/llamaindex_loader.py</code>). Neither is wired
              into this graph — FAISS is the only vector backend the live agent queries.
            </div>
          </div>
        </div>
      </section>

      <section style={styles.section} aria-label="Technology stack">
        <h2 style={styles.sectionTitle}>Technology Stack</h2>
        <div style={styles.tableWrap}>
          <table style={styles.table} aria-label="Tech stack">
            <thead>
              <tr>
                {["Component", "Technology", "Notes"].map((h) => (
                  <th key={h} style={styles.th}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TECH_STACK.map((row, i) => (
                <tr key={row.component} style={i % 2 === 0 ? {} : { background: "var(--surface-2)" }}>
                  <td style={{ ...styles.td, color: "var(--text-primary)", fontWeight: 500 }}>{row.component}</td>
                  <td style={{ ...styles.td, fontFamily: "var(--font-mono)", color: "var(--accent)", fontSize: "13px" }}>
                    {row.technology}
                  </td>
                  <td style={{ ...styles.td, color: "var(--text-muted)", fontSize: "13px" }}>{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    padding: "32px",
    maxWidth: "860px",
    margin: "0 auto",
  },
  title: {
    fontSize: "26px",
    fontWeight: 700,
    color: "var(--text-primary)",
    marginBottom: "8px",
  },
  subtitle: {
    fontSize: "15px",
    color: "var(--text-muted)",
    lineHeight: 1.6,
    marginBottom: "36px",
    maxWidth: "620px",
  },
  section: {
    marginBottom: "40px",
  },
  sectionTitle: {
    fontSize: "16px",
    fontWeight: 600,
    color: "var(--text-primary)",
    marginBottom: "16px",
    paddingBottom: "10px",
    borderBottom: "1px solid var(--border)",
  },
  pipeline: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 0,
  },
  stepCard: {
    background: "var(--surface-1)",
    border: "1px solid var(--border)",
    borderTop: "3px solid var(--accent)",
    borderRadius: "8px",
    padding: "16px 20px",
    width: "100%",
    maxWidth: "560px",
    textAlign: "center",
  },
  stepName: {
    fontSize: "15px",
    fontWeight: 600,
    marginBottom: "4px",
  },
  stepDetail: {
    fontSize: "13px",
    color: "var(--text-muted)",
    lineHeight: 1.6,
  },
  branchLabel: {
    textAlign: "center",
    fontSize: "12px",
    color: "var(--text-muted)",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    marginBottom: "8px",
  },
  branchRow: {
    display: "flex",
    gap: "12px",
    width: "100%",
    maxWidth: "760px",
    margin: "0 auto",
  },
  arrow: {
    display: "flex",
    justifyContent: "center",
    padding: "4px 0",
  },
  code: {
    fontFamily: "var(--font-mono)",
    fontSize: "0.9em",
    background: "var(--surface-2)",
    padding: "1px 5px",
    borderRadius: "4px",
  },
  noteGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
    gap: "14px",
  },
  noteCard: {
    background: "var(--surface-1)",
    border: "1px solid var(--border)",
    borderRadius: "10px",
    padding: "16px 18px",
  },
  noteTitle: {
    fontSize: "13.5px",
    fontWeight: 600,
    color: "var(--text-primary)",
    marginBottom: "8px",
  },
  noteBody: {
    fontSize: "12.5px",
    color: "var(--text-secondary)",
    lineHeight: 1.6,
  },
  tableWrap: {
    overflowX: "auto",
    borderRadius: "8px",
    border: "1px solid var(--border)",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: "14px",
  },
  th: {
    padding: "10px 16px",
    textAlign: "left",
    fontSize: "12px",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    color: "var(--text-muted)",
    background: "var(--surface-2)",
    borderBottom: "1px solid var(--border)",
    whiteSpace: "nowrap",
  },
  td: {
    padding: "11px 16px",
    borderBottom: "1px solid var(--border)",
    color: "var(--text-secondary)",
    verticalAlign: "top",
  },
};
