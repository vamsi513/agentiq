export type RouteDecision = "retrieval" | "web_search" | "direct";

export interface ChatRequest {
  query: string;
  session_id?: string;
}

export interface SourceItem {
  title: string;
  url: string;
  content: string;
  score: number;
  type: RouteDecision;
}

export interface ChatResponse {
  answer: string;
  sources: SourceItem[];
  route_decision: RouteDecision;
  session_id: string;
  turn_count: number;
  retrieval_score: number;
}

export interface HealthResponse {
  status: string;
  version: string;
}

/** A single Server-Sent Event chunk emitted by POST /chat/stream. */
export interface StreamChunk {
  type: "route" | "token" | "sources" | "done" | "error";
  data: RouteDecision | string | SourceItem[] | null;
}

/** A message in the client-side conversation view. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Present once the assistant message has finished streaming. */
  routeDecision?: RouteDecision;
  sources?: SourceItem[];
  latencyMs?: number;
  /** True while tokens are still arriving for this message. */
  streaming?: boolean;
  /** True if the message ended in an error. */
  error?: boolean;
}
