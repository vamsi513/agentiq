import { NextRequest, NextResponse } from "next/server";
import { API_BASE } from "@/lib/upstream";

const UPSTREAM = `${API_BASE}/chat/stream`;

// Never cache or statically optimize this route — every call is a fresh
// stream tied to a specific query/session.
export const dynamic = "force-dynamic";

// Time to wait for the backend to start responding at all (headers/first
// byte). A plain overall timeout would be wrong here — a real stream can
// legitimately run well past 15s while still actively producing tokens.
const FIRST_BYTE_TIMEOUT_MS = 15_000;

// Once streaming has started, how long we'll wait between chunks before
// treating the connection as hung and closing it. Resets on every chunk.
const IDLE_TIMEOUT_MS = 30_000;

/**
 * Wrap a ReadableStream so it self-terminates if no chunk arrives within
 * `idleMs` of the previous one — without this, a backend that stops
 * producing data mid-stream (network partition, hung LLM call) would leave
 * the connection open indefinitely with no defined failure state.
 */
function withIdleTimeout(source: ReadableStream<Uint8Array>, idleMs: number): ReadableStream<Uint8Array> {
  const reader = source.getReader();
  let timer: ReturnType<typeof setTimeout>;

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const timeoutPromise = new Promise<"timeout">((resolve) => {
        timer = setTimeout(() => resolve("timeout"), idleMs);
      });

      const result = await Promise.race([reader.read(), timeoutPromise]);
      clearTimeout(timer);

      if (result === "timeout") {
        controller.error(new Error("Stream idle timeout — no data received."));
        await reader.cancel().catch(() => {});
        return;
      }
      if (result.done) {
        controller.close();
        return;
      }
      controller.enqueue(result.value);
    },
    cancel(reason) {
      clearTimeout(timer);
      return reader.cancel(reason);
    },
  });
}

// Streaming proxy: forwards the SSE response from the AgentIQ backend to the
// browser as-is, chunk by chunk. This must NOT buffer the response — piping
// the upstream ReadableStream straight through (rather than awaiting
// upstream.text()/.json() first) is what keeps tokens arriving incrementally
// instead of all landing at once when the backend finishes.
export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json();

    const upstream = await fetch(UPSTREAM, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(FIRST_BYTE_TIMEOUT_MS),
    });

    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text().catch(() => "");
      return NextResponse.json(
        { error: text || `Upstream error (${upstream.status})` },
        { status: upstream.status || 502 }
      );
    }

    const sessionId = upstream.headers.get("X-Session-Id");
    const headers: Record<string, string> = {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    };
    if (sessionId) headers["X-Session-Id"] = sessionId;

    return new NextResponse(withIdleTimeout(upstream.body, IDLE_TIMEOUT_MS), {
      status: 200,
      headers,
    });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      return NextResponse.json(
        { error: "The backend didn't respond in time. Please try again." },
        { status: 504 }
      );
    }
    const message = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
