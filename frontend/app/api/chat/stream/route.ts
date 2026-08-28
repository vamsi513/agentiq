import { NextRequest, NextResponse } from "next/server";
import { API_BASE } from "@/lib/upstream";

const UPSTREAM = `${API_BASE}/chat/stream`;

// Never cache or statically optimize this route — every call is a fresh
// stream tied to a specific query/session.
export const dynamic = "force-dynamic";

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

    // Pass the upstream body straight through — no intermediate buffering.
    return new NextResponse(upstream.body, { status: 200, headers });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
