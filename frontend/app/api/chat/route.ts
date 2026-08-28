import { NextRequest, NextResponse } from "next/server";
import { API_BASE } from "@/lib/upstream";

const UPSTREAM = `${API_BASE}/chat`;

// Ceiling for the whole non-streaming request — the backend has no defined
// upper bound of its own, so without this a hung EC2/model call would leave
// the Vercel function (and the waiting browser) with no failure state at all.
const REQUEST_TIMEOUT_MS = 30_000;

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json();

    const upstream = await fetch(UPSTREAM, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });

    const data = await upstream.json();

    return NextResponse.json(data, { status: upstream.status });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      return NextResponse.json(
        { error: "The request took too long. Please try again." },
        { status: 504 }
      );
    }
    const message = err instanceof Error ? err.message : "Internal error";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
