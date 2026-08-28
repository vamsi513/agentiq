import type { ChatRequest, ChatResponse, StreamChunk } from "./types";

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    if (response.status >= 500) throw new Error("AgentIQ is temporarily unavailable. Please try again shortly.");
    if (response.status === 404) throw new Error("Chat endpoint not found. The backend may be restarting.");
    const text = await response.text().catch(() => "");
    throw new Error(text || `Request failed (${response.status})`);
  }

  return response.json() as Promise<ChatResponse>;
}

export interface StreamCallbacks {
  onToken: (chunk: string) => void;
  onSources: (sources: StreamChunk["data"] & unknown[]) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

/**
 * Open the streaming chat endpoint and dispatch each SSE event to the
 * matching callback as it arrives. Reads the response body incrementally
 * via a ReadableStream reader — never waits for the whole response.
 */
export async function streamChat(
  request: ChatRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    callbacks.onError(text || `Request failed (${response.status})`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line ("\n\n").
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        let parsed: StreamChunk;
        try {
          parsed = JSON.parse(jsonStr) as StreamChunk;
        } catch {
          continue;
        }

        switch (parsed.type) {
          case "token":
            if (typeof parsed.data === "string") callbacks.onToken(parsed.data);
            break;
          case "sources":
            if (Array.isArray(parsed.data)) callbacks.onSources(parsed.data as never);
            break;
          case "error":
            callbacks.onError(typeof parsed.data === "string" ? parsed.data : "Unknown error");
            break;
          case "done":
            callbacks.onDone();
            break;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
