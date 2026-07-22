import type { ChatMessage, SourceMeta } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/documents/upload`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ document_id: string; source: string; chunks_indexed: number }>;
}

export async function listDocuments() {
  const res = await fetch(`${API_BASE}/api/documents`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function streamChat(
  query: string,
  history: ChatMessage[],
  onToken: (token: string) => void,
  onSources: (sources: SourceMeta[]) => void,
  onError?: (message: string) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      history: history.map(({ role, content }) => ({ role, content })),
    }),
  });

  if (!res.ok || !res.body) {
    throw new Error(await res.text());
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const event of events) {
      const line = event.trim();
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice("data: ".length);
      if (payload === "[DONE]") continue;

      const parsed = JSON.parse(payload);
      if (parsed.type === "token") onToken(parsed.content as string);
      if (parsed.type === "sources") onSources(parsed.content as SourceMeta[]);
      if (parsed.type === "error") onError?.(parsed.content as string);
    }
  }
}
