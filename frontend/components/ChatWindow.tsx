"use client";

import { useRef, useState } from "react";
import { streamChat } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import DocumentUpload from "./DocumentUpload";
import MessageBubble from "./MessageBubble";

export default function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function handleSend() {
    const query = input.trim();
    if (!query || isStreaming) return;

    const history = messages;
    setMessages((prev) => [...prev, { role: "user", content: query }, { role: "assistant", content: "" }]);
    setInput("");
    setIsStreaming(true);

    try {
      await streamChat(
        query,
        history,
        (token) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + token };
            return next;
          });
        },
        (sources) => {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], sources };
            return next;
          });
        },
        (errorMessage) => {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], content: `Error: ${errorMessage}` };
            return next;
          });
        }
      );
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          ...next[next.length - 1],
          content: `Error: ${err instanceof Error ? err.message : "failed to get a response"}`,
        };
        return next;
      });
    } finally {
      setIsStreaming(false);
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }

  return (
    <div className="chat-window">
      <DocumentUpload />
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">Upload a document, then ask a question about it.</div>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          {isStreaming ? "Thinking..." : "Send"}
        </button>
      </form>
    </div>
  );
}
