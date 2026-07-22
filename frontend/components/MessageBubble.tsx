import type { ChatMessage } from "@/lib/types";
import SourceCitations from "./SourceCitations";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  return (
    <div className={`bubble ${message.role}`}>
      <div className="bubble-role">{message.role === "user" ? "You" : "Assistant"}</div>
      <div className="bubble-content">{message.content || (message.role === "assistant" ? "..." : "")}</div>
      {message.sources && message.sources.length > 0 && <SourceCitations sources={message.sources} />}
    </div>
  );
}
