export interface SourceMeta {
  marker: number;
  source: string;
  page: number | null;
  preview: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: SourceMeta[];
}
