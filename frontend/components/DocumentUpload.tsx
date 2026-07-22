"use client";

import { useRef, useState } from "react";
import { uploadDocument } from "@/lib/api";

export default function DocumentUpload() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleFile(file: File) {
    setBusy(true);
    setStatus(`Indexing ${file.name}...`);
    try {
      const result = await uploadDocument(file);
      setStatus(`Indexed "${file.name}" -- ${result.chunks_indexed} chunks`);
    } catch (err) {
      setStatus(`Failed to index "${file.name}": ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="upload-bar">
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />
      <button type="button" disabled={busy} onClick={() => inputRef.current?.click()}>
        {busy ? "Uploading..." : "Upload document"}
      </button>
      {status && <span className="upload-status">{status}</span>}
    </div>
  );
}
