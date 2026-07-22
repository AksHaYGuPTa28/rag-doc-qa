import type { SourceMeta } from "@/lib/types";

export default function SourceCitations({ sources }: { sources: SourceMeta[] }) {
  return (
    <div className="citations">
      <div className="citations-title">Sources</div>
      <ul>
        {sources.map((s) => (
          <li key={s.marker}>
            <span className="marker">[{s.marker}]</span>{" "}
            <span className="source-name">
              {s.source}
              {s.page ? `, p.${s.page}` : ""}
            </span>
            <div className="preview">{s.preview}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
