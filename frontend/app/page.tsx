import ChatWindow from "@/components/ChatWindow";

export default function Home() {
  return (
    <main className="page">
      <header className="page-header">
        <h1>Enterprise Document Q&amp;A</h1>
        <p>Ask questions about your internal documents, grounded with source citations.</p>
      </header>
      <ChatWindow />
    </main>
  );
}
