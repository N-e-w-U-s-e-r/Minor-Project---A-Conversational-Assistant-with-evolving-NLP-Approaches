import { useEffect, useState } from "react";
import "./DocumentRag.css";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:5000";

export default function DocumentRag() {
  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");

  async function loadDocuments() {
    try {
      const response = await fetch(
        `${API_BASE}/api/documents`
      );

      if (!response.ok) {
        throw new Error("Could not load documents.");
      }

      const data = await response.json();

      setDocuments(data.documents || []);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  async function uploadDocument(event) {
    event.preventDefault();

    if (!file) {
      setError("Select a document first.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(
        `${API_BASE}/api/documents`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Upload failed."
        );
      }

      setFile(null);

      // Reset the HTML file input.
      event.target.reset();

      await loadDocuments();

      setMessages((previous) => [
        ...previous,
        {
          role: "system",
          text:
            `Uploaded ${data.filename}. ` +
            `${data.chunks_indexed} chunks indexed.`,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();

    const userQuestion = question.trim();

    if (!userQuestion) {
      return;
    }

    setQuestion("");
    setError("");

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        text: userQuestion,
      },
    ]);

    setAsking(true);

    try {
      const response = await fetch(
        `${API_BASE}/api/rag/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: userQuestion,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Could not answer the question."
        );
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources || [],
          retrievedChunks: data.retrieved_chunks || 0,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="document-rag">
      <h1>Document Assistant</h1>

      <p>
        Upload your documents and ask questions
        about their contents.
      </p>

      <section className="rag-panel">
        <h2>Upload a document</h2>

        <form onSubmit={uploadDocument}>
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={(event) =>
              setFile(event.target.files?.[0] || null)
            }
          />

          <button
            type="submit"
            disabled={!file || uploading}
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </form>

        <p className="rag-note">
          Supported formats: PDF, DOCX, TXT and MD.
          Maximum file size: 15 MB.
        </p>
      </section>

      <section className="rag-panel">
        <h2>Uploaded documents</h2>

        {documents.length === 0 ? (
          <p>No documents uploaded yet.</p>
        ) : (
          <ul>
            {documents.map((document) => (
              <li key={document.filename}>
                {document.filename}
                {" "}
                ({Math.ceil(document.size_bytes / 1024)} KB)
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="rag-panel">
        <h2>Ask your documents</h2>

        <div className="rag-messages">
          {messages.map((message, index) => (
            <article
              className={`rag-message ${message.role}`}
              key={`${message.role}-${index}`}
            >
              <strong>
                {message.role === "user"
                  ? "You"
                  : message.role === "assistant"
                  ? "Assistant"
                  : "System"}
              </strong>

              <p>{message.text}</p>

              {message.sources?.length > 0 && (
                <div className="rag-sources">
                  <strong>Sources</strong>

                  <ul>
                    {message.sources.map(
                      (source, sourceIndex) => (
                        <li key={sourceIndex}>
                          {source.filename}

                          {source.page != null
                            ? `, page ${source.page}`
                            : ""}

                          {" — similarity: "}
                          {source.similarity.toFixed(3)}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              )}
            </article>
          ))}
        </div>

        <form onSubmit={askQuestion}>
          <textarea
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            placeholder="Ask something about your documents..."
            rows={3}
            maxLength={5000}
          />

          <button
            type="submit"
            disabled={!question.trim() || asking}
          >
            {asking ? "Thinking..." : "Ask"}
          </button>
        </form>
      </section>

      {error && (
        <p className="rag-error" role="alert">
          {error}
        </p>
      )}
    </main>
  );
}