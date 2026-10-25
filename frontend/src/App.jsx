import { useState } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import Chat from "./components/Chat/Chat";
import DocumentRag from "./DocumentRag";
import "./App.css";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedLevel, setSelectedLevel] = useState(1);

  const isKnowledgeBase = selectedLevel === 5;

  const levelNames = {
    1: "Rule-Based",
    2: "Statistical",
    3: "Classifier-Based",
    4: "Transformer-Based",
    5: "Knowledge Base",
  };

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        setOpen={setSidebarOpen}
        selectedLevel={selectedLevel}
        setSelectedLevel={setSelectedLevel}
      />

      <main className="main-content">
        <header className="workspace-header">
          <div className="workspace-heading">
            <div className="brand-mark">N</div>

            <div className="brand-copy">
              <h1>NLP Assistant</h1>
              <p>
                Five approaches. One conversational workspace.
              </p>
            </div>
          </div>

          <div className="current-level">
            <span className="current-level-dot" />
            <span>LEVEL {selectedLevel}</span>
            <span className="current-level-divider">/</span>
            <strong>{levelNames[selectedLevel]}</strong>
          </div>
        </header>

        <section
          className={`workspace-content ${
            isKnowledgeBase ? "knowledge-workspace" : "chat-workspace"
          }`}
          key={selectedLevel}
        >
          {isKnowledgeBase ? (
            <DocumentRag />
          ) : (
            <Chat selectedLevel={selectedLevel} />
          )}
        </section>

        <footer className="workspace-footer">
          <span className="footer-indicator" />
          <span>Conversational AI</span>
          <span className="footer-divider">·</span>
          <span>Five-level architecture</span>
        </footer>
      </main>
    </div>
  );
}

export default App;