import "./Sidebar.css";

const levels = [
  {
    id: 1,
    name: "Rule-Based",
    description: "Pattern matching and rules",
    icon: "01",
  },
  {
    id: 2,
    name: "Statistical",
    description: "Statistical language models",
    icon: "02",
  },
  {
    id: 3,
    name: "Classifier-Based",
    description: "Machine learning classifiers",
    icon: "03",
  },
  {
    id: 4,
    name: "Transformer-Based",
    description: "Transformer-powered NLP",
    icon: "04",
  },
  {
    id: 5,
    name: "Knowledge Base",
    description: "Document search and RAG",
    icon: "05",
  },
];

function Sidebar({
  open,
  setOpen,
  selectedLevel,
  setSelectedLevel,
}) {
  return (
    <aside
      className={`sidebar ${
        open ? "sidebar-open" : "sidebar-compact"
      }`}
    >
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="sidebar-logo">N</div>

          {open && (
            <div className="sidebar-brand-copy">
              <span className="sidebar-brand-title">NLP LAB</span>
              <span className="sidebar-brand-subtitle">
                Assistant workspace
              </span>
            </div>
          )}
        </div>

        <button
          type="button"
          className="sidebar-toggle"
          onClick={() => setOpen(!open)}
          aria-label={open ? "Collapse sidebar" : "Expand sidebar"}
          title={open ? "Collapse sidebar" : "Expand sidebar"}
        >
          {open ? "‹" : "›"}
        </button>
      </div>

      {open && (
        <div className="levels-title">
          <span>YOUR WORKSPACE</span>
          <span className="level-count">05</span>
        </div>
      )}

      <nav className="level-list" aria-label="Assistant levels">
        {levels.map((level) => {
          const selected = selectedLevel === level.id;

          return (
            <button
              type="button"
              key={level.id}
              className={`level-button ${
                selected ? "selected" : ""
              } ${level.id === 5 ? "knowledge-level" : ""}`}
              onClick={() => setSelectedLevel(level.id)}
              title={open ? level.name : `${level.name} — ${level.description}`}
              aria-current={selected ? "page" : undefined}
            >
              <span className="level-icon">{level.icon}</span>

              {open && (
                <span className="level-copy">
                  <span className="level-name">{level.name}</span>
                  <span className="level-description">
                    {level.description}
                  </span>
                </span>
              )}

              {open && selected && (
                <span className="level-active-dot" />
              )}
            </button>
          );
        })}
      </nav>

      {open && (
        <div className="sidebar-bottom">
          <div className="sidebar-bottom-mark">✦</div>

          <div>
            <span className="sidebar-bottom-title">
              Explore NLP
            </span>
            <span className="sidebar-bottom-description">
              Five approaches to intelligent assistance.
            </span>
          </div>
        </div>
      )}
    </aside>
  );
}

export default Sidebar;