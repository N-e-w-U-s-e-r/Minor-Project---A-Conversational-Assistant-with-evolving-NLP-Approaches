import "./Sidebar.css";

const levels = [
  {
    id: 1,
    name: "Rule-Based",
    icon: "①",
  },
  {
    id: 2,
    name: "Statistical",
    icon: "②",
  },
  {
    id: 3,
    name: "Classifier-Based",
    icon: "③",
  },
  {
    id: 4,
    name: "Transformer-Based",
    icon: "④",
  },
];

function Sidebar({
  open,
  setOpen,
  selectedLevel,
  setSelectedLevel,
}) {
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : "sidebar-compact"}`}>
      <div className="sidebar-header">
        {open && (
          <div className="sidebar-title">
            NLP Assistant
          </div>
        )}

        <button
          type="button"
          className="sidebar-toggle"
          onClick={() => setOpen(!open)}
          aria-label={open ? "Collapse sidebar" : "Expand sidebar"}
        >
          {open ? "‹" : "›"}
        </button>
      </div>

      {open && (
        <div className="levels-title">
          NLP Levels
        </div>
      )}

      <nav className="level-list">
        {levels.map((level) => (
          <button
            type="button"
            key={level.id}
            className={`level-button ${
              selectedLevel === level.id ? "selected" : ""
            }`}
            onClick={() => setSelectedLevel(level.id)}
            title={open ? "" : level.name}
          >
            <span className="level-icon">
              {level.icon}
            </span>

            {open && (
              <span className="level-name">
                {level.name}
              </span>
            )}
          </button>
        ))}
      </nav>
    </aside>
  );
}

export default Sidebar;