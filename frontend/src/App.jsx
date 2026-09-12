import { useState } from "react";
import Sidebar from "./components/Sidebar/Sidebar";
import Chat from "./components/Chat/Chat";
import "./App.css";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedLevel, setSelectedLevel] = useState(1);

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        setOpen={setSidebarOpen}
        selectedLevel={selectedLevel}
        setSelectedLevel={setSelectedLevel}
      />

      <main className="main-content">
        <Chat selectedLevel={selectedLevel} />
      </main>
    </div>
  );
}

export default App;