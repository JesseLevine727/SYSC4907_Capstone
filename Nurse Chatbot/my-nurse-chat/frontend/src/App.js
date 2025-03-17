import React, { useState } from 'react';
import ChatSidebar from './components/ChatSidebar';
import ChatWindow from './components/ChatWindow';
import './App.css';

function App() {
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [chatHistory, setChatHistory] = useState({});
  const [sessionFilePaths, setSessionFilePaths] = useState({});

  // "sessions" is a list of all session IDs we have in chatHistory
  const sessions = Object.keys(chatHistory);

  // Called when the user clicks "+ New Chat" in the sidebar
  const handleNewSession = (sessionId, filePath) => {
    // Create an empty chat array for the new session
    setChatHistory((prev) => ({
      ...prev,
      [sessionId]: []
    }));
    // Store the file path associated with this session
    setSessionFilePaths((prev) => ({
      ...prev,
      [sessionId]: filePath
    }));
    // Make this session active
    setActiveSessionId(sessionId);
  };

  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
  };

  // Used by ChatWindow to update the chat messages in state
  const updateHistory = (sessionId, messages) => {
    setChatHistory((prev) => ({
      ...prev,
      [sessionId]: messages
    }));
  };

  return (
    <div className="app-container">
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
      />

      {/* Render ChatWindow only if a session is active */}
      {activeSessionId && (
        <ChatWindow
          sessionId={activeSessionId}
          filePath={sessionFilePaths[activeSessionId]} // pass the file path for this session
          chatHistory={chatHistory}
          updateHistory={updateHistory}
        />
      )}
    </div>
  );
}

export default App;
