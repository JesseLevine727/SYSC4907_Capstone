import React, { useState } from 'react';
import ChatSidebar from './components/ChatSidebar';
import ChatWindow from './components/ChatWindow';
import './App.css';

function App() {
  const [activeSessionId, setActiveSessionId] = useState('session-1');
  const [chatHistory, setChatHistory] = useState({}); // {sessionId: [ {role, content}, ... ]}

  const sessions = Object.keys(chatHistory);

  const handleNewSession = () => {
    const newId = `session-${Date.now()}`;
    setChatHistory((prev) => ({
      ...prev,
      [newId]: []
    }));
    setActiveSessionId(newId);
  };

  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
  };

  const onUpdateHistory = (sessionId, messages) => {
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
      <ChatWindow
        sessionId={activeSessionId}
        chatHistory={chatHistory}
        onUpdateHistory={onUpdateHistory}
      />
    </div>
  );
}

export default App;
