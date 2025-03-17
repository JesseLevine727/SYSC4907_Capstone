import React, { useState } from 'react';
import './ChatSidebar.css';

function ChatSidebar({ sessions, activeSessionId, onNewSession, onSelectSession }) {
  const [customSessionId, setCustomSessionId] = useState('');
  const [filePath, setFilePath] = useState('');

  const handleCreateSession = () => {
    // If no custom session ID is provided, generate one
    const sessionId = customSessionId.trim() || `session-${Date.now()}`;

    // Pass both sessionId and filePath up to the parent
    onNewSession(sessionId, filePath);

    // Reset local inputs
    setCustomSessionId('');
    setFilePath('');
  };

  return (
    <div className="sidebar-container">
      <h2>Sessions</h2>

      {/* Input for custom session ID */}
      <input
        type="text"
        placeholder="Enter custom session ID"
        value={customSessionId}
        onChange={(e) => setCustomSessionId(e.target.value)}
      />

      {/* Input for file path */}
      <input
        type="text"
        placeholder="Enter file path"
        value={filePath}
        onChange={(e) => setFilePath(e.target.value)}
      />

      <button onClick={handleCreateSession}>+ New Chat</button>

      <ul>
        {sessions.map((session) => (
          <li
            key={session}
            className={session === activeSessionId ? 'active-session' : ''}
            onClick={() => onSelectSession(session)}
          >
            {session}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ChatSidebar;
