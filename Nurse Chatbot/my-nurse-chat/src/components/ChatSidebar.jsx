import React from 'react';
import './ChatSidebar.css';

function ChatSidebar({ sessions, activeSessionId, onNewSession, onSelectSession }) {
  return (
    <div className="sidebar-container">
      <h2>Sessions</h2>
      <button onClick={onNewSession}>+ New Chat</button>
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
