import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ChatWindow.css';

function ChatWindow({ sessionId, onUpdateHistory, chatHistory }) {
  const [input, setInput] = useState('');

  // Load history from backend on mount or session change
  useEffect(() => {
    if (sessionId) {
      axios.get(`http://localhost:5000/api/history?session_id=${sessionId}`)
        .then((res) => {
          onUpdateHistory(sessionId, res.data.messages);
        })
        .catch(console.error);
    }
  }, [sessionId]);

  const messages = chatHistory[sessionId] || [];

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };

    // Optimistically update local state
    onUpdateHistory(sessionId, [...messages, userMsg]);
    setInput('');

    // Call the backend
    try {
      const res = await axios.post('http://localhost:5000/api/chat', {
        session_id: sessionId,
        message: input,
        file_path: 'Kate_Data.json' // Or pass in whichever file you want
      });
      const reply = res.data.reply || 'No response';

      const aiMsg = { role: 'assistant', content: reply };
      onUpdateHistory(sessionId, [...messages, userMsg, aiMsg]);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="chat-window-container">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`message ${msg.role === 'assistant' ? 'assistant' : 'user'}`}
          >
            {msg.content}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default ChatWindow;
