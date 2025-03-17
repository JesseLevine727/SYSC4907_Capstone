import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ChatWindow.css';

function ChatWindow({ sessionId, filePath, chatHistory, updateHistory }) {
  const [input, setInput] = useState('');
  const messages = chatHistory[sessionId] || [];

  // Load existing history for this session
  useEffect(() => {
    axios.get(`http://localhost:5000/api/history?session_id=${sessionId}`)
      .then((res) => {
        updateHistory(sessionId, res.data.messages);
      })
      .catch(console.error);
  }, [sessionId, updateHistory]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    const updatedMessages = [...messages, userMsg];

    // Optimistically update the UI
    updateHistory(sessionId, updatedMessages);
    const currentInput = input;
    setInput('');

    try {
      // Include filePath in the request body
      const res = await axios.post('http://localhost:5000/api/chat', {
        session_id: sessionId,
        message: currentInput,
        file_path: filePath
      });
      const reply = res.data.reply || 'No response';

      const aiMsg = { role: 'assistant', content: reply };
      updateHistory(sessionId, [...updatedMessages, aiMsg]);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="chat-window-container">
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
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
