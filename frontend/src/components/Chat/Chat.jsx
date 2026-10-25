
import { useEffect, useRef, useState } from "react";
import Message from "../Message/Message";
import "./Chat.css";

function Chat({ selectedLevel }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messageEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  useEffect(() => {
    if (!loading) {
      inputRef.current?.focus();
    }
  }, [loading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [selectedLevel]);

  useEffect(() => {
    function handleSlash(event) {
      if (
        event.key === "/" &&
        document.activeElement !== inputRef.current
      ) {
        event.preventDefault();
        inputRef.current?.focus();
      }
    }

    window.addEventListener("keydown", handleSlash);

    return () => {
      window.removeEventListener("keydown", handleSlash);
    };
  }, []);

  async function sendMessage() {
    const message = input.trim();

    if (!message || loading) {
      return;
    }

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: message,
        level: selectedLevel,
      },
    ]);

    setInput("");
    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/api/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,
            level: selectedLevel,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();

        throw new Error(
          errorData.error || `Request failed: ${response.status}`
        );
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.response,
          level: data.level,
        },
      ]);
    } catch (error) {
      console.error("Chat request failed:", error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Sorry, I couldn't connect to the NLP backend.",
          level: selectedLevel,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  return (
    <section className="chat">

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-chat">
            <h1>NLP Conversational Assistant</h1>
            <p>
              Select an NLP level and start a conversation.
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <Message
            key={index}
            role={message.role}
            content={message.content}
            level={message.level}
          />
        ))}

        {loading && (
          <Message
            role="assistant"
            content="Thinking..."
            level={selectedLevel}
          />
        )}

        <div ref={messageEndRef} />
      </div>

      <div className="input-area">
        <div className="input-container">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message your assistant..."
            rows={1}
            disabled={loading}
          />

          <button
            type="button"
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            aria-label="Send message"
          >
            ↑
          </button>
        </div>
      </div>
    </section>
  );
}

export default Chat;
