
import "./Message.css";

function Message({ role, content, level }) {
  return (
    <div className={`message ${role}`}>
      {/* <div className="message-header">
        {level && (
          <span className="message-level">
            Level {level}
          </span>
        )}
      </div> */}

      <div className="message-content">
        {content}
      </div>
    </div>
  );
}

export default Message;

