import { React, useState } from "react";
import "./Nickname.css";

// component for Nickname typing
const Nickname = ({ onNicknameSubmit }) => {
  const [nickname, setNickname] = useState("");

  const handleInputChange = (e) => {
    setNickname(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (nickname.trim()) {
      onNicknameSubmit(nickname.trim());
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSubmit(e);
    }
  };

  return (
    <div className="nickname-container">
      <label className="nickname-label">Nickname:</label>
      <input
        type="text"
        value={nickname}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        className="nickname-input"
        placeholder="Enter your nickname"
        autoFocus
      />
    </div>
  );
};

export default Nickname;
