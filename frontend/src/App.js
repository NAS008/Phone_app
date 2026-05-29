import "./App.css";
import { Loader } from "./components";
import { QRCodePage } from "./pages";
import { useState, useMemo } from "react";
import { useLocation } from "react-router-dom";
import messageBusService from "./services/messageBusService";

const ADMIN = "NAS"

function App() {
  const location = useLocation();

  // Session ID from QR code URL: ?session_id=<uuid>
  const urlSessionId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get("session_id") || "";
  }, [location.search]);

  // null = not joined yet; ADMIN or any string = joined
  const [nickname, setNickname] = useState(null);
  // For ADMIN: the session they typed in (overrides urlSessionId)
  const [adminSessionId, setAdminSessionId] = useState(null);

  const isAdmin = nickname === ADMIN;
  const sessionId = isAdmin ? (adminSessionId || "") : urlSessionId;

  // Called by Loader when a normal user submits their nickname
  const handleNicknameSubmit = (submittedNickname) => {
    setNickname(submittedNickname);
    messageBusService
      .sendUserJoined(urlSessionId, submittedNickname)
      .catch((err) => console.warn("user_joined publish failed:", err));
  };

  // Called by Loader when ADMIN submits their session ID — jumps straight to main app
  const handleAdminSessionSubmit = (sid) => {
    setAdminSessionId(sid);
    setNickname(ADMIN);
    messageBusService
      .sendUserJoined(sid, ADMIN)
      .catch((err) => console.warn("user_joined publish failed:", err));
  };

  // ── Entry page ────────────────────────────────────────────────────────────
  if (!nickname) {
    return (
      <Loader
        isClaimMode={true}
        sessionId={urlSessionId}
        adminNickname={ADMIN}
        onNicknameSubmit={handleNicknameSubmit}
        onAdminSessionSubmit={handleAdminSessionSubmit}
        onComplete={() => {}}
      />
    );
  }

  // ── Main app ──────────────────────────────────────────────────────────────
  return (
    <div className="app-container">
      <QRCodePage nickname={nickname} sessionId={sessionId} isAdmin={isAdmin} />
    </div>
  );
}

export default App;
