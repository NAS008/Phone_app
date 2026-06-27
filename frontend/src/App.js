import "./App.css";
import { Loader } from "./components";
import { QRCodePage } from "./pages";
import { useState, useMemo, useEffect, lazy, Suspense } from "react";
import { useLocation } from "react-router-dom";
import messageBusService from "./services/messageBusService";

const ArchivistPhonePage = lazy(() => import("./pages/ArchivistPage/ArchivistPhonePage"));
const ArchivistPCPage    = lazy(() => import("./pages/ArchivistPage/ArchivistPCPage"));

function App() {
  const location = useLocation();
  const [adminNickname, setAdminNickname] = useState("NAS");

  useEffect(() => {
    messageBusService.fetchConfig()
      .then(cfg => { if (cfg?.adminNickname) setAdminNickname(cfg.adminNickname); })
      .catch(() => {});
  }, []);

  // Session ID from QR code URL: ?session_id=<uuid>
  const urlSessionId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get("session_id") || "";
  }, [location.search]);

  // null = not joined yet; adminNickname or any string = joined
  const [nickname, setNickname] = useState(null);
  const [appMode, setAppMode] = useState("artist"); // "artist" | "archivist"
  // For ADMIN: the session they typed in (overrides urlSessionId)
  const [adminSessionId, setAdminSessionId] = useState(null);

  const isAdmin = nickname === adminNickname;
  const sessionId = isAdmin ? (adminSessionId || "") : urlSessionId;

  // Called by Loader when a normal user submits their nickname
  const handleNicknameSubmit = (submittedNickname, mode = "artist") => {
    setNickname(submittedNickname);
    setAppMode(mode);
    if (mode === "artist") {
      messageBusService
        .sendUserJoined(urlSessionId, submittedNickname)
        .catch((err) => console.warn("user_joined publish failed:", err));
    }
  };

  // Called by Loader when ADMIN submits their session ID — jumps straight to main app
  const handleAdminSessionSubmit = (sid) => {
    setAdminSessionId(sid);
    setNickname(adminNickname);
    messageBusService
      .sendUserJoined(sid, adminNickname)
      .catch((err) => console.warn("user_joined publish failed:", err));
  };

  // ── /archivist — PC browser page (no login needed) ───────────────────────
  if (location.pathname === "/archivist") {
    return (
      <Suspense fallback={null}>
        <ArchivistPCPage />
      </Suspense>
    );
  }

  // ── Entry page ────────────────────────────────────────────────────────────
  if (!nickname) {
    return (
      <Loader
        isClaimMode={true}
        sessionId={urlSessionId}
        adminNickname={adminNickname}
        onNicknameSubmit={handleNicknameSubmit}
        onAdminSessionSubmit={handleAdminSessionSubmit}
        onComplete={() => {}}
      />
    );
  }

  // ── Archivist phone page ──────────────────────────────────────────────────
  if (appMode === "archivist") {
    return (
      <Suspense fallback={null}>
        <ArchivistPhonePage nickname={nickname} />
      </Suspense>
    );
  }

  // ── Main artist app ───────────────────────────────────────────────────────
  return (
    <div className="app-container">
      <QRCodePage nickname={nickname} sessionId={sessionId} isAdmin={isAdmin} />
    </div>
  );
}

export default App;
