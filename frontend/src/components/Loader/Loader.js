import { useState, useEffect, useRef, useCallback } from "react";
import "./Loader.css";
import { Nickname } from "../../components";
import officialLogo from "../../assets/logo/the-first-noncarbon-artist.png";

const LOADER_LINES_OF_TEXT = ["THE", "FIRST", "NONCARBON", "ARTIST"];

const Loader = ({
  onComplete,
  isQrCodePage = false,
  onNicknameSubmit = null,
  onAdminSessionSubmit = null,
  sessionId,
  isClaimMode = false,
  adminNickname = "NAS",
}) => {
  const [positions, setPositions] = useState([]);
  const [textLines, setTextLines] = useState([]);
  const [animationPhase, setAnimationPhase] = useState("rectangle");
  const [animationDone, setAnimationDone] = useState(false);
  const [showNickname, setShowNickname] = useState(false);

  const [nickname, setNickname] = useState("");
  const [adminSessionInput, setAdminSessionInput] = useState("");
  const [showScanQR, setShowScanQR] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const adminSessionRef = useRef(null);

  const pixelSize = 6;
  const rectWidth = 30;
  const rectHeight = 20;
  const totalWidth = rectWidth * pixelSize;
  const totalHeight = rectHeight * pixelSize;

  // Use refs to avoid dependency issues
  const onCompleteRef = useRef(onComplete);
  const onNicknameSubmitRef = useRef(onNicknameSubmit);

  useEffect(() => {
    onCompleteRef.current = onComplete;
    onNicknameSubmitRef.current = onNicknameSubmit;
  }, [onComplete, onNicknameSubmit]);

  // When user finishes typing the admin nickname, auto-focus the session input
  useEffect(() => {
    if (nickname === adminNickname && adminSessionRef.current) {
      adminSessionRef.current.focus();
    }
  }, [nickname, adminNickname]);

  const handleNicknameSubmit = useCallback((nick) => {
    if (onNicknameSubmitRef.current) {
      onNicknameSubmitRef.current(nick);
    }
    setAnimationPhase("complete");
  }, []);

  const isAdminTyped = nickname === adminNickname;

  const handleClaimSubmit = (e) => {
    e.preventDefault();
    setError(null);

    if (isAdminTyped) {
      // ── ADMIN path ──
      const sid = adminSessionInput.trim();
      if (!sid) {
        setError("Please enter a session ID");
        return;
      }
      setIsSubmitting(true);
      if (onAdminSessionSubmit) {
        onAdminSessionSubmit(sid);
      }
      return;
    }

    // ── Normal user path ──
    const trimmed = nickname.trim();
    if (!trimmed) {
      setError("Please enter a nickname");
      return;
    }
    if (!sessionId) {
      // No session from QR code URL — show scan card
      setShowScanQR(true);
      return;
    }
    if (trimmed.length > 50) {
      setError("Nickname must be 50 characters or less");
      return;
    }
    setIsSubmitting(true);
    if (onNicknameSubmitRef.current) {
      onNicknameSubmitRef.current(trimmed);
    }
  };

  const getTextPosition = (index) => {
    const verticalSpacing = 4;
    const topMargin = 3;
    return {
      width: `${(rectWidth - 6) * pixelSize}px`,
      left: `${3 * pixelSize}px`,
      top: `${(index * verticalSpacing + topMargin) * pixelSize}px`,
      fontSize: `${pixelSize * 2.8}px`,
    };
  };

  // Rectangle animation - only runs once on mount
  useEffect(() => {
    const startX = -Math.floor(rectWidth / 2);
    const startY = -Math.floor(rectHeight / 2);
    let rectangle = [];
    let currentPos = { x: startX, y: startY };
    let step = 0;
    let maxSteps = 2 * (rectWidth + rectHeight) - 4;

    const interval = setInterval(() => {
      rectangle.push({ ...currentPos });
      if (step < rectWidth - 1) {
        currentPos.x += 1;
      } else if (step < rectWidth + rectHeight - 2) {
        currentPos.y += 1;
      } else if (step < 2 * rectWidth + rectHeight - 3) {
        currentPos.x -= 1;
      } else if (step < 2 * rectWidth + 2 * rectHeight - 4) {
        currentPos.y -= 1;
      }
      step++;
      setPositions([...rectangle]);
      if (step >= maxSteps) {
        clearInterval(interval);
        setAnimationPhase("text");
      }
    }, 12);

    return () => clearInterval(interval);
  }, []);

  // Text animation - runs when phase changes to "text"
  useEffect(() => {
    if (animationPhase !== "text") return;

    let currentLines = [];
    let lineIndex = 0;
    const textInterval = setInterval(() => {
      if (lineIndex < LOADER_LINES_OF_TEXT.length) {
        currentLines.push(LOADER_LINES_OF_TEXT[lineIndex]);
        setTextLines([...currentLines]);
        lineIndex++;
        if (lineIndex === LOADER_LINES_OF_TEXT.length) {
          clearInterval(textInterval);
          if (isQrCodePage && onNicknameSubmitRef.current) {
            setTimeout(() => setShowNickname(true), 600);
          } else {
            setTimeout(() => setAnimationPhase("complete"), 600);
          }
        }
      } else {
        clearInterval(textInterval);
      }
    }, 400);

    return () => clearInterval(textInterval);
  }, [animationPhase, isQrCodePage]);

  // Complete animation
  useEffect(() => {
    if (animationPhase !== "complete") return;
    setAnimationDone(true);
    const timer = setTimeout(() => {
      if (onCompleteRef.current) onCompleteRef.current();
    }, 1000);
    return () => clearTimeout(timer);
  }, [animationPhase]);

  // ── Claim mode ────────────────────────────────────────────────────────────
  if (isClaimMode) {
    // ── "Scan the QR code" card ───────────────────────────────────────────
    if (showScanQR) {
      return (
        <div className="container claim-mode">
          <div className="claim-shell">
            <div className="claim-card">
              <div className="claim-visual">
                <img
                  src={officialLogo}
                  alt="The First NonCarbon Artist"
                  className="claim-logo-image"
                />
              </div>
              <div className="claim-panel">
                <div className="claim-copy">
                  <h1 className="claim-title">Scan the QR code to join</h1>
                </div>
                <button
                  className="claim-button"
                  onClick={() => {
                    setShowScanQR(false);
                    setNickname("");
                    setError(null);
                  }}
                >
                  Go Back
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // ── Main entry card (nickname + session, reactive) ─────────────────────
    return (
      <div className="container claim-mode">
        <div className="claim-shell">
          <div className="claim-card">
            <div className="claim-visual">
              <img
                src={officialLogo}
                alt="The First NonCarbon Artist"
                className="claim-logo-image"
              />
            </div>

            <div className="claim-panel">
              <div className="claim-copy">
                <h1 className="claim-title">
                  {isAdminTyped ? "Admin Access" : "Welcome!"}
                </h1>
              </div>

              <form onSubmit={handleClaimSubmit} className="claim-form">
                {/* ── Upper box: nickname input OR locked "ADMIN" display ── */}
                {isAdminTyped ? (
                  <div className="claim-session-panel">
                    <span className="claim-session-label">Nickname</span>
                    <span className="claim-session-id">{adminNickname}</span>
                  </div>
                ) : (
                  <input
                    id="claim-nickname"
                    type="text"
                    value={nickname}
                    onChange={(e) => {
                      setNickname(e.target.value);
                      setError(null);
                    }}
                    placeholder="Enter a nickname"
                    maxLength={50}
                    disabled={isSubmitting}
                    className="claim-input"
                    autoFocus
                  />
                )}

                {error && <div className="claim-error">{error}</div>}

                {/* ── Lower box: static session display OR editable session input ── */}
                {isAdminTyped ? (
                  <>
                    <input
                      id="admin-session"
                      ref={adminSessionRef}
                      type="text"
                      value={adminSessionInput}
                      onChange={(e) => {
                        setAdminSessionInput(e.target.value);
                        setError(null);
                      }}
                      placeholder="Paste session ID here"
                      disabled={isSubmitting}
                      className="claim-input"
                    />
                  </>
                ) : (
                  <div className="claim-session-panel">
                    <span className="claim-session-label">Session</span>
                    <span className="claim-session-id">{sessionId}</span>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={
                    isSubmitting ||
                    (isAdminTyped
                      ? !adminSessionInput.trim()
                      : !nickname.trim())
                  }
                  className="claim-button"
                >
                  {isSubmitting
                    ? isAdminTyped
                      ? "Connecting..."
                      : "Joining..."
                    : isAdminTyped
                    ? "Connect"
                    : "Join Session"}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Animated loader (non-claim mode) ──────────────────────────────────────
  return (
    <div className={`container ${animationDone ? "fade-out" : ""}`}>
      <div
        className="rectangle-animation-area"
        style={{ width: `${totalWidth}px`, height: `${totalHeight}px`, position: "relative" }}
      >
        {positions.map((pos, index) => (
          <div
            key={`pixel-${index}`}
            className="rectangle-pixel"
            style={{
              width: `${pixelSize}px`,
              height: `${pixelSize}px`,
              left: `${(pos.x + Math.floor(rectWidth / 2)) * pixelSize}px`,
              top: `${(pos.y + Math.floor(rectHeight / 2)) * pixelSize}px`,
            }}
          />
        ))}
        {textLines.map((line, index) => (
          <div
            key={`text-${index}`}
            className={`text-line ${textLines.length === LOADER_LINES_OF_TEXT.length ? "text-complete" : ""}`}
            style={getTextPosition(index)}
          >
            {line}
          </div>
        ))}
      </div>
      {showNickname && <Nickname onNicknameSubmit={handleNicknameSubmit} />}
    </div>
  );
};

export default Loader;
