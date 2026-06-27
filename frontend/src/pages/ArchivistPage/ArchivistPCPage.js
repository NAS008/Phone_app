import React, { useCallback, useEffect, useRef, useState } from "react";
import "./ArchivistPage.css";
import officialLogo from "../../assets/logo/the-first-noncarbon-artist.png";

const WS_URL       = process.env.REACT_APP_ARCHIVIST_WS_URL || "ws://localhost:8890";
const RECEIVE_RATE = 24000;

// ── Gapless PCM16 → AudioContext player ──────────────────────────────────────

function makePlayer(sampleRate) {
  const ctx = new AudioContext({ sampleRate });
  let nextAt = 0;

  return {
    play(b64) {
      const bytes   = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const int16   = new Int16Array(bytes.buffer);
      const f32     = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;

      const buf = ctx.createBuffer(1, f32.length, sampleRate);
      buf.getChannelData(0).set(f32);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);

      const start = Math.max(ctx.currentTime + 0.04, nextAt);
      src.start(start);
      nextAt = start + buf.duration;
    },
    resume() { if (ctx.state === "suspended") ctx.resume().catch(() => {}); },
    reset()  { nextAt = 0; },
    close()  { ctx.close().catch(() => {}); },
  };
}

// ── Component ─────────────────────────────────────────────────────────────────

const ArchivistPCPage = () => {
  const [userText,    setUserText]    = useState("");
  const [geminiText,  setGeminiText]  = useState("");
  const [statusState, setStatusState] = useState("connecting");
  const [connected,   setConnected]   = useState(false);

  const wsRef      = useRef(null);
  const playerRef  = useRef(null);
  const retryRef   = useRef(null);
  const retryN     = useRef(0);
  const endRef     = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState <= 1) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      retryN.current = 0;
      setConnected(true);
      setStatusState("idle");
      if (!playerRef.current) playerRef.current = makePlayer(RECEIVE_RATE);
      playerRef.current.resume();
    };

    ws.onmessage = ({ data }) => {
      let msg;
      try { msg = JSON.parse(data); } catch { return; }

      if (msg.type === "audio" && msg.data) {
        playerRef.current?.resume();
        playerRef.current?.play(msg.data);
      }
      if (msg.type === "user_transcript")   setUserText(msg.text || "");
      if (msg.type === "gemini_transcript") setGeminiText(msg.text || "");

      if (msg.type === "status") {
        setStatusState(msg.state || "idle");
        if (msg.state === "idle" || msg.state === "connected") playerRef.current?.reset();
      }
      if (msg.type === "interrupted") {
        playerRef.current?.reset();
        setStatusState("idle");
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setStatusState("connecting");
      const delay = Math.min(1000 * 2 ** retryN.current, 15000);
      retryN.current += 1;
      retryRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(retryRef.current);
      wsRef.current?.close();
      playerRef.current?.close();
    };
  }, [connect]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [userText, geminiText]);

  const dotClass =
    statusState === "user_speaking" ? "active"
    : statusState === "processing"  ? "processing"
    : "";

  const statusLabel = {
    connecting:    "Connecting...",
    connected:     "Idle",
    idle:          "Idle",
    user_speaking: "Listening...",
    processing:    "Processing...",
    interrupted:   "Interrupted",
  }[statusState] ?? statusState;

  return (
    <div className="archivist-page" onClick={() => playerRef.current?.resume()}>
      <header className="archivist-header">
        <img src={officialLogo} alt="The First NonCarbon Artist" className="archivist-header__logo" />
        <div className="archivist-header__copy">
          <h1>Archivist</h1>
          <p>PC output</p>
        </div>
      </header>

      <div className="archivist-pc-body">
        <div className="archivist-status-bar">
          <span className={`archivist-status-bar__dot${dotClass ? ` ${dotClass}` : ""}`} />
          {statusLabel}
          {!connected && (
            <span style={{ marginLeft: 6, color: "#6a3030", fontSize: 11 }}>
              ● {WS_URL}
            </span>
          )}
        </div>

        <div className="archivist-feed">
          {!userText && !geminiText && (
            <p className="archivist-feed__empty">
              {connected ? "Waiting for a conversation to start…" : "Waiting for local server…"}
            </p>
          )}

          {userText && (
            <div className="archivist-feed__line">
              <span className="archivist-feed__label">User</span>
              <p className="archivist-feed__text user">{userText}</p>
            </div>
          )}

          {geminiText && (
            <div className="archivist-feed__line">
              <span className="archivist-feed__label">Vera</span>
              <p className="archivist-feed__text gemini">{geminiText}</p>
            </div>
          )}

          <div ref={endRef} />
        </div>

        {!connected && (
          <p className="archivist-pc-hint">
            Run <code>python server/archivist_server.py</code> on this PC to connect.
          </p>
        )}
      </div>
    </div>
  );
};

export default ArchivistPCPage;
