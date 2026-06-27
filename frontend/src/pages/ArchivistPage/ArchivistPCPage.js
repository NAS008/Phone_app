import React, { useCallback, useEffect, useRef, useState } from "react";
import "./ArchivistPage.css";
import officialLogo from "../../assets/logo/the-first-noncarbon-artist.png";

const WS_URL = process.env.REACT_APP_ARCHIVIST_WS_URL || "ws://localhost:8890";
const RECEIVE_RATE = 24000; // Gemini output sample rate

// ── Gapless PCM16 playback scheduler ─────────────────────────────────────────

function createAudioPlayer(sampleRate) {
  const ctx = new AudioContext({ sampleRate });
  let nextPlayTime = ctx.currentTime;

  function scheduleChunk(base64Data) {
    const bytes  = Uint8Array.from(atob(base64Data), (c) => c.charCodeAt(0));
    const int16  = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const buffer = ctx.createBuffer(1, float32.length, sampleRate);
    buffer.getChannelData(0).set(float32);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime + 0.05, nextPlayTime);
    source.start(startAt);
    nextPlayTime = startAt + buffer.duration;
  }

  function resume() {
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
  }

  function reset() {
    nextPlayTime = ctx.currentTime;
  }

  function close() {
    ctx.close().catch(() => {});
  }

  return { scheduleChunk, resume, reset, close };
}

// ── Component ─────────────────────────────────────────────────────────────────

const ArchivistPCPage = () => {
  const [userTranscript,   setUserTranscript]   = useState("");
  const [geminiTranscript, setGeminiTranscript] = useState("");
  const [statusText,       setStatusText]       = useState("Connecting...");
  const [connected,        setConnected]        = useState(false);

  const wsRef      = useRef(null);
  const playerRef  = useRef(null);
  const retryRef   = useRef(null);
  const retryCount = useRef(0);
  const feedEndRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= 1) return; // already open/connecting

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      retryCount.current = 0;
      setConnected(true);
      setStatusText("Idle");

      if (!playerRef.current) {
        playerRef.current = createAudioPlayer(RECEIVE_RATE);
      }
      playerRef.current.resume();
    };

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === "audio" && msg.data) {
        playerRef.current?.resume();
        playerRef.current?.scheduleChunk(msg.data);
      }

      if (msg.type === "user_transcript") {
        setUserTranscript(msg.text || "");
      }

      if (msg.type === "gemini_transcript") {
        setGeminiTranscript(msg.text || "");
        if (msg.final) {
          // Keep displayed until next turn
        }
      }

      if (msg.type === "status") {
        const labels = {
          connected:     "Idle",
          idle:          "Idle",
          user_speaking: "Listening...",
          processing:    "Processing...",
          interrupted:   "Interrupted",
        };
        setStatusText(labels[msg.state] || msg.state);

        if (msg.state === "idle" || msg.state === "connected") {
          playerRef.current?.reset();
        }
      }

      if (msg.type === "interrupted") {
        playerRef.current?.reset();
        setStatusText("Interrupted");
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setStatusText("Disconnected — retrying...");
      const delay = Math.min(1000 * 2 ** retryCount.current, 15000);
      retryCount.current += 1;
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
    feedEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [userTranscript, geminiTranscript]);

  const dotState = statusText.startsWith("Listen") ? "active"
    : statusText.startsWith("Process") ? "processing"
    : "";

  return (
    <div className="archivist-page" onClick={() => playerRef.current?.resume()}>
      <header className="archivist-header">
        <img src={officialLogo} alt="The First NonCarbon Artist" className="archivist-header__logo" />
        <div className="archivist-header__copy">
          <h1>Archivist</h1>
          <p>Gemini Live — PC output</p>
        </div>
      </header>

      <div className="archivist-status">
        <span className={`archivist-status__dot${dotState ? ` ${dotState}` : ""}`} />
        {statusText}
        {!connected && (
          <span style={{ marginLeft: 6, color: "#ff7d7d", fontSize: 11 }}>
            ● {WS_URL}
          </span>
        )}
      </div>

      <div className="archivist-transcript archivist-transcript--pc archivist-pc-content">
        {!userTranscript && !geminiTranscript && (
          <p className="archivist-transcript__empty">
            Waiting for a conversation to start.
          </p>
        )}

        {userTranscript && (
          <div className="archivist-transcript__line">
            <span className="archivist-transcript__label">User</span>
            <p className="archivist-transcript__text user">{userTranscript}</p>
          </div>
        )}

        {geminiTranscript && (
          <div className="archivist-transcript__line">
            <span className="archivist-transcript__label">Vera</span>
            <p className="archivist-transcript__text gemini">{geminiTranscript}</p>
          </div>
        )}

        <div ref={feedEndRef} />
      </div>

      {!connected && (
        <p style={{ textAlign: "center", color: "#555", fontSize: 12, marginTop: "auto" }}>
          Run <code style={{ color: "#888" }}>python server/archivist_server.py</code> on the PC to connect.
        </p>
      )}
    </div>
  );
};

export default ArchivistPCPage;
