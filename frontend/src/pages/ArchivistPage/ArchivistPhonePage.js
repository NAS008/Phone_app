import React, { useCallback, useEffect, useRef, useState } from "react";
import "./ArchivistPage.css";
import officialLogo from "../../assets/logo/the-first-noncarbon-artist.png";

const API_URL      = process.env.REACT_APP_API_URL || "https://phoneapp-production-48e4.up.railway.app";
const CHUNK_MS     = 250;
const POLL_MS      = 500;
const TARGET_RATE  = 16000;

// ── Audio helpers ─────────────────────────────────────────────────────────────

function downsample(float32, fromRate, toRate) {
  if (fromRate === toRate) return float32;
  const ratio  = fromRate / toRate;
  const length = Math.floor(float32.length / ratio);
  const out    = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = float32[Math.floor(i * ratio)];
  return out;
}

function float32ToInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

function toBase64(buffer) {
  const bytes  = new Uint8Array(buffer);
  let   binary = "";
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

const MicIcon = ({ size = 48 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
  </svg>
);

// ── Component ─────────────────────────────────────────────────────────────────

const ArchivistPhonePage = ({ nickname }) => {
  const [pttActive,      setPttActive]      = useState(false);
  const [status,         setStatus]         = useState("idle");   // idle | speaking | processing
  const [transcript,     setTranscript]     = useState("");
  const [notice,         setNotice]         = useState(null);

  const audioCtxRef   = useRef(null);
  const processorRef  = useRef(null);
  const streamRef     = useRef(null);
  const pcmBufRef     = useRef([]);
  const chunkRef      = useRef(null);
  const pollRef       = useRef(null);
  const noticeRef     = useRef(null);
  const activeRef     = useRef(false);

  const flash = useCallback((msg) => {
    setNotice(msg);
    clearTimeout(noticeRef.current);
    noticeRef.current = setTimeout(() => setNotice(null), 2800);
  }, []);

  // ── Send buffered PCM to backend ──────────────────────────────────────────

  const flush = useCallback(async () => {
    if (!pcmBufRef.current.length) return;
    const bufs = pcmBufRef.current;
    pcmBufRef.current = [];

    const total  = bufs.reduce((n, a) => n + a.length, 0);
    const merged = new Float32Array(total);
    let   off    = 0;
    for (const b of bufs) { merged.set(b, off); off += b.length; }

    const rate      = audioCtxRef.current?.sampleRate ?? TARGET_RATE;
    const resampled = downsample(merged, rate, TARGET_RATE);
    const b64       = toBase64(float32ToInt16(resampled).buffer);

    fetch(`${API_URL}/api/archivist/audio`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ nickname, session_id: "", audio_bytes: b64, sample_rate: TARGET_RATE }),
    }).catch(() => {});
  }, [nickname]);

  // ── PTT start ─────────────────────────────────────────────────────────────

  const startPtt = useCallback(async (e) => {
    if (activeRef.current) return;
    e.currentTarget.setPointerCapture(e.pointerId);

    try {
      await fetch(`${API_URL}/api/archivist/ptt`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ action: "start", nickname, session_id: "" }),
      });
    } catch {
      flash("Network error — check connection.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
    } catch {
      flash("Microphone access denied.");
      return;
    }

    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    const source    = ctx.createMediaStreamSource(streamRef.current);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;
    processor.onaudioprocess = (ev) => {
      if (!activeRef.current) return;
      pcmBufRef.current.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(ctx.destination);

    activeRef.current = true;
    setPttActive(true);
    setStatus("speaking");
    setTranscript("");
    chunkRef.current = setInterval(flush, CHUNK_MS);
  }, [nickname, flush, flash]);

  // ── PTT stop ─────────────────────────────────────────────────────────────

  const stopPtt = useCallback(async () => {
    if (!activeRef.current) return;
    activeRef.current = false;
    clearInterval(chunkRef.current);
    await flush();

    processorRef.current?.disconnect();
    processorRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    pcmBufRef.current = [];

    setPttActive(false);
    setStatus("processing");

    fetch(`${API_URL}/api/archivist/ptt`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ action: "stop", nickname, session_id: "" }),
    }).catch(() => {});

    setTimeout(() => {
      if (!activeRef.current) setStatus("idle");
    }, 1500);
  }, [nickname, flush]);

  // ── Poll transcripts ──────────────────────────────────────────────────────

  useEffect(() => {
    let alive = true;
    let lastTs = 0;

    const poll = async () => {
      try {
        const res  = await fetch(`${API_URL}/api/archivist/transcripts?after=${lastTs}`);
        const data = await res.json();
        if (data.transcripts?.length) {
          const sorted = [...data.transcripts].sort((a, b) => a.ts - b.ts);
          const last   = sorted[sorted.length - 1];
          if (last.full) setTranscript(last.full);
          if (last.ts > lastTs) lastTs = last.ts;
        }
      } catch { /* ignore */ }
      if (alive) pollRef.current = setTimeout(poll, POLL_MS);
    };

    pollRef.current = setTimeout(poll, POLL_MS);
    return () => {
      alive = false;
      clearTimeout(pollRef.current);
    };
  }, []);

  // ── Cleanup ───────────────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      clearInterval(chunkRef.current);
      clearTimeout(noticeRef.current);
      clearTimeout(pollRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────

  const hint = pttActive ? "Release to send"
    : status === "processing" ? "Processing..."
    : "Hold to speak";

  return (
    <div className="archivist-page">
      {notice && <div className="archivist-notice">{notice}</div>}

      <header className="archivist-header">
        <img src={officialLogo} alt="The First NonCarbon Artist" className="archivist-header__logo" />
        <div className="archivist-header__copy">
          <h1>{nickname}</h1>
          <p>Archivist</p>
        </div>
      </header>

      <div className="archivist-center">
        {transcript && (
          <div className="archivist-said">
            <span className="archivist-said__label">You said</span>
            <p className="archivist-said__text">{transcript}</p>
          </div>
        )}

        <div className="archivist-ptt-wrap">
          <button
            className={`archivist-ptt-btn${pttActive ? " active" : ""}${status === "processing" ? " processing" : ""}`}
            onPointerDown={startPtt}
            onPointerUp={stopPtt}
            onPointerCancel={stopPtt}
            aria-label="Push to talk"
          >
            <MicIcon size={52} />
          </button>
          <span className="archivist-ptt-label">{hint}</span>
        </div>
      </div>
    </div>
  );
};

export default ArchivistPhonePage;
