import React, { useCallback, useEffect, useRef, useState } from "react";
import "./ArchivistPage.css";
import officialLogo from "../../assets/logo/the-first-noncarbon-artist.png";

const API_URL = process.env.REACT_APP_API_URL || "https://phoneapp-production-48e4.up.railway.app";
const AUDIO_CHUNK_MS   = 250;   // send a PCM chunk every 250 ms while PTT active
const TRANSCRIPT_POLL_MS = 500; // poll user transcripts every 500 ms
const TARGET_RATE = 16000;      // Gemini expects 16 kHz PCM16

function downsample(float32, fromRate, toRate) {
  if (fromRate === toRate) return float32;
  const ratio = fromRate / toRate;
  const length = Math.floor(float32.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = float32[Math.floor(i * ratio)];
  }
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

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

const MicIcon = () => (
  <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
  </svg>
);

const ArchivistPhonePage = ({ nickname }) => {
  const [pttActive, setPttActive]           = useState(false);
  const [userTranscript, setUserTranscript] = useState("");
  const [statusText, setStatusText]         = useState("idle");
  const [notice, setNotice]                 = useState(null);

  const audioCtxRef    = useRef(null);
  const processorRef   = useRef(null);
  const streamRef      = useRef(null);
  const pcmBufRef      = useRef([]);       // accumulated Float32 samples
  const chunkTimerRef  = useRef(null);
  const pollTimerRef   = useRef(null);
  const noticeTimerRef = useRef(null);
  const pttActiveRef   = useRef(false);

  const showNotice = useCallback((text) => {
    setNotice(text);
    if (noticeTimerRef.current) clearTimeout(noticeTimerRef.current);
    noticeTimerRef.current = setTimeout(() => setNotice(null), 2500);
  }, []);

  // ── Send one PCM chunk to the backend ────────────────────────────────────

  const sendChunk = useCallback(async () => {
    if (pcmBufRef.current.length === 0) return;
    const samples = pcmBufRef.current;
    pcmBufRef.current = [];

    try {
      const merged = new Float32Array(samples.reduce((acc, a) => acc + a.length, 0));
      let offset = 0;
      for (const s of samples) { merged.set(s, offset); offset += s.length; }

      const rate = audioCtxRef.current ? audioCtxRef.current.sampleRate : TARGET_RATE;
      const resampled = downsample(merged, rate, TARGET_RATE);
      const int16 = float32ToInt16(resampled);
      const b64   = arrayBufferToBase64(int16.buffer);

      await fetch(`${API_URL}/api/archivist/audio`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nickname,
          session_id: "",
          audio_bytes: b64,
          sample_rate: TARGET_RATE,
        }),
      });
    } catch {
      // drop silently — audio is real-time
    }
  }, [nickname]);

  // ── PTT start ────────────────────────────────────────────────────────────

  const startPtt = useCallback(async (e) => {
    if (pttActiveRef.current) return;
    e.currentTarget.setPointerCapture(e.pointerId);

    try {
      await fetch(`${API_URL}/api/archivist/ptt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "start", nickname, session_id: "" }),
      });

      pttActiveRef.current = true;
      setPttActive(true);
      setStatusText("speaking");
      setUserTranscript("");

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;

      const source    = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (ev) => {
        if (!pttActiveRef.current) return;
        pcmBufRef.current.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      chunkTimerRef.current = setInterval(sendChunk, AUDIO_CHUNK_MS);
    } catch (err) {
      showNotice("Microphone access denied.");
    }
  }, [nickname, sendChunk, showNotice]);

  // ── PTT stop ─────────────────────────────────────────────────────────────

  const stopPtt = useCallback(async () => {
    if (!pttActiveRef.current) return;
    pttActiveRef.current = false;
    setPttActive(false);
    setStatusText("processing");

    clearInterval(chunkTimerRef.current);
    await sendChunk(); // flush remaining samples

    // Tear down audio
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    pcmBufRef.current = [];

    await fetch(`${API_URL}/api/archivist/ptt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "stop", nickname, session_id: "" }),
    }).catch(() => {});

    setTimeout(() => {
      if (!pttActiveRef.current) setStatusText("idle");
    }, 1500);
  }, [nickname, sendChunk]);

  // ── Poll user transcripts ─────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    let lastTs = 0;

    const poll = async () => {
      if (cancelled) return;
      try {
        const res = await fetch(`${API_URL}/api/archivist/transcripts?after=${lastTs}`);
        const { transcripts } = await res.json();
        if (transcripts && transcripts.length > 0) {
          transcripts.sort((a, b) => a.ts - b.ts);
          const last = transcripts[transcripts.length - 1];
          if (last.full) setUserTranscript(last.full);
          if (last.ts > lastTs) lastTs = last.ts;
        }
      } catch {
        // ignore
      }
      if (!cancelled) pollTimerRef.current = setTimeout(poll, TRANSCRIPT_POLL_MS);
    };

    pollTimerRef.current = setTimeout(poll, TRANSCRIPT_POLL_MS);
    return () => {
      cancelled = true;
      clearTimeout(pollTimerRef.current);
    };
  }, []);

  // ── Cleanup on unmount ────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      clearInterval(chunkTimerRef.current);
      clearTimeout(noticeTimerRef.current);
      clearTimeout(pollTimerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, []);

  const statusLabel = {
    idle:       "Hold to speak",
    speaking:   "Listening...",
    processing: "Processing...",
  }[statusText] ?? "Hold to speak";

  const dotState = statusText === "speaking" ? "active" : statusText === "processing" ? "processing" : "";

  return (
    <div className="archivist-page">
      {notice && <div className="archivist-notice">{notice}</div>}

      <header className="archivist-header">
        <img src={officialLogo} alt="The First NonCarbon Artist" className="archivist-header__logo" />
        <div className="archivist-header__copy">
          <h1>Hello {nickname}</h1>
          <p>Archivist — live conversation</p>
        </div>
      </header>

      <div className="archivist-transcript">
        {userTranscript ? (
          <div className="archivist-transcript__line">
            <span className="archivist-transcript__label">You said</span>
            <p className="archivist-transcript__text user">{userTranscript}</p>
          </div>
        ) : (
          <p className="archivist-transcript__empty">
            Hold the button below to speak.
          </p>
        )}
      </div>

      <div className="archivist-status">
        <span className={`archivist-status__dot${dotState ? ` ${dotState}` : ""}`} />
        {statusLabel}
      </div>

      <div className="archivist-ptt-area">
        <button
          className={`archivist-ptt-btn${pttActive ? " active" : ""}`}
          onPointerDown={startPtt}
          onPointerUp={stopPtt}
          onPointerCancel={stopPtt}
          aria-label="Push to talk"
        >
          <MicIcon />
        </button>
        <span className="archivist-ptt-label">
          {pttActive ? "Release to send" : "Hold to speak"}
        </span>
      </div>
    </div>
  );
};

export default ArchivistPhonePage;
