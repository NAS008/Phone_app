import React, { useEffect, useRef, useState } from "react";
import "./StreamView.css";

const OFFER_URL =
  (process.env.REACT_APP_STREAM_URL || "https://192.168.68.65:8080") + "/offer";

const CloseIcon = (props) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const StreamView = ({ onClose }) => {
  const videoRef = useRef(null);
  const [status, setStatus] = useState("connecting...");

  useEffect(() => {
    const pc = new RTCPeerConnection();

    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
      }
      setStatus("live");
    };

    pc.onconnectionstatechange = () => {
      const s = pc.connectionState;
      if (s === "connected") setStatus("live");
      else if (s === "failed" || s === "closed" || s === "disconnected") setStatus("unavailable");
    };

    (async () => {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        const res = await fetch(OFFER_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
          }),
        });
        if (!res.ok) throw new Error(`${res.status}`);
        const answer = await res.json();
        await pc.setRemoteDescription(answer);
      } catch {
        setStatus("unavailable");
      }
    })();

    return () => {
      pc.close();
    };
  }, []);

  return (
    <div className="stream-view">
      <button
        type="button"
        className="stream-close-btn"
        onClick={onClose}
        aria-label="Close stream"
      >
        <CloseIcon />
      </button>
      {status !== "live" && (
        <div className="stream-status">{status}</div>
      )}
      <video
        ref={videoRef}
        className="stream-video"
        autoPlay
        playsInline
        muted
      />
    </div>
  );
};

export default StreamView;
