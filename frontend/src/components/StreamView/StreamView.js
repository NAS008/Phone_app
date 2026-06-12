import { useEffect, useRef, useState } from "react";
import "./StreamView.css";
import MessageBusService from "../../services/messageBusService";

// TURN relay is required on mobile data (CGNAT blocks direct/STUN connections).
// Cloudflare TURN credentials are short-lived, so they are fetched from the
// backend per connection; the static REACT_APP_TURN_* vars remain as fallback.
const buildStaticIceServers = () => {
  const servers = [{ urls: "stun:stun.l.google.com:19302" }];
  const turnUrl = process.env.REACT_APP_TURN_URL;
  if (turnUrl) {
    servers.push({
      urls: [`${turnUrl}?transport=udp`, `${turnUrl}?transport=tcp`],
      username: process.env.REACT_APP_TURN_USERNAME,
      credential: process.env.REACT_APP_TURN_PASSWORD,
    });
  }
  return servers;
};

const waitForIceGathering = (pc, timeoutMs = 4000) =>
  new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
      return;
    }
    let timeoutId = null;
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        clearTimeout(timeoutId);
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    // Don't wait forever — partial candidates are usually enough
    timeoutId = setTimeout(() => {
      pc.removeEventListener("icegatheringstatechange", check);
      resolve();
    }, timeoutMs);
  });

const StreamView = ({ sessionId, nickname, onClose }) => {
  const videoRef = useRef(null);
  const [status, setStatus] = useState("Connecting to the artwork…");

  useEffect(() => {
    let cancelled = false;
    let pc = null;

    const start = async () => {
      const turnServers = await MessageBusService.fetchIceServers();
      if (cancelled) return;

      pc = new RTCPeerConnection({
        iceServers: [...buildStaticIceServers(), ...(turnServers || [])],
      });

      pc.addTransceiver("video", { direction: "recvonly" });

      pc.ontrack = (event) => {
        if (cancelled) return;
        if (videoRef.current) {
          videoRef.current.srcObject = event.streams[0];
        }
      };

      pc.onconnectionstatechange = () => {
        if (cancelled) return;
        if (pc.connectionState === "connected") {
          setStatus("");
        } else if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
          setStatus("Stream unavailable");
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGathering(pc);

      const answer = await MessageBusService.sendWebRtcOffer(sessionId, nickname, {
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      });
      if (cancelled) return;
      if (!answer?.sdp || !answer?.type) {
        throw new Error("No answer from the artwork");
      }
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    };

    start().catch(() => {
      if (!cancelled) setStatus("Stream unavailable");
    });

    return () => {
      cancelled = true;
      if (pc) pc.close();
    };
  }, [nickname, sessionId]);

  return (
    <div className="stream-view">
      <video
        ref={videoRef}
        className="stream-view__video"
        autoPlay
        playsInline
        muted
      />
      {status && <div className="stream-view__status">{status}</div>}
      <button
        type="button"
        className="stream-view__close"
        onClick={onClose}
        aria-label="Close stream and go back"
      >
        ×
      </button>
    </div>
  );
};

export default StreamView;
