import { useEffect, useRef, useState } from "react";
import "./StreamView.css";
import MessageBusService from "../../services/messageBusService";

const waitForIceGathering = (pc, timeoutMs = 2000) =>
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
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
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

    const start = async () => {
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
      pc.close();
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
