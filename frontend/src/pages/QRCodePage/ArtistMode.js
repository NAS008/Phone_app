import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./ArtistMode.css";
import MessageBusService from "../../services/messageBusService";
import officialLogo from "../../assets/logo/the-first-noncarbon-artist.png";

const buildId = (prefix) =>
  `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const sliderPct = (value, min, max) =>
  `${(((value - min) / (max - min)) * 100).toFixed(1)}%`;

const createFeedMessage = (role, text, extra = {}) => ({
  id: buildId(role),
  role,
  text,
  ...extra,
});

const buildDownloadUrl = (url, filename) => {
  const parsedUrl = new URL(url, window.location.href);
  parsedUrl.searchParams.set("download", "1");
  if (filename) {
    parsedUrl.searchParams.set("filename", filename);
  }
  return parsedUrl.toString();
};

const shouldPreferNativeShare = () => {
  if (typeof navigator === "undefined" || !navigator.share) return false;

  const userAgent = navigator.userAgent || "";
  const platform = navigator.userAgentData?.platform || navigator.platform || "";
  const maxTouchPoints = navigator.maxTouchPoints || 0;
  const isIOS =
    /iPad|iPhone|iPod/i.test(userAgent) ||
    (platform === "MacIntel" && maxTouchPoints > 1);
  const isAndroid = /Android/i.test(userAgent);

  return isIOS || isAndroid;
};

const fileToDataUrl = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });

const loadImage = (src) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = src;
  });

const prepareImageData = async (file) => {
  const rawDataUrl = await fileToDataUrl(file);
  const image = await loadImage(rawDataUrl);
  const maxSide = 1400;
  let width = image.width;
  let height = image.height;

  if (Math.max(width, height) > maxSide) {
    const ratio = maxSide / Math.max(width, height);
    width = Math.round(width * ratio);
    height = Math.round(height * ratio);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Canvas is not available");
  }

  context.drawImage(image, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.86);
};

const formatSeconds = (value) => {
  const minutes = Math.floor(value / 60)
    .toString()
    .padStart(2, "0");
  const seconds = (value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
};

const buildWelcomeFeed = (nickname) => [
  createFeedMessage(
    "assistant",
    `Hello ${nickname || "there"}. This session now behaves like a small AI studio. Send a prompt, image, photo, or voice note to influence the next artwork.`
  ),
  createFeedMessage(
    "assistant",
    "Use the plus button for Gallery or Take Photo, and the mic for a voice note.",
    { subtle: true }
  ),
];

const RECEIVED_ACKNOWLEDGMENTS = [
  "Conjuring pigments...",
  "Distilling intention...",
  "Weaving light...",
  "Summoning form...",
  "Dreaming in colour...",
  "Sculpting the unseen...",
  "Chasing the signal...",
  "Folding time into texture...",
  "Reading the static...",
  "Unravelling pattern...",
  "Breathing into the canvas...",
  "Gathering luminance...",
  "Parsing the sublime...",
  "Reaching for edges...",
  "Dissolving into pixels...",
  "Negotiating with entropy...",
  "Translating gesture...",
  "Hunting the frequency...",
  "Composing silence...",
  "Sketching in probabilities...",
  "Rendering intention...",
  "Stirring colour fields...",
  "Listening to the image...",
  "Crystallising thought...",
  "Bending light toward meaning...",
];

const pickAcknowledgment = () =>
  RECEIVED_ACKNOWLEDGMENTS[Math.floor(Math.random() * RECEIVED_ACKNOWLEDGMENTS.length)];

const INTRO_TYPING_INTERVAL = 42;
const INTRO_TYPING_TARGET = 70;
const INTRO_MESSAGE_GAP = 620;
const GYRO_SEND_INTERVAL = 100;
const TRANSCRIPT_TYPING_TARGET = 60;
const TRANSCRIPT_TYPING_INTERVAL = 22;

const PlusIcon = (props) => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden="true"
    {...props}
  >
    <path
      d="M12 5V19"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M5 12H19"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const SendIcon = (props) => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden="true"
    {...props}
  >
    <path
      d="M12 5.25L12 18.75"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M18.75 12L12 5.25L5.25 12"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const MicIcon = (props) => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
  </svg>
);

const CameraIcon = (props) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.7"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <path d="M4 7h3l1.6-2h6.8L17 7h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z" />
    <circle cx="12" cy="13" r="3.5" />
  </svg>
);

const GalleryIcon = (props) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.7"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <circle cx="9" cy="10" r="1.5" />
    <path d="M21 16l-5.5-5.5L7 19" />
  </svg>
);

const GyroIcon = (props) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <circle cx="12" cy="12" r="2.2" />
    <ellipse cx="12" cy="12" rx="8" ry="4.8" />
    <path d="M12 4C16.4 4 20 7.58 20 12C20 16.42 16.4 20 12 20" />
    <path d="M12 4C7.6 4 4 7.58 4 12C4 16.42 7.6 20 12 20" />
  </svg>
);

const SettingsIcon = (props) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.7"
    strokeLinecap="round"
    aria-hidden="true"
    {...props}
  >
    <line x1="3" y1="6" x2="21" y2="6" />
    <line x1="3" y1="12" x2="21" y2="12" />
    <line x1="3" y1="18" x2="21" y2="18" />
    <circle cx="8" cy="6" r="2.2" fill="currentColor" stroke="none" />
    <circle cx="16" cy="12" r="2.2" fill="currentColor" stroke="none" />
    <circle cx="8" cy="18" r="2.2" fill="currentColor" stroke="none" />
  </svg>
);

const VideoIcon = (props) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
    width="18" height="18" aria-hidden="true" {...props}>
    <rect x="2" y="6" width="15" height="12" rx="2" />
    <path d="M17 9l5-3v12l-5-3V9z" />
  </svg>
);

const HeartIcon = (props) => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
  </svg>
);

// Rotate device-frame acceleration into world frame using euler angles from DeviceOrientationEvent.
// Produces metres/s² in world space (x=East, y=North, z=Up).
const deviceToWorldAccel = (ax, ay, az, alpha, beta, gamma) => {
  const a = (alpha ?? 0) * (Math.PI / 180);
  const b = (beta  ?? 0) * (Math.PI / 180);
  const g = (gamma ?? 0) * (Math.PI / 180);
  const cA = Math.cos(a), sA = Math.sin(a);
  const cB = Math.cos(b), sB = Math.sin(b);
  const cG = Math.cos(g), sG = Math.sin(g);
  return {
    x: (cA * cG - sA * sB * sG) * ax + (-sA * cB) * ay + (cA * sG + sA * sB * cG) * az,
    y: (sA * cG + cA * sB * sG) * ax + (cA * cB) * ay + (sA * sG - cA * sB * cG) * az,
    z: (-cB * sG) * ax + sB * ay + (cB * cG) * az,
  };
};

// Mean absolute per-channel difference between two ImageData objects (0–255 scale, alpha ignored).
// Returns the average |Δchannel| across all RGB channels and all pixels.
const frameDiff = (a, b) => {
  const d1 = a.data, d2 = b.data;
  let sum = 0;
  for (let i = 0; i < d1.length; i += 4) {
    sum += Math.abs(d1[i] - d2[i]) + Math.abs(d1[i + 1] - d2[i + 1]) + Math.abs(d1[i + 2] - d2[i + 2]);
  }
  return sum / (d1.length / 4 * 3); // divide by total RGB channel count → true per-channel mean
};

const ArtistMode = ({ sessionId, nickname, isAdmin }) => {
  const [prompt, setPrompt] = useState("");
  const [draftImage, setDraftImage] = useState(null);
  const [draftLabel, setDraftLabel] = useState("");
  const [feed, setFeed] = useState([]);
  const [notice, setNotice] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isAttachmentMenuOpen, setIsAttachmentMenuOpen] = useState(false);
  const [audioWidgetPhase, setAudioWidgetPhase] = useState("idle");
  const [audioWidgetText, setAudioWidgetText] = useState("");
  const [audioWidgetDisplayText, setAudioWidgetDisplayText] = useState("");
  const [isGyroEnabled, setIsGyroEnabled] = useState(false);
  const [isLikePulsing, setIsLikePulsing] = useState(false);
  const [isVideoRequesting, setIsVideoRequesting] = useState(false);
  const [isSettingsPanelOpen, setIsSettingsPanelOpen] = useState(false);
  const [settingsMode, setSettingsMode] = useState(0);
  const [settingsShape, setSettingsShape] = useState(0);
  const [settingsZoom, setSettingsZoom] = useState(1.1);
  const [settingsConstraintsOn, setSettingsConstraintsOn] = useState(true);
  const [settingsGoBackOn, setSettingsGoBackOn] = useState(true);
  const [settingsGradientOn, setSettingsGradientOn] = useState(false);

  const galleryInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const audioInputRef = useRef(null);
  const attachmentMenuRef = useRef(null);
  const recorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const noticeTimeoutRef = useRef(null);
  const refreshTimeoutsRef = useRef([]);
  const audioWidgetTimeoutRef = useRef(null);
  const audioWidgetIntervalRef = useRef(null);
  const introTimeoutsRef = useRef([]);
  const composerTextareaRef = useRef(null);
  const feedEndRef = useRef(null);
  const introSequenceEndsRef = useRef(0);
  const gyroDataRef = useRef({
    alpha: null,
    beta: null,
    gamma: null,
    absolute: null,
  });
  const lastSentGyroRef = useRef(null);
  const gyroTransportBlockedRef = useRef(false);
  const positionCameraStreamRef = useRef(null);
  const videoRef = useRef(null);
  const motionDataRef = useRef({ x: 0, y: 0, z: 0 });
  const positionRef = useRef({ x: 0, y: 0, z: 0 });
  const velocityRef = useRef({ x: 0, y: 0, z: 0 });
  const prevFrameDataRef = useRef(null);
  const posLastTimeRef = useRef(0);
  const likeTimeoutRef = useRef(null);
  const messageReceivedPollRef = useRef(null);
  const lastMessageReceivedMsRef = useRef(0);

  const flashNotice = useCallback((type, text) => {
    if (noticeTimeoutRef.current) {
      clearTimeout(noticeTimeoutRef.current);
    }
    setNotice({ type, text });
    noticeTimeoutRef.current = setTimeout(() => {
      setNotice(null);
      noticeTimeoutRef.current = null;
    }, 2600);
  }, []);

  const sendSetting = useCallback(async (key, value) => {
    try {
      await MessageBusService.sendSettings({ sessionId, nickname, [key]: value });
    } catch {
      flashNotice("error", "Setting failed to send.");
    }
  }, [flashNotice, nickname, sessionId]);

  const appendFeed = useCallback((message) => {
    setFeed((current) => [...current, message]);
  }, []);

  const appendReceivedAcknowledgment = useCallback(() => {
    appendFeed(createFeedMessage("assistant", pickAcknowledgment()));
  }, [appendFeed]);

  const clearIntroTimers = useCallback(() => {
    introTimeoutsRef.current.forEach((timeoutId) => clearTimeout(timeoutId));
    introTimeoutsRef.current = [];
  }, []);

  const scheduleIntroTimeout = useCallback((callback, delay) => {
    const timeoutId = setTimeout(callback, delay);
    introTimeoutsRef.current.push(timeoutId);
    return timeoutId;
  }, []);

  const replayWelcomeFeed = useCallback(
    (nextNickname) => {
      clearIntroTimers();
      const welcomeMessages = buildWelcomeFeed(nextNickname);
      setFeed([]);

      let delay = 180;

      welcomeMessages.forEach((message) => {
        const fullText = message.text;
        const step = Math.max(1, Math.ceil(fullText.length / INTRO_TYPING_TARGET));
        const steps = Math.ceil(fullText.length / step);

        scheduleIntroTimeout(() => {
          setFeed((current) => [...current, { ...message, text: "", typing: true }]);

          let cursor = 0;
          const revealChunk = () => {
            cursor = Math.min(fullText.length, cursor + step);

            setFeed((current) =>
              current.map((item) =>
                item.id === message.id
                  ? {
                      ...item,
                      text: fullText.slice(0, cursor),
                      typing: cursor < fullText.length,
                    }
                  : item
              )
            );

            if (cursor < fullText.length) {
              scheduleIntroTimeout(revealChunk, INTRO_TYPING_INTERVAL);
            }
          };

          revealChunk();
        }, delay);

        delay += steps * INTRO_TYPING_INTERVAL + INTRO_MESSAGE_GAP;
      });

      introSequenceEndsRef.current = Date.now() + Math.max(0, delay - INTRO_MESSAGE_GAP);
    },
    [clearIntroTimers, scheduleIntroTimeout]
  );

  const clearAudioWidgetTimers = useCallback(() => {
    if (audioWidgetTimeoutRef.current) {
      clearTimeout(audioWidgetTimeoutRef.current);
      audioWidgetTimeoutRef.current = null;
    }
    if (audioWidgetIntervalRef.current) {
      clearInterval(audioWidgetIntervalRef.current);
      audioWidgetIntervalRef.current = null;
    }
  }, []);

  const resetAudioWidget = useCallback(() => {
    clearAudioWidgetTimers();
    setAudioWidgetPhase("idle");
    setAudioWidgetText("");
    setAudioWidgetDisplayText("");
  }, [clearAudioWidgetTimers]);

  const clearDraftImage = useCallback(() => {
    setDraftImage(null);
    setDraftLabel("");
  }, []);

  const handleDownloadImage = useCallback(
    async (url, filename = "noncarbon-artwork.jpg") => {
      try {
        if (shouldPreferNativeShare()) {
          // Convert to File so iOS share sheet shows "Save to Photos"
          try {
            const res = await fetch(url);
            const blob = await res.blob();
            const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
            if (navigator.canShare && navigator.canShare({ files: [file] })) {
              await navigator.share({ files: [file], title: "NonCarbon artwork" });
              return;
            }
          } catch (shareError) {
            if (shareError?.name === "AbortError") return;
          }
        }

        // Desktop: trigger download via hidden link
        const downloadUrl = url.startsWith("data:") ? url : buildDownloadUrl(url, filename);
        const link = document.createElement("a");
        link.href = downloadUrl;
        link.download = filename;
        link.rel = "noopener,noreferrer";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (error) {
        window.open(url, "_blank", "noopener,noreferrer");
        flashNotice("success", "Opened artwork download.");
      }
    },
    [flashNotice]
  );

  const handleShareRequest = useCallback(async () => {
    if (likeTimeoutRef.current) clearTimeout(likeTimeoutRef.current);
    setIsLikePulsing(true);
    likeTimeoutRef.current = setTimeout(() => {
      setIsLikePulsing(false);
      likeTimeoutRef.current = null;
    }, 700);

    try {
      await MessageBusService.sendUserLike(sessionId, nickname);
      flashNotice("success", "Like sent.");
    } catch {
      flashNotice("error", "Like failed to send.");
    }
  }, [flashNotice, nickname, sessionId]);

  const handleVideoRequest = useCallback(async () => {
    if (isVideoRequesting) return;
    setIsVideoRequesting(true);
    try {
      await MessageBusService.sendUserVideo(sessionId, nickname);
      flashNotice("success", "Video requested…");
    } catch {
      flashNotice("error", "Video request failed.");
    } finally {
      setIsVideoRequesting(false);
    }
  }, [flashNotice, isVideoRequesting, nickname, sessionId]);

  useEffect(() => {
    if (!sessionId) return undefined;

    let cancelled = false;

    const poll = async () => {
      if (cancelled) return;
      try {
        const { messages } = await MessageBusService.fetchAiMessages(
          lastMessageReceivedMsRef.current,
          sessionId
        );
        const fresh = messages || [];
        if (fresh.length > 0) {
          fresh.sort((a, b) => (a.received_at_ms || 0) - (b.received_at_ms || 0));
          for (const msg of fresh) {
            const text = msg.text || null;
            const audioSrc = msg.audio_base64
              ? `data:audio/mpeg;base64,${msg.audio_base64}`
              : null;
            const mime = msg.image_mime_type || "image/jpeg";
            const imageSrc = msg.video_url
              ? `${MessageBusService.apiUrl}${msg.video_url}`
              : msg.image_base64
              ? `data:${mime};base64,${msg.image_base64}`
              : null;
            const imageExt = msg.video_url ? "gif" : mime === "image/gif" ? "gif" : "jpg";

            // AI_MESSAGE always shows in the feed regardless of which user sent it
            if (text || audioSrc || imageSrc) {
              appendFeed(
                createFeedMessage("assistant", text || "", {
                  audio: audioSrc,
                  image: imageSrc,
                  senderNickname: msg.nickname || "NonCarbon Artist",
                  downloadUrl: imageSrc || null,
                  downloadName: imageSrc ? `noncarbon-artwork-${msg.received_at_ms || Date.now()}.${imageExt}` : null,
                })
              );
            }

            if ((msg.received_at_ms || 0) > lastMessageReceivedMsRef.current) {
              lastMessageReceivedMsRef.current = msg.received_at_ms;
            }
          }
        }
      } catch {
        // silently ignore poll errors
      }
      if (!cancelled) {
        messageReceivedPollRef.current = setTimeout(poll, 2000);
      }
    };

    messageReceivedPollRef.current = setTimeout(poll, 2000);

    return () => {
      cancelled = true;
      if (messageReceivedPollRef.current) {
        clearTimeout(messageReceivedPollRef.current);
        messageReceivedPollRef.current = null;
      }
    };
  }, [appendFeed, sessionId]);


  const stopMediaStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  const stageAudioTranscript = useCallback(
    (transcript) => {
      const finalText = transcript?.trim();

      if (!finalText) {
        clearAudioWidgetTimers();
        setAudioWidgetPhase("reveal");
        setAudioWidgetText("Could not transcribe voice.");
        setAudioWidgetDisplayText("Could not transcribe voice.");
        audioWidgetTimeoutRef.current = setTimeout(() => resetAudioWidget(), 1500);
        return;
      }

      clearAudioWidgetTimers();
      setAudioWidgetPhase("reveal");
      setAudioWidgetText(finalText);
      setAudioWidgetDisplayText("");

      let index = 0;
      const step = Math.max(1, Math.ceil(finalText.length / TRANSCRIPT_TYPING_TARGET));

      audioWidgetIntervalRef.current = setInterval(() => {
        index = Math.min(finalText.length, index + step);
        setAudioWidgetDisplayText(finalText.slice(0, index));

        if (index >= finalText.length) {
          clearAudioWidgetTimers();
          audioWidgetTimeoutRef.current = setTimeout(async () => {
            try {
              await MessageBusService.sendUserMessage({
                text: finalText,
                audioBlob: null,
                imageData: draftImage || null,
                sessionId,
                nickname,
              });
              appendFeed(
                createFeedMessage("user", finalText, {
                  inputType: "voice",
                  ...(draftImage ? { image: draftImage } : {}),
                })
              );
              if (draftImage) clearDraftImage();
              appendReceivedAcknowledgment();
            } catch (error) {
              appendFeed(
                createFeedMessage(
                  "assistant",
                  `Voice note failed: ${error.message || "unknown error"}`,
                  { error: true }
                )
              );
              flashNotice("error", "Voice note failed to send.");
            }
            resetAudioWidget();
          }, 1300);
        }
      }, TRANSCRIPT_TYPING_INTERVAL);
    },
    [
      appendFeed,
      appendReceivedAcknowledgment,
      clearAudioWidgetTimers,
      clearDraftImage,
      draftImage,
      flashNotice,
      nickname,
      resetAudioWidget,
      sessionId,
    ]
  );

  const uploadAudio = useCallback(
    async (audioBlob) => {
      clearAudioWidgetTimers();
      setAudioWidgetPhase("processing");
      setAudioWidgetText("");
      setAudioWidgetDisplayText("");
      setIsSubmitting(true);

      try {
        const result = await MessageBusService.sendAudioForTranscription(
          audioBlob,
          sessionId,
          nickname
        );
        const transcript = result?.transcript?.trim();

        if (result?.status !== "success" || !transcript) {
          throw new Error(result?.transcription_error || "Audio transcription failed.");
        }

        flashNotice("success", "Voice note transcribed.");
        stageAudioTranscript(transcript);
      } catch (error) {
        clearAudioWidgetTimers();
        setAudioWidgetPhase("error");
        setAudioWidgetText("Voice note failed.");
        setAudioWidgetDisplayText("Voice note failed.");
        audioWidgetTimeoutRef.current = setTimeout(() => resetAudioWidget(), 1500);
        flashNotice("error", "Voice note failed to send.");
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      appendFeed,
      clearAudioWidgetTimers,
      flashNotice,
      nickname,
      resetAudioWidget,
      sessionId,
      stageAudioTranscript,
    ]
  );

  const startRecording = useCallback(async () => {
    if (
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === "undefined"
    ) {
      audioInputRef.current?.click();
      return;
    }

    try {
      clearAudioWidgetTimers();
      setIsAttachmentMenuOpen(false);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredMime =
        MediaRecorder.isTypeSupported &&
        MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : MediaRecorder.isTypeSupported &&
              MediaRecorder.isTypeSupported("audio/webm")
            ? "audio/webm"
            : "";

      mediaStreamRef.current = stream;
      audioChunksRef.current = [];

      const recorder = preferredMime
        ? new MediaRecorder(stream, { mimeType: preferredMime })
        : new MediaRecorder(stream);

      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        audioChunksRef.current = [];
        recorderRef.current = null;
        stopMediaStream();
        if (blob.size > 0) {
          await uploadAudio(blob);
        }
      };

      recorder.start();
      setIsRecording(true);
      setAudioWidgetPhase("recording");
      setAudioWidgetText("");
      setAudioWidgetDisplayText("");
      flashNotice("success", "Recording started. Tap Audio again to send.");
    } catch (error) {
      flashNotice("error", "Microphone access was blocked.");
    }
  }, [clearAudioWidgetTimers, flashNotice, stopMediaStream, uploadAudio]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
      setAudioWidgetPhase("processing");
      setAudioWidgetText("");
      setAudioWidgetDisplayText("");
    } else {
      stopMediaStream();
      resetAudioWidget();
    }
    setIsRecording(false);
  }, [resetAudioWidget, stopMediaStream]);

  const handleAudioAction = useCallback(() => {
    if (isSubmitting) return;
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, isSubmitting, startRecording, stopRecording]);

  const handleAudioFileChange = useCallback(
    async (event) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file) return;
      setIsAttachmentMenuOpen(false);
      await uploadAudio(file);
    },
    [uploadAudio]
  );

  const stopPositionCamera = useCallback(() => {
    if (positionCameraStreamRef.current) {
      positionCameraStreamRef.current.getTracks().forEach((t) => t.stop());
      positionCameraStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    positionRef.current = { x: 0, y: 0, z: 0 };
    velocityRef.current = { x: 0, y: 0, z: 0 };
    prevFrameDataRef.current = null;
    posLastTimeRef.current = 0;
  }, []);

  const handleGyroToggle = useCallback(async () => {
    if (isGyroEnabled) {
      setIsGyroEnabled(false);
      stopPositionCamera();
      flashNotice("success", "Position tracking turned off.");
      return;
    }

    // Request orientation permission (iOS 13+)
    try {
      const OrientationCtor = typeof window !== "undefined" ? window.DeviceOrientationEvent : undefined;
      if (OrientationCtor && typeof OrientationCtor.requestPermission === "function") {
        if ((await OrientationCtor.requestPermission()) !== "granted") {
          flashNotice("error", "Orientation access was not granted.");
          return;
        }
      }
    } catch {
      flashNotice("error", "Orientation access was blocked.");
      return;
    }

    // Request motion permission (iOS 13+, separate from orientation)
    try {
      const MotionCtor = typeof window !== "undefined" ? window.DeviceMotionEvent : undefined;
      if (MotionCtor && typeof MotionCtor.requestPermission === "function") {
        if ((await MotionCtor.requestPermission()) !== "granted") {
          flashNotice("error", "Motion access was not granted.");
          return;
        }
      }
    } catch {
      flashNotice("error", "Motion access was blocked.");
      return;
    }

    // Start rear camera for zero-velocity detection (ZUPT)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 64 }, height: { ideal: 48 } },
      });
      positionCameraStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch {
      // Camera unavailable — position tracking degrades to pure IMU
    }

    setIsGyroEnabled(true);
    flashNotice("success", "Position tracking turned on.");
  }, [flashNotice, isGyroEnabled, stopPositionCamera]);

  const handleImageSelection = useCallback(
    async (file, sourceLabel) => {
      if (!file) return;

      try {
        const dataUrl = await prepareImageData(file);
        setDraftImage(dataUrl);
        setDraftLabel(`${sourceLabel}: ${file.name || "image"}`);
        setIsAttachmentMenuOpen(false);
        flashNotice("success", `${sourceLabel} is ready to send.`);
      } catch (error) {
        flashNotice("error", `Couldn't prepare the ${sourceLabel.toLowerCase()}.`);
      }
    },
    [flashNotice]
  );

  const handleSubmit = useCallback(async () => {
    const trimmedPrompt = prompt.trim();
    if ((!trimmedPrompt && !draftImage) || isSubmitting) return;

    appendFeed(
      createFeedMessage(
        "user",
        trimmedPrompt || draftLabel || "Image attached.",
        draftImage ? { image: draftImage } : {}
      )
    );

    setIsSettingsPanelOpen(false);
    setIsSubmitting(true);
    try {
      await MessageBusService.sendSensorFrameWithTranscriptToMessageBus(
        draftImage,
        trimmedPrompt,
        sessionId,
        nickname
      );
      flashNotice("success", "Message sent.");

      setPrompt("");
      clearDraftImage();
      appendReceivedAcknowledgment();
    } catch (error) {
      appendFeed(
        createFeedMessage(
          "assistant",
          `Send failed: ${error.message || "unknown error"}`,
          { error: true }
        )
      );
      flashNotice("error", "Message failed to send.");
    } finally {
      setIsSubmitting(false);
    }
  }, [
    appendFeed,
    appendReceivedAcknowledgment,
    clearDraftImage,
    draftImage,
    draftLabel,
    flashNotice,
    isSubmitting,
    nickname,
    prompt,
    sessionId,
  ]);

  useEffect(() => {
    replayWelcomeFeed(nickname);
    introSequenceEndsRef.current = 0;
    setPrompt("");
    clearDraftImage();
    resetAudioWidget();
    setIsAttachmentMenuOpen(false);
  }, [clearDraftImage, nickname, replayWelcomeFeed, resetAudioWidget, sessionId]);

  useEffect(() => {
    if (!isRecording) {
      setRecordingSeconds(0);
      return undefined;
    }

    const startedAt = Date.now();
    const intervalId = setInterval(() => {
      setRecordingSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 300);
    return () => clearInterval(intervalId);
  }, [isRecording]);

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [feed]);

  useEffect(() => {
    const el = composerTextareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [prompt]);


  useEffect(() => {
    if (!isGyroEnabled) return undefined;

    const handleOrientation = (event) => {
      gyroDataRef.current = {
        alpha: event.alpha ?? null,
        beta: event.beta ?? null,
        gamma: event.gamma ?? null,
        absolute: event.absolute ?? null,
      };
    };

    window.addEventListener("deviceorientation", handleOrientation);
    return () => {
      window.removeEventListener("deviceorientation", handleOrientation);
    };
  }, [isGyroEnabled]);

  // DeviceMotionEvent listener — stores gravity-removed acceleration in device frame.
  useEffect(() => {
    if (!isGyroEnabled) return undefined;

    const handleMotion = (event) => {
      const a = event.acceleration;
      motionDataRef.current = { x: a?.x ?? 0, y: a?.y ?? 0, z: a?.z ?? 0 };
    };

    window.addEventListener("devicemotion", handleMotion);
    return () => window.removeEventListener("devicemotion", handleMotion);
  }, [isGyroEnabled]);

  // Position integration loop + ZUPT from camera + transmission.
  useEffect(() => {
    if (!isGyroEnabled || !sessionId) {
      gyroTransportBlockedRef.current = false;
      lastSentGyroRef.current = null;
      positionRef.current = { x: 0.5, y: 0.5, z: 0 };
      velocityRef.current = { x: 0, y: 0, z: 0 };
      prevFrameDataRef.current = null;
      posLastTimeRef.current = 0;
      return undefined;
    }

    let cancelled = false;
    const STATIONARY_DIFF = 6;      // mean per-channel diff threshold (0–255 scale) for ZUPT
    const FRAME_W = 64, FRAME_H = 48;
    // Tilt range (degrees) that maps to full-speed movement; smaller = more sensitive.
    const TILT_RANGE = 30;
    // Max pointer speed in normalized units per second at full tilt.
    const MAX_SPEED = 0.8;

    positionRef.current = { x: 0.5, y: 0.5, z: 0 };
    velocityRef.current = { x: 0, y: 0, z: 0 };

    const offscreen = document.createElement("canvas");
    offscreen.width = FRAME_W;
    offscreen.height = FRAME_H;
    const ctx = offscreen.getContext("2d");

    posLastTimeRef.current = performance.now();

    // 10 Hz — camera ZUPT + velocity integration
    const frameIntervalId = setInterval(() => {
      if (cancelled || !ctx) return;

      const video = videoRef.current;
      const now = performance.now();
      const dt = Math.min((now - posLastTimeRef.current) / 1000, 0.1);
      posLastTimeRef.current = now;

      // Camera ZUPT: compare consecutive frames; if stationary, kill velocity.
      let isStationary = false;
      if (video && video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);
        const frame = ctx.getImageData(0, 0, FRAME_W, FRAME_H);
        const prev = prevFrameDataRef.current;
        prevFrameDataRef.current = frame;
        if (prev && frameDiff(prev, frame) < STATIONARY_DIFF) {
          isStationary = true;
        }
      }

      const { alpha, beta, gamma } = gyroDataRef.current;

      if (isStationary) {
        velocityRef.current = { x: 0, y: 0, z: 0 };
        positionRef.current = { ...positionRef.current, z: 0 };
        return;
      }

      // Tilt angle (clamped to ±TILT_RANGE) drives velocity, not position.
      // gamma: tilt right → positive vx (neutral = 0°).
      // beta: neutral upright portrait = 90°, so offset by 90 before scaling.
      const vx = Math.max(-1, Math.min(1, (gamma ?? 0) / TILT_RANGE)) * MAX_SPEED;
      const vy = Math.max(-1, Math.min(1, ((beta ?? 90) - 90) / TILT_RANGE)) * MAX_SPEED;

      const pos = positionRef.current;
      positionRef.current = {
        x: Math.max(0, Math.min(1, pos.x + vx * dt)),
        y: Math.max(0, Math.min(1, pos.y + vy * dt)),
        z: positionRef.current.z,
      };

      // z = world-frame acceleration magnitude → drives vz force in sim
      const { x: ax, y: ay, z: az } = motionDataRef.current;
      const w = deviceToWorldAccel(ax, ay, az, alpha, beta, gamma);
      positionRef.current.z = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    }, 100);

    // 320 ms — transmit position to the bus
    const txIntervalId = setInterval(async () => {
      if (cancelled || gyroTransportBlockedRef.current || isSubmitting) return;

      const { x, y, z } = positionRef.current;
      const prev = lastSentGyroRef.current;
      const changed =
        !prev ||
        Math.abs(x - (prev.x ?? 0)) > 0.001 ||
        Math.abs(y - (prev.y ?? 0)) > 0.001 ||
        Math.abs(z - (prev.z ?? 0)) > 0.001;

      if (!changed) return;

      try {
        await MessageBusService.sendUserGesture({ x, y, z, sessionId, nickname });
        lastSentGyroRef.current = { x, y, z };
      } catch {
        gyroTransportBlockedRef.current = true;
        flashNotice("error", "Position tracking failed.");
      }
    }, GYRO_SEND_INTERVAL);

    return () => {
      cancelled = true;
      clearInterval(frameIntervalId);
      clearInterval(txIntervalId);
      gyroTransportBlockedRef.current = false;
      lastSentGyroRef.current = null;
    };
  }, [flashNotice, isGyroEnabled, isSubmitting, nickname, sessionId]);

  useEffect(() => {
    const refreshTimeouts = refreshTimeoutsRef.current;

    return () => {
      stopMediaStream();
      stopPositionCamera();
      if (noticeTimeoutRef.current) clearTimeout(noticeTimeoutRef.current);
      if (likeTimeoutRef.current) clearTimeout(likeTimeoutRef.current);
      if (messageReceivedPollRef.current) clearTimeout(messageReceivedPollRef.current);
      refreshTimeouts.forEach((timeoutId) => clearTimeout(timeoutId));
      clearAudioWidgetTimers();
      clearIntroTimers();
    };
  }, [clearAudioWidgetTimers, clearIntroTimers, stopMediaStream, stopPositionCamera]);

  useEffect(() => {
    if (!isAttachmentMenuOpen) return undefined;

    const handlePointerDown = (event) => {
      if (
        attachmentMenuRef.current &&
        !attachmentMenuRef.current.contains(event.target)
      ) {
        setIsAttachmentMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
    };
  }, [isAttachmentMenuOpen]);

  const canSend = prompt.trim().length > 0 || Boolean(draftImage);

  const audioWidgetLabel = useMemo(() => {
    if (audioWidgetPhase === "recording") return "Listening";
    if (audioWidgetPhase === "processing") return "Transcribing";
    if (audioWidgetPhase === "error") return "Audio Error";
    if (audioWidgetPhase === "reveal") return "Transcript";
    return "";
  }, [audioWidgetPhase]);

  const audioWidgetCopy = useMemo(() => {
    if (audioWidgetPhase === "recording") {
      return `Capturing voice note ${formatSeconds(recordingSeconds)}`;
    }
    if (audioWidgetPhase === "processing") {
      return "Transcribing your voice note...";
    }
    if (audioWidgetPhase === "error") {
      return audioWidgetDisplayText || audioWidgetText || "Voice note failed.";
    }
    if (audioWidgetPhase === "reveal") {
      return audioWidgetDisplayText || audioWidgetText;
    }
    return "";
  }, [audioWidgetDisplayText, audioWidgetPhase, audioWidgetText, recordingSeconds]);

  return (
    <div className="artist-mode">
      {notice && (
        <div className={`qr-notice ${notice.type === "error" ? "error" : "success"}`}>
          {notice.text}
        </div>
      )}

      <header className="assistant-header">
        <div className="assistant-brand-mark" aria-hidden="true">
          <img
            src={officialLogo}
            alt="The First NonCarbon Artist"
            className="assistant-brand-image"
          />
        </div>
        <div className="assistant-copy">
          <h1>{nickname ? `Hello ${nickname}` : "Hello"}</h1>
          <p>
            Guide the artwork with an idea, image or voice note
          </p>
        </div>
      </header>

      {/* Keeping the current-canvas preview block in place for later iteration.
      <section className="preview-card">
        <div className="preview-meta">
          <div>
            <span className="preview-label">Current canvas</span>
            <strong>
              {draftImage
                ? "Attached image ready to send"
                : selectedArtwork
                  ? "Latest artwork from this session"
                  : "Nothing generated yet"}
            </strong>
          </div>
          <span className="preview-chip">Session {sessionLabel}</span>
        </div>

        <div className="preview-frame">
          {previewImage ? (
            <img src={previewImage} alt="Session preview" className="preview-image" />
          ) : (
            <div className="preview-empty">
              Send something to start this conversation.
            </div>
          )}
        </div>

        {draftImage ? (
          <div className="draft-chip">
            <span>{draftLabel || "Image attached"}</span>
            <button type="button" onClick={clearDraftImage}>
              Remove
            </button>
          </div>
        ) : (
          <p className="preview-note">
            {selectedArtwork
              ? "Tap a thumbnail below to switch between your recent outputs."
              : "Nothing has come back from this QR session yet."}
          </p>
        )}
      </section>
      */}

      {!(isAdmin && isSettingsPanelOpen) && (
      <section className="conversation-panel">
        <div className="conversation-feed">
          {feed.map((message) => (
            <div
              key={message.id}
              className={`message-row ${message.role === "user" ? "user" : "assistant"}`}
            >
              <div
                className={`message-bubble ${message.role} ${message.error ? "error" : ""}`}
              >
                <span className="bubble-label">
                  {message.role === "user"
                    ? `${nickname || "You"}${
                        message.inputType === "voice"
                          ? " • Voice"
                          : message.image
                            ? " • Image"
                            : ""
                      }`
                    : (message.senderNickname || "NonCarbon Artist")}
                </span>
                <p className={`bubble-text ${message.typing ? "typing" : ""}`}>
                  {message.text}
                </p>
                {message.audio && (
                  <audio
                    controls
                    className="bubble-audio"
                    src={message.audio}
                    preload="metadata"
                  />
                )}
                {message.image && (
                  <div className="bubble-image-wrap">
                    <img
                      src={message.image}
                      alt={message.downloadUrl ? "Generated artwork" : "Sent attachment"}
                      className="bubble-image"
                    />
                    {message.downloadUrl && (
                      <button
                        type="button"
                        className="bubble-download"
                        onClick={() =>
                          handleDownloadImage(message.downloadUrl, message.downloadName)
                        }
                        aria-label="Download artwork"
                      >
                        <svg
                          width="18"
                          height="18"
                          viewBox="0 0 24 24"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                          aria-hidden="true"
                        >
                          <path
                            d="M12 4V14"
                            stroke="currentColor"
                            strokeWidth="1.8"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M8 10L12 14L16 10"
                            stroke="currentColor"
                            strokeWidth="1.8"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M5 19H19"
                            stroke="currentColor"
                            strokeWidth="1.8"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={feedEndRef} />
        </div>
      </section>
      )}

      {isAdmin && isSettingsPanelOpen && (
        <section className="settings-panel">
          <div className="settings-inner">
            <div className="settings-section">
              <div className="settings-row">
                <span className="settings-label">AI Mode</span>
                <span className="settings-value">{settingsMode}</span>
              </div>
              <input
                type="range"
                className="settings-slider"
                min="0"
                max="2"
                step="1"
                value={settingsMode}
                style={{ "--pct": sliderPct(settingsMode, 0, 2) }}
                onChange={(e) => setSettingsMode(parseInt(e.target.value, 10))}
                onMouseUp={(e) => sendSetting("mode", parseInt(e.target.value, 10))}
                onTouchEnd={() => sendSetting("mode", settingsMode)}
              />
            </div>

            <div className="settings-section">
              <div className="settings-row">
                <span className="settings-label">Ray Shape</span>
                <span className="settings-value">{settingsShape}</span>
              </div>
              <input
                type="range"
                className="settings-slider"
                min="0"
                max="4"
                step="1"
                value={settingsShape}
                style={{ "--pct": sliderPct(settingsShape, 0, 4) }}
                onChange={(e) => setSettingsShape(parseInt(e.target.value, 10))}
                onMouseUp={(e) => sendSetting("shape", parseInt(e.target.value, 10))}
                onTouchEnd={() => sendSetting("shape", settingsShape)}
              />
            </div>

            <div className="settings-section">
              <div className="settings-row">
                <span className="settings-label">Field of View</span>
                <span className="settings-value">{settingsZoom.toFixed(1)}</span>
              </div>
              <input
                type="range"
                className="settings-slider"
                min="0.1"
                max="1.2"
                step="0.1"
                value={settingsZoom}
                style={{ "--pct": sliderPct(settingsZoom, 0.1, 1.2) }}
                onChange={(e) => setSettingsZoom(parseFloat(e.target.value))}
                onMouseUp={(e) => sendSetting("zoom", parseFloat(e.target.value))}
                onTouchEnd={() => sendSetting("zoom", settingsZoom)}
              />
            </div>

            <div className="settings-section settings-row">
              <span className="settings-label">Constraints</span>
              <button
                type="button"
                role="switch"
                aria-checked={settingsConstraintsOn}
                className={`settings-toggle${settingsConstraintsOn ? " on" : ""}`}
                onClick={() => {
                  const next = !settingsConstraintsOn;
                  setSettingsConstraintsOn(next);
                  sendSetting("constraints_on", next);
                }}
              >
                <span className="settings-toggle__thumb" />
              </button>
            </div>

            <div className="settings-section settings-row">
              <span className="settings-label">Go Back</span>
              <button
                type="button"
                role="switch"
                aria-checked={settingsGoBackOn}
                className={`settings-toggle${settingsGoBackOn ? " on" : ""}`}
                onClick={() => {
                  const next = !settingsGoBackOn;
                  setSettingsGoBackOn(next);
                  sendSetting("go_back_on", next);
                }}
              >
                <span className="settings-toggle__thumb" />
              </button>
            </div>

            <div className="settings-section settings-row">
              <span className="settings-label">Gradient</span>
              <button
                type="button"
                role="switch"
                aria-checked={settingsGradientOn}
                className={`settings-toggle${settingsGradientOn ? " on" : ""}`}
                onClick={() => {
                  const next = !settingsGradientOn;
                  setSettingsGradientOn(next);
                  sendSetting("gradient_on", next);
                }}
              >
                <span className="settings-toggle__thumb" />
              </button>
            </div>
          </div>
        </section>
      )}

      <section className="composer-panel open">
        {audioWidgetPhase !== "idle" && (
          <div className={`audio-widget ${audioWidgetPhase}`}>
            <div className={`audio-widget__orb ${audioWidgetPhase}`} aria-hidden="true" />
            <div className="audio-widget__copy">
              <span className="audio-widget__eyebrow">{audioWidgetLabel}</span>
              <p
                className={`audio-widget__text ${
                  audioWidgetPhase === "reveal" && audioWidgetDisplayText !== audioWidgetText
                    ? "typing"
                    : ""
                }`}
              >
                {audioWidgetCopy}
              </p>
              {audioWidgetPhase === "processing" && (
                <div className="audio-widget__dots" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
              )}
            </div>
          </div>
        )}

        {draftImage && (
          <div className="composer-status attachment">
            <span>{draftLabel || "Image attached"}</span>
            <button type="button" onClick={clearDraftImage} aria-label="Remove attached image">
              <span aria-hidden="true">×</span>
            </button>
          </div>
        )}

        <textarea
          ref={composerTextareaRef}
          className="composer-textarea"
          placeholder=""
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          maxLength={300}
          disabled={isSubmitting}
          rows={1}
        />

        <div className="composer-actions">
          <div className="action-cluster">
            <div className="attachment-menu" ref={attachmentMenuRef}>
              <button
                type="button"
                className="icon-button action-button"
                onClick={() => setIsAttachmentMenuOpen((open) => !open)}
                disabled={isSubmitting || isRecording}
                aria-label="Open attachment options"
                aria-expanded={isAttachmentMenuOpen}
              >
                <PlusIcon />
              </button>

              {isAttachmentMenuOpen && (
                <div className="attachment-popover">
                  <button
                    type="button"
                    className="attachment-popover__item"
                    onClick={() => {
                      setIsAttachmentMenuOpen(false);
                      cameraInputRef.current?.click();
                    }}
                  >
                    <CameraIcon />
                    <span>Take Photo</span>
                  </button>
                  <button
                    type="button"
                    className="attachment-popover__item"
                    onClick={() => {
                      setIsAttachmentMenuOpen(false);
                      galleryInputRef.current?.click();
                    }}
                  >
                    <GalleryIcon />
                    <span>Gallery</span>
                  </button>
                </div>
              )}
            </div>

            <button
              type="button"
              className={`icon-button action-button ${isGyroEnabled ? "toggled" : ""}`}
              onClick={handleGyroToggle}
              disabled={isSubmitting}
              aria-label={isGyroEnabled ? "Disable gyroscope" : "Enable gyroscope"}
              aria-pressed={isGyroEnabled}
              title={isGyroEnabled ? "Gyroscope on" : "Gyroscope off"}
            >
              <GyroIcon />
            </button>
          </div>

          <button
            type="button"
            className={`icon-button action-button${isLikePulsing ? " like-active" : ""}`}
            onClick={handleShareRequest}
            disabled={isSubmitting}
            aria-label="Like — request image share"
          >
            <HeartIcon />
          </button>

          <button
            type="button"
            className={`icon-button action-button${isVideoRequesting ? " toggled" : ""}`}
            onClick={handleVideoRequest}
            disabled={isVideoRequesting}
            aria-label="Request video clip"
            title="Get animated clip"
          >
            <VideoIcon />
          </button>

          {isAdmin && (
            <button
              type="button"
              className={`icon-button action-button${isSettingsPanelOpen ? " toggled" : ""}`}
              onClick={() => setIsSettingsPanelOpen((open) => !open)}
              aria-label="Open settings"
              aria-expanded={isSettingsPanelOpen}
            >
              <SettingsIcon />
            </button>
          )}

          <div className="action-cluster action-cluster--right">
            <button
              type="button"
              className={`icon-button action-button ${isRecording ? "recording" : ""}`}
              onClick={handleAudioAction}
              disabled={isSubmitting}
              aria-label={isRecording ? "Stop audio recording" : "Record voice"}
            >
              <MicIcon />
            </button>

            <button
              type="button"
              className="icon-button send-button"
              onClick={handleSubmit}
              disabled={!canSend || isSubmitting || isRecording}
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          </div>
        </div>

        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            handleImageSelection(file, "Camera capture");
          }}
          style={{ display: "none" }}
        />

        <input
          ref={galleryInputRef}
          type="file"
          accept="image/*"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            handleImageSelection(file, "Gallery image");
          }}
          style={{ display: "none" }}
        />

        <input
          ref={audioInputRef}
          type="file"
          accept="audio/*"
          capture
          onChange={handleAudioFileChange}
          style={{ display: "none" }}
        />
      </section>

      {/* Hidden video for camera-based zero-velocity detection */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: "none" }}
        aria-hidden="true"
      />
    </div>
  );
};

export default ArtistMode;
