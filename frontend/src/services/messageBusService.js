import { encode } from "@msgpack/msgpack";

class MessageBusService {
  constructor() {
    this.apiUrl = process.env.REACT_APP_API_URL || "https://phoneapp-production-48e4.up.railway.app";
  }

  // ── Low-level helpers ──────────────────────────────────────────────────────

  async postMsgpack(endpoint, payload) {
    const packed = encode(payload);
    const response = await fetch(`${this.apiUrl}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/msgpack" },
      body: packed,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Server error: ${response.status} - ${text}`);
    }
    return response.json();
  }

  async blobToBytes(blob) {
    if (!blob) return null;
    return new Uint8Array(await blob.arrayBuffer());
  }

  async dataUrlToBytes(dataUrl) {
    if (!dataUrl || typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) {
      return null;
    }
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new Uint8Array(await blob.arrayBuffer());
  }

  // ── Bus publish methods ────────────────────────────────────────────────────

  /**
   * Publish USER_JOINED — call once when the user submits their nickname.
   * app_server validates that session_id matches the active session.
   */
  async sendUserJoined(sessionId, nickname) {
    return this.postMsgpack("/api/publish/user_joined", {
      session_id: String(sessionId),
      nickname: String(nickname),
    });
  }

  /**
   * Publish USER_MESSAGE — text prompt and/or image/audio attachment.
   */
  async sendUserMessage({ text = null, audioBlob = null, imageData = null, sessionId, nickname }) {
    const audio_bytes = await this.blobToBytes(audioBlob);
    const image_bytes = await this.dataUrlToBytes(imageData);
    return this.postMsgpack("/api/publish/user_message", {
      session_id: String(sessionId),
      nickname: String(nickname),
      text: text || null,
      audio_bytes: audio_bytes || null,
      image_bytes: image_bytes || null,
    });
  }

  /**
   * Publish USER_LIKE — user tapped the heart / like button.
   * app_pc responds by sending the current rendered frame as an AI_MESSAGE.
   */
  async sendUserLike(sessionId, nickname) {
    return this.postMsgpack("/api/publish/user_like", {
      session_id: String(sessionId),
      nickname: String(nickname),
    });
  }

  /**
   * Publish USER_GESTURE — gyroscope / position data.
   * app_pc uses x, y, z to move the pointer in the particle sim.
   */
  async sendUserGesture({ x, y, z = 0.0, sessionId, nickname }) {
    return this.postMsgpack("/api/publish/user_gesture", {
      session_id: String(sessionId),
      nickname: String(nickname),
      x: Number(x),
      y: Number(y),
      z: Number(z),
    });
  }

  /**
   * Publish SETTINGS — controls AI mode, ray shape, sim constraints, zoom.
   */
  async sendSettings({ sessionId, nickname, ...settings }) {
    return this.postMsgpack("/api/publish/settings", {
      session_id: String(sessionId),
      nickname: String(nickname),
      ...settings,
    });
  }

  // ── Polling ────────────────────────────────────────────────────────────────

  /**
   * Fetch the server's current Unix time in milliseconds.
   * Used to sync the poll cursor to the server clock and eliminate clock-skew
   * filtering (where server time < client Date.now() hides all messages).
   * Falls back to Date.now() on any error.
   */
  async fetchServerTime() {
    try {
      const response = await fetch(`${this.apiUrl}/api/server_time`);
      if (!response.ok) return Date.now();
      const { ts } = await response.json();
      return typeof ts === "number" ? ts : Date.now();
    } catch {
      return Date.now();
    }
  }

  /**
   * Fetch AI_MESSAGEs stored by app_phone.
   * Returns { success, messages: [{ text, session_id, nickname, received_at_ms,
   *   audio_base64?, image_base64? }] }
   */
  async fetchAiMessages(afterMs = 0, sessionId = null) {
    let url = `${this.apiUrl}/api/ai_messages?after=${afterMs}`;
    if (sessionId) url += `&sessionId=${encodeURIComponent(sessionId)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    return response.json();
  }

  // ── Convenience wrappers (used by ArtistMode) ─────────────────────────────

  /**
   * Send a text prompt with an optional image data-URL.
   */
  async sendSensorFrameWithTranscriptToMessageBus(imageData, transcript, sessionId, nickname) {
    if (!imageData && !transcript) {
      throw new Error("Both image and transcript cannot be empty");
    }
    return this.sendUserMessage({
      text: transcript || null,
      imageData: typeof imageData === "string" ? imageData : null,
      audioBlob: imageData instanceof Blob ? imageData : null,
      sessionId,
      nickname,
    });
  }

  /**
   * Send an audio blob as a USER_MESSAGE.
   */
  async sendAudioMessage(audioBlob, sessionId, nickname) {
    if (!audioBlob) return null;
    return this.sendUserMessage({ audioBlob, sessionId, nickname });
  }

  /**
   * Send audio to the backend for speech-to-text transcription.
   * Returns { success, status, transcript } on success.
   */
  async sendAudioForTranscription(audioBlob, sessionId, nickname = null) {
    if (!audioBlob) return null;
    const audio_bytes = await this.blobToBytes(audioBlob);
    return this.postMsgpack("/api/publish/audio", {
      session_id: String(sessionId || ""),
      nickname: String(nickname || ""),
      audio_bytes,
    });
  }
}

const messageBusService = new MessageBusService();
export default messageBusService;
