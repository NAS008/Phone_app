import os
import time
import cv2
import numpy as np
import sounddevice as sd
import mediapipe as mp

class Audio:
    def __init__(self, sample_rate=16000, channels=1, dtype=np.int16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.stream = None
        self.chunks = []

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        self.chunks.append(indata.copy())

    def start_recording(self):
        if self.recording:
            return

        self.chunks = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self.stream.start()
        self.recording = True

    def stop_recording(self):
        if not self.recording:
            return b""

        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.recording = False

        if not self.chunks:
            return b""

        audio = np.concatenate(self.chunks, axis=0)
        return audio.tobytes()

    def get_audio_bytes(self, audio):
        audio = np.asarray(audio, dtype=self.dtype)
        return audio.tobytes()

    def decode_audio_bytes(self, audio_bytes):
        audio = np.frombuffer(audio_bytes, dtype=self.dtype)
        if self.channels > 1:
            audio = audio.reshape(-1, self.channels)
        return audio
   
class Mouse:
    def __init__(self, width, height):
        self.W = width
        self.H = height
        self.on = False
        self.x = 0.5
        self.y = 0.5

    def mouse_callback(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.x = 1.0 - x / self.W
            self.y = y / self.H
                
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.on = True

class Camera:
    WRIST = 16
    
    LANDMARK_NAMES = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
        19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index"
    }

    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]

    KEY_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(
        self,
        model_path,
        camera_candidates=None,
        width=640,
        height=480,
        num_poses=1,
        mirror=True,
        show_labels=True,
        visibility_threshold=0.4,
    ):
        self.mp = mp
        self.model_path = model_path
        self.camera_candidates = camera_candidates or [
            (1, cv2.CAP_DSHOW),
            (0, cv2.CAP_DSHOW),
            (1, cv2.CAP_MSMF),
            (0, cv2.CAP_MSMF),
            (1, cv2.CAP_ANY),
            (0, cv2.CAP_ANY),
            (2, cv2.CAP_DSHOW),
            (2, cv2.CAP_ANY),
        ]
        self.width = width
        self.height = height
        self.num_poses = num_poses
        self.mirror = mirror
        self.show_labels = show_labels
        self.visibility_threshold = visibility_threshold

        self.cap = None
        self.landmarker = None
        self.latest_result = None
        self.latest_timestamp = -1
        self.start_time = None
        self.last_sent_ts = -1

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model not found: {self.model_path}')

    def _result_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result
        self.latest_timestamp = timestamp_ms

    def _create_landmarker(self):
        BaseOptions = self.mp.tasks.BaseOptions
        PoseLandmarker = self.mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = self.mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = self.mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=self.num_poses,
            result_callback=self._result_callback,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _open_camera(self):
        for index, backend in self.camera_candidates:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                self.cap = cap
                print(f'Opened camera index={index}, backend={backend}, shape={frame.shape}')
                return

            cap.release()

        raise RuntimeError('Could not open any camera. Try changing camera index manually.')

    def _valid_landmark(self, lm, margin=0.15):
        if hasattr(lm, 'visibility') and lm.visibility < self.visibility_threshold:
            return False
        return (-margin <= lm.x <= 1 + margin) and (-margin <= lm.y <= 1 + margin)

    def _get_timestamp_ms(self):
        timestamp_ms = int((time.perf_counter() - self.start_time) * 1000)
        if timestamp_ms <= self.last_sent_ts:
            timestamp_ms = self.last_sent_ts + 1
        self.last_sent_ts = timestamp_ms
        return timestamp_ms

    def start(self):
        self._create_landmarker()
        self._open_camera()
        self.start_time = time.perf_counter()

    def read(self):
        if self.cap is None or self.landmarker is None:
            raise RuntimeError('PoseCamera.start() must be called before read().')

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, self._failure_canvas(), None, None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(mp_image, self._get_timestamp_ms())

        canvas = self.draw(frame, self.latest_result)
        return True, canvas, frame, self.latest_result

    def _failure_canvas(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(canvas, 'No camera frame', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return canvas

    def draw(self, frame, result=None):
        h, w = frame.shape[:2]
        canvas = np.zeros_like(frame)

        result = self.latest_result if result is None else result
        if result is None or not result.pose_landmarks:
            cv2.putText(canvas, 'Camera OK - no pose detected yet', (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            return canvas

        landmarks = result.pose_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        gray = (192, 192, 192)
        white = (255, 255, 255)
        dim = (170, 170, 170)
        red = (0, 0, 255)

        for i, j in self.CONNECTIONS:
            if self._valid_landmark(landmarks[i]) and self._valid_landmark(landmarks[j]):
                cv2.line(canvas, pts[i], pts[j], gray, 2, cv2.LINE_AA)

        if self._valid_landmark(landmarks[0]) and self._valid_landmark(landmarks[11]) and self._valid_landmark(landmarks[12]):
            shoulder_mid = (
                (pts[11][0] + pts[12][0]) // 2,
                (pts[11][1] + pts[12][1]) // 2
            )
            cv2.line(canvas, pts[0], shoulder_mid, gray, 2, cv2.LINE_AA)

        for idx in self.KEY_LANDMARKS:
            lm = landmarks[idx]
            if not self._valid_landmark(lm):
                continue

            x, y = pts[idx]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            cv2.circle(canvas, (x, y), 4, gray, -1, cv2.LINE_AA)

            if self.show_labels:
                name = self.LANDMARK_NAMES[idx]
                xyz_text = f'({lm.x:.2f}, {lm.y:.2f}, {lm.z:.2f})'
                text_x = min(x + 10, w - 220)
                text_y = max(y - 10, 20)
                cv2.putText(canvas, name, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1, cv2.LINE_AA)
                cv2.putText(canvas, xyz_text, (text_x, text_y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, dim, 1, cv2.LINE_AA)

        highlight_chain = [(0, 16), (16, 28), (28, 27), (27, 15), (15, 0)]
        for i, j in highlight_chain:
            if self._valid_landmark(landmarks[i]) and self._valid_landmark(landmarks[j]):
                cv2.line(canvas, pts[i], pts[j], red, 3, cv2.LINE_AA)

        cv2.putText(canvas, 'Pose detected', (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return canvas

    def get_hand_raw(self, result=None, clamp=True):
        result = self.latest_result if result is None else result
        if result is None or not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0][self.WRIST]
        if not self._valid_landmark(lm):
            return None

        x, y, z = lm.x, lm.y, lm.z

        if clamp:
            x = float(np.clip(x, 0.0, 1.0))
            y = float(np.clip(y, 0.0, 1.0))
            z = float(np.clip(z, -1.0, 1.0))

        return {"x": x, "y": y, "z": z}
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

