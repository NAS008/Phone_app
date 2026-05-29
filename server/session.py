import cv2
import json
import uuid
import qrcode
import numpy as np
from urllib.parse import urlencode

class Session:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session_id = ""
        self.active_users = {}

    def create_session(self):
        self.session_id = str(uuid.uuid4())
        self.active_users = {}
        return self.session_id

    def get_qr_url(self):
        if not self.session_id:
            raise ValueError("✗ Session: No active session")

        return f"{self.base_url}?{urlencode({'session_id': self.session_id})}"

    def join_user(self, session_id, nickname):
        nickname = nickname.strip()

        if not nickname:
            return False

        if str(session_id) != str(self.session_id):
            return False

        self.active_users[nickname] = True
        return True

    def is_user_joined(self, nickname):
        return nickname in self.active_users

    def is_valid_message(self, session_id, nickname):
        if str(session_id) != str(self.session_id):
            return False

        return nickname in self.active_users

    def remove_user(self, nickname):
        if nickname in self.active_users:
            del self.active_users[nickname]
            return True
        return False

    def export_state(self):
        return json.dumps({
            "session_id": self.session_id,
            "active_users": list(self.active_users.keys())
        })

    def import_state(self, data):
        obj = json.loads(data)
        self.session_id = obj.get("session_id", "")
        self.active_users = {nickname: True for nickname in obj.get("active_users", [])}

    def generate_qr_code(self):
        qr_url = self.get_qr_url()

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=1,
        )
        qr.add_data(qr_url)
        qr.make(fit=True)

        pil_img = qr.make_image(
            fill_color="white",
            back_color=(0, 0, 0, 0)
        ).convert("RGBA")

        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]
        border_px = max(4, h // 25)

        cv_img = cv2.copyMakeBorder(
            cv_img,
            top=border_px,
            bottom=border_px,
            left=border_px,
            right=border_px,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        cv2.imwrite(r"..\..\brand\qrcode.png", cv_img)
        return cv_img