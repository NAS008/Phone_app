# Cloudflare TURN (Realtime) — dynamic short-lived relay credentials.
# Cloudflare's free TURN service has no static username/password: credentials
# are minted via API and expire after `ttl` seconds. Both the PC (aiortc) and
# the phone backend (which hands them to the browser) fetch through this class.
# Dashboard setup: Cloudflare → Realtime → TURN Server → create key, then set
# CF_TURN_KEY_ID and CF_TURN_API_TOKEN in the environment.

import json
import threading
import time
import urllib.request

class CloudflareTurn:
    """Fetches and caches Cloudflare ICE servers in browser iceServers format."""

    API_URL = "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"

    def __init__(self, key_id, api_token, ttl=86400):
        self.key_id = key_id
        self.api_token = api_token
        self.ttl = ttl
        self._lock = threading.Lock()
        self._servers = None
        self._expires_at = 0.0

    @property
    def enabled(self):
        return bool(self.key_id and self.api_token)

    def get_ice_servers(self):
        """Cached [{"urls": [...], "username"?, "credential"?}] or None on failure."""
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            if now < self._expires_at:
                return self._servers
            try:
                self._servers = self._fetch()
                # Refresh well before the credentials actually expire
                self._expires_at = now + self.ttl * 0.8
            except Exception as exc:
                print(f"TURN: Cloudflare credential fetch failed - {exc}")
                self._servers = None
                self._expires_at = now + 60  # back off instead of hammering the API
            return self._servers

    def _fetch(self):
        request = urllib.request.Request(
            self.API_URL.format(key_id=self.key_id),
            data=json.dumps({"ttl": self.ttl}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "User-Agent": "TFNCA/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        servers = payload.get("iceServers")
        if not servers:
            raise ValueError(f"unexpected response: {payload}")
        return servers
