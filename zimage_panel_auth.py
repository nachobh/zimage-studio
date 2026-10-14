"""Small session gate for a panel that can bind to the LAN."""
from __future__ import annotations

from collections import defaultdict, deque
from http.cookies import SimpleCookie
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time

from zimage_conversations import STORE_DIR

SESSION_LIFETIME = 86400
# Sessions are persisted; any older than this are dropped when the panel starts.
SESSION_PURGE_AFTER = 90 * 86400


def _digest(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


class PanelAuth:
    def __init__(self, enabled):
        self.enabled = enabled
        self.sessions = {}
        self.attempts = defaultdict(deque)
        self.lock = threading.Lock()
        self.password_file = Path(STORE_DIR) / 'panel-password'
        self.sessions_file = Path(STORE_DIR) / 'panel-sessions.json'
        if enabled:
            self.password_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            password = os.environ.get('ZIMAGE_PANEL_PASSWORD')
            if not password:
                try:
                    fd = os.open(self.password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'w') as output:
                        output.write(secrets.token_urlsafe(24) + '\n')
                except FileExistsError:
                    pass
                password = self.password_file.read_text().strip()
            if len(password) < 12:
                raise ValueError('ZIMAGE_PANEL_PASSWORD debe tener al menos 12 caracteres')
            self.password = password
            self._load_sessions()

    def _load_sessions(self):
        """Restore persisted sessions (stored hashed), dropping those older than 3 months."""
        try:
            stored = json.loads(self.sessions_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
        limit = time.time() - SESSION_PURGE_AFTER
        with self.lock:
            for key, item in stored.items():
                if (isinstance(item, dict) and isinstance(item.get('csrf'), str)
                        and isinstance(item.get('created'), (int, float))
                        and isinstance(item.get('expires'), (int, float))
                        and item['created'] >= limit):
                    self.sessions[key] = (item['csrf'], item['expires'], item['created'])
            self._save_locked()

    def _save_locked(self):
        data = {key: {'csrf': csrf, 'expires': expires, 'created': created}
                for key, (csrf, expires, created) in self.sessions.items()}
        try:
            self.sessions_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, temp = tempfile.mkstemp(prefix='.panel-sessions-', dir=self.sessions_file.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as out:
                    json.dump(data, out)
                os.chmod(temp, 0o600)
                os.replace(temp, self.sessions_file)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        except OSError:
            # Persistence is a convenience; the in-memory session still works.
            pass

    def login(self, password, address):
        if not self.enabled: return '', ''
        if not isinstance(password, str): return None
        with self.lock:
            current = time.time()
            attempts = self.attempts[address]
            while attempts and attempts[0] < current - 60: attempts.popleft()
            if len(attempts) >= 6: return None
            if not hmac.compare_digest(password.encode('utf-8'), self.password.encode('utf-8')):
                attempts.append(current)
                return None
            attempts.clear()
            token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            self.sessions[_digest(token)] = (csrf, current + SESSION_LIFETIME, current)
            self._save_locked()
            return token, csrf

    def _token(self, cookie_header):
        cookie = SimpleCookie()
        try: cookie.load(cookie_header or '')
        except Exception: return None
        morsel = cookie.get('zimage_session')
        return morsel.value if morsel is not None else None

    def session(self, cookie_header):
        if not self.enabled: return ('', '')
        token = self._token(cookie_header)
        if token is None: return None
        key = _digest(token)
        with self.lock:
            value = self.sessions.get(key)
            if not value or value[1] < time.time():
                if self.sessions.pop(key, None) is not None:
                    self._save_locked()
                return None
            return token, value[0]

    def logout(self, cookie_header):
        """Invalidate the session carried by the cookie; True if one existed."""
        if not self.enabled: return False
        token = self._token(cookie_header)
        if token is None: return False
        with self.lock:
            removed = self.sessions.pop(_digest(token), None) is not None
            if removed:
                self._save_locked()
            return removed
