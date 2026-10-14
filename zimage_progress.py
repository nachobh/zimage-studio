"""Small thread-safe, bounded progress registry; no prompts or image data."""
import threading
import time


class Progress:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}

    def update(self, key, phase, step=0, total=0):
        if not isinstance(key, str) or not key or len(key) > 100:
            return
        with self.lock:
            now = time.monotonic()
            if key not in self.jobs:
                if len(self.jobs) >= 256:
                    self.jobs.pop(next(iter(self.jobs)))
                self.jobs[key] = {'started': now}
            self.jobs[key].update(phase=phase, step=step, total=total, updated=now)

    def get(self, key):
        with self.lock:
            job = self.jobs.get(key)
            if not job or time.monotonic() - job['updated'] > 3600:
                return {'status': 'ok', 'phase': 'waiting', 'step': 0, 'total': 0}
            return {'status': 'ok', 'phase': job['phase'], 'step': job['step'],
                    'total': job['total'], 'elapsed': round(time.monotonic()-job['started'], 1)}


progress = Progress()
