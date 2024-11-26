from contextlib import contextmanager
from threading import Lock

from tinydb import TinyDB


class TinyThreadSafeDB:
    def __init__(self, db_path: str):
        self.db = TinyDB(db_path)
        self._lock = Lock()

    @contextmanager
    def atomic_operation(self):
        """Context manager for thread-safe database operations"""
        with self._lock:
            yield self.db

    def insert(self, data: dict):
        """Thread-safe insertion of preference data"""
        with self.atomic_operation() as db:
            db.insert(data)
