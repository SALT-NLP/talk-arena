from contextlib import asynccontextmanager
from threading import Lock as TLock
from asyncio import Lock as ALock

from tinydb import TinyDB
from tinydb.table import Table as TinyDBTable
import uuid

class UUIDTable(TinyDBTable):
    document_id_class = uuid.UUID

    def _get_next_id(self):
        return uuid.uuid4()


class UUIDB(TinyDB):
    table_class = UUIDTable


class TinyThreadSafeDB:
    def __init__(self, db_path: str):
        self.db = UUIDB(db_path)
        self._lock1 = TLock()
        self._lock2 = ALock()

    @asynccontextmanager
    async def atomic_operation(self):
        """Context manager for thread-safe database operations"""
        with self._lock1:
            async with self._lock2:
                yield self.db

    async def insert(self, data: dict):
        """Thread-safe insertion of preference data"""
        async with self.atomic_operation() as db:
            db.insert(data)
