"""
SQLite 存储层：记忆数据的持久化和检索。

表结构:
    memories:
        id               INTEGER PRIMARY KEY AUTOINCREMENT
        user_id          TEXT NOT NULL
        session_id       TEXT NOT NULL
        content          TEXT NOT NULL
        tags             TEXT          -- JSON 序列化的标签列表
        importance       REAL DEFAULT 0.5  -- 重要性评分 0~1
        created_time     TEXT NOT NULL      -- ISO 格式
        schedule         TEXT NOT NULL      -- ScheduleState JSON
        status           TEXT DEFAULT 'active'  -- active / archived / forgotten

索引:
    idx_memories_user      ON user_id
    idx_memories_session   ON session_id
    idx_memories_next_rev  ON (status, json_extract(schedule, '$.next_review_time'))
    idx_memories_status    ON status
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .ebbinghaus import ScheduleState, initial_schedule

logger = logging.getLogger("astrbot_plugin_mementy")


@dataclass
class MemoryRecord:
    """一条完整的记忆记录。"""

    id: Optional[int] = None
    user_id: str = ""
    session_id: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    created_time: Optional[datetime] = None
    schedule: Optional[ScheduleState] = None
    status: str = "active"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "content": self.content,
            "tags": self.tags,
            "importance": self.importance,
            "created_time": self.created_time.isoformat() if self.created_time else None,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "status": self.status,
        }


class MemoryStorage:
    """SQLite 存储管理器（线程安全，使用 per-thread 连接）。"""

    _CREATE_MEMORIES_TABLE = """
    CREATE TABLE IF NOT EXISTS memories (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         TEXT    NOT NULL,
        session_id      TEXT    NOT NULL,
        content         TEXT    NOT NULL,
        tags            TEXT    DEFAULT '[]',
        importance      REAL    DEFAULT 0.5,
        created_time    TEXT    NOT NULL,
        schedule        TEXT    NOT NULL,
        status          TEXT    DEFAULT 'active'
    );
    """

    _CREATE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);",
        "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);",
        "CREATE INDEX IF NOT EXISTS idx_memories_next_rev ON memories(status, created_time);",
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    @property
    def _conn(self) -> sqlite3.Connection:
        """获取当前线程的连接。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA foreign_keys=ON;")
        return self._local.conn

    def _init_schema(self):
        conn = self._conn
        conn.execute(self._CREATE_MEMORIES_TABLE)
        for idx_sql in self._CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.commit()

    # ------------------------------------------------------------------
    # 基础 CRUD
    # ------------------------------------------------------------------
    def add_memory(
        self,
        user_id: str,
        session_id: str,
        content: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        schedule: Optional[ScheduleState] = None,
        now: Optional[datetime] = None,
    ) -> int:
        """
        新增一条记忆，返回其自增 ID。
        自动填充 created_time 和初始 schedule。
        """
        now = now or datetime.now()
        schedule = schedule or initial_schedule(now)
        conn = self._conn
        cur = conn.execute(
            """
            INSERT INTO memories (user_id, session_id, content, tags, importance,
                                  created_time, schedule, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                user_id,
                session_id,
                content,
                json.dumps(tags or [], ensure_ascii=False),
                importance,
                now.isoformat(),
                json.dumps(schedule.to_dict()),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def update_schedule(self, memory_id: int, schedule: ScheduleState):
        """更新指定记忆的调度状态。"""
        conn = self._conn
        conn.execute(
            "UPDATE memories SET schedule = ? WHERE id = ?",
            (json.dumps(schedule.to_dict()), memory_id),
        )
        conn.commit()

    def update_status(self, memory_id: int, status: str):
        """更新记忆状态 (active / archived / forgotten)。"""
        conn = self._conn
        conn.execute(
            "UPDATE memories SET status = ? WHERE id = ?",
            (status, memory_id),
        )
        conn.commit()

    def get_by_id(self, memory_id: int) -> Optional[MemoryRecord]:
        conn = self._conn
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def delete(self, memory_id: int):
        conn = self._conn
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query_by_user(self, user_id: str, limit: int = 50) -> List[MemoryRecord]:
        conn = self._conn
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_time DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query_by_session(self, session_id: str, limit: int = 50) -> List[MemoryRecord]:
        conn = self._conn
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE session_id = ? AND status = 'active'
            ORDER BY created_time DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query_due(self, user_id: str, now: Optional[datetime] = None) -> List[MemoryRecord]:
        """
        查询当前到期需要复习的记忆。
        按 importance 降序返回，优先处理重要的。
        """
        now = now or datetime.now()
        now_iso = now.isoformat()
        conn = self._conn
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND status = 'active'
              AND json_extract(schedule, '$.next_review_time') IS NOT NULL
              AND json_extract(schedule, '$.next_review_time') <= ?
            ORDER BY importance DESC, json_extract(schedule, '$.next_review_time') ASC
            """,
            (user_id, now_iso),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def query_recent(
        self, user_id: str, session_id: Optional[str] = None, limit: int = 20
    ) -> List[MemoryRecord]:
        """
        查询用户最近创建的记忆（不考虑到期时间），作为兜底检索。
        """
        conn = self._conn
        if session_id:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND session_id = ? AND status = 'active'
                ORDER BY created_time DESC
                LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND status = 'active'
                ORDER BY created_time DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self, user_id: str, status: str = "active") -> int:
        conn = self._conn
        row = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ? AND status = ?",
            (user_id, status),
        ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def purge_forgotten(self, user_id: str, now: Optional[datetime] = None) -> int:
        """
        清理长时间未复习且已标记为 forgotten 的记忆。
        返回删除数量。
        """
        conn = self._conn
        cur = conn.execute(
            "DELETE FROM memories WHERE user_id = ? AND status = 'forgotten'",
            (user_id,),
        )
        conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        schedule_dict = json.loads(row["schedule"])
        tags = json.loads(row["tags"]) if row["tags"] else []
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=row["content"],
            tags=tags,
            importance=row["importance"],
            created_time=datetime.fromisoformat(row["created_time"]),
            schedule=ScheduleState.from_dict(schedule_dict),
            status=row["status"],
        )
