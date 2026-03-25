"""SQLite-based episodic memory store."""
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    episode_id: str
    problem_id: str
    problem_text: str
    topic: str
    difficulty: int
    trace: List[Dict[str, Any]]
    final_answer: str
    verified: bool
    failure_mode: Optional[str]
    skills_used: List[str]
    duration_seconds: float
    num_steps: int
    timestamp: float


@dataclass
class Step:
    step_id: str
    episode_id: str
    step_number: int
    action_type: str
    skill_id: Optional[str]
    input_summary: str
    output_text: str
    timestamp: float


_CREATE_EPISODES = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id       TEXT PRIMARY KEY,
    problem_id       TEXT,
    problem_text     TEXT,
    topic            TEXT,
    difficulty       INTEGER,
    trace            TEXT,
    final_answer     TEXT,
    verified         INTEGER,
    failure_mode     TEXT,
    skills_used      TEXT,
    duration_seconds REAL,
    num_steps        INTEGER,
    timestamp        REAL
);
"""

_CREATE_STEPS = """
CREATE TABLE IF NOT EXISTS steps (
    step_id      TEXT PRIMARY KEY,
    episode_id   TEXT,
    step_number  INTEGER,
    action_type  TEXT,
    skill_id     TEXT,
    input_summary TEXT,
    output_text  TEXT,
    timestamp    REAL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);
"""


class EpisodeStore:
    """SQLite-backed store for episode traces."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_CREATE_EPISODES)
        cur.execute(_CREATE_STEPS)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_episode(self, episode_data: Dict[str, Any]) -> str:
        """Store a complete episode. Returns episode_id."""
        eid = episode_data.get("episode_id") or str(uuid.uuid4())
        cur = self._conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO episodes VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eid,
                episode_data.get("problem_id", ""),
                episode_data.get("problem_text", ""),
                episode_data.get("topic", ""),
                int(episode_data.get("difficulty", 1)),
                json.dumps(episode_data.get("trace", [])),
                str(episode_data.get("final_answer", "")),
                int(bool(episode_data.get("verified", False))),
                episode_data.get("failure_mode"),
                json.dumps(episode_data.get("skills_used", [])),
                float(episode_data.get("duration_seconds", 0.0)),
                int(episode_data.get("num_steps", 0)),
                float(episode_data.get("timestamp", time.time())),
            ),
        )
        # store individual steps
        for step in episode_data.get("trace", []):
            self.store_step(eid, step)
        self._conn.commit()
        return eid

    def store_step(self, episode_id: str, step: Dict[str, Any]) -> None:
        sid = step.get("step_id") or str(uuid.uuid4())
        cur = self._conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?,?,?)""",
            (
                sid,
                episode_id,
                int(step.get("step_number", 0)),
                step.get("action_type", ""),
                step.get("skill_id"),
                step.get("input_summary", ""),
                step.get("output_text", ""),
                float(step.get("timestamp", time.time())),
            ),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        return Episode(
            episode_id=row["episode_id"],
            problem_id=row["problem_id"],
            problem_text=row["problem_text"],
            topic=row["topic"],
            difficulty=row["difficulty"],
            trace=json.loads(row["trace"] or "[]"),
            final_answer=row["final_answer"],
            verified=bool(row["verified"]),
            failure_mode=row["failure_mode"],
            skills_used=json.loads(row["skills_used"] or "[]"),
            duration_seconds=row["duration_seconds"],
            num_steps=row["num_steps"],
            timestamp=row["timestamp"],
        )

    def get_recent(self, n: int = 10) -> List[Episode]:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_by_id(self, episode_id: str) -> Optional[Episode]:
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        return self._row_to_episode(row) if row else None

    def get_similar_episodes(
        self, problem_text: str, topic: str, limit: int = 5
    ) -> List[Episode]:
        """Simple keyword-based similarity (same topic, most recent)."""
        cur = self._conn.cursor()
        rows = cur.execute(
            """SELECT * FROM episodes WHERE topic = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (topic, limit * 3),
        ).fetchall()
        episodes = [self._row_to_episode(r) for r in rows]
        # crude token overlap scoring
        query_tokens = set(problem_text.lower().split())
        scored = []
        for ep in episodes:
            ep_tokens = set(ep.problem_text.lower().split())
            overlap = len(query_tokens & ep_tokens)
            scored.append((overlap, ep))
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:limit]]

    def get_stats(self, topic: Optional[str] = None) -> Dict[str, Any]:
        cur = self._conn.cursor()
        where = "WHERE topic = ?" if topic else ""
        params = (topic,) if topic else ()
        row = cur.execute(
            f"""SELECT
                  COUNT(*) AS total,
                  SUM(verified) AS successes,
                  AVG(duration_seconds) AS avg_duration,
                  AVG(num_steps) AS avg_steps
               FROM episodes {where}""",
            params,
        ).fetchone()
        total = row["total"] or 0
        successes = row["successes"] or 0
        return {
            "total_episodes": total,
            "success_count": successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_duration": row["avg_duration"] or 0.0,
            "avg_steps": row["avg_steps"] or 0.0,
        }

    def count(self) -> int:
        cur = self._conn.cursor()
        return cur.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
