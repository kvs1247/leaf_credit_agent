"""
LEAF Credit Agent — Evidence Ledger
Immutable SQLite store. Every layer artifact is hashed and sealed.
Sprint 3: Added summaries table, clear_all_data, get_all_summaries.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from models.schemas import LedgerEntry

DB_PATH = Path(__file__).parent.parent / "data" / "evidence_ledger.db"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_ledger():
    """Create all tables if they do not exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_ledger (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id  TEXT NOT NULL,
                layer           TEXT NOT NULL,
                artifact_json   TEXT NOT NULL,
                artifact_hash   TEXT NOT NULL,
                sealed_at       TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id      TEXT NOT NULL UNIQUE,
                applicant_name      TEXT,
                amount_requested    REAL,
                purpose             TEXT,
                decision            TEXT,
                approval_probability REAL,
                confidence          TEXT,
                scenario_tag        TEXT,
                timestamp           TEXT NOT NULL
            )
        """)
        conn.commit()


def seal_artifact(application_id: str, layer: str, artifact: dict) -> LedgerEntry:
    """Hash and store a layer artifact. Immutable once sealed."""
    artifact_json = json.dumps(artifact, default=str, sort_keys=True)
    artifact_hash = hashlib.sha256(artifact_json.encode()).hexdigest()[:12]
    sealed_at = datetime.now()

    entry = LedgerEntry(
        application_id=application_id,
        layer=layer,
        artifact_json=artifact_json,
        artifact_hash=artifact_hash,
        sealed_at=sealed_at
    )

    with _get_connection() as conn:
        conn.execute("""
            INSERT INTO evidence_ledger
                (application_id, layer, artifact_json, artifact_hash, sealed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (entry.application_id, entry.layer, entry.artifact_json,
              entry.artifact_hash, entry.sealed_at.isoformat()))
        conn.commit()

    return entry


def write_summary(
    application_id: str,
    applicant_name: str,
    amount_requested: float,
    purpose: str,
    decision: str,
    approval_probability: float,
    confidence: str,
    scenario_tag: Optional[str] = None,
):
    """Write one summary row after a successful run."""
    with _get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO summaries
                (application_id, applicant_name, amount_requested, purpose,
                 decision, approval_probability, confidence, scenario_tag, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (application_id, applicant_name, amount_requested, purpose,
              decision, approval_probability, confidence, scenario_tag,
              datetime.now().isoformat()))
        conn.commit()


def get_all_summaries() -> List[dict]:
    """Retrieve all summary rows ordered by most recent first."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM summaries ORDER BY timestamp DESC
        """).fetchall()
    return [dict(row) for row in rows]


def get_layer_artifact(application_id: str, layer: str) -> Optional[dict]:
    """Retrieve and deserialise a specific layer artifact."""
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT artifact_json FROM evidence_ledger
            WHERE application_id = ? AND layer = ?
            ORDER BY id DESC LIMIT 1
        """, (application_id, layer)).fetchone()
    if row:
        return json.loads(row["artifact_json"])
    return None


def retrieve_application_ledger(application_id: str) -> List[LedgerEntry]:
    """Retrieve all sealed artifacts for a given application."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM evidence_ledger
            WHERE application_id = ? ORDER BY id ASC
        """, (application_id,)).fetchall()
    return [
        LedgerEntry(
            application_id=row["application_id"],
            layer=row["layer"],
            artifact_json=row["artifact_json"],
            artifact_hash=row["artifact_hash"],
            sealed_at=datetime.fromisoformat(row["sealed_at"])
        )
        for row in rows
    ]


def clear_all_data():
    """
    Delete all data from both tables.
    Used by Reset Demo Data button.
    In production this would require multi-factor authorisation.
    In this PoC it exists for demonstration and testing purposes.
    """
    with _get_connection() as conn:
        conn.execute("DELETE FROM evidence_ledger")
        conn.execute("DELETE FROM summaries")
        conn.commit()
