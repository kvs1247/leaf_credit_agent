"""
LEAF Credit Agent — Evidence Ledger
Immutable SQLite store. Every layer artifact is hashed and sealed.
This is what regulators and auditors retrieve during an audit.
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
    """Create the Evidence Ledger table if it does not exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_ledger (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id TEXT NOT NULL,
                layer       TEXT NOT NULL,
                artifact_json TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                sealed_at   TEXT NOT NULL
            )
        """)
        conn.commit()


def seal_artifact(application_id: str, layer: str, artifact: dict) -> LedgerEntry:
    """
    Hash and store a layer artifact.
    Once sealed, artifacts cannot be modified — only retrieved.
    This immutability is what makes the Evidence Ledger auditable.
    """
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
        """, (
            entry.application_id,
            entry.layer,
            entry.artifact_json,
            entry.artifact_hash,
            entry.sealed_at.isoformat()
        ))
        conn.commit()

    return entry


def retrieve_application_ledger(application_id: str) -> List[LedgerEntry]:
    """Retrieve all sealed artifacts for a given application."""
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM evidence_ledger
            WHERE application_id = ?
            ORDER BY id ASC
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
