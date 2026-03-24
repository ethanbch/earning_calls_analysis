from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .models import TranscriptRecord
from .state import StateStore
from .utils import ensure_dir


@dataclass(slots=True)
class PersistResult:
    stored: bool
    path: Path | None
    reason: str


class Storage:
    def __init__(self, *, raw_dir: Path, processed_dir: Path, logger) -> None:
        self.raw_dir = ensure_dir(raw_dir)
        self.processed_dir = ensure_dir(processed_dir)
        self.manifest_parquet = self.processed_dir / "koyfin_manifest.parquet"
        self.index_parquet = self.processed_dir / "koyfin_index.parquet"
        self.logger = logger
        self._lock = threading.RLock()
        self._uid_to_path: dict[str, str] = {}
        self._hash_to_uid: dict[str, str] = {}
        self._load_index_cache()

    def _load_index_cache(self) -> None:
        if not self.index_parquet.exists():
            return
        try:
            df = pd.read_parquet(self.index_parquet)
        except Exception:
            return
        for _, row in df.iterrows():
            uid = str(row.get("transcript_uid") or "")
            raw_path = str(row.get("raw_path") or "")
            content_hash = str(row.get("sha256_raw_text") or "")
            if uid and raw_path:
                self._uid_to_path[uid] = raw_path
            if uid and content_hash:
                self._hash_to_uid[content_hash] = uid

    def _target_path(self, record: TranscriptRecord) -> Path:
        day = (record.transcript_date or record.fetch_timestamp_utc[:10]).split("-")
        yyyy, mm, dd = day[0], day[1], day[2]
        out_dir = ensure_dir(self.raw_dir / yyyy / mm / dd)
        return out_dir / f"{record.transcript_uid}.json"

    def persist_raw_transcript(self, record: TranscriptRecord) -> PersistResult:
        with self._lock:
            if record.transcript_uid in self._uid_to_path:
                return PersistResult(False, Path(self._uid_to_path[record.transcript_uid]), "uid_exists")
            if record.sha256_raw_text in self._hash_to_uid:
                return PersistResult(False, None, "hash_duplicate")

            target = self._target_path(record)
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(target)

            self._uid_to_path[record.transcript_uid] = str(target)
            self._hash_to_uid[record.sha256_raw_text] = record.transcript_uid
            return PersistResult(True, target, "stored")

    def export_parquets(self, state: StateStore) -> None:
        with self._lock:
            manifest_rows = [dict(r) for r in state.fetch_manifest_export_rows()]
            index_rows = [dict(r) for r in state.fetch_index_export_rows()]

            ensure_dir(self.manifest_parquet.parent)
            pd.DataFrame(manifest_rows).to_parquet(self.manifest_parquet, index=False)
            pd.DataFrame(index_rows).to_parquet(self.index_parquet, index=False)
