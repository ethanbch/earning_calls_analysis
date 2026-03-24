from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass(slots=True)
class DateWindow:
    start: date
    end: date

    def key(self) -> tuple[str, str]:
        return (self.start.isoformat(), self.end.isoformat())


@dataclass(slots=True)
class ManifestRow:
    transcript_uid: str
    window_start: str
    window_end: str
    result_page: int
    result_position: int
    title: str | None
    company_name: str | None
    row_text: str
    source_url: str | None
    dom_key: str | None
    data_id: str | None
    is_earnings_call: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptRecord:
    transcript_uid: str
    source: str
    fetch_timestamp_utc: str
    transcript_date: str | None
    company_name: str | None
    title: str | None
    event_type: str | None
    source_url: str | None
    participants: list[str]
    speaker_blocks: list[dict[str, Any]]
    raw_text: str
    raw_html: str | None
    window_start: str
    window_end: str
    result_page: int
    result_position: int
    sha256_raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NetworkProbeResult:
    http_possible: bool
    api_endpoint: str | None
    api_method: str | None
    request_headers: dict[str, str]
    request_payload: str | None
    notes: str


@dataclass(slots=True)
class StrategyBenchmark:
    strategy: str
    sample_size: int
    succeeded: int
    failed: int
    transcripts_per_min: float
    avg_latency_seconds: float
    projected_hours: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostProcessSegment:
    speaker: str | None
    role: str | None
    text: str
    segment_type: str
