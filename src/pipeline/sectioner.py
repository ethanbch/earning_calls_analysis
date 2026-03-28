"""Label each speaker turn with a section tag.

Section ∈ {Prepared, Q, A, O}:
  - Prepared : executive speaking *before* the Q&A session starts
  - Q        : analyst asking a question during Q&A
  - A        : executive answering during Q&A
  - O        : operator / moderator

The Q&A boundary is detected when the first Analyst turn appears, or when
the Operator introduces a question-and-answer segment.

DataFrame in → DataFrame out.
"""

from __future__ import annotations

import logging
import re
from typing import List

import pandas as pd


def label_sections(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Add a ``section`` column to a segments DataFrame."""
    if df.empty:
        df = df.copy()
        df["section"] = pd.Series(dtype=str)
        return df

    out = df.sort_values(["transcript_id", "segment_order"]).copy()
    sections: List[str] = []

    for _, grp in out.groupby("transcript_id", sort=False):
        qa_started = False
        has_seen_executive = False
        for _, row in grp.iterrows():
            stype = row["speaker_type"]
            seg_text = str(row.get("segment_text", ""))

            if stype == "Executive":
                has_seen_executive = True

            if stype == "Analyst":
                qa_started = True
                sec = "Q"
            elif stype == "Executive":
                sec = "A" if qa_started else "Prepared"
            elif stype == "Operator":
                sec = "O"
                # Only trigger Q&A from Operator text if we've already
                # seen at least one Executive turn (i.e. prepared remarks
                # have started). This prevents "[Operator Instructions]"
                # at the very top of the call from flipping qa_started.
                if has_seen_executive and re.search(
                    r"question-and-answer|q&a session|open.+(?:line|floor).+(?:question|q&a)",
                    seg_text,
                    flags=re.IGNORECASE,
                ):
                    qa_started = True
            else:
                sec = "O" if qa_started else "Prepared"

            sections.append(sec)

    out["section"] = sections
    logger.info(
        "Section distribution: %s",
        out["section"].value_counts().to_dict(),
    )
    return out
