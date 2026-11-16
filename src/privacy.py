"""Shared PII matching and redaction helpers."""

from __future__ import annotations

import re

_presidio_analyzer = None

PII_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:\+?\d[\d ()-]{7,}\d)\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b"),
)

PII_HINTS = (
    "name",
    "patient",
    "address",
    "born",
    "phone",
    "email",
    "ssn",
    "social security",
    "passport",
    "license",
    "doctor",
)


def _known_entities(pii_entities) -> list[str]:
    return sorted(
        {
            str(entity).strip()
            for entity in pii_entities or []
            if str(entity).strip()
        },
        key=len,
        reverse=True,
    )


def detect_pii_spans(text: str, pii_entities=None) -> list[tuple[int, int]]:
    """Return spans matching labels, Presidio entities, or common formats."""
    if not text:
        return []

    spans = []
    for entity in _known_entities(pii_entities):
        for match in re.finditer(re.escape(entity), text, flags=re.IGNORECASE):
            spans.append(match.span())

    for pattern in PII_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(text))

    candidate_for_ner = any(
        hint in text.lower()
        for hint in PII_HINTS
    )
    if not spans and candidate_for_ner:
        global _presidio_analyzer
        if _presidio_analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine

                _presidio_analyzer = AnalyzerEngine()
            except Exception:
                _presidio_analyzer = False

        if _presidio_analyzer:
            try:
                spans.extend(
                    (result.start, result.end)
                    for result in _presidio_analyzer.analyze(
                        text=text,
                        language="en",
                    )
                )
            except Exception:
                pass

    unique_spans = sorted(set(spans))
    merged = []
    for start, end in unique_spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def contains_pii(text: str, pii_entities=None) -> bool:
    return bool(detect_pii_spans(text, pii_entities))


def redact_pii(text: str, pii_entities=None) -> str:
    """Replace detected PII with a neutral marker before remote processing."""
    spans = detect_pii_spans(text, pii_entities)
    if not spans:
        return text

    redacted = text
    for start, end in reversed(spans):
        redacted = redacted[:start] + "[REDACTED]" + redacted[end:]
    return redacted