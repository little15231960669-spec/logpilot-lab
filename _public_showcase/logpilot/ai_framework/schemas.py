"""Structured output schemas for LangChain log parsing."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LogParseResult(BaseModel):
    template: str = Field(description="Log template with variables replaced by <*>.")
    variables: list[str] = Field(description="Extracted variables in occurrence order.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0 to 1.")
    reason: str = Field(description="Brief explanation for the extraction.")

