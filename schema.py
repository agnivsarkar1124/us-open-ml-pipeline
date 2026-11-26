"""Structured output the agent must produce. Validated by Pydantic before grounding checks."""

from typing import Literal

from pydantic import BaseModel, Field

SourceTool = Literal["predict_match", "player_profile", "head_to_head"]


class Factor(BaseModel):
    claim: str = Field(description="One specific, evidence-backed point, e.g. a rating or stat edge.")
    source: SourceTool = Field(description="The tool whose output supports this claim.")


class ScoutingReport(BaseModel):
    player_a: str
    player_b: str
    predicted_winner: str = Field(description="Must equal the winner returned by predict_match.")
    win_probability_pct: float = Field(description="Must equal the confidence returned by predict_match.")
    key_factors: list[Factor] = Field(min_length=2, max_length=5)
    upset_path: str = Field(description="How the underdog could still win, using only tool data.")
    summary: str = Field(description="2-4 sentence scouting summary that answers the user's question.")
