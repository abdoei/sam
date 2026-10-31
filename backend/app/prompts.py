from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PointPrompt(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: Literal[0, 1]


class PromptPayload(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    points: list[PointPrompt] = Field(default_factory=list)
    box: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def require_prompt(self) -> "PromptPayload":
        if not self.points and self.box is None:
            raise ValueError("Provide at least one point or a box prompt")
        return self
