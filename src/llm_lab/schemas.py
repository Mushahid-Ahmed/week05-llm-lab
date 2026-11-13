from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Category = Literal[
    "vehicle",
    "property",
    "travel",
    "health",
    "other",
    "context-dependent",
]


class ClaimAnalysis(BaseModel):
    """Schema for a validated preliminary claim analysis."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    summary: str = Field(min_length=1)
    missing_information: list[str]
    needs_human_review: bool
    rationale: str = Field(min_length=1)

    @field_validator("summary", "rationale")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("missing_information")
    @classmethod
    def _clean_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]

        if any(not item for item in cleaned):
            raise ValueError("items must not be blank")

        return cleaned


def parse_analysis(raw: str) -> ClaimAnalysis:
    """Parse model output into the schema, raising ValueError on failure."""

    try:
        return ClaimAnalysis.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(_summarize(exc)) from exc
    except ValueError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def _summarize(exc: ValidationError) -> str:
    parts = []

    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")

    return "Output failed schema validation -> " + "; ".join(parts)
