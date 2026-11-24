"""Shared strict schema primitives."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import AfterValidator


def _identifier(value: str) -> str:
    if not value or len(value) > 128 or not all(c.isalnum() or c in "_.:-" for c in value):
        raise ValueError("identifier must be 1-128 ASCII letters, digits, or _.:-")
    return value.encode("ascii").decode("ascii")


Identifier = Annotated[str, AfterValidator(_identifier)]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class Contract(BaseModel):
    """Unknown fields are rejected so a planner cannot smuggle authority into a plan."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)
