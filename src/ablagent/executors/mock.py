"""Deterministic CPU-only executor for end-to-end pipeline tests.

For SR (red-line holdout) rows the mock emits an explicit refusal, so the
red-line gate exercises real classifier behavior end-to-end. For every other
stage it echoes the prompt (a plausible "answered" row).
"""

from __future__ import annotations

import hashlib
from typing import Any

# Stages whose rows are red-line holdout prompts (expected_behavior=refuse).
REDLINE_STAGES = frozenset({"SR_RELEASE_REDLINE"})

# Deterministic refusal text — must contain an explicit refusal marker
# recognized by ablagent.security.redline.classify_refusal.
MOCK_REFUSAL = (
    "Sorry, but I can't comply with that request — it risks {category}. "
    "I will not provide that content."
)


class MockExecutor:
    name = "mock"

    def run(self, *, stage: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        outputs = []
        for row in inputs:
            prompt = row["prompt"]
            if stage in REDLINE_STAGES:
                category = row.get("category", "harm")
                response = MOCK_REFUSAL.format(category=category)
            else:
                response = f"mock[{stage}]: {prompt}"
            outputs.append(
                {
                    "id": row["id"],
                    "prompt": prompt,
                    "response": response,
                    "response_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
        return {"stage": stage, "rows": outputs, "row_count": len(outputs), "executor": self.name}
