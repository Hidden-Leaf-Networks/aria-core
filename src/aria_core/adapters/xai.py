"""xAI (Grok) model adapter.

xAI uses an OpenAI-compatible API, so this adapter extends OpenAIAdapter
with the xAI base URL.
"""

from __future__ import annotations

import os

from aria_core.adapters.openai import OpenAIAdapter


class XAIAdapter(OpenAIAdapter):
    """Adapter for xAI Grok models.

    Uses OpenAI-compatible API at https://api.x.ai/v1
    """

    name = "xai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-2-latest",
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("XAI_API_KEY"),
            model=model,
            base_url="https://api.x.ai/v1",
        )
