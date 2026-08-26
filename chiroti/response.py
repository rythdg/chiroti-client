"""The object chiroti.ask() returns — text plus whatever metadata the server sent."""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel


@dataclass
class ChirotiResponse:
    text: str
    token_count: Optional[int] = None    # completion tokens
    prompt_tokens: Optional[int] = None
    ttft: Optional[float] = None         # seconds to first token
    tps: Optional[float] = None          # completion_tokens / decode time
    reasoning: Optional[str] = None      # the model's thinking content, if any
    parsed: Optional[BaseModel] = None   # set only when output_format= was used

    def __str__(self) -> str:
        return self.text
