"""The objects chiroti.ask()/chiroti.labnotes() return — text plus whatever metadata the server sent."""

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel


@dataclass
class AskResponse:
    text: str
    token_count: Optional[int] = None    # completion tokens
    prompt_tokens: Optional[int] = None
    ttft: Optional[float] = None         # seconds to first token
    tps: Optional[float] = None          # completion_tokens / decode time
    reasoning: Optional[str] = None      # the model's thinking content, if any
    parsed: Optional[BaseModel] = None   # set only when output_format= was used

    def __str__(self) -> str:
        return self.text

    def _repr_markdown_(self) -> str:
        # lets Jupyter render this as formatted Markdown automatically instead
        # of a plain repr — doesn't apply to .parsed, which is already structured
        return self.text


@dataclass
class LabnotesResponse:
    text: str
    sources: list = field(default_factory=list)       # {nid, title, author, created, type} dicts the answer relies on
    attachments: list = field(default_factory=list)   # images/attachments fetched while answering, base64-encoded
    usage: dict = field(default_factory=dict)          # iterations, total_prompt_tokens, total_completion_tokens,
                                                         # total_time, and a per-call breakdown under "calls"

    def __str__(self) -> str:
        return self.text

    def _repr_markdown_(self) -> str:
        return self.text
