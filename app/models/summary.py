from enum import Enum
from pydantic import Field

from app.schemas.base import BaseRequest, get_response
from app.core.summary import headings


# Get heading types
HEADING_TYPES = {k for k in headings.HEADING_PROMPTS.keys()}


# Define supported default summary types
class SummaryType(str, Enum):
    title = headings.get_title              # Suggested content titles
    subtitle = headings.get_subtitle        # Candidate content subtitles 
    description = headings.get_description  # Basic content summaries
    outline = headings.get_outline          # A list of content section key points or themes


class SummaryRequest(BaseRequest):
    summary: SummaryType = Field(..., description="Return the specified summary type")
    top_n: int | None = Field(None, description="Return the top N summaries of the specified type")
    n_sections: int | None = Field(None, description="The number of content sections to outline")
