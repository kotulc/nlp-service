from enum import Enum
from pydantic import Field

from app.models.base import BaseRequest


# Define supported default summary types
class SummaryType(str, Enum):
    title = "title"             # Feasible content titles
    subtitle = "subtitle"       # Feasible content subtitles 
    description = "description" # Basic content summary
    outline = "outline"         # A list of content section key points or themes


class SummaryRequest(BaseRequest):
    summary: SummaryType = Field(..., description="Return the specified summary type")
    top_n: int | None = Field(None, description="Return the top N summaries of the specified type")
    n_sections: int | None = Field(None, description="The number of content sections to outline")
