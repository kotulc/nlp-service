from enum import Enum
from pydantic import Field
from typing import List

from app.models.schemas import BaseResponse, BaseRequest, get_response
from app.core.summary.summary import SummaryType 


class SummaryResults(BaseModel):
    summaries: List[str] = Field(..., description="The summaries generated from the supplied content")
    scores: List[float] = Field(..., description="The scores for each returned summary")


class SummaryResponse(BaseResponse):
    results: SummaryResults = Field(..., description="The extracted summaries of each requested type")


class SummaryRequest(BaseRequest):
    summary: SummaryType = Field(SummaryType.description, description="Return the specified summary type")
    top_n: int | None = Field(None, description="Return the top N summaries of the specified type")
    n_sections: int | None = Field(None, description="The number of content sections to outline")
