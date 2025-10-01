from pydantic import BaseModel, Field
from typing import List

from app.schemas.schemas import BaseResponse, BaseRequest


class SummaryResults(BaseModel):
    summaries: List[str] = Field(..., description="The summaries generated from the supplied content")
    scores: List[float] = Field(..., description="The scores for each returned summary")


class SummaryResponse(BaseResponse):
    result: SummaryResults = Field(..., description="The extracted summaries of each requested type")


class SummaryRequest(BaseRequest):
    summary: str = Field(default="description", description="Return the specified summary type")
    n_sections: int | None = Field(None, description="The number of content sections to outline", gt=0)
    top_n: int | None = Field(None, description="Return the top N summaries of the specified type", gt=0)
