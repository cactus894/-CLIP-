from pydantic import BaseModel


class VideoProcessResponse(BaseModel):
    status: str
    message: str
    start_time: float
    end_time: float
    output_url: str | None
