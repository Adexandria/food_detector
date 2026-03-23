from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    interacting: bool = Field(..., description="Indicates if a person is interacting with food")
    results: dict = Field(..., description="The predicted label")

