from pydantic import BaseModel, Field

class PredictionLabel(BaseModel):
    specific_dish: str = Field(..., description="Whether the user wants specific dishes or not")
    cuisine: str = Field(..., description="The type of cuisine the user is interested in")
    parent_category: str = Field(..., description="The parent dish or category the user is interested in")
    food_groups: list[str] = Field(..., description="The food group the user is interested in")

class PredictionResponse(BaseModel):
    predicted_label: PredictionLabel = Field(..., description="The predicted label for the food item")