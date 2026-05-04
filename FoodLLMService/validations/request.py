from pydantic import BaseModel, Field

class PredictionLabel(BaseModel):
    Specific_dish: str = Field(..., description="Whether the user wants specific dishes or not")
    Cuisine: str = Field(..., description="The type of cuisine the user is interested in")
    Parent_category: str = Field(..., description="The parent dish or category the user is interested in")
    Food_group: list[str] = Field(..., description="The food group the user is interested in")

class PredictionResponse(BaseModel):
    predicted_label: PredictionLabel = Field(..., description="The predicted label for the food item")