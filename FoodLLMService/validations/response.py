from pydantic import BaseModel, Field

class llmResponse(BaseModel):
    message: str = Field(..., description="The response message from the LLM")
    action: str = Field(..., description="The action to be performed by the robot")
    expression: str = Field(..., description="The facial expression to be displayed by the robot")
    pause_ms: int = Field(..., description="The duration of the pause before the next message in milliseconds")
    
class Response(BaseModel):
    llm_response: list[llmResponse] = Field(..., description="The response from the LLM")
    is_task_completed: bool = Field(..., description="Indicates if the task is completed")