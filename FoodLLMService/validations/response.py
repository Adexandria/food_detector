from pydantic import BaseModel, Field
import re

class llmResponse(BaseModel):
    message: str = Field(..., description="The response message from the LLM")
    action: str = Field(..., description="The action to be performed by the robot")
    expression: str = Field(..., description="The facial expression to be displayed by the robot")
    pause_ms: int = Field(..., description="The duration of the pause before the next message in milliseconds")
    
class Response(BaseModel):
    response_segments: list[llmResponse] = Field(..., description="The response from the LLM")
    is_task_completed: bool = Field(..., description="Indicates if the task is completed")


def create_response(text: str) -> Response:
    llm_response =  text.split('[')
    action = re.search(r"Action:\s*([^,\]]+)", text).group(1).strip()
    expression = re.search(r"Expression:\s*([^,\]]+)", text).group(1).strip()
    response = llmResponse(
        message=llm_response[0].strip(),
        action=action,
        expression=expression,
        pause_ms=0
    )
    return Response(response_segments=[response], is_task_completed=False)