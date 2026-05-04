from typing import Annotated, Literal

from pydantic import BaseModel, Field

class ModelParams(BaseModel):
    name: Annotated[
        Literal["resnet", "mobilenet"],
        Field(
            description="The model to use for prediction. Must be either 'resnet' or 'mobilenet'."
        ),
    ] = "resnet"