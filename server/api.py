from fastapi import APIRouter, Depends, Body, Query,UploadFile, File ,HTTPException
from predict_resnet import predict as predict_resnet
from predict_mobilenet import predict as predict_mobilenet
from validations.modelparams import ModelParams
from PIL import Image
from typing import Annotated, Literal
from utilities.load_model import load_model
import torch
import io

router = APIRouter(
    tags=["detect"]
)

@router.post("/detect")
async def detect_image(model : Annotated[ModelParams,  Query()],image: UploadFile = File(...)):

    try:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file= await image.read()

        image_bytes = Image.open(io.BytesIO(file)).convert("RGB")

        image_tensor, proposed_model = load_model(image_bytes, model.name, device)
        
        if(model.name == "resnet"):
            predicted_label = predict_resnet(proposed_model, image_tensor, device)
        else:
            predicted_label = predict_mobilenet(proposed_model, image_tensor, device)

        return {"predicted_label": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
