from fastapi import APIRouter, Depends, Body, Query,UploadFile, File ,HTTPException
from validations.response import Response
from validations.request import PredictionResponse
from models.inference import generate_llm_response, generate_food_nutritional_response
import userService
from PIL import Image
from typing import Annotated, Literal
import torch
import io

router = APIRouter()

@router.get("/main")
async def main(user_message: str = Query(...)) -> Response:
    try:
        user_profile = userService.read_user_profile("jane_smith")
        llm_response = generate_llm_response(user_message, user_profile)
        return Response(llm_response=llm_response, is_task_completed=True)
    except Exception as e:
        print(f"Error in /main endpoint: {e}")
        return Response(llm_response=[], is_task_completed=False)


@router.post("/set-food")
async def set_food(response: PredictionResponse = Body(...)):
    try:
        dish = response.predicted_label.Specific_dish
        if dish.lower() == "unknown":
            dish = response.predicted_label.Food_group[0] if response.predicted_label.Food_group else "unknown dish"
        
        nutritional_response = generate_food_nutritional_response(dish)

        if(userService.get_user_if_exist("jane_smith")):
            userService.update_user_food_profile("jane_smith", dish, nutritional_response)
            return {"message": "Food information updated successfully"}
            
        return HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))