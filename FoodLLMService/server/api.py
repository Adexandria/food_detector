from fastapi import APIRouter, Depends, Body, Query,HTTPException
from validations.response import Response, create_llm_response
from validations.request import PredictionResponse
from models.inference import generate_llm_response, generate_food_nutritional_response
import userService


router = APIRouter()

@router.get("/main")
async def main(user_message: str = Query(...)) -> Response:
    try:
        user_profile = userService.read_user_profile("jane_smith")

        user_message_history = userService.get_recent_user_message_history("jane_smith")

        assistant_response = generate_llm_response(user_message, user_profile, user_message_history)

        is_task_completed, cleaned_response = userService.add_user_message_history("jane_smith", user_message, assistant_response)

        return create_llm_response(cleaned_response, is_task_completed=is_task_completed)
    except Exception as e:
        print(f"Error in /main endpoint: {e}")
        return Response(response_segments=[], is_task_completed=True)


@router.post("/set-food")
async def set_food(response: PredictionResponse = Body(...)):
    try:
        dish = response.predicted_label.specific_dish
        if dish.lower() == "uncertain":
            dish = response.predicted_label.food_groups[0] if response.predicted_label.food_groups else "unknown dish"
        
        cuisine = response.predicted_label.cuisine
        if cuisine.lower() == "uncertain":
            cuisine = "unknown cuisine"

        food_groups = response.predicted_label.food_groups[1:] if len(response.predicted_label.food_groups) > 1 else "unknown food groups"

        nutritional_response = generate_food_nutritional_response(dish, cuisine, food_groups)

        if(userService.get_user_if_exist("jane_smith")):
            userService.update_user_food_profile("jane_smith", dish, nutritional_response)
            return {"message": "Food information updated successfully"}
            
        return HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))