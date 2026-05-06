import os

DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))

USERPROFILE_PATH = os.path.join(DIRECTORY_PATH, "data", "user_profiles.json")

USER_PROFILE_KEYWORDS = {
        "NAME": "name",
        "AGE": "age",
        "DIETARY_PREFERENCES": "dietary_preferences",
        "ALLERGIES": "allergies",
        "HEALTH_CONDITIONS": "health_conditions",
        "FOOD_CONSUMED": "food_consumed",
        "DISH": "dish",
        "TIME": "time",
        "NUTRITIONAL_INFO": "nutritional_info"
    }
    
