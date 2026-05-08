from config import  USERPROFILE_PATH, USER_PROFILE_KEYWORDS, MESSAGES_PATH, MESSAGE_KEYWORDS
from datetime import datetime
import json



def read_user_profile(user_id):
    with open(USERPROFILE_PATH, 'r') as f:
        user_profiles = json.load(f)
    
    user = user_profiles.get(user_id)

    if not user:
        raise ValueError(f"User with ID {user_id} not found in user profiles.")

    name = user[USER_PROFILE_KEYWORDS["NAME"]]

    age = user[USER_PROFILE_KEYWORDS["AGE"]]

    dietary_preferences = user[USER_PROFILE_KEYWORDS["DIETARY_PREFERENCES"]]

    allergies = user[USER_PROFILE_KEYWORDS["ALLERGIES"]]

    health_conditions = user[USER_PROFILE_KEYWORDS["HEALTH_CONDITIONS"]]

    food_consumed = user[USER_PROFILE_KEYWORDS["FOOD_CONSUMED"]]

    formated_text = generate_text_context(food_consumed)
    return  f"""
    {name}, aged {age}, has dietary preferences of {dietary_preferences} and is allergic to {allergies}. They have the following health conditions: {health_conditions}.  
    """ + formated_text


def get_recent_user_message_history(user_id):
    with open(MESSAGES_PATH, 'r') as f:
        message_history = json.load(f)
    
    user_messages = message_history.get(user_id, [])
    
    if not user_messages:
        message_history[user_id] = []
        with open(MESSAGES_PATH, 'w') as f:
            json.dump(message_history, f, indent=4)
    
    current_date = datetime.now().date()
    
    # Filter messages from today
    today_messages = [
        msg for msg in user_messages
        if datetime.fromisoformat(msg[MESSAGE_KEYWORDS["TIMESTAMP"]]).date() == current_date
    ]
    
    if not today_messages:
        return []
    
    return (
        []
        if (
            today_messages[-1][MESSAGE_KEYWORDS["IS_TASK_COMPLETED"]]
            and today_messages[-2][MESSAGE_KEYWORDS["IS_TASK_COMPLETED"]]
        )
        else [
            {
                "role": msg[MESSAGE_KEYWORDS["USER"]],
                "content": msg[MESSAGE_KEYWORDS["CONTENT"]]
            }
            for msg in today_messages
            ]
        )
    


def add_user_message_history(user_id, user_message, assistant_response)-> tuple[bool, str]:
    with open(MESSAGES_PATH, 'r') as f:
        all_message_history = json.load(f)
        
    user_messages = all_message_history.get(user_id, [])

    is_task_completed = "[TASK_COMPLETED]" in assistant_response

    cleaned_response = assistant_response.replace("[TASK_COMPLETED]", "").strip()

    user_messages.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().date().strftime("%Y-%m-%d"),
        "is_task_completed": is_task_completed
    })
    user_messages.append({
        "role": "assistant",
        "content": cleaned_response,
        "timestamp": datetime.now().date().strftime("%Y-%m-%d"),
        "is_task_completed": is_task_completed
    })

    all_message_history[user_id] = user_messages

    with open(MESSAGES_PATH, 'w') as f:
        json.dump(all_message_history, f, indent=4)

    return is_task_completed, cleaned_response


def generate_text_context(food_consumed):
    text_template = ""
    total_food_consumed = len(food_consumed)
    if total_food_consumed == 0:
        return text_template + " No food consumed."
    
    food_item = food_consumed[total_food_consumed - 1]

    dish = food_item[USER_PROFILE_KEYWORDS["DISH"]]

    time = food_item[USER_PROFILE_KEYWORDS["TIME"]]

    nutritional_info = food_item[USER_PROFILE_KEYWORDS["NUTRITIONAL_INFO"]]

    day = convert_time_to_context(time)

    text_template += f" They are eating a meal consisting of {dish} at {day}({time}), which has the following nutritional profile: {nutritional_info}."
    
    return text_template
    

def update_user_food_profile(user_id, dish, nutritional_info):
    with open(USERPROFILE_PATH, 'r') as f:
        user_profiles = json.load(f)
    
    user = user_profiles.get(user_id, {})

    food_consumed = {
        USER_PROFILE_KEYWORDS["DISH"]: dish,
        USER_PROFILE_KEYWORDS["TIME"]: datetime.now().isoformat(),
        USER_PROFILE_KEYWORDS["NUTRITIONAL_INFO"]: nutritional_info
    }
    current_food_consumed = user.get(USER_PROFILE_KEYWORDS["FOOD_CONSUMED"], [])

    print(f"Current food consumed for user {user_id}: {current_food_consumed}")

    current_food_consumed.append(food_consumed)

    user[USER_PROFILE_KEYWORDS["FOOD_CONSUMED"]] = current_food_consumed

    user_profiles[user_id] = user

    with open(USERPROFILE_PATH, 'w') as f:
        json.dump(user_profiles, f, indent=4)


def get_user_if_exist(user_id)-> bool:
    with open(USERPROFILE_PATH, 'r') as f:
        user_profiles = json.load(f)
    
    return user_id in user_profiles


def create_user_profile(user_id, name, age, dietary_preferences, allergies, health_conditions):
    with open(USERPROFILE_PATH, 'r') as f:
        user_profiles = json.load(f)
    
    user_profiles[user_id] = {
        USER_PROFILE_KEYWORDS["NAME"]: name,
        USER_PROFILE_KEYWORDS["AGE"]: age,
        USER_PROFILE_KEYWORDS["DIETARY_PREFERENCES"]: dietary_preferences,
        USER_PROFILE_KEYWORDS["ALLERGIES"]: allergies,
        USER_PROFILE_KEYWORDS["HEALTH_CONDITIONS"]: health_conditions
    }

    with open(USERPROFILE_PATH, 'w') as f:
        json.dump(user_profiles, f, indent=4)


def  update_user_profile(user_id, name=None, age=None, dietary_preferences=None, allergies=None, health_conditions=None):
    with open(USERPROFILE_PATH, 'r') as f:
        user_profiles = json.load(f)
    
    user = user_profiles.get(user_id, {})

    if name is not None:
        user[USER_PROFILE_KEYWORDS["NAME"]] = name
    if age is not None:
        user[USER_PROFILE_KEYWORDS["AGE"]] = age
    if dietary_preferences is not None:
        user[USER_PROFILE_KEYWORDS["DIETARY_PREFERENCES"]] = dietary_preferences
    if allergies is not None:
        user[USER_PROFILE_KEYWORDS["ALLERGIES"]] = allergies
    if health_conditions is not None:
        user[USER_PROFILE_KEYWORDS["HEALTH_CONDITIONS"]] = health_conditions

    user_profiles[user_id] = user

    with open(USERPROFILE_PATH, 'w') as f:
        json.dump(user_profiles, f, indent=4)


def convert_time_to_context(time_str):
    if not time_str:
        return "unknown time"
    
    dt = datetime.fromisoformat(time_str)

    hour = dt.hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"
