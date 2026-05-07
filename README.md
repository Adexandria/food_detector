
# 🥗 Nutrition Microservices

This project contains **two microservices** designed to support **nutrition and dietary monitoring for elderly individuals** through food detection and conversational AI.

----------

# 📦 Services Overview

```text
┌──────────────────────┐
│   Food Service       │
│  (Food Detection)    │
└─────────┬────────────┘
          │ Detected Food
          ▼
┌──────────────────────┐
│  Food LLM Service    │
│ (Nutrition Assistant)│
└──────────────────────┘

```

----------

# 🍱 Food Service

This service simulates how a robot:

1.  Detects a person holding food
    
2.  Processes the image
    
3.  Classifies the food item using classifers

The classifier uses pretrained:

-   **ResNet**
    
-   **MobileNet V3**
    

Models are available on Kaggle:

👉 `https://www.kaggle.com/code/adeolawuraolaade/food-inference`

----------

# ⚙️ Setup

## 1️⃣ Create Virtual Environment

```bash
python -m venv .venv

```

----------

## 2️⃣ Install Dependencies

```bash
pip3 install torch torchvision
pip install -r requirements.txt

```

----------

# ▶️ Run Detection Pipeline

```bash
python detection_pipeline.py --use_resnet

```

----------

# 🌐 API Usage

The API supports two models:

Model

Parameter

ResNet

`resnet`

MobileNet V3 Large

`mobilenet`

----------

# 🚀 Run API Locally

```bash
python app.py

```

----------

# ☁️ Hosted API

## Using cURL

```bash
curl -X POST \
"https://papri-ka-food-detector.hf.space/detect?name=mobilenet" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "image=@160194.jpg;type=image/jpeg"

```

----------

# 📘 Swagger Documentation

Docs:

```text
https://papri-ka-food-detector.hf.space/docs#/detect/detect_image_detect_post

```

### Request URL

```text
https://papri-ka-food-detector.hf.space/detect?name=mobilenet

```

### Parameters

Choose one:

-   `resnet`
    
-   `mobilenet`
    

### Upload

-   Upload an image for food detection
    

----------

# ✅ Example Response

```json
{
  "predicted_label": {
    "specific_dish": "baby_back_ribs",
    "cuisine": "sty_american",
    "parent_category": "Meat-Centric Dishes",
    "food_groups": [
      "ff_bread",
      "ff_meat",
      "ff_soup",
      "ff_vegetable"
    ]
  }
}

```

----------

# 🤖 Food LLM Service

This service uses **GPT-4.1-mini** to simulate conversations between the robot and the elderly person.

----------

# 🧠 Features

## `/main`

Engages in conversation using:

-   User health status
    
-   Previously consumed food
    
-   Nutritional context
    

----------

## `/set-food`

Used to:

1.  Receive food detected by the Food Service
    
2.  Generate nutritional insights
    
3.  Store food information
    
4.  Support future conversations with the elderly user
    

----------

# 🔄 System Workflow

```text
        Image Input
              │
              ▼
     ┌────────────────┐
     │ Food Detector  │
     │ ResNet/MobileNet
     └───────┬────────┘
             │
             ▼
     Classified Food
             │
             ▼
     ┌────────────────┐
     │ Food LLM       │
     │ Nutrition AI   │
     └───────┬────────┘
             │
             ▼
   Nutrition Guidance &
   Elderly Conversation

```

----------

# 🛠️ Tech Stack

-   Python
    
-   PyTorch
    
-   ResNet
    
-   MobileNet V3
    
-   FastAPI
    
-   Swagger / OpenAPI
    
-   GPT-4.1-mini
