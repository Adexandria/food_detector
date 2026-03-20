# Getting Started
This simulation demonstrates how the robot identifies a person holding food and processes the image using a food classifier. The classifier relies on a pre-trained ResNet model available on [Kaggle](https://www.kaggle.com/code/adeolawuraolaade/food-inference).

### 1. Set up the virtual environment
First, create the virtual environment:
```bash
python -m venv .venv
```
### 2. Install dependencies
```bash
pip3 install torch torchvision
pip install -r requirements.txt
```

### Run Pipeline
```bash
python detection_pipeline.py --use_resnet
```

## API Usage

The API currently supports two models: **ResNet** and **MobileNet**.  
Pass the model name as a parameter to select which one to use.

### Available Models

-   **ResNet** → `resnet`
    
-   **MobileNet V3 Large** → `mobilenet`
    

----------

## Run API Locally

python app.py

----------

## Use Hosted API

### Using `curl`

curl  -X POST \  
  "https://papri-ka-food-detector.hf.space/detect?name=mobilenet" \  
  -H  "accept: application/json" \  
  -H  "Content-Type: multipart/form-data" \  
  -F  "image=@160194.jpg;type=image/jpeg"

----------

### Using Swagger

-   **Request URL:**  
    `https://papri-ka-food-detector.hf.space/detect?name=mobilenet`
    
-   **Parameter:**  
    Choose either `resnet` or `mobilenet`
    
-   **Image:**  
    Upload the image to detect
    

----------

## Example Response
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
