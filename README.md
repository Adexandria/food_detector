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

### API Usage
To run API locally
```bash
python app.py
```



