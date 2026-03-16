from fastapi import APIRouter, Depends, Body
from predict import predict

router = APIRouter(
    tags=["detect"]
)


@router.post("/detect")
async def detect_image(image: bytes = Body(..., media_type="image/jpeg")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10  # must match your fine-tuned model
    proposed_model = models.resnet18(weights=None)  # no pre-trained needed
    proposed_model.fc = nn.Linear(proposed_model.fc.in_features, num_classes)

    checkpoint_path = "models/food_classifier.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found. Ensure the file exists and the path is correct.")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    proposed_model.load_state_dict(checkpoint['model_state_dict'])
    proposed_model.eval()
    predicted_label = predict(proposed_model, image, device)
    return {"predicted_label": predicted_label}
