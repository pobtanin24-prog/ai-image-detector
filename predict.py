import sys
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# -------------------------
# Load model
# -------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("ai_detector.pth", map_location="cpu")
)

model.eval()

# -------------------------
# Image preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# Load image
# -------------------------

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# -------------------------
# Predict
# -------------------------

with torch.no_grad():
    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, 1)

classes = ["FAKE", "REAL"]

print("Prediction:", classes[prediction.item()])
print(f"Confidence: {confidence.item() * 100:.2f}%")