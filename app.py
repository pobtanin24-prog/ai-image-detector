import torch
import gradio as gr

from torch import nn
from torchvision import models, transforms
from PIL import Image


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = models.resnet18(weights=None)

# 2 output classes: FAKE / REAL
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load(
        "ai_detector.pth",
        map_location="cpu"
    )
)

model.eval()


# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# IMPORTANT:
# ImageFolder sorted our training classes as:
# ['FAKE', 'REAL']

classes = ["AI Generated", "Real"]


# ==========================================
# 3. PREDICTION
# ==========================================

def predict(image):

    if image is None:
        return {}

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    return {
        "AI Generated": float(probabilities[0]),
        "Real": float(probabilities[1])
    }


# ==========================================
# 4. GRADIO WEB APP
# ==========================================

demo = gr.Interface(

    fn=predict,

    inputs=gr.Image(
        type="pil",
        label="Upload Image"
    ),

    outputs=gr.Label(
        num_top_classes=2,
        label="Prediction"
    ),

    title="AI-Generated Image Detector",

    description=(
        "Upload an image to estimate whether it is "
        "AI-generated or real."
    ),

    flagging_mode="never"
)


# ==========================================
# 5. START APP
# ==========================================

if __name__ == "__main__":
    demo.launch(share=True)