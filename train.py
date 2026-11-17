import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------------
# 1. Settings
# -------------------------

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -------------------------
# 2. Prepare images
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    "data/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "data/test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))


# -------------------------
# 3. Load pretrained ResNet18
# -------------------------

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final classifier
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)


# -------------------------
# 4. Training setup
# -------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# 5. Train
# -------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )


# -------------------------
# 6. Test
# -------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")


# -------------------------
# 7. Save model
# -------------------------

torch.save(
    model.state_dict(),
    "ai_detector.pth"
)

print("Model saved as ai_detector.pth")