import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score
import os

# ✅ Local imports (no 'train.' prefix)
from dataset import RetinopathyDataset
from augmentations import get_train_transforms, get_val_transforms

# File paths
train_csv = "data/raw/train.csv"
image_dir = "data/raw/train_images"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoaders
full_dataset = RetinopathyDataset(train_csv, image_dir, transform=get_train_transforms())
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Setup
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_f1 = 0

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.numpy())

    val_f1 = f1_score(val_targets, val_preds, average='macro')
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"✅ Saved new best model with F1: {val_f1:.4f}")

print("✅ Training complete.")
