# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from dataset import OffroadDataset, get_transforms
from model import UNet
from iou_metrics import calculate_iou, print_iou_results

# =============================================
# SETUP
# =============================================
print(f"Using device: {Config.DEVICE}")
print(f"(GPU = fast, CPU = slow but works)")

# =============================================
# LOAD DATA
# =============================================
print("\nLoading dataset...")

train_transform = get_transforms(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, is_training=True)
val_transform   = get_transforms(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, is_training=False)

full_dataset = OffroadDataset(
    image_dir=Config.IMAGE_DIR,
    mask_dir=Config.MASK_DIR,
    transform=train_transform
)

total_size = len(full_dataset)
train_size = int(total_size * Config.TRAIN_SPLIT)
val_size   = total_size - train_size

print(f"Total images:      {total_size}")
print(f"Training images:   {train_size}")
print(f"Validation images: {val_size}")

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# num_workers=0 is IMPORTANT for Windows — fixes the multiprocessing error!
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=0    # <-- THIS fixes your error!
)
val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=0    # <-- THIS fixes your error!
)

# =============================================
# BUILD MODEL
# =============================================
print("\nBuilding U-Net model...")
model = UNet(in_channels=3, num_classes=Config.NUM_CLASSES)
model = model.to(Config.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters!")

# =============================================
# LOSS FUNCTION AND OPTIMIZER
# =============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# =============================================
# TRAINING LOOP
# =============================================
print(f"\nStarting training for {Config.EPOCHS} epochs...")

best_val_iou = 0.0
train_losses = []
val_ious     = []

for epoch in range(1, Config.EPOCHS + 1):

    # ---- TRAINING PHASE ----
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")

    for images, masks in loop:
        images = images.to(Config.DEVICE)
        masks  = masks.to(Config.DEVICE)

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(predictions, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- VALIDATION PHASE ----
    model.eval()
    all_ious = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Val]  "):
            images = images.to(Config.DEVICE)
            masks  = masks.to(Config.DEVICE)

            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1)

            _, batch_iou = calculate_iou(
                preds.cpu(), masks.cpu(), Config.NUM_CLASSES
            )
            all_ious.append(batch_iou)

    avg_val_iou = np.mean(all_ious)
    val_ious.append(avg_val_iou)

    scheduler.step(avg_train_loss)

    print(f"\nEpoch {epoch:3d}/{Config.EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val mIoU: {avg_val_iou:.4f}")

    # Save best model
    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print(f"   Best model saved! (mIoU: {best_val_iou:.4f})")

# =============================================
# SAVE TRAINING GRAPHS
# =============================================
print("\nSaving training graphs...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, color="red", label="Train Loss")
ax1.set_title("Training Loss Over Time")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(val_ious, color="blue", label="Val mIoU")
ax2.set_title("Validation IoU Over Time")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("mIoU")
ax2.legend()

plt.tight_layout()
plt.savefig("training_history.png")
print("Saved as training_history.png")

print(f"\nTraining complete!")
print(f"Best mIoU achieved: {best_val_iou:.4f}")
print(f"Model saved to: {Config.MODEL_SAVE_PATH}")