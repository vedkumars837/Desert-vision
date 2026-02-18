# predict.py
# Use this AFTER training to see your model's predictions visually

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from model import UNet

# Color map for visualization (one color per class)
CLASS_COLORS = {
    0: (0,   0,   0),    # background = black
    1: (0,   255, 0),    # vegetation = green
    2: (128, 128, 128),  # rock = gray
    3: (0,   0,   255),  # sky = blue
    4: (255, 165, 0),    # ground = orange
    5: (255, 255, 0),    # vehicle_path = yellow
}

def predict_single_image(image_path, model, device):
    """Run the model on one image and return the prediction"""
    
    # Load and preprocess image
    image = np.array(Image.open(image_path).convert("RGB"))
    original_image = image.copy()
    
    transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return original_image, prediction


def colorize_mask(mask):
    """Convert a mask (numbers) to a colorful RGB image"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


# ---- MAIN ----
if __name__ == "__main__":
    # Load trained model
    print("Loading trained model...")
    model = UNet(in_channels=3, num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    print("✅ Model loaded!")
    
    # Test on an image (change this path to any image you want to test)
    test_image_path = "dataset/images/image_001.png"
    
    original, prediction = predict_single_image(test_image_path, model, Config.DEVICE)
    color_pred = colorize_mask(prediction)
    
    # Show results side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(color_pred)
    axes[1].set_title("Model Prediction")
    axes[1].axis("off")
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=np.array(c)/255, label=name)
                      for (cls_id, c), (_, name) in 
                      zip(CLASS_COLORS.items(), Config.CLASSES.items())]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()
    print("✅ Prediction saved as prediction_result.png")

