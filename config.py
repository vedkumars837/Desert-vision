import torch

class Config:
    # --- DATA SETTINGS ---
    DATA_DIR  = "data/"
    IMAGE_DIR = "data/images/"
    MASK_DIR  = "data/masks/"

    # --- YOUR ACTUAL CLASSES (based on mask values we found) ---
    # These numbers come directly from your mask files!
    MASK_VALUES = [200, 300, 500, 550, 800, 7100, 10000]
    
    CLASSES = {
        0: "class_200",    # Whatever 200 represents in your dataset
        1: "class_300",    # Whatever 300 represents
        2: "class_500",    # Whatever 500 represents
        3: "class_550",    # Whatever 550 represents
        4: "class_800",    # Whatever 800 represents
        5: "class_7100",   # Whatever 7100 represents
        6: "class_10000",  # Whatever 10000 represents
    }
    NUM_CLASSES = 7   # We found exactly 7 unique values!

    # --- IMAGE SETTINGS ---
    # Your images are 540x960, we'll resize to make training faster
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH  = 256

    # --- TRAINING SETTINGS ---
    BATCH_SIZE    = 4      # Keeping low since we're on CPU
    EPOCHS        = 50
    LEARNING_RATE = 0.001
    TRAIN_SPLIT   = 0.8

    # --- SAVE SETTINGS ---
    MODEL_SAVE_PATH = "best_model.pth"

    # --- HARDWARE ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"