import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

class OffroadDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.transform   = transform
        self.mask_values = Config.MASK_VALUES  # [200, 300, 500, 550, 800, 7100, 10000]
        
        self.images = sorted(os.listdir(image_dir))
        print(f"✅ Found {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def convert_mask(self, mask):
        """
        This is the KEY function!
        Converts big numbers → simple class numbers
        
        200   → 0
        300   → 1
        500   → 2
        550   → 3
        800   → 4
        7100  → 5
        10000 → 6
        """
        new_mask = np.zeros_like(mask, dtype=np.int64)
        for class_idx, val in enumerate(self.mask_values):
            new_mask[mask == val] = class_idx
        return new_mask
    
    def __getitem__(self, index):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[index])
        image    = np.array(Image.open(img_path).convert("RGB"))
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.images[index])
        mask      = np.array(Image.open(mask_path))
        
        # Convert mask values to class indices!
        mask = self.convert_mask(mask)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]
        
        return image, mask.long()


def get_transforms(image_height, image_width, is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(image_height, image_width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_height, image_width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])