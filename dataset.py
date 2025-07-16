from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['id_code'] + ".png")
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        label = int(row['diagnosis'])

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
