import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torchvision.transforms as T

class SkillSequenceDataset(Dataset):
    def __init__(self, data_dir, max_len=5):
        self.image_dir = os.path.join(data_dir, "images")
        with open(os.path.join(data_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)
        self.sample_ids = list(self.labels.keys())
        self.max_len = max_len
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        head_img = Image.open(os.path.join(self.image_dir, f"{sample_id}_head.png")).convert("RGB")
        flan_img = Image.open(os.path.join(self.image_dir, f"{sample_id}_flan.png")).convert("RGB")

        head_tensor = self.transform(head_img)
        flan_tensor = self.transform(flan_img)
        skill_seq = self.labels[sample_id]

        # Padding
        padded_seq = skill_seq + [-1] * (self.max_len - len(skill_seq))
        attention_mask = [1]*len(skill_seq) + [0]*(self.max_len - len(skill_seq))

        return {
            "head": head_tensor,
            "flan": flan_tensor,
            "labels": torch.tensor(padded_seq),
            "mask": torch.tensor(attention_mask),
        }
