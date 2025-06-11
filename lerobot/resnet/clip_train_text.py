import os

import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open_clip
import gc

# Config
NUM_SKILLS = 6
MAX_LEN = 5
BATCH_SIZE = 16
EPOCHS = 35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval().to(DEVICE)

# Dataset class
class SkillDataset(Dataset):
    def __init__(self, json_path, image_dir, preprocess):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        entry = self.data[key]

        head_path = os.path.join(self.image_dir, f"{key}_head.png")
        flan_path = os.path.join(self.image_dir, f"{key}_flan.png")

        head_img = self.preprocess(Image.open(head_path).convert("RGB"))
        flan_img = self.preprocess(Image.open(flan_path).convert("RGB"))
        text_prompt = entry['text']
        label = entry['skill_sequence'] + [5] * (MAX_LEN - len(entry['skill_sequence']))

        return head_img, flan_img, text_prompt, torch.tensor(label, dtype=torch.long)

# Model
class ClipSkillPredictor(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512, num_skills=NUM_SKILLS, max_len=MAX_LEN):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=2
        )
        self.skill_token = nn.Parameter(torch.randn(max_len, hidden_dim))
        self.output_head = nn.Linear(hidden_dim, num_skills)

    def forward(self, fused_feat):
        B = fused_feat.size(0)
        context = self.embedding(fused_feat).unsqueeze(0)  # [1, B, D]
        tgt = self.skill_token.unsqueeze(1).expand(MAX_LEN, B, -1)  # [T, B, D]
        out = self.decoder(tgt, context)
        logits = self.output_head(out.transpose(0, 1))  # [B, T, num_skills]
        return logits

# Feature extraction
def extract_features_batch(heads, flans):
    with torch.no_grad():
        heads, flans = heads.to(DEVICE), flans.to(DEVICE)
        head_feats = clip_model.encode_image(heads)
        flan_feats = clip_model.encode_image(flans)
    return torch.cat([head_feats, flan_feats], dim=-1)  # [B, 1024]

def extract_text_features_batch(texts):
    with torch.no_grad():
        tokens = tokenizer(texts).to(DEVICE)
        text_feats = clip_model.encode_text(tokens)  # [B, 512]
    return text_feats

# Main training
if __name__ == "__main__":
    print("Starting training...")

    dataset = SkillDataset("data/labels_with_text1.json", "data/images", preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ClipSkillPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for head, flan, texts, labels in dataloader:
            head, flan, labels = head.to(DEVICE), flan.to(DEVICE), labels.to(DEVICE)

            vision_feat = extract_features_batch(head, flan)  # [B, 1024]
            text_feat = extract_text_features_batch(texts)    # [B, 512]
            fused_feat = torch.cat([vision_feat, text_feat], dim=-1)  # [B, 1536]

            logits = model(fused_feat)
            loss = F.cross_entropy(logits.view(-1, NUM_SKILLS), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'clip_skill_model.pt')

    # ===== 验证并打印预测序列 =====
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (head, flan, texts, labels) in enumerate(dataloader):
            head, flan, labels = head.to(DEVICE), flan.to(DEVICE), labels.to(DEVICE)

            vision_feat = extract_features_batch(head, flan)  # 只用图像特征
            # 不拼接文本特征，保持推理简洁
            dummy_text_feat = torch.zeros((head.size(0), 512), device=DEVICE)
            fused_feat = torch.cat([vision_feat, dummy_text_feat], dim=-1)

            logits = model(fused_feat)
            preds = torch.argmax(logits, dim=-1)

            for i in range(head.size(0)):
                key = dataset.keys[idx * BATCH_SIZE + i]
                print()
                print(f"Sample ID: {key}")
                print(f"True label: {labels[i].tolist()}")
                print(f"Predicted : {preds[i].tolist()}")
                print()
                print("-" * 50)


                mask = labels[i] != -1
                correct += ((preds[i] == labels[i]) & mask).sum().item()
                total += mask.sum().item()

    print(f"\nValidation Accuracy: {correct / total:.4f}")

    gc.collect()
    torch.cuda.empty_cache()
