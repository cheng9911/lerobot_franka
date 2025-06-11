# train.py
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
gc.collect()
torch.cuda.empty_cache()
# Config
NUM_SKILLS = 6
MAX_LEN = 5
BATCH_SIZE = 16
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval().to(DEVICE)
class SkillDataset(Dataset):
    def __init__(self, data_dir, preprocess,image_dir):
        self.data_dir = data_dir
        self.transform = preprocess
        with open(data_dir, 'r') as f:
            self.labels_dict = json.load(f)
        self.keys = list(self.labels_dict.keys())
        # self.keys = sorted(list(self.labels_dict.keys()))

        self.image_dir = image_dir
        print("\n==== JSON 标签预览（前10个） ====")
        for i, key in enumerate(self.keys[:10]):
            print(f"{key} => {self.labels_dict[key]}")
        print("================================\n")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        head_path = os.path.join(self.image_dir,  f"{key}_head.png")
        flan_path = os.path.join(self.image_dir, f"{key}_flan.png")
        # print("Sample key:", key)
        # print("Head path:", head_path)
        # print("flan_path:", flan_path)


        head_img = self.transform(Image.open(head_path).convert("RGB"))
        flan_img = self.transform(Image.open(flan_path).convert("RGB"))
        label = self.labels_dict[key]  # already a list like [1, 2, 3]

        # pad label to max_len with -1
        # label = label + [0] * (MAX_LEN - len(label))
        label = label + [5] * (MAX_LEN - len(label))

        return head_img, flan_img, torch.tensor(label, dtype=torch.long)
# class SkillDataset(Dataset):
#     def __init__(self, label_path, image_dir):
#         with open(label_path, 'r') as f:
#             self.data = json.load(f)
#         self.image_dir = image_dir

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         head = preprocess(Image.open(os.path.join(self.image_dir, item['head']))).to(DEVICE)
#         flan = preprocess(Image.open(os.path.join(self.image_dir, item['flan']))).to(DEVICE)

#         label = item['actions'] + [-1] * (MAX_LEN - len(item['actions']))
#         return head, flan, torch.tensor(label, dtype=torch.long, device=DEVICE)

class ClipSkillPredictor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_skills=NUM_SKILLS, max_len=MAX_LEN):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, dropout=0.1),
            num_layers=2,
        )
        self.skill_token = nn.Parameter(torch.randn(max_len, hidden_dim))
        self.output_head = nn.Linear(hidden_dim, num_skills)
        self.max_len = max_len

    def forward(self, vision_feat, labels=None):
        B = vision_feat.size(0)
        context = self.embedding(vision_feat).unsqueeze(0)  # [1, B, D]
        tgt = self.skill_token.unsqueeze(1).expand(self.max_len, B, -1)  # [T, B, D]
        out = self.decoder(tgt, context)  # [T, B, D]
        logits = self.output_head(out.transpose(0, 1))  # [B, T, num_skills]
        return logits
    # def forward(self, vision_feat, labels=None):
    #     B = vision_feat.size(0)
    #     context = self.embedding(vision_feat).unsqueeze(0)  # [1, B, D]
    #     tgt = self.skill_token.unsqueeze(1).expand(self.max_len, B, -1)  # [T, B, D]

    #     # Mask: [B, T] → True 表示是 padding
    #     if labels is not None:
    #         padding_mask = (labels == -1)  # [B, T]
    #     else:
    #         padding_mask = None

    #     out = self.decoder(tgt, context, tgt_key_padding_mask=padding_mask)  # [T, B, D]
    #     logits = self.output_head(out.transpose(0, 1))  # [B, T, num_skills]
    #     return logits
def extract_features_batch(heads, flans):
    with torch.no_grad():
        heads = heads.to(DEVICE)
        flans = flans.to(DEVICE)
        head_feats = clip_model.encode_image(heads)  # [B, 512]
        flan_feats = clip_model.encode_image(flans)  # [B, 512]
    return torch.cat([head_feats, flan_feats], dim=-1)  # [B, 1024]

def extract_features(head, flan):
    head = head.unsqueeze(0).to(DEVICE)  # add batch dim & move to GPU
    flan = flan.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        head_feat = clip_model.encode_image(head)
        flan_feat = clip_model.encode_image(flan)
    return torch.cat([head_feat, flan_feat], dim=-1).squeeze(0)  # [1024]
    # return torch.cat([head_feat, flan_feat], dim=-1)

# def extract_features(head, flan):
#     with torch.no_grad():
#         head_feat = clip_model.encode_image(head.unsqueeze(0))
#         flan_feat = clip_model.encode_image(flan.unsqueeze(0))
#     return torch.cat([head_feat, flan_feat], dim=-1).squeeze(0)  # [1024]

# Prepare dataset
if __name__ == "__main__":
    # Prepare
    print("Starting training...")

    train_dataset = SkillDataset('data/labels1.json', preprocess,'data/images')
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model and optimizer
    model = ClipSkillPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for head, flan, labels in dataloader:
            head, flan, labels = head.to(DEVICE), flan.to(DEVICE), labels.to(DEVICE)
            # vision_feats = torch.stack([extract_features(h, f) for h, f in zip(head, flan)])
            vision_feats = torch.stack([extract_features(h, f) for h, f in zip(head, flan)])  # 每个是 [1024]，stack 后 [B, 1024]
            logits = model(vision_feats)
            # loss = F.cross_entropy(logits.view(-1, NUM_SKILLS), labels.view(-1), ignore_index=-1)
            loss = F.cross_entropy(logits.view(-1, NUM_SKILLS), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'clip_skill_model.pt')
    # ====== 在训练集上测试模型准确率 ======
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (head, flan, labels) in enumerate(dataloader):
            head, flan, labels = head.to(DEVICE), flan.to(DEVICE), labels.to(DEVICE)
            vision_feats = torch.stack([extract_features(h, f) for h, f in zip(head, flan)])
            logits = model(vision_feats)  # [B, T, num_skills]
            preds = torch.argmax(logits, dim=-1)  # [B, T]

            for i in range(head.size(0)):
                # 获取样本 key（图像 ID）
                key = train_dataset.keys[idx * BATCH_SIZE + i]
                label_seq = labels[i].tolist()
                pred_seq = preds[i].tolist()

                print(f"Sample ID: {key}")
                print(f"Head image path: {os.path.join(train_dataset.image_dir, f'{key}_head.png')}")
                print(f"Flan image path: {os.path.join(train_dataset.image_dir, f'{key}_flan.png')}")
                print(f"True label: {label_seq}")
                print(f"Predicted:  {pred_seq}")
                print("-" * 50)

                # Token-level accuracy统计
                mask = labels[i] != -1
                correct += ((preds[i] == labels[i]) & mask).sum().item()
                total += mask.sum().item()

    print(f"\nValidation Accuracy: {correct / total:.4f}")
    

    # ============ 清理显存 ============
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    del dataloader
