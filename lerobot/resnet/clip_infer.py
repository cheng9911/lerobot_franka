import torch
from PIL import Image
import os
import open_clip
from clip_train import ClipSkillPredictor, extract_features, preprocess, DEVICE, NUM_SKILLS, MAX_LEN

# 载入模型
model = ClipSkillPredictor().to(DEVICE)
model.load_state_dict(torch.load('clip_skill_model.pt', map_location=DEVICE))
model.eval()

# 加载 CLIP 模型（只要你没卸载它）
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval().to(DEVICE)

# 设置图像路径（假设在 data/images 下）
image_id = "0048"
print(f"正在处理图像 ID: {image_id}")
head_path = f"data/images/{image_id}_head.png"
flan_path = f"data/images/{image_id}_flan.png"

# 加载图像并提取特征
head_img = preprocess(Image.open(head_path).convert("RGB"))
flan_img = preprocess(Image.open(flan_path).convert("RGB"))
vision_feat = extract_features(head_img, flan_img).unsqueeze(0).to(DEVICE)  # [1, 1024]

# 预测
with torch.no_grad():
    logits = model(vision_feat)  # [1, MAX_LEN, NUM_SKILLS]
    pred = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # [MAX_LEN]

print(f"预测技能序列（图像 ID: {image_id}）: {pred}")
