import torch
from PIL import Image
import open_clip
from clip_train_text import ClipSkillPredictor, extract_features_batch, preprocess, DEVICE
import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
# 模型加载
model = ClipSkillPredictor().to(DEVICE)
model.load_state_dict(torch.load('clip_skill_model.pt', map_location=DEVICE))
model.eval()

# 图像路径
image_id = "0069"
head_path = f"data/images/{image_id}_head.png"
flan_path = f"data/images/{image_id}_flan.png"

# 加载并预处理图像
head_img = preprocess(Image.open(head_path).convert("RGB"))
flan_img = preprocess(Image.open(flan_path).convert("RGB"))

# 加一个 batch 维度 → [1, 3, H, W]
head_img = head_img.unsqueeze(0).to(DEVICE)
flan_img = flan_img.unsqueeze(0).to(DEVICE)

# 提取图像特征并构造模型输入
with torch.no_grad():
    vision_feat = extract_features_batch(head_img, flan_img)  # 返回 [1, 1024]
    dummy_text_feat = torch.zeros((head_img.size(0), 512), device=DEVICE)
    fused_feat = torch.cat([vision_feat, dummy_text_feat], dim=-1)  # [1, 1536]

    logits = model(fused_feat)  # 输出 [1, MAX_LEN, NUM_SKILLS]
    pred = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # [MAX_LEN]

print(f"图像 ID: {image_id} → 预测技能序列: {pred}")
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()