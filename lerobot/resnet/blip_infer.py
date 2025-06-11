from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import torch.nn as nn
from blip_train import SkillPredictor, generate_text_and_skill
# 你之前定义的 SkillPredictor 类和 generate_text_and_skill 函数放在这里
# （或从你的模块 import）

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SKILL_LEN = 5
NUM_SKILLS = 6

# 加载模型
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(DEVICE)
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
skill_tokenizer = blip_processor.tokenizer
skill_predictor = SkillPredictor(len(skill_tokenizer)).to(DEVICE)

# 加载训练好的权重
# ckpt = torch.load("blip2_skill_multitask.pt", map_location=DEVICE)
ckpt = torch.load("blip2_skill_multitask.pt", map_location=DEVICE, weights_only=False)

blip_model.load_state_dict(ckpt['blip_model'])
skill_predictor.load_state_dict(ckpt['skill_predictor'])

print("✅ 模型加载成功")

# 进行推理
text, skills = generate_text_and_skill("data/images/0001_head.png", "data/images/0001_flan.png",skill_predictor)

print("生成文本：", text)
print("预测技能序列：", skills)
