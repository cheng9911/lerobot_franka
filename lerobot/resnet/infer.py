# infer.py
import torch
from model import SkillPlanner
from PIL import Image
import torchvision.transforms as T

# === Config ===
num_skills = 5
max_len = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "clip_skill_model.pt"

# === Load model ===
model = SkillPlanner(num_skills=num_skills, max_len=max_len).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Preprocess image ===
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, 224, 224]

def infer(head_path, flan_path):
    head = load_image(head_path).to(device)
    flan = load_image(flan_path).to(device)

    with torch.no_grad():
        pred_seq = model(head, flan)  # [1, T]
    return pred_seq.squeeze(0).tolist()

# === Example ===
if __name__ == "__main__":
    head_path = "data/images/0005_head.png"
    flan_path = "data/images/0005_flan.png"
    pred = infer(head_path, flan_path)
    print("Predicted skill sequence:", pred)
