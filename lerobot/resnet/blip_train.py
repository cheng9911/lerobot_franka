from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # 减小batch size，避免显存爆炸
EPOCHS = 20
LR = 1e-4

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(DEVICE)
# 先冻结所有参数
# 冻结所有参数
for param in blip_model.parameters():
    param.requires_grad = False

# 解冻视觉编码器
for param in blip_model.vision_model.parameters():
    param.requires_grad = True

# 解冻语言模型最后一层 decoder block
# for param in blip_model.language_model.model.decoder.layers[-1].parameters():
#     param.requires_grad = True
# 例如解冻解码器最后3层
num_layers_to_unfreeze = 3
decoder_layers = blip_model.language_model.model.decoder.layers
for i in range(-num_layers_to_unfreeze, 0):
    for param in decoder_layers[i].parameters():
        param.requires_grad = True

# 解冻语言模型输出头
for param in blip_model.language_model.lm_head.parameters():
    param.requires_grad = True
class MultiModalDataset(Dataset):
    def __init__(self, json_path, image_dir, processor):
        import json
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        entry = self.data[key]
        head_img = Image.open(os.path.join(self.image_dir, f"{key}_head.png")).convert("RGB")
        flan_img = Image.open(os.path.join(self.image_dir, f"{key}_flan.png")).convert("RGB")
        combined_img = Image.new("RGB", (head_img.width + flan_img.width, head_img.height))
        combined_img.paste(head_img, (0, 0))
        combined_img.paste(flan_img, (head_img.width, 0))
        text = entry['text']
        return {
            "image": combined_img,
            "text": text
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]

    pixel_values = blip_processor(images=images, return_tensors="pt").pixel_values

    inputs = blip_processor.tokenizer(texts, padding=True, return_tensors="pt")

    labels = blip_processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids
    labels[labels == blip_processor.tokenizer.pad_token_id] = -100

    batch = {
        "pixel_values": pixel_values,
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels
    }
    return batch


def train():
    dataset = MultiModalDataset("data/labels_with_text1.json", "data/images", blip_processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # optimizer = torch.optim.Adam(blip_model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, blip_model.parameters()),
    lr=LR
)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度

    for epoch in range(EPOCHS):
        blip_model.train()
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # 混合精度上下文
                outputs = blip_model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} loss: {total_loss/len(dataloader):.4f}")
        torch.cuda.empty_cache()  # 清理显存碎片

def test(head_img_path, flan_img_path):
    blip_model.eval()
    head_img = Image.open(head_img_path).convert("RGB")
    flan_img = Image.open(flan_img_path).convert("RGB")
    combined_img = Image.new("RGB", (head_img.width + flan_img.width, head_img.height))
    combined_img.paste(head_img, (0, 0))
    combined_img.paste(flan_img, (head_img.width, 0))

    inputs = blip_processor(images=combined_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = blip_model.generate(
    **inputs,
    max_new_tokens=60,
    no_repeat_ngram_size=2,
    num_beams=5,
    early_stopping=True
)

        generated_text = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    # print("Pretrained BLIP-2 output:", generated_text)
    return generated_text


if __name__ == "__main__":
    train()

    example_head = "data/images/0001_head.png"
    example_flan = "data/images/0001_flan.png"
    output_text = test(example_head, example_flan)
    print("Generated text:", output_text)
