# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from lerobot.resnet.dataest import SkillSequenceDataset
# dataset import SkillSequenceDataset
from model import SkillPlanner
import os

# === Config ===
data_path = "./data"
num_skills = 5
max_len = 5
batch_size = 16
epochs = 60
lr = 1e-4
save_path = "model.pt"

# === Dataset ===
dataset = SkillSequenceDataset(data_path, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkillPlanner(num_skills=num_skills, max_len=max_len).to(device)

# === Loss / Optimizer ===
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=lr)

# === Training loop ===
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        head = batch['head'].to(device)
        flan = batch['flan'].to(device)
        labels = batch['labels'].to(device)

        logits = model(head, flan, labels=labels)  # [B, T, num_skills]
        loss = criterion(logits.view(-1, num_skills), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# === Optional eval: measure accuracy ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in dataloader:
        head = batch['head'].to(device)
        flan = batch['flan'].to(device)
        labels = batch['labels'].to(device)

        preds = model(head, flan)  # [B, T]
        for pred_seq, label_seq in zip(preds, labels):
            label_seq = label_seq[label_seq != -1]
            pred_seq = pred_seq[:len(label_seq)]
            if torch.equal(pred_seq.cpu(), label_seq.cpu()):
                correct += 1
            total += 1
print(f"Eval accuracy: {correct}/{total} = {correct/total:.2%}")
