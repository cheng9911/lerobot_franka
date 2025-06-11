from PIL import Image
import requests
import torch
print(torch.__version__)
from transformers import CLIPProcessor , CLIPModel
model= CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image=Image.open("data/images/0048_head.png")
text = ['a panda toy on the table','a tiger toy  on the table','a rabbit toy on the table','a spiderman toy on the table','a doll toy on the table']
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
for i in range(len(text)):
    print(text[i], probs[0,i])