import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import sys
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import sys
import os
import warnings
from pathlib import Path
from typing import Optional
from PIL import Image
import pickle 

new_list = []
with open("./final_models/cubs_daataset/class_names_cubs.pickle","rb") as fn:
    new_list = pickle.load(fn)
print(new_list)

new_list = []
with open("final_models/cubs_dataset/class_names_cubs.pickle","rb") as fn:
    new_list = pickle.load(fn)

class_labels = {}

for i,val in enumerate(new_list):
    class_labels[i] = val.lower().replace("'","").replace(" ","-")

print(class_labels)

device="cuda"
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(new_list))
model.load_state_dict(torch.load('./final_models/cubs_dataset/cubs_model_finetuned.pth'))
# Move model to appropriate device

model = model.to(device)
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return input_tensor


def predict_apple(image_path):
    input_tensor = preprocess_image(image_path)

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor.cuda())

    _, predicted_idx = torch.max(output, 1)


    predicted_label = class_labels.get(predicted_idx.detach().cpu().item(), 'Unknown')

    return predicted_label


target_concept = sys.argv[1]
target_path = sys.argv[2]


p = Path(target_path,"eval_generated_images")

image_paths = list(p.iterdir())

from torchvision import transforms
convert_tensor = transforms.ToTensor()

img_paths = []
for path_name in image_paths:
    pn = str(path_name).split("/")[-1]
    img_path = Image.open(path_name)
    img_paths.append(path_name)

# Sample image path
correct = 0
total = 0

print(img_paths[:4])
for img_path in img_paths:
    # Predict if the image is of an apple or not
    predicted_label = predict_apple(img_path)

    print(f'Predicted Label: {predicted_label}, actual class: {target_concept}')

    if predicted_label==target_concept:
        correct+=1
    total+=1
    
print("img path:",img_path)
print("total :",total)

print("for target :",target_concept)
print("final acc :",correct/total)
print("#"*100)
