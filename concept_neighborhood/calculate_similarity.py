from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import torch 
import os 
import numpy as np
from tqdm import tqdm

def get_features(image_path,model,processor):
    filenames = os.listdir(image_path)
    features = None
    features_prev = None

    for img_path in filenames:
        full_path = os.path.join(image_path,img_path)
        image = Image.open(full_path)
        inputs = processor(images=image, return_tensors="pt").to('cuda:3')
        image_features = model.get_image_features(**inputs)
        image_features = image_features.detach().cpu()
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        if features is None:
             features = image_features
        else:
            features = torch.cat((features,image_features),dim=0)
     
    return features.numpy(),features.mean(dim=0)


def get_similarity_scores(feature1,feature2,logit_scale=1.0):
    
    similarity = torch.nn.functional.cosine_similarity(feature1,feature2) * logit_scale

    return similarity


def get_image_features(dataset_path,save_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to('cuda:3')
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # processor = processor.to('cuda:1')
    
    omit_list = [] # here you can append some of the classes you want to omit for calculating similarity scores
    logit_scale = model.logit_scale.exp()   
    logit_scale = logit_scale.cpu()
    
    objects = os.listdir(dataset_path)

    all_features = {}
    import sys
    for so in objects:
        selected_obj = so

        with open(save_path+so+".txt","w") as f:
            sys.stdout = f 

            similarity_scores = {}

            features,feature1 = get_features(os.path.join(dataset_path,selected_obj,"generated_images"),model,processor)
            image_features = []
            image_labels = []
            label = 0
            label_to_name = {}
            for obj in tqdm(objects):
                if obj.lower() in omit_list or obj.lower().replace("_","-") in omit_list:
                     continue
                label_to_name[label] = obj
                obj_path = os.path.join(dataset_path,obj,"generated_images")
                fts,features = get_features(obj_path,model,processor)
                image_features.extend(list(fts))
                image_labels.extend([label]*len(fts))
                label += 1
                score = get_similarity_scores(feature1.unsqueeze(0),features.unsqueeze(0),logit_scale)
                similarity_scores[obj] = score.item()
                all_features[obj] = {'feats':features.unsqueeze(0),'scores':{}}
            sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
            for pair, score in sorted_scores:
                    print(f"{pair}: {score}")

            print("Extracted all features :: ",len(similarity_scores))
            sys.stdout = sys.__stdout__


dataset_path = "cubs_dataset" # this should have path of dataset dir which consists of each class name images in a manner dir_path/class_1/generated_image; dir_path/class_2/generated_images
save_path = "final_scores/cubs_dataset/" # here the similarity scores for each class with other classes would be stored in a txt file for each class

get_image_features(dataset_path,save_path)

