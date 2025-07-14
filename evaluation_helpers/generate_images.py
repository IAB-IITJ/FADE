import os
import sys

from omegaconf import OmegaConf
#from src.train_attn_modified_acc import main as attn_component
from inference_generate import main as test_sampling


model_id = sys.argv[1] # model path
output_dir = sys.argv[2] #output path where generated images would be stored
original_dir = None

objects_to_generate = ["blue jay","florida jay","green jay"]

print("model id :",model_id,"out dir :",output_dir)
for i,obj in enumerate(objects_to_generate):
    
    multi_concept = None
    print("Inferencing for A photo of "+obj," and saving at "+output_dir+"/"+obj)
    test_sampling(OmegaConf.create({
        "pretrained_model_name_or_path":model_id,
        "multi_concept": multi_concept
    }),output_dir+"/"+obj,csv_path="sample.csv",extra_prompt="A photo of a "+obj,extra_prompt_folder=None)

