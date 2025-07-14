import os
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import copy

def main(args,output_folder_root,csv_path,extra_prompt,extra_prompt_folder):

    prompts = []       

    if True or args.multi_concept is not None:
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        model_id = "CompVis/stable-diffusion-v1-4" 
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        

         
        pipe.load_lora_weights(args.pretrained_model_name_or_path)
        lora_unet = copy.deepcopy(pipe.unet)
        pipe.unet = lora_unet
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)       
        print("loaded model weights!!!!!")
        
        def dummy(images, **kwargs):
            return images, False
        pipe.safety_checker = dummy
        pipe.enable_vae_slicing()


    extra_folder = output_folder_root+"/eval_generated_images"
    os.makedirs(extra_folder, exist_ok=True)

    seed_lists = [1,2,3,4,5] 
    total = 1 

    print("will be saved at :",extra_folder)
    if extra_prompt is not None:
        print(f'Inferencing: {extra_prompt}')
        cf = 0
        for itr in range(total):
            torch.manual_seed(seed_lists[itr])
            images = pipe(extra_prompt, num_inference_steps=50, guidance_scale=8, num_images_per_prompt=25).images

            for i, im in enumerate(images):
                cf = cf +1
                im.save(f"{extra_folder}/o_{extra_prompt.replace(' ', '-')}_{cf}.jpg")  
