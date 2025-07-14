The code is extensively borrowed from below: 
SPM : https://github.com/Con6924/SPM 
FMN : https://github.com/SHI-Labs/Forget-Me-Not 

1. Concept Neighborhood 
    Steps to Run:
        cd concept_neighborhood
        python calculate_similarity.py

    Description:
    The script generates similarity scores for each class with other classes to get adjacent concepts.
        Adjust below parameters:
            dataset_path:  this should have path of dataset dir which consists of each class name images in a manner dir_path/class_1/generated_image; dir_path/class_2/generated_images
            save_path : here the similarity scores for each class with other classes would be stored in a txt file for each class

2. Concept Fade
    Steps to Run:
        cd fade 
        python train_fade_cr.py --config_file <config_path>

    Description:
    This is the training script for the proposed FADE algortihm.
    Please refer fade/configs/blue_jay/config.yaml and fade/configs/blue_jay/prompt.yaml to define the target concept for unlearning along with retain concepts(top-k adjacent concepts used for adjacency loss)
    Also, refer the above files to set guide_strength(for guidance loss); retain_strength(for adjacency loss) and exp_strength(for erase loss)


3. Evaluation Helper Scripts 
    Steps to Run: 
        cd evaluation_helpers
        python generate_images.py <model_path> <output_path>
    
    Description:
        The above script helps to generate images from the trained model. 
        Please provide full path of the trained model in <model_path> and the path where you want to store the images as <output_path>
        Also, provide objects for which you wnat to generate images in objects_to_generate list in generate_images.py 
        To mention the total number of images to be generated with desired seeds, please adjust total and seeds_list variables in inference_generate.py

4. Classification Helper Scripts: 
    Steps to Run:
        cd classification_helpers
        python birds_all_classification.py <target_class> <dir_path>

    Description:
        The above script provides classification accuracy of the desired class images (for class of CUBS dataset for reference).
        Please provide <target_class> as the class for which you are providing the Images like "blue-jay" from Cubs dataset.
        Please provide <dir_path> as the path of the directory where the images are stored for the corresponding target class. 

