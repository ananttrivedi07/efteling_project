from glob import glob
import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import kagglehub
import shutil


def show_random_images_per_class(base_path, num_images=5):
    """
    Displays random images per class from a dataset folder.
    """
    
    base_path = os.path.join(base_path, "dataset-resized")
    classes = [cls for cls in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, cls))]
    
    for cls in classes:
        class_path = os.path.join(base_path, cls)
        
        # Only keep valid image files
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg'))]
        
        if len(images) == 0:
            print(f"No images found in class {cls}")
            continue
        
        sampled_images = random.sample(images, min(num_images, len(images)))
        
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Class: {cls}", fontsize=14)
        
        for i, img_name in enumerate(sampled_images):
            img_path = os.path.join(class_path, img_name)
            
            try:
                img = Image.open(img_path)
                
                plt.subplot(1, num_images, i + 1)
                plt.imshow(img)
                plt.axis('off')
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
        
        plt.show()

    
def run_data_processing(root_folder, show_raw_data=False):   
    """
    Get data from trashnet dataset, copy to local folder, and optionally create plots of random images per class.
    """ 
    path = kagglehub.dataset_download("feyzazkefe/trashnet")

    target_dir = os.path.join(os.getcwd(), root_folder)
    os.makedirs(target_dir, exist_ok=True)

    shutil.copytree(path, target_dir, dirs_exist_ok=True)

    print("Dataset copied to:", target_dir)
    if show_raw_data:
        show_random_images_per_class(target_dir)    