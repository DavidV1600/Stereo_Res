import os
import json
import shutil
from PIL import Image
import random
from pathlib import Path
import numpy as np

def prepare_tinyperson_for_yolo(dataset_path, output_path):
    """
    Prepare TinyPerson dataset for YOLO training
    
    Args:
        dataset_path: Root path containing the TinyPerson dataset
        output_path: Where to save the YOLO-formatted dataset
    """
    # Create necessary directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'test'), exist_ok=True)
    
    # Load JSON files
    train_json_path = os.path.join(dataset_path, 'tiny_set_train.json')
    test_json_path = os.path.join(dataset_path, 'tiny_set_test.json')
    
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Process train data
    process_split(train_data, dataset_path, output_path, is_train=True)
    
    # Process test data
    process_split(test_data, dataset_path, output_path, is_train=False)
    
    # Create dataset.yaml
    create_dataset_yaml(output_path)
    
    print(f"Dataset preparation complete at {output_path}")


def process_split(data, dataset_path, output_path, is_train=True):
    """Process a specific split (train/test)"""
    # Extract images and annotations
    images = data['images']
    annotations = data['annotations']
    
    # Create mapping from image_id to filename
    image_map = {img['id']: img['file_name'] for img in images}
    
    # Group annotations by image_id
    anno_by_image = {}
    for anno in annotations:
        img_id = anno['image_id']
        if img_id not in anno_by_image:
            anno_by_image[img_id] = []
        anno_by_image[img_id].append(anno)
    
    # Determine which split we're processing
    split_name = 'train' if is_train else 'test'
    print(f"Processing {split_name} split with {len(images)} images and {len(annotations)} annotations")
    
    # For training set, create a train/val split
    if is_train:
        # Get all image IDs
        all_image_ids = list(image_map.keys())
        
        # Randomly select 20% for validation
        random.seed(42)  # For reproducibility
        val_ids = set(random.sample(all_image_ids, int(len(all_image_ids) * 0.2)))
        train_ids = set(all_image_ids) - val_ids
        
        print(f"Split train set into {len(train_ids)} training and {len(val_ids)} validation images")
        
        # Process training set
        train_success = 0
        for img_id in train_ids:
            success = process_image(
                img_id, 
                image_map, 
                anno_by_image, 
                dataset_path, 
                os.path.join(output_path, 'images', 'train'),
                os.path.join(output_path, 'labels', 'train')
            )
            if success:
                train_success += 1
        
        # Process validation set
        val_success = 0
        for img_id in val_ids:
            success = process_image(
                img_id, 
                image_map, 
                anno_by_image, 
                dataset_path, 
                os.path.join(output_path, 'images', 'val'),
                os.path.join(output_path, 'labels', 'val')
            )
            if success:
                val_success += 1
        
        print(f"Successfully processed {train_success}/{len(train_ids)} training and {val_success}/{len(val_ids)} validation images")
        
        # If validation is empty, create some from training
        if val_success == 0 and train_success > 0:
            print("Warning: No validation images were processed successfully. Creating validation set from training images...")
            
            # Get list of successfully processed training images
            train_images_dir = os.path.join(output_path, 'images', 'train')
            train_labels_dir = os.path.join(output_path, 'labels', 'train')
            
            processed_train_images = os.listdir(train_images_dir)
            
            if processed_train_images:
                # Select 20% for validation
                val_count = max(1, int(len(processed_train_images) * 0.2))
                val_images = random.sample(processed_train_images, val_count)
                
                # Move these to validation folder
                for img_name in val_images:
                    # Move image
                    shutil.move(
                        os.path.join(train_images_dir, img_name),
                        os.path.join(output_path, 'images', 'val', img_name)
                    )
                    
                    # Move corresponding label
                    label_name = Path(img_name).stem + '.txt'
                    label_path = os.path.join(train_labels_dir, label_name)
                    if os.path.exists(label_path):
                        shutil.move(
                            label_path,
                            os.path.join(output_path, 'labels', 'val', label_name)
                        )
                
                print(f"Created validation set with {val_count} images from training set")
    else:
        # Process test set
        test_success = 0
        for img_id in image_map.keys():
            success = process_image(
                img_id, 
                image_map, 
                anno_by_image, 
                dataset_path, 
                os.path.join(output_path, 'images', 'test'),
                os.path.join(output_path, 'labels', 'test')
            )
            if success:
                test_success += 1
        
        print(f"Successfully processed {test_success}/{len(image_map)} test images")


def process_image(img_id, image_map, anno_by_image, dataset_path, img_output_dir, label_output_dir):
    """Process a single image and its annotations"""
    # Get filename
    filename = image_map[img_id]
    
    # Check if this is from the train or test split
    split_folder = 'train' if 'train' in img_output_dir else 'test'
    
    # Handle the case where filename might already contain subfolder information
    # e.g., "labeled_images/youtube_V0003_I0000480.jpg"
    img_path = None
    if '/' in filename:
        subfolder, actual_filename = filename.split('/', 1)
        potential_path = os.path.join(dataset_path, split_folder, subfolder, actual_filename)
        if os.path.exists(potential_path):
            img_path = potential_path
    else:
        # Try different potential folders if the filename doesn't include subfolder
        potential_folders = ['labeled_dense_images', 'labeled_images', 'no_label_images', 'pure_bg_images']
        
        for folder in potential_folders:
            temp_path = os.path.join(dataset_path, split_folder, folder, filename)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
    
    # If we still haven't found the image, try one more approach - in case the image is directly in the split folder
    if img_path is None:
        direct_path = os.path.join(dataset_path, split_folder, filename)
        if os.path.exists(direct_path):
            img_path = direct_path
    
    # If we still haven't found the image, report and return
    if img_path is None or not os.path.exists(img_path):
        print(f"Warning: Could not find image {filename} in {split_folder} folder structure")
        return False
    
    # Make sure the actual_filename is defined for output
    if '/' in filename:
        _, actual_filename = filename.split('/', 1)
    else:
        actual_filename = filename
    
    # Copy the image - use just the actual filename without subfolder for output
    shutil.copy(img_path, os.path.join(img_output_dir, actual_filename))
    
    # Create YOLO-format label
    if img_id in anno_by_image:
        create_yolo_label(
            img_path,
            anno_by_image[img_id],
            os.path.join(label_output_dir, Path(actual_filename).stem + '.txt')
        )
    else:
        # If no annotations but we still want to include this image, create an empty label file
        with open(os.path.join(label_output_dir, Path(actual_filename).stem + '.txt'), 'w') as f:
            pass  # Create empty file
    
    return True


def create_yolo_label(img_path, annotations, label_path):
    """Convert COCO annotations to YOLO format"""
    try:
        # Get image dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        with open(label_path, 'w') as f:
            for anno in annotations:
                # Skip annotations marked as ignore
                if anno.get('ignore', False):
                    continue
                
                # Get category_id and bbox
                category_id = anno['category_id']
                bbox = anno['bbox']  # [x, y, width, height]
                
                # Skip invalid boxes
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue
                
                # Convert to YOLO format
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # Ensure values are in valid range [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # For YOLO, person is class 0
                class_id = 0  # Map category_id 1 to YOLO class 0
                
                # Write to file
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")


def create_dataset_yaml(output_path):
    """Create YAML configuration file for YOLO"""
    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_path)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        f.write("  0: person\n")


if __name__ == "__main__":
    prepare_tinyperson_for_yolo(
        dataset_path="./data2",  # Update with your actual path
        output_path="./data2/output"  # Update with your desired output path
    )
