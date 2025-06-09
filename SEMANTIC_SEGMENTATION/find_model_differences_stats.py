from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import xml.etree.ElementTree as ET
import json

def load_models():
    """Load the trained models"""
    # Load baseline model
    baseline_path = 'runs/detect/tinyperson_baseline16/weights/best.pt'
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline model not found at {baseline_path}. Run train_yoloV8.py first.")
    baseline_model = YOLO(baseline_path)
    
    # Load SR model
    sr_path = 'runs/detect/tinyperson_sr_10017/weights/best.pt'
    if not os.path.exists(sr_path):
        raise FileNotFoundError(f"SR model not found at {sr_path}. Run train_yoloV8_upscaled.py first.")
    sr_model = YOLO(sr_path)
    
    return baseline_model, sr_model

def find_test_images():
    """Find test images in the dataset directories"""
    # Try both original and SR dataset test directories
    test_dirs = [
        './data2/output/images/test/',
        './data2/output_sr_4x/images/test/'
    ]
    
    images = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in ['jpg', 'jpeg', 'png']:
                images.extend(glob(f"{test_dir}/*.{ext}"))
    
    if not images:
        raise FileNotFoundError("No test images found. Please check your dataset paths.")
    
    print(f"Found {len(images)} test images")
    return images

def compare_predictions(baseline_model, sr_model, images, conf_threshold=0.25):
    """Find cases where models differ significantly in their predictions"""
    results_dir = './comparison_differences'
    os.makedirs(results_dir, exist_ok=True)
    
    sr_found_baseline_missed = []
    baseline_false_positives = []
    
    for i, img_path in enumerate(images):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(images)}...")
        
        # Get predictions from both models
        baseline_results = baseline_model.predict(img_path, conf=conf_threshold)[0]
        sr_results = sr_model.predict(img_path, conf=conf_threshold)[0]
        
        # Extract boxes, scores and class IDs
        baseline_boxes = baseline_results.boxes.xyxy.cpu().numpy() if len(baseline_results.boxes) > 0 else np.array([])
        baseline_scores = baseline_results.boxes.conf.cpu().numpy() if len(baseline_results.boxes) > 0 else np.array([])
        
        sr_boxes = sr_results.boxes.xyxy.cpu().numpy() if len(sr_results.boxes) > 0 else np.array([])
        sr_scores = sr_results.boxes.conf.cpu().numpy() if len(sr_results.boxes) > 0 else np.array([])
        
        # Case 1: SR model found objects but baseline missed them
        if len(sr_boxes) > 0 and len(baseline_boxes) == 0:
            sr_found_baseline_missed.append((img_path, sr_boxes, sr_scores, baseline_boxes, baseline_scores))
        
        # Case 2: SR model found more objects than baseline (at least 2 more)
        elif len(sr_boxes) >= len(baseline_boxes) + 2:
            sr_found_baseline_missed.append((img_path, sr_boxes, sr_scores, baseline_boxes, baseline_scores))
        
        # Case 3: Baseline model found objects but SR model thinks they aren't there
        # This might indicate false positives in the baseline model
        if len(baseline_boxes) > 0 and len(sr_boxes) == 0:
            baseline_false_positives.append((img_path, baseline_boxes, baseline_scores, sr_boxes, sr_scores))
    
    return sr_found_baseline_missed, baseline_false_positives

def visualize_differences(sr_found_baseline_missed, baseline_false_positives, max_examples=5):
    """Create visualization of the differences between model predictions"""
    results_dir = './comparison_differences'
    
    # Visualize SR model finding people that baseline missed
    print(f"\nFound {len(sr_found_baseline_missed)} examples where SR model found people that baseline missed")
    for i, (img_path, sr_boxes, sr_scores, baseline_boxes, baseline_scores) in enumerate(sr_found_baseline_missed[:max_examples]):
        if i >= max_examples:
            break
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"SR model found people that baseline missed - Example {i+1}", fontsize=16)
        
        # Show original image with baseline predictions
        axes[0].imshow(img)
        axes[0].set_title("Baseline Model Predictions")
        for box, score in zip(baseline_boxes, baseline_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-10, f"{score:.2f}", color='red', fontsize=10, backgroundcolor='white')
        
        # Show original image with SR model predictions
        axes[1].imshow(img)
        axes[1].set_title("SR Model Predictions")
        for box, score in zip(sr_boxes, sr_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-10, f"{score:.2f}", color='green', fontsize=10, backgroundcolor='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"sr_found_baseline_missed_{i+1}.png"))
        plt.close()
    
    # Visualize baseline false positives
    print(f"Found {len(baseline_false_positives)} examples where baseline model may have false positives")
    for i, (img_path, baseline_boxes, baseline_scores, sr_boxes, sr_scores) in enumerate(baseline_false_positives[:max_examples]):
        if i >= max_examples:
            break
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Potential baseline false positives - Example {i+1}", fontsize=16)
        
        # Show original image with baseline predictions
        axes[0].imshow(img)
        axes[0].set_title("Baseline Model Predictions")
        for box, score in zip(baseline_boxes, baseline_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-10, f"{score:.2f}", color='red', fontsize=10, backgroundcolor='white')
        
        # Show original image with SR model predictions
        axes[1].imshow(img)
        axes[1].set_title("SR Model Predictions")
        for box, score in zip(sr_boxes, sr_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-10, f"{score:.2f}", color='green', fontsize=10, backgroundcolor='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"baseline_false_positives_{i+1}.png"))
        plt.close()

def parse_voc_boxes(xml_path):
    """Parse VOC XML annotation and return list of boxes: [xmin, ymin, xmax, ymax]"""
    boxes = []
    if not os.path.exists(xml_path):
        return boxes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'person':
            continue
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def iou(boxA, boxB):
    # Compute intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def analyze_size_bins(baseline_model, sr_model, test_images, annotation_dir, size_bins=[0, 32, 64, 128, 256, 512, 99999]):
    """Analyze recall for different object size bins"""
    object_stats = []  # [area, detected_baseline, detected_sr]
    for img_path in test_images:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(annotation_dir, img_name + '.xml')
        gt_boxes = parse_voc_boxes(xml_path)
        if not gt_boxes:
            continue
        baseline_results = baseline_model.predict(img_path, conf=0.25)[0]
        sr_results = sr_model.predict(img_path, conf=0.25)[0]
        baseline_boxes = baseline_results.boxes.xyxy.cpu().numpy() if len(baseline_results.boxes) > 0 else []
        sr_boxes = sr_results.boxes.xyxy.cpu().numpy() if len(sr_results.boxes) > 0 else []
        for gt_box in gt_boxes:
            w = gt_box[2] - gt_box[0]
            h = gt_box[3] - gt_box[1]
            area = w * h
            detected_baseline = any(iou(gt_box, pred_box) > 0.5 for pred_box in baseline_boxes)
            detected_sr = any(iou(gt_box, pred_box) > 0.5 for pred_box in sr_boxes)
            object_stats.append([area, detected_baseline, detected_sr])
    object_stats = np.array(object_stats)
    if object_stats.size == 0:
        print("No ground truth objects found for any test image. Check your annotation_dir and image/annotation matching.")
        return
    print("\nRecall by object size bin:")
    for i in range(len(size_bins)-1):
        bin_mask = (object_stats[:,0] >= size_bins[i]) & (object_stats[:,0] < size_bins[i+1])
        bin_objects = object_stats[bin_mask]
        if len(bin_objects) == 0:
            continue
        recall_baseline = bin_objects[:,1].mean()
        recall_sr = bin_objects[:,2].mean()
        print(f"Size {size_bins[i]}-{size_bins[i+1]}: Baseline recall={recall_baseline:.2f}, SR recall={recall_sr:.2f}, N={len(bin_objects)}")

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Build a mapping from image_id to file_name
    imgid_to_name = {img['id']: img['file_name'] for img in data['images']}
    # Build a mapping from file_name to list of boxes
    name_to_boxes = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        file_name = imgid_to_name[img_id]
        box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        name_to_boxes.setdefault(file_name, []).append(box)
    return name_to_boxes

def main():
    # Create results directory
    results_dir = './comparison_differences'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    baseline_model, sr_model = load_models()
    
    # Find test images
    test_images = find_test_images()
    
    # Compare predictions
    print("\nComparing model predictions...")
    sr_found_baseline_missed, baseline_false_positives = compare_predictions(
        baseline_model, sr_model, test_images
    )
    
    # Visualize differences
    print("\nVisualizing differences...")
    visualize_differences(sr_found_baseline_missed, baseline_false_positives)
    
    # === NEW: Analyze recall by object size bins ===
    annotation_dir = './data2/output/Annotations/test'  # <-- update as needed
    analyze_size_bins(baseline_model, sr_model, test_images, annotation_dir)
    
    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    main() 