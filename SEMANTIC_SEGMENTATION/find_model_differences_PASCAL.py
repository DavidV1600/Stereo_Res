from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_models():
    """Load the trained YOLOv8 models for VOC baseline and upscaled"""
    baseline_path = 'runs/detect/voc_orig_baseline6/weights/best.pt'
    sr_path = 'runs/detect/voc_upscaled_baseline3/weights/best.pt'
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline model not found at {baseline_path}")
    if not os.path.exists(sr_path):
        raise FileNotFoundError(f"SR model not found at {sr_path}")
    
    baseline_model = YOLO(baseline_path)
    sr_model = YOLO(sr_path)
    
    return baseline_model, sr_model

def find_test_images():
    """Find test images in the original and upscaled VOC datasets"""
    test_dirs = [
        './data/VOC_YOLO/images/val/',
    ]
    
    images = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in ['jpg', 'jpeg', 'png']:
                images.extend(glob(f"{test_dir}/*.{ext}"))
    
    if not images:
        raise FileNotFoundError("No test images found in expected VOC directories.")
    
    print(f"Found {len(images)} test images")
    return images

def compare_predictions(baseline_model, sr_model, images, conf_threshold=0.25):
    """Compare predictions and detect differences"""
    results_dir = './comparison_differences'
    os.makedirs(results_dir, exist_ok=True)
    
    sr_found_baseline_missed = []
    baseline_false_positives = []
    
    for i, img_path in enumerate(images):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(images)}...")
        
        baseline_results = baseline_model.predict(img_path, conf=conf_threshold)[0]
        sr_results = sr_model.predict(img_path, conf=conf_threshold)[0]
        
        baseline_boxes = baseline_results.boxes.xyxy.cpu().numpy() if baseline_results.boxes else np.array([])
        baseline_scores = baseline_results.boxes.conf.cpu().numpy() if baseline_results.boxes else np.array([])
        
        sr_boxes = sr_results.boxes.xyxy.cpu().numpy() if sr_results.boxes else np.array([])
        sr_scores = sr_results.boxes.conf.cpu().numpy() if sr_results.boxes else np.array([])
        
        if len(sr_boxes) > 0 and len(baseline_boxes) == 0:
            sr_found_baseline_missed.append((img_path, sr_boxes, sr_scores, baseline_boxes, baseline_scores))
        elif len(sr_boxes) >= len(baseline_boxes) + 2:
            sr_found_baseline_missed.append((img_path, sr_boxes, sr_scores, baseline_boxes, baseline_scores))
        if len(baseline_boxes) > 0 and len(sr_boxes) == 0:
            baseline_false_positives.append((img_path, baseline_boxes, baseline_scores, sr_boxes, sr_scores))
    
    return sr_found_baseline_missed, baseline_false_positives

def visualize_differences(sr_found_baseline_missed, baseline_false_positives, max_examples=5):
    """Save visual comparisons of model differences"""
    results_dir = './comparison_differences'
    
    print(f"\nFound {len(sr_found_baseline_missed)} examples where SR model found extra objects")
    for i, (img_path, sr_boxes, sr_scores, baseline_boxes, baseline_scores) in enumerate(sr_found_baseline_missed[:max_examples]):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"SR model found extra objects - Example {i+1}", fontsize=16)
        
        axes[0].imshow(img)
        axes[0].set_title("Baseline Model Predictions")
        for box, score in zip(baseline_boxes, baseline_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', linewidth=2, fill=False)
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-10, f"{score:.2f}", color='red', fontsize=10, backgroundcolor='white')
        
        axes[1].imshow(img)
        axes[1].set_title("SR Model Predictions")
        for box, score in zip(sr_boxes, sr_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='green', linewidth=2, fill=False)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-10, f"{score:.2f}", color='green', fontsize=10, backgroundcolor='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"sr_found_extra_{i+1}.png"))
        plt.close()

    print(f"Found {len(baseline_false_positives)} examples where baseline model may have false positives")
    for i, (img_path, baseline_boxes, baseline_scores, sr_boxes, sr_scores) in enumerate(baseline_false_positives[:max_examples]):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Potential baseline false positives - Example {i+1}", fontsize=16)
        
        axes[0].imshow(img)
        axes[0].set_title("Baseline Model Predictions")
        for box, score in zip(baseline_boxes, baseline_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', linewidth=2, fill=False)
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-10, f"{score:.2f}", color='red', fontsize=10, backgroundcolor='white')
        
        axes[1].imshow(img)
        axes[1].set_title("SR Model Predictions")
        for box, score in zip(sr_boxes, sr_scores):
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='green', linewidth=2, fill=False)
            axes[1].add_patch(rect)
            axes[1].text(x1, y1-10, f"{score:.2f}", color='green', fontsize=10, backgroundcolor='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"baseline_fp_{i+1}.png"))
        plt.close()

def main():
    print("Loading models...")
    baseline_model, sr_model = load_models()

    print("Finding test images...")
    test_images = find_test_images()

    print("Comparing predictions...")
    sr_found_extra, baseline_fp = compare_predictions(baseline_model, sr_model, test_images)

    print("Visualizing differences...")
    visualize_differences(sr_found_extra, baseline_fp)

    print("\nAll results saved in './comparison_differences/'")

if __name__ == "__main__":
    main()
