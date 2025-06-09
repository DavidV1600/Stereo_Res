from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import numpy as np

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

def fix_yaml_paths():
    """Ensure the dataset.yaml files point to the correct directories"""
    # Fix SR dataset.yaml
    sr_yaml_path = './data2/output_sr_4x/dataset.yaml'
    with open(sr_yaml_path, 'r') as f:
        yaml_content = f.read()
    
    # Make sure it points to the SR data
    sr_path = os.path.abspath('./data2/output_sr_4x')
    print(f"Checking SR YAML path: {sr_yaml_path}")
    print(f"Current content: {yaml_content}")
    
    if "output_sr_4x" not in yaml_content:
        print("Fixing SR YAML path...")
        yaml_content = yaml_content.replace('path:', f'path: {sr_path}')
        with open(sr_yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Fixed {sr_yaml_path} to point to {sr_path}")

def evaluate_models(baseline_model, sr_model):
    """Evaluate both models on original and upscaled data"""
    results = {}
    
    # First, fix the dataset.yaml files to make sure they point to the right locations
    fix_yaml_paths()
    
    # Evaluate baseline model on original data
    print("Evaluating baseline model on original data...")
    results['baseline_on_original'] = baseline_model.val(data='./data2/output/dataset.yaml')
    
    # Evaluate SR model on original data
    print("Evaluating SR model on original data...")
    results['sr_on_original'] = sr_model.val(data='./data2/output/dataset.yaml')
    
    # Evaluate baseline model on SR data
    print("Evaluating baseline model on upscaled data...")
    results['baseline_on_sr'] = baseline_model.val(data='./data2/output_sr_4x/dataset.yaml')
    
    # Evaluate SR model on SR data
    print("Evaluating SR model on upscaled data...")
    results['sr_on_sr'] = sr_model.val(data='./data2/output_sr_4x/dataset.yaml')
    
    return results

def visual_comparison(baseline_model, sr_model):
    """Compare models visually on some test images"""
    # Choose a few challenging test images (small objects)
    test_images = [
        './data2/output/images/test/image1.jpg',  # Replace with actual test images
        './data2/output/images/test/image2.jpg'
    ]
    
    # Create results directory
    results_dir = './comparison_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run predictions with both models
    for i, img_path in enumerate(test_images):
        if not os.path.exists(img_path):
            print(f"Warning: Test image {img_path} not found. Skipping.")
            continue
            
        print(f"Processing test image {i+1}...")
        
        # Baseline model prediction
        baseline_results = baseline_model.predict(
            img_path, 
            conf=0.25, 
            save=True,
            save_dir=os.path.join(results_dir, f'baseline_image{i+1}')
        )
        
        # SR model prediction
        sr_results = sr_model.predict(
            img_path, 
            conf=0.25, 
            save=True,
            save_dir=os.path.join(results_dir, f'sr_image{i+1}')
        )
        
        print(f"Saved prediction results for image {i+1}")

def plot_comparison(results):
    """Create comparison charts"""
    # Using the correct metric attributes from YOLO v8
    metrics = ['map50', 'map', 'mp', 'mr']  # mp=mean precision, mr=mean recall
    conditions = ['Original Data', 'Upscaled Data']
    
    # Extract values for plotting - using the correct attributes
    baseline_values = [
        [results['baseline_on_original'].box.map50, results['baseline_on_original'].box.map, 
         results['baseline_on_original'].box.mp, results['baseline_on_original'].box.mr],
        [results['baseline_on_sr'].box.map50, results['baseline_on_sr'].box.map, 
         results['baseline_on_sr'].box.mp, results['baseline_on_sr'].box.mr]
    ]
    
    sr_values = [
        [results['sr_on_original'].box.map50, results['sr_on_original'].box.map, 
         results['sr_on_original'].box.mp, results['sr_on_original'].box.mr],
        [results['sr_on_sr'].box.map50, results['sr_on_sr'].box.map, 
         results['sr_on_sr'].box.mp, results['sr_on_sr'].box.mr]
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot for original data
    axes[0].bar(x - width/2, baseline_values[0], width, label='Baseline Model')
    axes[0].bar(x + width/2, sr_values[0], width, label='SR-Trained Model')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance on Original Data')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['mAP@50', 'mAP@50-95', 'Mean Precision', 'Mean Recall'])
    axes[0].legend()
    
    # Plot for upscaled data
    axes[1].bar(x - width/2, baseline_values[1], width, label='Baseline Model')
    axes[1].bar(x + width/2, sr_values[1], width, label='SR-Trained Model')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance on Upscaled Data')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['mAP@50', 'mAP@50-95', 'Mean Precision', 'Mean Recall'])
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('./comparison_results/metrics_comparison.png')
    plt.close()
    
    # Calculate improvement percentages
    improvement_original = ((sr_values[0][0] / baseline_values[0][0]) - 1) * 100  # mAP50 improvement on original data
    improvement_upscaled = ((sr_values[1][0] / baseline_values[1][0]) - 1) * 100  # mAP50 improvement on upscaled data
    
    # Create improvement plot
    fig, ax = plt.subplots(figsize=(8, 6))
    improvements = [improvement_original, improvement_upscaled]
    ax.bar(['Original Data', 'Upscaled Data'], improvements, color=['blue', 'orange'])
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Percentage Improvement from SR Training')
    
    # Add value labels on bars
    for i, v in enumerate(improvements):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('./comparison_results/improvement_comparison.png')
    print(f"Saved comparison charts to ./comparison_results/")

def main():
    # Create results directory
    os.makedirs('./comparison_results', exist_ok=True)
    
    # Load the trained models
    print("Loading trained models...")
    baseline_model, sr_model = load_models()
    
    # Evaluate both models
    print("\nEvaluating models...")
    results = evaluate_models(baseline_model, sr_model)
    
    # Create visual examples
    print("\nGenerating visual comparisons...")
    visual_comparison(baseline_model, sr_model)
    
    # Plot comparison charts
    print("\nCreating comparison charts...")
    plot_comparison(results)
    
    # Print summary
    print("\n----- COMPARISON SUMMARY -----")
    print(f"Baseline mAP50 on original data: {results['baseline_on_original'].box.map50:.4f}")
    print(f"SR model mAP50 on original data: {results['sr_on_original'].box.map50:.4f}")
    
    # Calculate improvement percentage
    improvement = ((results['sr_on_original'].box.map50 / results['baseline_on_original'].box.map50) - 1) * 100
    print(f"Improvement from super-resolution: {improvement:.2f}%")
    
    print("\nComparison complete! Results saved to ./comparison_results/")

if __name__ == "__main__":
    main()
