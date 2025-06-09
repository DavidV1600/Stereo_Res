from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import numpy as np

def load_models():
    """Load the trained YOLOv8 models"""
    baseline_path = 'runs/detect/voc_orig_baseline6/weights/best.pt'
    upscaled_path = 'runs/detect/voc_upscaled_baseline3/weights/best.pt'
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline model not found at {baseline_path}")
    if not os.path.exists(upscaled_path):
        raise FileNotFoundError(f"Upscaled model not found at {upscaled_path}")
    
    baseline_model = YOLO(baseline_path)
    sr_model = YOLO(upscaled_path)
    
    return baseline_model, sr_model

def evaluate_models(baseline_model, sr_model):
    """Evaluate both models on original and upscaled VOC datasets"""
    results = {}
    
    print("Evaluating baseline model on original data...")
    results['baseline_on_original'] = baseline_model.val(data='./data/VOC_YOLO/dataset.yaml')

    print("Evaluating SR model on original data...")
    results['sr_on_original'] = sr_model.val(data='./data/VOC_YOLO/dataset.yaml')

    print("Evaluating baseline model on upscaled data...")
    results['baseline_on_sr'] = baseline_model.val(data='./data/VOC_train_upscaled_4x_yolo/dataset.yaml')

    print("Evaluating SR model on upscaled data...")
    results['sr_on_sr'] = sr_model.val(data='./data/VOC_train_upscaled_4x_yolo/dataset.yaml')
    
    return results

def plot_comparison(results):
    metrics = ['map50', 'map', 'mp', 'mr']
    x = np.arange(len(metrics))
    width = 0.35

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

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].bar(x - width/2, baseline_values[0], width, label='Baseline')
    axes[0].bar(x + width/2, sr_values[0], width, label='SR-Trained')
    axes[0].set_title("Performance on Original Data")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['mAP@50', 'mAP@50-95', 'Precision', 'Recall'])
    axes[0].legend()

    axes[1].bar(x - width/2, baseline_values[1], width, label='Baseline')
    axes[1].bar(x + width/2, sr_values[1], width, label='SR-Trained')
    axes[1].set_title("Performance on Upscaled Data")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['mAP@50', 'mAP@50-95', 'Precision', 'Recall'])
    axes[1].legend()

    plt.tight_layout()
    os.makedirs('./comparison_results', exist_ok=True)
    plt.savefig('./comparison_results/voc_metrics_comparison.png')
    plt.close()

    imp1 = ((sr_values[0][0] / baseline_values[0][0]) - 1) * 100
    imp2 = ((sr_values[1][0] / baseline_values[1][0]) - 1) * 100

    fig, ax = plt.subplots()
    ax.bar(['Original Data', 'Upscaled Data'], [imp1, imp2], color=['blue', 'orange'])
    ax.set_title("mAP@50 Improvement (SR vs. Baseline)")
    ax.set_ylabel("Improvement (%)")
    for i, val in enumerate([imp1, imp2]):
        ax.text(i, val + 0.5, f"{val:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('./comparison_results/voc_improvement.png')
    print("Comparison charts saved to ./comparison_results/")

def main():
    print("Loading models...")
    baseline_model, sr_model = load_models()
    
    print("\nEvaluating models...")
    results = evaluate_models(baseline_model, sr_model)

    print("\nPlotting results...")
    plot_comparison(results)

    print("\nSummary (high precision):")
    print(f"Baseline mAP@50 (orig):    {results['baseline_on_original'].box.map50:.8f}")
    print(f"SR mAP@50 (orig):          {results['sr_on_original'].box.map50:.8f}")
    print(f"Baseline mAP@50 (upscaled): {results['baseline_on_sr'].box.map50:.8f}")
    print(f"SR mAP@50 (upscaled):       {results['sr_on_sr'].box.map50:.8f}")

    # Calculate deltas
    delta_orig = results['sr_on_original'].box.map50 - results['baseline_on_original'].box.map50
    delta_upscaled = results['sr_on_sr'].box.map50 - results['baseline_on_sr'].box.map50

    print("\nImprovement Analysis:")
    print(f"mAP@50 improvement on original data:  {delta_orig:.8f} ({(delta_orig/results['baseline_on_original'].box.map50)*100:.4f}%)")
    print(f"mAP@50 improvement on upscaled data: {delta_upscaled:.8f} ({(delta_upscaled/results['baseline_on_sr'].box.map50)*100:.4f}%)")

if __name__ == "__main__":
    main()