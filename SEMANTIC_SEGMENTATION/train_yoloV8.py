from ultralytics import YOLO
import os

def train_baseline_model():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the nano version for faster training
    
    # Train the model on your TinyPerson dataset
    results = model.train(
        data='./data2/output/dataset.yaml',  # Path to your dataset.yaml file
        epochs=40,                          # Number of epochs to train for
        imgsz=4000,                          # Image size
        batch=1,                           # Batch size (reduce if you get OOM errors)
        patience=15,                        # Early stopping patience
        name='tinyperson_baseline'          # Name for this run/experiment
    )
    
    return model, results

if __name__ == "__main__":
    # Train the baseline model
    print("Training baseline model...")
    model, results = train_baseline_model()
    
    # Evaluate the baseline model
    metrics = model.val()
    print(f"Baseline validation metrics: {metrics}")
    
    # Save model in ONNX format for easier deployment
    save_dir = './saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Export baseline model
    model.export(format='onnx', save_dir=os.path.join(save_dir, 'baseline'))
    print(f"Baseline model saved to {os.path.join(save_dir, 'baseline')}")
    
    print("Baseline training complete!")
