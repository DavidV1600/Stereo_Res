from ultralytics import YOLO
import os

def train_sr_model():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the nano version for faster training
    
    # Train the model on your upscaled TinyPerson dataset
    results = model.train(
        data='./data2/output_sr_4x/dataset.yaml',  # Path to upscaled dataset
        epochs=40,                          # Number of epochs to train for
        imgsz=4000,                          # Image size
        batch=1,                           # Batch size (reduce if you get OOM errors)
        patience=15,                        # Early stopping patience
        name='tinyperson_sr_100'                # Name for this run/experiment
    )
    
    return model, results

if __name__ == "__main__":
    # Train the super-resolution model
    print("Training super-resolution model...")
    model, results = train_sr_model()
    
    # Evaluate the SR model
    metrics = model.val()
    print(f"SR model validation metrics: {metrics}")
    
    # Save model in ONNX format for easier deployment
    save_dir = './saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Export SR model
    model.export(format='onnx', save_dir=os.path.join(save_dir, 'sr'))
    print(f"SR model saved to {os.path.join(save_dir, 'sr')}")
    
    print("Super-resolution training complete!")
