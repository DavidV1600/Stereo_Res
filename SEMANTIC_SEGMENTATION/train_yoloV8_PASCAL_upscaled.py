from ultralytics import YOLO
import os

def train_model(data_yaml, exp_name, epochs=1, imgsz=3560, batch=1):
    model = YOLO('yolov8n.pt')  # You can also try yolov8s.pt for better accuracy
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        name=exp_name
    )
    return model, results

def evaluate_and_export(model, model_name):
    metrics = model.val()
    print(f"{model_name} validation metrics: {metrics}")

    save_dir = f'./saved_models/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    model.export(format='onnx', save_dir=save_dir)
    print(f"{model_name} model exported to {save_dir}")

if __name__ == "__main__":
    print("Training on upscaled VOC...")
    model_upscaled, _ = train_model(
        data_yaml='./data/VOC_train_upscaled_4x_yolo/dataset.yaml',  # adjust path
        exp_name='voc_upscaled_baseline'
    )
    evaluate_and_export(model_upscaled, 'voc_upscaled')
    print("Training complete on both datasets.")
