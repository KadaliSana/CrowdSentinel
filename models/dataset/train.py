from ultralytics import YOLO

def train_yolo26():
    # 1. Load the YOLO26 model
    # Options: yolo26n.pt (nano), yolo26s.pt (small), yolo26m.pt (medium), yolo26l.pt (large)
    model = YOLO("yolo26s.pt") 

    # 2. Train the model
    # Replace 'path/to/data.yaml' with your dataset configuration file
    results = model.train(
        data="data.yaml",   # Path to your dataset yaml
        epochs=100,                  # Number of training epochs
        imgsz=640,                   # Image size (default is 640)
        batch=-1,                    # Batch size (set -1 for AutoBatch)
        device=0,                    # Device to run on (e.g. 0 or 0,1,2,3 or 'cpu')
        optimizer="MuSGD",           # YOLO26's optimized hybrid optimizer
        patience=50,                 # Early stopping patience
        save=True,                   # Save train checkpoints and predict results
        project="YOLO26_Training",    # Project name
        name="custom_run_1",         # Run name
        exist_ok=True                # Overwrite existing run if necessary
    )

    print("Training complete. Best weights saved to:", results.save_dir)

if __name__ == "__main__":
    train_yolo26()
