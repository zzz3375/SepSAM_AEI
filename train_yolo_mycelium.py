from ultralytics import YOLO
import os

# === PATH TO YOUR DATASET ===
DATA_YAML_PATH = "mycelium_yolo_dataset/data.yaml"  # Adjust if in a different folder

# === MODEL SELECTION ===
# You can also try "yolov8s.pt" or "yolov8m.pt" if you want more accuracy
BASE_MODEL = "yolov8n.pt"

# === TRAINING CONFIG ===
EPOCHS = 50
IMAGE_SIZE = 640  # Can also try 512 or 416

def train():
    # Check that data.yaml exists
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"data.yaml not found at: {DATA_YAML_PATH}")

    # Load the base model
    model = YOLO(BASE_MODEL)

    # Train
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=16,
        project="runs",
        name="mycelium_yolo_train"
    )

    print("✅ Training complete!")
    print("➡️ Trained model saved to:", model.ckpt_path)

if __name__ == "__main__":
    train()
