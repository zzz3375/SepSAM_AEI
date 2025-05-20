from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
import numpy as np
import cv2
import os
import datetime
from PIL import Image
        
# === CONFIG ===
YOLO_MODEL_PATH = "models/best.pt"
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
INPUT_FOLDER = "mycelium"
OUTPUT_FOLDER = "output"


# === INIT MODELS ===
yolo = YOLO(YOLO_MODEL_PATH)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

# === LOGGING ===
def getCurrentTime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

log_filename = f"output_log_{getCurrentTime()}.txt"

def log(message: str):
    print(message)
    with open(log_filename, "a") as f:
        f.write(message + "\n")

# === SEGMENT FUNCTION ===
def segment_with_yolo_and_sam(image_path: str, output_name: str):
    try:
        image = cv2.imread(image_path)
        if image is None:
            log(f"Could not load image: {image_path}")
            return

        # Step 1: YOLO Detection
        yolo_results = yolo(image)
        boxes = yolo_results[0].boxes.xyxy

        if len(boxes) == 0:
            log(f"No bounding box detected in {image_path}")
            return

        # Use the first box
        box = boxes[0].cpu().numpy()
        box = np.array([box])  # Shape (1, 4), format [x1, y1, x2, y2]

        # Step 2: SAM Segmentation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        masks, scores, logits = predictor.predict(box=box, multimask_output=False)
        mask = masks[0]

        # Apply the mask to the original image
        masked_image = image_rgb.copy()
        masked_image[~mask] = 0

        # Save output
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        Image.fromarray(masked_image).save(os.path.join(OUTPUT_FOLDER, output_name))
        log(f"Segmented and saved: {output_name}")

    except Exception as e:
        log(f"Error processing {image_path}: {e}")

# === IMAGE PROCESSING ===
def process_images(base_folder):
    test_folders = [f for f in os.listdir(base_folder) if f.startswith("myceliumtest")]

    for test_folder in sorted(test_folders, key=lambda x: int(x.replace("myceliumtest", ""))):
        test_path = os.path.join(base_folder, test_folder)
        subfolders = sorted(os.listdir(test_path))

        if not subfolders:
            continue

        first_timestamp = parse_timestamp(subfolders[0])

        for subfolder in subfolders:
            if "temperature" in subfolder.lower():
                log(f"Skipping temp folder: {test_folder}/{subfolder}")
                continue

            subfolder_path = os.path.join(test_path, subfolder)
            current_timestamp = parse_timestamp(subfolder)
            if not current_timestamp:
                continue
            hours_elapsed = int((current_timestamp - first_timestamp).total_seconds() // 3600)

            for img_num in range(1, 5):
                image_path = os.path.join(subfolder_path, f"{img_num}.jpg")
                if os.path.exists(image_path):
                    new_name = f"test{test_folder.replace('myceliumtest', '')}_h{hours_elapsed}_{img_num}.jpg"
                    if os.path.exists(os.path.join(OUTPUT_FOLDER, new_name)):
                        log(f"Already exists: {new_name}")
                        continue
                    log(f"Processing: {image_path}")
                    segment_with_yolo_and_sam(image_path, new_name)
                else:
                    log(f"Missing file: {image_path}")

# === TIMESTAMP PARSING ===
def parse_timestamp(folder_name):
    try:
        date_part, time_part = folder_name.split("___")
        year, month, day = map(int, date_part.split("-"))
        hour, minute = map(int, time_part.split("-"))
        return datetime.datetime(year + 2000, month, day, hour, minute)
    except Exception as e:
        log(f"Error parsing timestamp from {folder_name}: {e}")
        return None

# === MAIN ===
if __name__ == "__main__":
    process_images(INPUT_FOLDER)
