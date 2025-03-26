from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import os
import datetime


def getCurrentTime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

start_date = getCurrentTime()  # Replaced colons with dashes

# Use the formatted date in the filename
filename = f"output_log_{start_date}.txt"


# Load the SAM model
checkpoint = "models/sam_vit_h.pth"
sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
predictor = SamPredictor(sam)

def log(message: str):
    """Prints a message and appends it to output_log.txt."""
    print(message)  # Print to console
    with open(filename, "a") as log_file:  # Open file in append mode
        log_file.write(message + "\n")  # Append message with a newline


def generate_points(image_np: np.ndarray, foreground_points: int = 2, background_points: int = 0):
    """Generate foreground and background points with labels for segmentation."""
    height, width = image_np.shape[:2]
    background_difference = 11  # Controls background point positioning

     # Foreground points (middle horizontal line)
    x_foreground_positions = np.linspace(width // 10, 9 * width // 10, foreground_points, dtype=int)
    y_foreground_position = height // 2  # Fixed at middle height

    print(f"asked for points: {foreground_points}\n generated x: {x_foreground_positions}\ngenerated y: {y_foreground_position}")

    fg_points = np.array([[x, y_foreground_position] for x in x_foreground_positions])

    # Background points (top & bottom) - Only create if background_points > 0
    top_bg_points = bottom_bg_points = np.empty((0, 2), dtype=int)  # Empty arrays if no background points

    if background_points > 0:
        x_background_positions = np.linspace(width // 10, 9 * width // 10, background_points, dtype=int)
        y_bottom_position = height // background_difference
        y_top_position = y_bottom_position * (background_difference - 1)

        top_bg_points = np.array([[x, y_top_position] for x in x_background_positions])
        bottom_bg_points = np.array([[x, y_bottom_position] for x in x_background_positions])

    # Combine all points only if they exist (non-empty arrays)
    all_points = fg_points
    if top_bg_points.size > 0:
        all_points = np.vstack((all_points, top_bg_points))
    if bottom_bg_points.size > 0:
        all_points = np.vstack((all_points, bottom_bg_points))

    # Assign labels: foreground (1), background (0)
    fg_labels = np.ones(len(fg_points), dtype=int)  # Foreground = 1
    bg_labels = np.zeros(len(top_bg_points) + len(bottom_bg_points), dtype=int)  # Background = 0
    point_labels = np.concatenate((fg_labels, bg_labels))

    return all_points, point_labels

def generate_segment(filename: str, new_name: str, foreground_points: int = 2, background_points: int = 0): 
    # Load the image
    try:
        image = Image.open(filename)
        image_np = np.array(image)
    except Exception as e:
        log(f"couldn't load image: {e}")
        return

    
    points, point_labels = generate_points(image_np, foreground_points, background_points)

    # Check if points were generated
    if points.size == 0:
        log(f"Skipping {filename}, no points generated.")
        return

    # Predict the mask using the multiple points as prompts
    try:
        predictor.set_image(image_np)
        masks, scores, logits = predictor.predict(point_coords=points, point_labels=point_labels)
    except Exception as e:
        log(f"Error during mask prediction for {filename}: {e}")
        return

    # Choose the first mask (adjust the index if necessary)
    mask = masks[0]

    # Apply the mask to the image
    masked_image = image_np.copy()
    masked_image[mask == 0] = 0  # Set background pixels to black

    # Save the resulting image
    output_full = Image.fromarray(masked_image)
    output_full.save(f"output/{new_name}")


def process_images(base_folder):
    test_folders = [f for f in os.listdir(base_folder) if f.startswith("myceliumtest")]
    
    for test_folder in sorted(test_folders, key=lambda x: int(x.replace("myceliumtest", ""))):
        test_path = os.path.join(base_folder, test_folder)
        subfolders = sorted(os.listdir(test_path))

        if not subfolders:
            continue
        
        # Extract the timestamp of the first image set
        first_timestamp = parse_timestamp(subfolders[0])
        
        for subfolder in subfolders:
            if not subfolder.lower().find("temperature") == -1:
                log(f"continueing invalid subfolder: {test_folder}/{subfolder}")
                continue

            subfolder_path = os.path.join(test_path, subfolder)
            current_timestamp = parse_timestamp(subfolder)
            hours_elapsed = int((current_timestamp - first_timestamp).total_seconds() // 3600)
            
            for img_num in range(1, 5):  # Images are named 1.jpg to 4.jpg
                image_path = os.path.join(subfolder_path, f"{img_num}.jpg")
                if os.path.exists(image_path):
                    restrict_foreground = img_num % 2 == 0  # Restrict foreground only if img_num is even
                    if not restrict_foreground:
                        new_name = f"test{test_folder.replace('myceliumtest', '')}_h{hours_elapsed}_{img_num}.jpg"
                        if os.path.exists(f"output/{new_name}"):
                            log(f"file already exists: {new_name}")
                            continue
                        log(f"Starting at {getCurrentTime()}\nProcessed {image_path} -> {new_name}")
                        generate_segment(image_path, new_name)
                        log(f"finished at: {getCurrentTime()}")
                    else:
                        log(f"sciping image, not side view: {image_path}")
                else: 
                    log(f"couldn't find file ${image_path}")

def parse_timestamp(folder_name):
    try:
        date_part, time_part = folder_name.split("___")
        year, month, day = map(int, date_part.split("-"))
        hour, minute = map(int, time_part.split("-"))
        return datetime.datetime(year + 2000, month, day, hour, minute)  # Assuming year is in YY format
    except Exception as e:
        log(f"Error while generating name for: {folder_name}: {e}")
        return
    
def is_subfolder_before(subfolder1: str, subfolder2: str) -> bool:
    """
    Compares two subfolder names based on their timestamps and returns True if subfolder1 is before subfolder2.
    """
    
    timestamp1 = parse_timestamp(subfolder1)
    timestamp2 = parse_timestamp(subfolder2)
    
    if timestamp1 and timestamp2:
        return timestamp1 < timestamp2
    else:
        return False  # If parsing fails, return False by default


process_images("mycelium")