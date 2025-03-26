import os
import re

def remove_images(folder: str, test_name: str, angle: int, min_hours: int):
    pattern = re.compile(rf"^{test_name}_h(\d+)_({angle})\.jpg$")
    
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            hours = int(match.group(1))
            if hours < min_hours:
                file_path = os.path.join(folder, filename)
                os.remove(file_path)
                print(f"Removed: {filename}")

remove_images("output", "test3", 3, 700)