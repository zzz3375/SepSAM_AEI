using_colab = False
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread('D:/GAIN-LQ-personal/SAM/1-canny-2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "C:/Users/DELL T366001/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

# Find and remove the largest area mask
largest_area = 0
largest_mask_index = -1

for i, mask in enumerate(masks):
    area = np.sum(mask['segmentation'])
    if area > largest_area:
        largest_area = area
        largest_mask_index = i

if largest_mask_index != -1:
    del masks[largest_mask_index]



print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


import random

def generate_distinct_color(index):
    """Generate a distinct color based on the index."""
    random.seed(index)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Create an RGBA image to support transparency
mask_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

# Iterate over each mask and draw it on the blank image with different colors and transparency
for i, mask in enumerate(masks):
    segmentation = mask['segmentation']
    color = generate_distinct_color(i)
    alpha = 88  # Set transparency level (0-255)
    rgba_color = (*color, alpha)
    mask_rgba = np.zeros_like(mask_image, dtype=np.uint8)
    mask_rgba[segmentation == 1] = rgba_color
    mask_image = cv2.addWeighted(mask_image, 1, mask_rgba, 0.5, 0)

# Convert the mask image to BGR before saving
mask_image_bgr = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2BGR)

# Save the mask image as a JPG file
cv2.imwrite('D:/GAIN-LQ-personal/SAM/colored-masks-canny.jpg', mask_image_bgr)