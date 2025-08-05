#%%
# import subprocess
# import shutil
import numpy as np

# image processing
from ultralytics import YOLO
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from segment_anything import sam_model_registry, SamPredictor

# visulization
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics.utils.ops import scale_masks
from pathlib import Path
from PIL import Image
import time

#%%
def yolo_predict(source, debug=True, yolo_model_path = "best.pt", conf_thresh=0):
    model_path = yolo_model_path
    source_image = cv2.imread(source)
    h,w=source_image.shape[:2]
    model = YOLO(model_path)
    results = model.predict(source,  device="cuda:0")
    # mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1,2,0).squeeze(-1).astype(int)*255
    original_shape = source_image.shape[:2]
    conf = results[0].boxes.conf.cpu().detach().numpy()
    if results[0].masks is not None:
        mask_raw_stack = scale_masks(results[0].masks.data[None,:],original_shape).cpu().numpy()
    else: mask_raw_stack = np.zeros([1,1]+list(source_image.shape[:2]), dtype = np.uint8)
    mask_raw = mask_raw_stack.squeeze(axis=0)[conf>conf_thresh].sum(axis=0)  
    mask_raw = ( mask_raw >0 ).astype(np.uint8) *255
    ret, mask_raw = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
    # if debug: 
    cv2.imwrite(r"tmp\yolo_raw_result.png",mask_raw)
    return mask_raw, conf

def mask2points(mask_raw, debug=True, sampling_points = 12) -> np.ndarray:
    skeleton_bool = skeletonize(mask_raw)
    dist = distance_transform_edt(mask_raw)
    if debug: np.save(r"tmp\distance-inplace", dist)
    dist[skeleton_bool==0]=0
    skeleton_img = skeleton_bool.astype(np.uint8)*255
    if debug: 
        cv2.imwrite(r"tmp\skeleton.png",skeleton_img)
        np.save(r"tmp\skeleton_length", skeleton_bool.sum())
    h,w = skeleton_bool.shape[:2]
    skeleton_index = np.where(skeleton_bool.ravel())[0]
    y = skeleton_index//w
    x = skeleton_index%w
    points = np.array([x,y]).T

    # sample
    n = points.shape[0]
    step = n//sampling_points
    # step = 80
    # points = points[int(step/2):-1-int(step/2):step]
    points = points[::step] if step > 0 else points
    return points

def sam_prompt(source, points, debug=True):
    source_image = cv2.imread(source)
    # sample

    input_label = np.ones(points.shape[:1])
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(source_image)
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=input_label,
        multimask_output=False,
    )
    if debug:
        cv2.imwrite(r"tmp\SAM_prompting.png",masks[scores.argmax()].astype(int)*255)
        cv2.imwrite(r"tmp\original_image.jpg",source_image)
        print("SAM score:", scores[0])
    # plt.figure(figsize=(10,10))
    # plt.imshow(source_image)
    # show_points(points, input_label, plt.gca())
    # plt.axis('on')
    # plt.savefig(r"tmp\points_prompting.jpg")
    
    for p in points: 
        # p = np.round(p)
        source_image = cv2.circle(source_image,p,10,(255,0,0),4)
    if debug: cv2.imwrite(r"tmp\points_prompting.jpg",source_image)
    return masks[0].astype(np.uint8)*255, scores[0]

def morno_coorrection(img, debug=1):
    """deplicated"""
    m = np.ceil(max(img.shape[:2])*0.015).astype(int)
    n = np.ceil(max(img.shape[:2])*0.007).astype(int)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (m,m))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (n,n))
    # out = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel_open)
    out = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel_close)
    if debug: cv2.imwrite(r"tmp\SAM_prompting_morno.png", out)        
    return out

def contour_aera(con,mask):
    tmp = np.zeros_like(mask)
    mask_i = cv2.fillPoly(tmp,con,255)
    aera = np.sum(mask_i>0)
    # print(aera)
    return aera

def contour_optimization(mask,yolo_conf, debug=1):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.drawContours(np.zeros_like(mask), contours, -1, 255, 1)
    if debug: cv2.imwrite(r"tmp\contours.png",out)
    contours_sorted = sorted(contours, key = lambda con: contour_aera(con, mask=mask), reverse=1)
    contours_accepted = contours_sorted[:sum(yolo_conf>0.3)]
    out = cv2.drawContours(np.zeros_like(mask), contours_accepted, -1, 255, 1)
    if debug: cv2.imwrite(r"tmp\contours_filtered.png", out)
    out = np.zeros_like(mask)
    for con in contours_accepted:
        out = cv2.fillPoly(out,[con.squeeze(axis=1)],255)
    if debug: cv2.imwrite(r"tmp\SAM_contour_filter.png", out)
    return out

def sam_seg_crack_by_prompt(source, debug=0, sampling_points = 12, yolo_thresh=0.5):
    Path("tmp").mkdir(parents=1,exist_ok=1)
    t0=time.time()
    mask_yolo, yolo_conf = yolo_predict(source,debug, conf_thresh=yolo_thresh)
    print("YOLO inferring time:", time.time()-t0, "s")

    t1=time.time()
    points = mask2points(mask_yolo,debug, sampling_points=sampling_points)
    print("points sampling time:", time.time()-t1, "s")
    
    t2=time.time()
    mask_sam, sam_scores = sam_prompt(source,points,debug)
    print("SAM inferring time:", time.time()-t2, "s")

    t3 = time.time()
    mask_sam = contour_optimization(mask_sam,yolo_conf,debug)
    print("contour optimization time:", time.time()-t3, "s")
    # mask_sam = morno_coorrection(mask_sam,debug)


    t4 = time.time()

    # contradictary analysis: 2 model should have coherent prediction
    conflict = (mask_sam - mask_sam*mask_yolo).astype(bool).sum()/ (mask_yolo).astype(bool).sum() if (mask_yolo).astype(bool).sum() > 0 else 100
    if  conflict < 1.5 and sam_scores > 0.85:
        mask_accepted = mask_sam
        if debug: print("accept SAM")
    else: 
        mask_accepted = mask_yolo # prevent the condition that YOLO raw result is not reliable, making some sampling point out of the crack
        if debug: print("accept yolo")
    if debug: 
        cv2.imwrite(r"tmp\accepted_result.png",mask_accepted)     

    print("confliction time:", time.time()-t4, "s")   

    return mask_accepted

# %% Demonstration

def show_distance_and_axis(source, h1h2w1w2 = [470, 580, 400, 510]):
    sam_seg_crack_by_prompt(source, yolo_thresh=0.0)
    # distance = np.load(r"tmp\distance-inplace.npy")

    colormap = plt.get_cmap('viridis') 

    img = np.asarray(Image.open(source))
    mask = cv2.imread(r"tmp\accepted_result.png", cv2.IMREAD_GRAYSCALE)
    distance = distance_transform_edt(mask).astype(np.uint8)
    array=distance
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    heatmap_rgb = (colormap(normalized_array)[:, :, :3] * 255).astype(np.uint8) 
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    distance = heatmap_bgr

    

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    edge = cv2.drawContours(np.zeros_like(mask), contours, -1, 255, 1)
     

    ske = np.asarray(Image.open(r"tmp\skeleton.png"))
    ske = ske[:,:,None].repeat(3,-1)
    ske[edge>127]=[0,255,0]

    points = mask2points(mask, sampling_points=60)
    # for p in points:
    #     ske = cv2.circle(ske,p,3,(255,0,0),1)
    crack_statistics = ske+edge[:,:,None].repeat(3,-1)
    # cv2.imwrite(r"result_demonstration\figures\skeleton.png",ske)
    cv2.imwrite(r"result_demonstration\figures\crack_statistics.png",crack_statistics)
    cv2.imwrite(r"result_demonstration\figures\distance.png",distance)

def show_agent_filter():
    source = r"data\crack_dataset_cleaned\混凝土桥梁裂缝optic_disc_seg\JPEGImages\H0021.jpg"
    # h1h2w1w2 = [470, 580, 400, 510]
    h1h2w1w2 = [0,512,512,1024]
    h1, h2, w1, w2 = h1h2w1w2
    # show_distance_and_axis(source, h1h2w1w2)
    sam_seg_crack_by_prompt(source)
    distance = np.load(r"tmp\distance-inplace.npy")
    distance_heatmap = (distance/distance.max()*255).astype(np.uint8)
    # distance_heatmap = np.stack([np.zeros_like(distance_heatmap), np.zeros_like(distance_heatmap), distance_heatmap], axis=-1)
    distance_heatmap = cv2.applyColorMap(distance_heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(r"result_demonstration\figures\heatmap.jpg",distance_heatmap[h1:h2, w1:w2])


    ret, sam_raw = cv2.threshold(np.asarray(Image.open(r"tmp\SAM_prompting.jpg")),127,255,cv2.THRESH_BINARY)
    con, hie = cv2.findContours(sam_raw,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    plt.rcParams.update({"font.size":8})
    n = 3
    ax = plt.subplot(1,n,1)
    plt.imshow(sam_raw[h1:h2, w1:w2], cmap="gray")
    ax.set_title("SAM Advise")
    ax.set_axis_off()
    
    ax = plt.subplot(1,n,2)
    sam_raw = sam_raw[:,:,None].repeat(3,-1)
    sam_raw = np.zeros_like(sam_raw)
    sam_raw = cv2.drawContours(sam_raw, con, -1, (255,0,0), 1)
    ax.imshow(sam_raw[h1:h2, w1:w2], cmap="gray")
    ax.set_title("Border Tracing and Archive")
    ax.set_axis_off()

    ax = plt.subplot(1,n,3)
    sam_filtered = np.array(Image.open(r"tmp\SAM_contour_filter.jpg"))
    ax.imshow(sam_filtered[h1:h2, w1:w2], cmap="gray")
    ax.set_axis_off()
    ax.set_title("Output")
    # plt.show()
    plt.savefig(r"2nd.svg", dpi=600)


    
    pass
# 11451
# %%
if __name__ == '__main__':

    source = r"C:\Users\13694\SFM-COLMAP\output\Mei_Jing_Pavement\dense\images\IMG_20250511_134025.jpg"
    sam_seg_crack_by_prompt(source, debug=1, sampling_points = 30, yolo_thresh=0)
    pass
