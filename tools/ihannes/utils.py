import math

import numpy as np
import torch
import pycocotools.mask as mask_util
from detectron2.structures import Instances


def find_centroid(mask):
    y_center, x_center = np.argwhere(mask==1).sum(1) / mask.sum()
    x_center, y_center = int(x_center), int(y_center)

    return x_center, y_center    


def mask_closest_center(pred_classes, pred_masks, draw=None, img=None):
    '''
    pred_classes: torch.tensor, shape (N,), int in range [0, num_categories). 
                  N is the number of instances detected in the image
    pred_masks: torch.tensor, shape (N, H, W), binary masks. 
    '''
    if draw is not None and img is None:
        raise Exception

    img_x_center, img_y_center = pred_masks.shape[2] // 2, pred_masks.shape[1] // 2

    min_dist = None
    min_dist_idx_cls = -1
    for idx_inst in range(pred_classes.shape[0]):
        bin_mask = pred_masks[idx_inst]
        if (np.unique(bin_mask) == np.array([False])).all():
            #print('SKIP INSTANCE, NO MASK PREDICTED')
            # It enters here when mask r-cnn predicts a box
            # but no mask inside it.
            # If it happens for all the N instances detected in 
            # the image, the program returns -1
            continue

        mask_x_center, mask_y_center = find_centroid(bin_mask)
        
        if draw is not None:
            pass

        cur_dist = math.dist(
            (img_x_center, img_y_center), (mask_x_center, mask_y_center)
        )
        if min_dist is None or cur_dist < min_dist:
            min_dist = cur_dist
            min_dist_idx_cls = pred_classes[idx_inst].item()

    return min_dist_idx_cls 


def rle_to_binmask(mask, height, width):
    # from detectron2.utils.visualizer.GenericMask.init
    assert isinstance(mask, dict)
    assert "counts" in mask and "size" in mask
    if isinstance(m["counts"], list):
        h, w = m["size"]
        assert h == height and w == width
        m = mask_util.frPyObjects(m, h, w)
    mask = mask_util.decode(mask)[:, :]
    return mask


def get_centroid_coords(bin_masks, height, width):
    # TODO WHAT IF THERE IS NO PREDICTED INSTANCE? DO THESE KEYS STILL 
    #      EXIST? MAYBE YES BUT WITH ZERO ITEMS
    # bin_masks.shape n_instances, h, w
    assert bin_masks.ndim == 3, bin_masks.shape
    assert bin_masks.dtype == torch.bool
    centroids = []
    for b_m in bin_masks:
        if (np.unique(b_m) == np.array([False])).all():
            # It happend when mask r-cnn predicts a box but no mask inside 
            # it. 
            continue
        x, y = find_centroid(b_m)
        centroids.append((x, y))
    return centroids 
    

def centroid_closest_to_img_ctr(centroids, height, width):
    assert isinstance(centroids, list)
    if len(centroids) == 0:
        return None

    x_img_ctr, y_img_ctr = width//2, height//2
    min_dist = None
    min_dist_x, min_dist_y = None, None
    for c in centroids:
        x, y = c
        cur_dist = math.dist((x_img_ctr, y_img_ctr), (x, y))
        if min_dist is None or cur_dist < min_dist:
            min_dist = cur_dist
            min_dist_x, min_dist_y = x, y

    return min_dist_x, min_dist_y