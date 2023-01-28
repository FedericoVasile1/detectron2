import os
import argparse
import pathlib
import glob

import cv2


def get_semseg_dicts(dataset_base_folder):
    # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts
    # Return a list where each element is a dict with the
    # following fields: file_name, height, width, image_id, 
    # sem_seg_filename

    # images_path: something/RGBxxx/rgb_*.png
    # semseg_path: something/SemanticSegmentationxxx/semantic_*.png

    dataset_base_folder = os.path.join(
        'datasets', 'synthetic', 'frames', dataset_base_folder
    )
    if not os.path.isdir(dataset_base_folder):
        raise Exception
    images_path = glob.glob(os.path.join(dataset_base_folder, 'RGB*')) 
    if len(images_path) != 1:
        raise Exception
    images_path = images_path[0]
    semseg_path = glob.glob(
        os.path.join(dataset_base_folder, 'Detectron2_SemanticSegmentation*')
    )
    if len(semseg_path) != 1:
        raise Exception
    semseg_path = semseg_path[0]

    dataset_dicts = []

    height, width = None, None
    imgs = glob.glob(os.path.join(images_path, 'rgb_*.png')) 
    for i in imgs:
        record = {}

        record['file_name'] = i
        if height is None or width is None:
            height, width = cv2.imread(i).shape[:2]
        record['height'], record['width'] = height, width
        # img name example: rgb_11.png
        # the corresponding semantic segmenation ground truth is segmentation_11.png
        record['image_id'] = os.path.basename(i).split('.')[0].split('_')[1]

        semseg_gt = os.path.basename(i).replace('rgb', 'segmentation')
        semseg_gt = os.path.join(semseg_path, semseg_gt)
        #if not os.path.isfile(semseg_gt):
        #    raise Exception
        record['sem_seg_file_name'] = semseg_gt

        dataset_dicts += record,

    return dataset_dicts

