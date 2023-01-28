import argparse
import shutil
import glob
import os

import cv2
import numpy as np

CLASSES_LIST = ['background', 'object_part', 'power_no3', 'pinch_3', 
                'lateral_no3', 'pinch_no3']

def main(semseg_folder_path, new_semseg_folder_path):
    color_to_preshape = {
        # TODO: CURRENTLY THE COLORS ARE TAKEN BY HAND BY LOOKING AT SOME IMAGES.
        #       INSTEAD, TAKE THEM FROM THE CORRECT .JSON
        str([0, 0, 0]): CLASSES_LIST[0],
        str([255, 0, 0]): CLASSES_LIST[1],
        str([0, 255, 0]): CLASSES_LIST[2],
        str([0, 0, 255]): CLASSES_LIST[3],
        str([4, 235, 255]): CLASSES_LIST[4],
        str([255, 255, 255]): CLASSES_LIST[5]
    }

    semseg_filenames = glob.glob(os.path.join(semseg_folder_path, 'segmentation_*.png'))
    for s_f in semseg_filenames:
        gt = cv2.imread(s_f)
        assert gt.shape[-1] == 3, gt.shape
        assert gt.ndim == 3, gt.ndim

        new_gt = np.zeros((gt.shape[0], gt.shape[1]), dtype=gt.dtype)
        unique_colors = np.unique(gt.reshape(-1, gt.shape[2]), axis=0)
        ids = []
        for u_q in unique_colors:
            bin_mask = (gt == u_q).prod(axis=-1).astype(gt.dtype)
            assert set(np.unique(bin_mask).tolist()).issubset(set([0, 1])), np.unique(bin_mask)
            assert bin_mask.ndim == 2, bin_mask.ndim

            preshape = color_to_preshape[str(u_q.tolist())]
            prsh_id = CLASSES_LIST.index(preshape)
            bin_mask[bin_mask == 1] = prsh_id
            new_gt += bin_mask

            ids.append(prsh_id)

        assert set(np.unique(new_gt).tolist()) == set(ids), np.unique(new_gt)
        cv2.imwrite(
            os.path.join(new_semseg_folder_path, os.path.basename(s_f)), new_gt
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semseg_folder_path',type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.semseg_folder_path):
        raise Exception
    foldername = os.path.basename(args.semseg_folder_path)
    new_semseg_folder_path = 'Detectron2_' + foldername
    new_semseg_folder_path = args.semseg_folder_path.replace(foldername, new_semseg_folder_path)
    #new_semseg_folder_path = os.path.join(os.getcwd(), new_semseg_folder_path)
    if os.path.isdir(new_semseg_folder_path):
        shutil.rmtree(new_semseg_folder_path)
    os.mkdir(new_semseg_folder_path)
    
    main(args.semseg_folder_path, new_semseg_folder_path)
