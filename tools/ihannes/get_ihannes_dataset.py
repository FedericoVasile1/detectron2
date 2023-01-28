import os
import pathlib
import glob

from PIL import Image

from .metadatas import (
    VIDEO_DIRS, 
    INSTANCES
)


def get_ihannesframes_dicts(test_set):
    dataset_base_folder = os.path.join(
        'datasets', 'real', 'frames', 'iHannesDataset'
    )
    if not os.path.isdir(dataset_base_folder):
        raise Exception

    dataset_dicts = []
    video_paths = VIDEO_DIRS[test_set]
    for v_p in video_paths:
        full_path = os.path.join(dataset_base_folder, v_p)
        imgs = glob.glob(os.path.join(full_path, "*.jpg"))
        imgs.sort()

        preshape_gt = pathlib.Path(imgs[0]).parts[-4]
        instance_name = pathlib.Path(imgs[0]).parts[-5]
        seq_num = pathlib.Path(imgs[0]).parts[-2]
        s_or_m = None
        if instance_name in INSTANCES["single"]:
            s_or_m = "single"
        elif instance_name in INSTANCES["multi"]:
            s_or_m = "multi"
        else:
            raise Exception

        for i in imgs:
            record = {}

            record["file_name"] = i
            width, height = Image.open(i).size
            record["height"], record["width"] = height, width
            record["image_id"] = os.path.basename(i).split('.')[0]
            
            record["preshape_gt"] = preshape_gt
            record["instance_name"] = instance_name
            record["seq_num"] = seq_num
            record["s_or_m"] = s_or_m

            dataset_dicts += record,

    return dataset_dicts

