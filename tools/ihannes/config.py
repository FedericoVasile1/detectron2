from detectron2.config import CfgNode as CN


def add_my_config(cfg):
    cfg.INPUT.BLUR = CN()
    cfg.INPUT.BLUR.ENABLED = False
    cfg.INPUT.BLUR.PROB_APPLY = 0.3
    cfg.INPUT.BLUR.RADIUS_MIN = 1.5
    cfg.INPUT.BLUR.RADIUS_MAX = 3.0

    cfg.DATASETS.IHANNES = CN()
    cfg.DATASETS.IHANNES.TEST_SET = "test_same_person"
    cfg.DATASETS.IHANNES.LAST_FRAME_ID = 60
    cfg.DATASETS.IHANNES.SAVE_OPTION = "wrong"
    cfg.DATASETS.IHANNES.EVAL_TYPE = "centroid"