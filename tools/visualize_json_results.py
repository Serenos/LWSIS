#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from adet.modeling.condinst import register_lidar_annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO dataset."
    )
    parser.add_argument("--input", default='nuscene_v1.0-mini_val_with_maskv1.0.json', help="JSON file produced by the model")
    parser.add_argument("--output", default='./vis_nuseg_train_gt/', help="output directory")
    parser.add_argument("--show",  default=False, help="show the result")
    parser.add_argument("--dataset", default="tbox_refine_nuscene_v1.0-trainval_train_v1.0")
    args = parser.parse_args()

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    for i, dic in enumerate(tqdm.tqdm(dicts)):
        if os.path.basename(dic['file_name']) == 'n008-2018-08-29-16-04-13-0400__CAM_FRONT__1535573350412404.jpg':
            break

    for i, dic in enumerate(tqdm.tqdm(dicts)):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        if args.output:
            cv2.imwrite(os.path.join(args.output, basename), vis_gt[:, :, ::-1])
            # cv2.imwrite(os.path.join(origin_ouput, basename), img[:, :, ::-1])
        if args.show:
            print(basename)
            cv2.imshow("window", vis_gt[:, :, ::-1])
            cv2.waitKey()

#python tools/visualize_json_results.py --input  projects/LidarSup/annotations/nuscene_v1.0-mini_val_balance_with_maskv01.json --output ~/Documents/nus_minival_segm_00/  --dataset nuscene_v1.0-mini_val_balance_with_maskv01