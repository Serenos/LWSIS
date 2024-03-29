import logging
import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin import _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json

# NuScenes dataset in coco format
def register_nuscenes_instances_with_points_and_box(name, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance segmentation with point annotation.

    The point annotation json does not have "segmentation" field, instead,
    it has "point_coords" and "point_labels" fields.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, ["bbox", "sample_token", "cam", "ann_token", "point_coords", "point_labels", "3din_coords", "3dout_coords"])
    )



#NuScenes
json_root = os.path.join(os.environ['HOME'], 'datasets/nuscenes/annotations')
image_root = os.path.join(os.environ['HOME'],'datasets/nuscenes')

# tight 2D box
h2_path = os.path.join(json_root, 'tbox_refine_nuscene_v1.0-trainval_train_v4.1.json')
register_nuscenes_instances_with_points_and_box('tbox_refine_nuscene_v1.0-trainval_train_v4.1', h2_path, image_root)

h3_path = os.path.join(json_root, 'tbox_refine_nuscene_v1.0-trainval_val_v1.0.json')
register_nuscenes_instances_with_points_and_box('tbox_refine_nuscene_v1.0-trainval_val_v1.0', h3_path, image_root)

