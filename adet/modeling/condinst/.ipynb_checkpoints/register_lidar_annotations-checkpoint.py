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

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
#     MetadataCatalog.get(name).set(
#         json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
#     )

# _root = os.environ['HOME']
_root = '/cpfs2/shared/public'
json_root = os.path.join(_root, 'dataset/nuscenes/annotations')
img_root =os.path.join(_root, 'dataset/nuscenes')


# json_train02 = 'refine2_nuscene_v1.0-mini_train_with_maskv1.0.json'
# json_val02 = 'refine2_nuscene_v1.0-mini_val_with_maskv2.0.json'
# json_train02_path = os.path.join(json_root, json_train02)
# json_val02_path = os.path.join(json_root, json_val02)
# register_nuscenes_instances_with_points_and_box('refine2_nuscene_v1.0-mini_train_with_maskv1.0', json_train02_path, img_root)
# register_nuscenes_instances_with_points_and_box('refine2_nuscene_v1.0-mini_val_with_maskv2.0', json_val02_path, img_root)

# json_train01 = 'refine2_nuscene_v1.0-mini_val_with_maskv1.0.json'
# json_val01 = 'nuscene_v1.0-trainval_val_with_maskv1.0.json'
# json_train01_path = os.path.join(json_root, json_train01)
# json_val01_path = os.path.join(json_root, json_val01)
# register_nuscenes_instances_with_points_and_box('refine2_nuscene_v1.0-mini_val_with_maskv1.0', json_train01_path, img_root)
# register_nuscenes_instances_with_points_and_box('nuscene_v1.0-trainval_val_with_maskv1.0', json_val01_path, img_root)


'''
human annotation results
'''
# h1 = "refine_nuscene_v1.0-mini_val_with_maskv1.0.json"
# h1_path = os.path.join(json_root, h1)
# register_nuscenes_instances_with_points_and_box("refine_nuscene_v1.0-mini_val_with_maskv1.0", h1_path, img_root)


h2 = "tbox_refine_nuscene_v1.0-trainval_val_v1.0.json"
h2_path = os.path.join(json_root, h2)
register_nuscenes_instances_with_points_and_box("tbox_refine_nuscene_v1.0-trainval_val_v1.0", h2_path, img_root)



#tight box
# h6 = "refine2_nuscene_v1.0-mini_train_with_maskv1.0_tight_box.json"
# h6_path = os.path.join(json_root, h6)
# register_nuscenes_instances_with_points_and_box("refine2_nuscene_v1.0-mini_train_with_maskv1.0_tight_box", h6_path, img_root)

# h7 = "refine2_nuscene_v1.0-mini_val_with_maskv1.0_tight_box.json"
# h7_path = os.path.join(json_root, h7)
# register_nuscenes_instances_with_points_and_box("refine2_nuscene_v1.0-mini_val_with_maskv1.0_tight_box", h7_path, img_root)

#3d point

h8 = "tbox_refine2_nuscene_v1.0-trainval_train_v4.1.json"
h8_path = os.path.join(json_root, h8)
register_nuscenes_instances_with_points_and_box("tbox_refine2_nuscene_v1.0-trainval_train_v4.1", h8_path, img_root)

h9 = "tbox_refine2_nuscene_v1.0-mini_val_v4.1.json"
h9_path = os.path.join(json_root, h9)
register_nuscenes_instances_with_points_and_box("tbox_refine2_nuscene_v1.0-mini_val_v4.1", h9_path, img_root)

h10 = "tbox_refine2_nuscene_v1.0-mini_train_v4.1.json"
h10_path = os.path.join(json_root, h10)
register_nuscenes_instances_with_points_and_box("tbox_refine2_nuscene_v1.0-mini_train_v4.1", h10_path, img_root)


