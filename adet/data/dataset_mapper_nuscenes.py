# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Union
import torch

import detectron2.data.detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as T
from detectron2.config import configurable
from detectron2.structures import BoxMode
from .detection_utils_nuscenes import annotations_to_instances, transform_instance_annotations
from .detection_utils import build_augmentation
from .augmentation import RandomCropWithInstance

__all__ = ["LidarSupDatasetMapper",]

logger = logging.getLogger(__name__)

def random_sample_balance(sample_num, pc_inbox, pc_outbox, bbox):
    xc, yc, w, h = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]

    # padding or down-sampling to same size
    if pc_inbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=False)
    elif pc_inbox.shape[0] == 0:
        pc_inbox = (np.random.rand(sample_num, 2) - 0.5) * min(w, h) / 2 + np.array([xc, yc])
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=False)
    else:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=True)
    pc_inbox = pc_inbox[idx, :]

    if pc_outbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=False)
    elif pc_outbox.shape[0] == 0:
        pc_outbox = np.random.rand(sample_num, 2) * min(w, h) / 5 + np.array([bbox[0], bbox[1]])
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=False)
    else:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=True)
    pc_outbox = pc_outbox[idx, :]

    pc_inbox_label = np.concatenate((pc_inbox[:, :2], np.ones((pc_inbox.shape[0], 1))), axis=1)
    pc_outbox_label = np.concatenate((pc_outbox[:, :2], np.zeros((pc_outbox.shape[0], 1))), axis=1)

    pc_with_label = np.concatenate((pc_inbox_label, pc_outbox_label), axis=0)
    np.random.shuffle(pc_with_label)
    point_coords = pc_with_label[:, :2]
    point_label = pc_with_label[:, 2]
    return point_coords, point_label

class LidarSupDatasetMapper(DatasetMapper):
    """
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            sample_points: subsample points at each iteration
        """
        self.sample_points = cfg.INPUT.SAMPLE_POINTS
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.use_3dpoints = cfg.INPUT.USE_3DPOINTS
        self.sample_num = cfg.INPUT.SAMPLE_NUM
        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {self.augmentation}")
        logger.info(f"Point Augmentations used in {mode}: sample {self.sample_points} points")
        
        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict


        if "annotations" in dataset_dict:
            # Maps points from the closed interval [0, image_size - 1] on discrete
            # image coordinates to the half-open interval [x1, x2) on continuous image
            # coordinates. We use the continuous-discrete conversion from Heckbert
            # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
            # where d is a discrete coordinate and c is a continuous coordinate.
            for ann in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    ann.pop("segmentation", None)
                #To use denser pla points 
                # lidar_in_box = np.array(ann["3din_coords"])
                # lidar_out_box = np.array(ann["3dout_coords"])
                # N_in, N_out = lidar_in_box.shape[0], lidar_out_box.shape[0]
                # if N_in != 0:
                #     lidar_in_box = lidar_in_box[:, 3:]
                # if N_out != 0:
                #     lidar_out_box = lidar_out_box[:, 3:]
                # point_coords, point_labels = random_sample_balance(50, lidar_in_box, lidar_out_box, np.array(ann["bbox"]))
                # ann["point_coords"], ann["point_labels"] = point_coords, point_labels

                point_coords_wrt_image = np.array(ann["point_coords"]).astype(np.float)
                point_coords_wrt_image = point_coords_wrt_image + 0.5
                ann["point_coords"] = point_coords_wrt_image

                # sample point num
                if self.sample_num < len(ann["point_coords"]):
                    ann["point_coords"] = ann["point_coords"][:self.sample_num]
                    ann["point_labels"] = ann["point_labels"][:self.sample_num]           

            annos = [
                # also need to transform point coordinates
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    self.use_3dpoints,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos,
                image_shape,
                self.sample_points,
                self.use_3dpoints,
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            #dataset_dict["instances"] = instances
        return dataset_dict
