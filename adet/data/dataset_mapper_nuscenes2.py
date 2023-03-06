# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from typing import List, Union
import torch
import os
import detectron2.data.detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as T
from detectron2.config import configurable
from detectron2.structures import BoxMode
from .detection_utils_nuscenes import annotations_to_instances, transform_instance_annotations
from .detection_utils import build_augmentation
from .augmentation import RandomCropWithInstance


try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points, points_in_box
    from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
except:
    print("nuScenes devkit not Found!")

import matplotlib.pyplot as plt
from .utils import random_sample_balance, remove_overlap, filter_with_2dbox, map_point_to_img
#from .nuScenes_viz import nuscene_vis
import time

__all__ = ["LidarSupDatasetMapper2",]

logger = logging.getLogger(__name__)


class LidarSupDatasetMapper2(DatasetMapper):
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
        if self.use_3dpoints:
            self.nus_dataroot = os.path.join(os.environ['HOME'], 'datasets/nuscenes')
            self.nusc = NuScenes(version='v1.0-mini', dataroot=self.nus_dataroot)
            #self.nus_vis = nuscene_vis('v1.0-mini', self.nus_dataroot, self.nusc)
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
        tic = time.time()
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

        if self.use_3dpoints:
            
            '''load and process lidar annotation'''
            ann_perimg = dataset_dict["annotations"]
            sample_token, cam = ann_perimg[0]['sample_token'], ann_perimg[0]['cam']
            sample_record = self.nusc.get("sample", sample_token)
            CAM_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            POINT_SENSOR = 'LIDAR_TOP'
            point_data_token = sample_record['data'][POINT_SENSOR]
            point_data = self.nusc.get('sample_data', point_data_token)
            lidar_path = point_data['filename']
            full_lidar_path = os.path.join(self.nusc.dataroot, lidar_path)
            pc = LidarPointCloud.from_file(full_lidar_path)
            pc_temp = copy.deepcopy(pc)

            camera_token = sample_record['data'][cam]
            cam_data = self.nusc.get('sample_data', camera_token)
            
            point2d, coloring, img, mask = map_point_to_img(self.nusc, pc_temp, point_data, cam_data)
            ## vis
            # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            # ax.imshow(img)
            '''remove overlap'''
            #point3d, lidarseg = pc.points[:, mask], lidarseg_temp[mask]
            # point3d = pc.points[:, mask]
            # depth_map = np.zeros((img.size[1], img.size[0]))
            # loc2index = np.zeros((img.size[1], img.size[0]), dtype=int)
            # point2d = point2d[:2, :].astype(int)
            # depth_map[point2d[1, :], point2d[0, :]] = coloring
            # loc2index[point2d[1, :], point2d[0, :]] = [i for i in range(point2d.shape[1])]

            # refine_depth_map = copy.deepcopy(depth_map)
            # refine_depth_map = remove_overlap(depth_img=refine_depth_map)

            # mask = np.ones(point2d.shape[1])
            # temp = np.logical_and(depth_map > 0, refine_depth_map == 0)
            # fliter_loc = temp.nonzero()
            # points_index = loc2index[fliter_loc]
            # mask[points_index] = 0
            # mask = mask.astype(np.bool8)
            # #pc2d, pc3d, depth, lidarseg = point2d[:, mask], point3d[:, mask], coloring[mask], lidarseg[mask]
            # pc2d, pc3d, depth = point2d[:, mask], point3d[:, mask], coloring[mask]
            point3d = pc.points[:, mask]
            pc2d, pc3d, depth = point2d, point3d, coloring
            for ann in ann_perimg:
                ann_token = ann['ann_token']
                # box_coord = view_points(box.corners(), camera_intrinsic, normalize=True).T[:, :2].tolist()
                # final_coord = post_process_coords(box_coord)
                # min_x, min_y, max_x, max_y = final_coord
                # bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
                bbox = ann['bbox']
                # pointcloud
                _, box_lidar_frame, _ = self.nusc.get_sample_data(sample_record['data'][POINT_SENSOR], selected_anntokens=[ann_token])
                box_lidar_frame = box_lidar_frame[0]

                logits = points_in_box(box_lidar_frame, pc3d[:3, :])

                pc2d_inbox = copy.deepcopy(pc2d)
                pc2d_outbox = copy.deepcopy(pc2d)
                pc2d_inbox = pc2d_inbox[:, logits]
                pc2d_outbox = pc2d_outbox[:, ~logits]

                depth_inbox = copy.deepcopy(depth)
                depth_outbox = copy.deepcopy(depth)
                depth_inbox = depth_inbox[logits]
                depth_outbox = depth_outbox[~logits]
                
                pc3d_inbox = copy.deepcopy(pc3d)
                pc3d_inbox = pc3d_inbox[:, logits]
                pc3d_outbox = copy.deepcopy(pc3d)
                pc3d_outbox = pc3d_outbox[:, ~logits]

                pc2d_inbox, depth_inbox, maskin = filter_with_2dbox(pc2d_inbox, img.size[1], img.size[0], depth_inbox)
                pc2d_outbox, depth_outbox, maskout = filter_with_2dbox(pc2d_outbox, img.size[1], img.size[0], depth_outbox, bbox)
                pc3d_inbox = pc3d_inbox[:, maskin]
                pc3d_outbox = pc3d_outbox[:, maskout]
                pc3d_inbox = np.concatenate((pc3d_inbox, pc2d_inbox), axis=0)
                pc3d_outbox = np.concatenate((pc3d_outbox, pc2d_outbox), axis=0)

                # random sample SAMPLE_NUM points for pos/neg
                #point_coords, point_label = random_sample_balance(10, pc2d_inbox, pc2d_outbox, bbox)
                
                ann['3din_coords'] = pc3d_inbox.T.tolist()
                ann['3dout_coords'] = pc3d_outbox.T.tolist()
                #visualize = 1
                # if visualize:
                #     self.nus_vis.cv_image_withbox(img, pc3d_inbox, pc3d_outbox, bbox)
                    #self.nus_vis.plot_scene_withbox(sample_record['token'], [box_lidar_frame])
        toc = time.time()
        print('load and process lidar info consume {}'.format(toc-tic))
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
                point_coords_wrt_image = np.array(ann["point_coords"]).astype(np.float)
                point_coords_wrt_image = point_coords_wrt_image + 0.5
                ann["point_coords"] = point_coords_wrt_image
            

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


