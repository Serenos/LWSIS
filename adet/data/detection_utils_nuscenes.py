# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch

# fmt: off
from detectron2.data.detection_utils import \
    annotations_to_instances as base_annotations_to_instances
from detectron2.data.detection_utils import \
    transform_instance_annotations as base_transform_instance_annotations

# fmt: on


def annotations_to_instances(annos, image_size, sample_points=0, use_3d_point=0):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width
        sample_points (int): subsample points at each iteration

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_point_coords", "gt_point_labels", if they can be obtained from `annos`.
            This is the format that builtin models with point supervision expect.
    """
    target = base_annotations_to_instances(annos, image_size, 'bitmask')

    # assert "point_coords" in annos[0]
    # assert "point_labels" in annos[0]
    # assert "segmentation" not in annos[0], "Please remove mask annotation"

    if len(annos) and "point_labels" in annos[0]:
        point_coords = []
        point_labels = []
        lidar_in_box_list = []
        lidar_out_box_list = []
        for i, _ in enumerate(annos):
            # Already in the image coordinate system
            point_coords_wrt_image = np.array(annos[i]["point_coords"])
            point_labels_wrt_image = np.array(annos[i]["point_labels"])

            if sample_points > 0:
                random_indices = np.random.choice(
                    point_coords_wrt_image.shape[0],
                    sample_points,
                    replace=point_coords_wrt_image.shape[0] < sample_points,
                ).astype(int)
                point_coords_wrt_image = point_coords_wrt_image[random_indices]
                point_labels_wrt_image = point_labels_wrt_image[random_indices]
                assert point_coords_wrt_image.shape[0] == point_labels_wrt_image.size

            #point_coords_wrt_image, point_labels_wrt_image = lidar_padding2(point_coords_wrt_image, point_labels_wrt_image)
            point_coords.append(point_coords_wrt_image)
            point_labels.append(point_labels_wrt_image)

            if use_3d_point:
                lidar_in_box = np.array(annos[i]["3din_coords"])
                lidar_out_box = np.array(annos[i]["3dout_coords"])
                lidar_in_box, lidar_out_box = lidar_padding(lidar_in_box, lidar_out_box)
                lidar_in_box_list.append(lidar_in_box)
                lidar_out_box_list.append(lidar_out_box)

        point_coords = torch.stack([torch.from_numpy(x) for x in point_coords])
        point_labels = torch.stack([torch.from_numpy(x) for x in point_labels])
        target.gt_point_coords = point_coords
        target.gt_point_labels = point_labels
        if use_3d_point:
            lidar_in_box = torch.stack([torch.from_numpy(x) for x in lidar_in_box_list])
            lidar_out_box = torch.stack([torch.from_numpy(x) for x in lidar_out_box_list])
            target.gt_lidar_in = lidar_in_box
            target.gt_lidar_out = lidar_out_box

    return target

def point_padding(point_coords, point_labels, max_points=100):
    '''
    padding or downsampling the lidarpoints with a maximum number.
    '''
    N = point_coords.shape[0]
    if N >= max_points:
        idx = np.random.choice(N, max_points, replace=False)
        point_coords =  point_coords[idx, :]
        point_labels = point_labels[idx]

    else:
        point_coords_new = np.zeros((max_points, 2))
        point_labels_new = -np.ones(max_points)
        if N == 0:
            return point_coords_new, point_labels_new
        point_coords_new[:N] = point_coords
        point_labels_new[:N] = point_labels
        point_coords =  point_coords_new
        point_labels = point_labels_new

    return point_coords, point_labels

def lidar_padding(lidar_in_box, lidar_out_box, max_points=100):
    '''
    padding or downsampling the lidarpoints with a maximum number.
    '''
    N = lidar_in_box.shape[0]
    dim = 6 #5 or 6
    if N >= max_points:
        idx = np.random.choice(N, max_points, replace=False)
        lidar_in_box =  lidar_in_box[idx, :]
    else:
        lidar_in_box_new = np.zeros((max_points, dim))
        if N != 0:
            lidar_in_box_new[:N] = lidar_in_box
        lidar_in_box = lidar_in_box_new

    N = lidar_out_box.shape[0]
    if N >= max_points:
        idx = np.random.choice(N, max_points, replace=False)
        lidar_out_box =  lidar_out_box[idx, :]
    else:
        lidar_out_box_new = np.zeros((max_points, dim))
        if N != 0:
            lidar_out_box_new[:N] = lidar_out_box
        lidar_out_box = lidar_out_box_new

    return lidar_in_box, lidar_out_box

def transform_instance_annotations(annotation, transforms, image_size, use_3d_point=0):
    """
    Apply transforms to box, and point annotations of a single instance.
    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for points.
    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
    Returns:
        dict:
            the same input dict with fields "bbox", "point_coords", "point_labels"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    annotation = base_transform_instance_annotations(annotation, transforms, image_size)


    # assert "segmentation" not in annotation
    # assert "point_coords" in annotation
    # assert "point_labels" in annotation

    point_coords = annotation["point_coords"]
    if len(point_coords) == 0:
        return annotation

    point_labels = np.array(annotation["point_labels"]).astype(np.float)
    point_coords = np.array(point_coords).astype(np.float)
    point_coords = transforms.apply_coords(point_coords)

    # Set all out-of-boundary points to "unlabeled"
    inside = (point_coords >= np.array([0, 0])) & (point_coords <= np.array(image_size[::-1]))
    inside = inside.all(axis=1)
    point_labels[~inside] = -1

    annotation["point_coords"] = point_coords
    annotation["point_labels"] = point_labels

    if use_3d_point:
        if len(annotation['3din_coords']) == 0:
            return annotation
        lidar_in_box = np.array(annotation['3din_coords']).astype(np.float)
        if lidar_in_box.shape[1] == 6:# [x,y,z,i,u,v]
            lidar_in_box[:, 4:6] = transforms.apply_coords(lidar_in_box[:, 4:6]+0.5)
        elif lidar_in_box.shape[1] == 5: #[x,y,z,u,v]
            lidar_in_box[:, 3:5] = transforms.apply_coords(lidar_in_box[:, 3:5]+0.5)
        annotation['3din_coords'] = lidar_in_box

        if len(annotation['3dout_coords']) == 0:
            return annotation
        lidar_out_box = np.array(annotation['3dout_coords']).astype(np.float)
        if lidar_out_box.shape[1] == 6:
            lidar_out_box[:, 4:6] = transforms.apply_coords(lidar_out_box[:, 4:6]+0.5)
        elif lidar_out_box.shape[1] == 5:
            lidar_out_box[:, 3:5] = transforms.apply_coords(lidar_out_box[:, 3:5]+0.5)
        annotation['3dout_coords'] = lidar_out_box

    return annotation


def annotations_to_instances_all_points(annos, image_size, sample_points=0):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width
        sample_points (int): subsample points at each iteration

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_point_coords", "gt_point_labels", if they can be obtained from `annos`.
            This is the format that builtin models with point supervision expect.
    """
    target = base_annotations_to_instances(annos, image_size)

    assert "point_coords" in annos[0]
    assert "point_labels" in annos[0]
    assert "segmentation" not in annos[0], "Please remove mask annotation"

    if len(annos) and "point_labels" in annos[0]:
        point_coords = []
        point_labels = []
        for i, _ in enumerate(annos):
            # Already in the image coordinate system
            point_coords_wrt_image = np.array(annos[i]["point_coords"])
            point_labels_wrt_image = np.array(annos[i]["point_labels"])

            if sample_points > 0:
                random_indices = np.random.choice(
                    point_coords_wrt_image.shape[0],
                    sample_points,
                    replace=point_coords_wrt_image.shape[0] < sample_points,
                ).astype(int)
                point_coords_wrt_image = point_coords_wrt_image[random_indices]
                point_labels_wrt_image = point_labels_wrt_image[random_indices]
                assert point_coords_wrt_image.shape[0] == point_labels_wrt_image.size

            point_coords.append(point_coords_wrt_image)
            point_labels.append(point_labels_wrt_image)

        point_coords = torch.stack([torch.from_numpy(x) for x in point_coords])
        point_labels = torch.stack([torch.from_numpy(x) for x in point_labels])
        target.gt_point_coords = point_coords
        target.gt_point_labels = point_labels

    return target