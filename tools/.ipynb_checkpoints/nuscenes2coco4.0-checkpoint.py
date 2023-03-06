'''
Time:   2021/9/29 下午9:01
Author: Lixiang
E-mail: 1075099620@qq.com / 3120211007@bit.edu.cn
Project: https://github.com/Serenos/LidarSup
About: 4.0 generate datasets with 3d coords
'''
import copy
import json
from re import M
import numpy as np
import os
import sys
import tqdm
import time
from pyquaternion import Quaternion
from PIL import Image
import matplotlib.pyplot as plt
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points, points_in_box
    from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
except:
    print("nuScenes devkit not Found!")

import json
from pycocotools.coco import COCO
from detectron2.evaluation.fast_eval_api import COCOeval_opt
import pycocotools.mask as mask_util
import copy
import numpy as np
import os
import tqdm

import pycocotools.mask as mask_utils
from detectron2.utils.file_io import PathManager
import tqdm
import matplotlib.pyplot as plt

from utils import random_sample_balance, random_sample, random_sample_balance2, random_sample_balance3, remove_overlap, keep_all

#from nuScenes_viz import nuscene_vis

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

CLASS2ID = {
    'car': 0,
    'truck': 1,
    'trailer': 2,
    'bus': 3,
    'construction_vehicle': 4,
    'bicycle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'barrier': 9
}

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
           'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
           'barrier')


classname_to_index = {  # RGB.
    "noise": 0,  # Black.
    "animal": 1,  # Steelblue
    "human.pedestrian.adult": 2,  # Blue
    "human.pedestrian.child": 3,  # Skyblue,
    "human.pedestrian.construction_worker": 4,  # Cornflowerblue
    "human.pedestrian.personal_mobility": 5,  # Palevioletred
    "human.pedestrian.police_officer": 6,  # Navy,
    "human.pedestrian.stroller": 7,  # Lightcoral
    "human.pedestrian.wheelchair": 8,  # Blueviolet
    "movable_object.barrier": 9,  # Slategrey
    "movable_object.debris": 10,  # Chocolate
    "movable_object.pushable_pullable": 11,  # Dimgrey
    "movable_object.trafficcone": 12,  # Darkslategrey
    "static_object.bicycle_rack": 13,  # Rosybrown
    "vehicle.bicycle": 14,  # Crimson
    "vehicle.bus.bendy": 15,  # Coral
    "vehicle.bus.rigid": 16,  # Orangered
    "vehicle.car": 17,  # Orange
    "vehicle.construction": 18,  # Darksalmon
    "vehicle.emergency.ambulance": 19,
    "vehicle.emergency.police": 20,  # Gold
    "vehicle.motorcycle": 21,  # Red
    "vehicle.trailer": 22,  # Darkorange
    "vehicle.truck": 23,  # Tomato
    "flat.driveable_surface": 24,  # nuTonomy green
    "flat.other": 25,
    "flat.sidewalk": 26,
    "flat.terrain": 27,
    "static.manmade": 28,  # Burlywood
    "static.other": 29,  # Bisque
    "static.vegetation": 30,  # Green
    "vehicle.ego": 31
}

# sample num for a instance
SAMPLE_NUM = 100

# using cam and lidar data
CAM_SENSOR = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
POINT_SENSOR = 'LIDAR_TOP'


def map_point_to_img(nusc, pc, pointsensor, cam):
    img = Image.open(os.path.join(nusc.dataroot, cam['filename']))

    # projection
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    pose_record = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
    pc.translate(np.array(pose_record['translation']))

    pose_record = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(pose_record['translation']))
    pc.rotate(Quaternion(pose_record['rotation']).rotation_matrix.T)

    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    depths = pc.points[2, :]
    coloring = depths

    camera_intrinsic = np.array(cs_record['camera_intrinsic'])
    points = view_points(pc.points[:3, :], camera_intrinsic, normalize=True)

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask].astype(np.int16)
    coloring = coloring[mask]

    return points, coloring, img, mask

def filter_with_2dbox(points, h, w, depth, bbox=None):
    mask = np.ones(points.shape[1], dtype=bool)
    if bbox:
        mask = np.logical_and(mask, depth > 1)
        mask = np.logical_and(mask, points[0, :] > max(bbox[0] + 1, 1))
        mask = np.logical_and(mask, points[0, :] < min(bbox[0] + bbox[2] - 1, w - 1))
        mask = np.logical_and(mask, points[1, :] > max(bbox[1] + 1, 1))
        mask = np.logical_and(mask, points[1, :] < min(bbox[1] + bbox[3] - 1, h - 1))
    else:
        mask = np.logical_and(mask, depth > 1)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < w - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < h - 1)
    points = points[:2, mask]
    depth = depth[mask]
    return points, depth, mask

def split_scene(nusc, version):
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    scene_names = [s['name'] for s in nusc.scene]
    train_scenes = list(filter(lambda x: x in scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in scene_names, val_scenes))
    train_scenes = set([
        nusc.scene[scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        nusc.scene[scene_names.index(s)]['token']
        for s in val_scenes
    ])
    print('train_scenes: ', len(train_scenes))
    print('val_scenes: ', len(val_scenes))
    return train_scenes, val_scenes

def prepare_coco_data(version='v1.0-mini', split='val', dataroot='/home/PJLAB/lixiang/datasets/nuscenes'):
    nusc = NuScenes(version=version, dataroot=dataroot)
    #nusc_vis = nuscene_vis('v1.0-mini', dataroot, nusc)
    train_scenes, val_scenes = split_scene(nusc, version)
    scene = train_scenes if split == 'train' else val_scenes
    # create coco_data for nuScenes
    new_coco_dataset = {}
    # -COCO_CATEGORY
    new_coco_dataset['categories'] = []
    for i, name in enumerate(CLASSES):
        new_coco_dataset['categories'].append(
            {
                'id': i,
                'name': name,
                'supercategory': 'mark'
            }
        )
    new_coco_dataset['images'] = []
    new_coco_dataset['annotations'] = []
    ann_index = 0
    for sample_index, sample_record in enumerate(tqdm.tqdm(nusc.sample)):
        if sample_record['scene_token'] not in scene:
            continue
        # -COCO_IMG: a sample includes 6 img from different CAM_SENSOR
        for index, cam in enumerate(CAM_SENSOR):
            cam_data_token = sample_record['data'][cam]
            cam_data = nusc.get('sample_data', cam_data_token)
            img_path = cam_data['filename']
            full_img_path = os.path.join(nusc.dataroot, img_path)
            height, width = cam_data['height'], cam_data['width']

            coco_image = {
                'file_name': img_path,
                'height': height,
                'width': width,
                'id': 6 * sample_index + index
            }
            new_coco_dataset['images'].append(coco_image)

        point_data_token = sample_record['data'][POINT_SENSOR]
        point_data = nusc.get('sample_data', point_data_token)
        lidar_path = point_data['filename']
        full_lidar_path = os.path.join(nusc.dataroot, lidar_path)
        pc = LidarPointCloud.from_file(full_lidar_path)
        fliter_ann_num = 0
        for j, cam in enumerate(CAM_SENSOR):

            camera_token = sample_record['data'][cam]
            cam_data = nusc.get('sample_data', camera_token)
            pc_temp = copy.deepcopy(pc)
            #lidarseg_temp = copy.deepcopy(lidarseg_sample)
            point2d, coloring, img, mask = map_point_to_img(nusc, pc_temp, point_data, cam_data)
            ### vis
            # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            # ax.imshow(img)
            # fig3d = plt.figure()
            # ax3d = fig3d.add_subplot(111, projection='3d')
            ###
            remove_overlap_flag = 1
            if remove_overlap_flag:
                #point3d, lidarseg = pc.points[:, mask], lidarseg_temp[mask]
                point3d = pc.points[:, mask]
                depth_map = np.zeros((img.size[1], img.size[0]))
                loc2index = np.zeros((img.size[1], img.size[0]), dtype=int)
                point2d = point2d[:2, :].astype(int)
                depth_map[point2d[1, :], point2d[0, :]] = coloring
                loc2index[point2d[1, :], point2d[0, :]] = [i for i in range(point2d.shape[1])]

                refine_depth_map = copy.deepcopy(depth_map)
                refine_depth_map = remove_overlap(depth_img=refine_depth_map)

                mask = np.ones(point2d.shape[1])
                temp = np.logical_and(depth_map > 0, refine_depth_map == 0)
                fliter_loc = temp.nonzero()
                points_index = loc2index[fliter_loc]
                mask[points_index] = 0
                mask = mask.astype(np.bool8)
                #pc2d, pc3d, depth, lidarseg = point2d[:, mask], point3d[:, mask], coloring[mask], lidarseg[mask]
                pc2d, pc3d, depth = point2d[:, mask], point3d[:, mask], coloring[mask]
            else:    
                point3d = pc.points[:, mask]
                pc2d, pc3d, depth = point2d, point3d, coloring
            data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_data['token'], box_vis_level=1)
            for box in boxes:
                if box.name not in NameMapping:
                    continue
                box_coord = view_points(box.corners(), camera_intrinsic, normalize=True).T[:, :2].tolist()
                final_coord = post_process_coords(box_coord)
                min_x, min_y, max_x, max_y = final_coord
                bbox = [min_x, min_y, max_x-min_x, max_y-min_y]

                # pointcloud
                _, box_lidar_frame, _ = nusc.get_sample_data(sample_record['data'][POINT_SENSOR], selected_anntokens=[box.token])
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
                point_coords, point_label = random_sample_balance(SAMPLE_NUM//2, pc2d_inbox, pc2d_outbox, bbox)

                coco_ann = {
                    'area': height * width,
                    'image_id': 6 * sample_index + CAM_SENSOR.index(cam),
                    'bbox': bbox,
                    'category_id': CLASS2ID[NameMapping[box.name]],
                    'id': ann_index,
                    'point_coords': point_coords,
                    'point_labels': point_label,
                    # '3din_coords': pc3d_inbox.T.tolist(),
                    # '3dout_coords': pc3d_outbox.T.tolist(),
                    #'box_3d': box_lidar_frame.corners().tolist(),
                    # extra info to link to origin nuscenes datasets
                    # 'sample_token': sample_record['token'],
                    # 'cam': cam,
                    # 'ann_token': box.token,
                }
                ann_index += 1
                new_coco_dataset['annotations'].append(coco_ann)

    print('found {} categories, {} images, {} annotations.'.format(len(new_coco_dataset['categories']), len(new_coco_dataset['images']), len(new_coco_dataset['annotations'])))
    return new_coco_dataset

class nuscenes_matcher:
    def __init__(self, input):
        self.input = input
        self.json_root = os.path.join(os.environ['HOME'], 'datasets/nuscenes/annotations')
        self.nus_without_segm_path = os.path.join(self.json_root, input)
        self.pseudo_segm_path = os.path.join(self.json_root, 'refine_nuscenes_train.segm.json')
        bitmask = np.zeros((900,1600), order='F', dtype='uint8')
        self.encode_mask = mask_util.encode(bitmask)
        if isinstance(self.encode_mask['counts'], bytes):
            self.encode_mask['counts'] = self.encode_mask['counts'].decode()
        self.coco_gt = COCO(self.nus_without_segm_path)
        for ann in self.coco_gt.anns:
            self.coco_gt.anns[ann]['iscrowd'] = 0
        coco_dt = self.coco_gt.loadRes(self.pseudo_segm_path)
        iou_type = 'bbox'
        self.coco_eval = COCOeval_opt(self.coco_gt, coco_dt, iou_type)
        p = self.coco_eval.params
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))

        self.coco_eval.params = p
        self.coco_eval._prepare()
        catIds = p.catIds
        computeIoU = self.coco_eval.computeIoU
        self.coco_eval.ious = {
            (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
        }
        self.ious = copy.deepcopy(self.coco_eval.ious)

    def match(self):
        gt_annos = self.coco_gt.dataset['annotations']
        valid_flag_list = []
        #for image level
        for img_id in tqdm.tqdm(self.coco_gt.getImgIds()):
            anns_per_img_index = self.coco_gt.getAnnIds(img_id)
            anns_per_img = [gt_annos[i] for i in anns_per_img_index]
            cats = set([x['category_id'] for x in anns_per_img])

            for cat in enumerate(cats):
                anns_index = self.coco_gt.getAnnIds(img_id, cat[1])
                anns = [gt_annos[i] for i in anns_index]
                pseudo_gt_ann = self.coco_eval._dts[img_id, cat[1]]
                iou = self.ious[img_id, cat[1]]
                areas = np.asarray([x['bbox'][2]*x['bbox'][3] for x in anns])
                sorted_idxs = np.argsort(-areas).tolist()
                #for annos level
                match_list = []
                for idx in sorted_idxs:
                    gt_ann = anns[idx]
                    if len(iou) == 0:
                        match_list.append(-1)
                        continue
                    if np.max(iou[:,idx]) < 0.1:
                        match_list.append(-1)
                        continue
                    match_ind = np.argmax(iou[:,idx], axis=0)
                    if match_ind in match_list:
                        match_list.append(-1)
                        continue
                    # if pseudo_gt_ann[match_ind]['score'] < 0.3:
                    #     match_list.append(-1)
                    #     continue
                    match_list.append(match_ind)
                
                for i,idx in enumerate(sorted_idxs):
                    if match_list[i] == -1:
                        valid_flag_list.append(False)
                        continue
                    anns[idx]['segmentation'] = pseudo_gt_ann[match_list[i]]['segmentation']
                    anns[idx]['area'] = pseudo_gt_ann[match_list[i]]['area']
                    valid_flag_list.append(True)

        gt_annos = [gt_annos[i] for i in range(len(gt_annos)) if 'segmentation' in gt_annos[i]]
        print('{}/{} ann remaining.'.format(len(gt_annos), len(valid_flag_list)))

        self.coco_gt.dataset['annotations'] = gt_annos
        save_path = os.path.join(self.json_root, 'refine2_'+self.input)
        with open(save_path, 'w') as f:
            json.dump(self.coco_gt.dataset, f)
        return save_path

def get_point_annotations_fromlidar(input_filename, num_points_per_instance):
    with PathManager.open(input_filename, "r") as f:
        coco_json = json.load(f)

    coco_annos = coco_json.pop("annotations")
    coco_points_json = copy.deepcopy(coco_json)

    imgs = {}
    for img in coco_json["images"]:
        imgs[img["id"]] = img

    new_annos = []
    padding_num = 0
    for i, ann in enumerate(tqdm.tqdm(coco_annos)):

        N = len(ann['point_coords'])
        segm = ann["segmentation"]
        t = imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            #exception
            segm = [segm[i] for i in range(len(segm))  if len(segm[i])]

            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif type(segm["counts"]) == list:
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        #calculate lidar accuracy
        #mask = mask_utils.decode(rle)
        new_ann = copy.deepcopy(ann)
        #tight box
        tbox = mask_utils.toBbox(rle)
        new_ann["bbox"] = tbox.tolist()
        #new_ann["point_coords"] = point_coords.tolist()
        #new_ann["point_labels"] = point_labels.tolist()
        new_annos.append(new_ann)

    coco_points_json["annotations"] = new_annos

    print("found {} anns".format(len(new_annos)))
    
    #_root = os.environ['HOME']
    json_root = os.path.join('/cpfs2/shared/public', 'dataset/nuscenes/annotations')
    file_name = os.path.basename(input_filename)
    output_path = os.path.join(json_root, 'tbox_'+file_name)
    with PathManager.open(output_path, "w") as f:
        json.dump(coco_points_json, f)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    return output_path


if __name__ == '__main__':
    version = 'v1.0-trainval' #sys.argv[1]
    split = 'train' #sys.argv[2]
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    # _root = os.environ['HOME']
    # json_root =os.path.join('/cpfs2/shared/public', 'dataset/nuscenes/annotations')
    json_root = os.path.join('/cpfs2/shared/public', 'dataset/nuscenes/annotations')
    if version not in available_vers:
        raise ValueError('unknown version')
    if split not in ['train', 'val']:
        raise ValueError('unknown split')

#     print('1.converting nuscens_{}_{} to coco format......'.format(version, split))
#     dataroot = os.path.join('/cpfs2/shared/public', 'dataset/nuscenes')
#     new_coco_dataset = prepare_coco_data(version=version, split=split, dataroot=dataroot)
#     output_filename = 'nuscene_{}_{}_{}.json'.format(version, split, 'v4.3')

#     out_path = os.path.join(json_root, output_filename)
#     with open(out_path, 'w') as f:
#         json.dump(new_coco_dataset, f)
#     print("{} done.".format(output_filename))

    # print('2.filter block object')
    # nusMatcher = nuscenes_matcher(output_filename)
    # output_filename = nusMatcher.match()
    # print("{} done.".format(output_filename))

    # output_filename = os.path.join(json_root, 'refine_nuscene_{}_{}_{}.json'.format(version, split, 'v1.0')) # for sensebee dataset called refine
    # #output_filename = os.path.join(json_root, 'refine2_nuscene_{}_{}_{}.json'.format(version, split, 'v1.16')) # for pseudo dataset called refine2
    print('3.calculate lidar acc and tight the box')
    output_filename = get_point_annotations_fromlidar(output_filename, 20)
    print("{} done.".format(output_filename))