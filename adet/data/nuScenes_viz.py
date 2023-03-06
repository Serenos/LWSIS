import argparse
import psutil
import numpy as np
from pyquaternion import Quaternion

#from lyft_dataset_sdk.lyftdataset import LyftDataset
#from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
#from lyft_dataset_sdk.utils.geometry_utils import view_points
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from pypcd import pypcd
#matplotlib inline
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
#import mayavi.mlab as mlab
## pcl related for visualization
#import pcl
#import pcl.pcl_visualization as viewer
## Open3D Related
import open3d as o3d

import time
import pandas as p
import cv2

from detectron2.utils.visualizer import VisImage

# label to detection label
general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

## modified by zdf
## The following is for nuScenes visualization
def get_lidar_points_nuScenes(nusc, token, multisweep=False, sweep_num=10):

    if(multisweep):
        try:
            s_record = nusc.get("sample", token)
            sample_data_token = s_record["data"]["LIDAR_TOP"]
        except:
            sample_data_token = token

        #sample_data_token = token
        sd_record  = nusc.get('sample_data', sample_data_token)

        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        #ref_sd_token = sample_rec['data'][ref_chan]
        #ref_sd_record = nusc.get('sample_data', ref_sd_token)
        pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=sweep_num)
        points = pc.points.transpose()
    else:

        #my_sample   = nusc.sample[fid]
        try:
            s_record = nusc.get("sample", token)
            sample_data_token = s_record["data"]["LIDAR_TOP"]
        except:
            sample_data_token = token
        pointsensor_token = s_record['data']['LIDAR_TOP']
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points.transpose()

    return points


def plot_lidar_with_depth_nuScenes(nusc, token, multisweep, sweep_num, gt_boxes = None, pred_boxes= None, pred_scores = None, pc=None, fid=0):
    '''plot given sample'''
    #print(f'Plotting sample, token: {sample["token"]}')
    #lidar_token = sample["data"]["LIDAR_TOP"]
    if pc is None:
        pc = get_lidar_points_nuScenes(nusc, token, multisweep, sweep_num)
    ## get the bounding boxes
    #my_sample = nusc.sample[fid]
    ## multi sweeps lidar point cloud
    #sample_data_token = my_sample['data']['LIDAR_TOP']

    try:
        s_record = nusc.get("sample", token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = token


    #if (gt_boxes is None):
        #boxes = nusc.get_boxes(sample_data_token)
    data_path, gt_boxes_wt_filter, cam_intrinsic = nusc.get_sample_data(sample_data_token)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Sample {:06d}".format(fid), width=1920, height=1080, left=50, top=50, visible=True)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    ## prepare the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc[:, 0:3].reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(255.0 * np.ones([pc.shape[0], 3], np.float32).reshape(-1, 3))
    vis.add_geometry(point_cloud)

    # prepare the ground truth bounding boxes
    # for box in gt_boxes_wt_filter:
    #     corners = view_points(box.corners(), view=np.eye(3), normalize=False)
    #     corners = corners.T.tolist()
    #     #detection_name = general_to_detection[box.name]
    #     #viewer = draw_gt_boxes3d_open3d([corners.T], viewer, color=(0, 1, 0), Name = detection_name)
    #     lines = [[0, 1], [1, 2], [2, 3], [3, 0],
    #              [4, 5], [5, 6], [6, 7], [7, 4],
    #              [0, 4], [1, 5], [2, 6], [3, 7]]
    #     colors = [[0, 0, 1] for i in range(len(lines))]
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = Vector3dVector(corners)
    #     line_set.lines = Vector2iVector(lines)
    #     line_set.colors = Vector3dVector(colors)
    #     vis.add_geometry(line_set)

    if (gt_boxes is not None):
        for box in gt_boxes:
            corners = view_points(box.corners(), view=np.eye(3), normalize=False)
            corners = corners.T.tolist()
            # detection_name = general_to_detection[box.name]
            # viewer = draw_gt_boxes3d_open3d([corners.T], viewer, color=(0, 1, 0), Name = detection_name)
            lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                     [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            colors = [[0, 1, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)


    if(pred_boxes is not None):
        # prepare the prediction bounding boxes
        #for box in pred_boxes:
        for id in range(len(pred_boxes)):
            box  = pred_boxes[id]

            score = pred_scores[id]
            if(score < 0.1):
                continue

            corners = view_points(box.corners(), view=np.eye(3), normalize=False)
            corners = corners.T.tolist()
            # detection_name = general_to_detection[box.name]
            # viewer = draw_gt_boxes3d_open3d([corners.T], viewer, color=(0, 1, 0), Name = detection_name)
            lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                     [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)

    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(2)
    # vis.capture_screen_image("/home/junbo/data/nuscenes/v1.0-mini/D1/visual/{}.png".format(token))
    # plt.imshow(np.asarray(image))
    # plt.imsave(, \
    #            np.asarray(image))


    vis.run()
    vis.destroy_window()


def plot_one_sample_nuScenes(nusc, token, multisweep, sweep_num, gt_boxes= None, pred_boxes = None, pred_scores = None):
    ''' plots only one sample's top lidar point cloud '''
    ## get the bounding boxes
    plot_lidar_with_depth_nuScenes(nusc, token, multisweep, sweep_num, gt_boxes, pred_boxes, pred_scores)

class nuscene_vis:
    def __init__(self, version, dataroot, nusc=None) -> None:
        if nusc:
            self.nusc = nusc
        else:
            self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.dataroot = dataroot
    def plot_scene_withbox(self, token, gt_boxes):
        try:
            s_record = self.nusc.get("sample", token)
            sample_data_token = s_record["data"]["LIDAR_TOP"]
        except:
            sample_data_token = token
        pointsensor_token = s_record['data']['LIDAR_TOP']
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        pc = pc.points.transpose()

        data_path, gt_boxes_wt_filter, cam_intrinsic = self.nusc.get_sample_data(sample_data_token)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Sample {}".format(token), width=1920, height=1080, left=50, top=50, visible=True)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 2.0

        ## prepare the point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc[:, 0:3].reshape(-1, 3))
        point_cloud.colors = o3d.utility.Vector3dVector(255.0 * np.ones([pc.shape[0], 3], np.float32).reshape(-1, 3))
        vis.add_geometry(point_cloud)


        if (gt_boxes is not None):
            for box in gt_boxes:
                corners = view_points(box.corners(), view=np.eye(3), normalize=False)
                corners = corners.T.tolist()
                # detection_name = general_to_detection[box.name]
                # viewer = draw_gt_boxes3d_open3d([corners.T], viewer, color=(0, 1, 0), Name = detection_name)
                lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                        [4, 5], [5, 6], [6, 7], [7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]]
                colors = [[0, 1, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)

        # vis.poll_events()
        # vis.update_renderer()
        # time.sleep(2)
        # vis.capture_screen_image("/home/junbo/data/nuscenes/v1.0-mini/D1/visual/{}.png".format(token))
        # plt.imshow(np.asarray(image))
        # plt.imsave(, \
        #            np.asarray(image))

        vis.run()
        vis.destroy_window()
        

    def plot_image_withbox(self, img, pc3d_inbox, pc3d_outbox, bbox):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(img)
        ax.scatter(np.array(pc3d_inbox)[4, :], np.array(pc3d_inbox)[5, :], c='r', s=10)
        ax.scatter(np.array(pc3d_outbox)[4, :], np.array(pc3d_outbox)[5, :], c='b', s=5)
        ax.axis('off')
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        plt.show()
        plt.close()

    def cv_image_withbox(self, img, pc3d_inbox, pc3d_outbox, bbox):
        img = np.array(img)[:,:,::-1]
        output = VisImage(img, scale=1)
        output.ax.add_patch(
            mpl.patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor=np.array([0.857, 0.857, 0.857]),
                # linewidth=linewidth * self.output.scale,
                # alpha=alpha,
                # linestyle=line_style,
            )
        )
        output.ax.scatter(np.array(pc3d_inbox)[4, :], np.array(pc3d_inbox)[5, :], c='r', s=10)
        output.ax.scatter(np.array(pc3d_outbox)[4, :], np.array(pc3d_outbox)[5, :], c='b', s=5)
        image = output.get_image()
        cv2.imshow("window", image)
        cv2.waitKey()
## read key
import sys
import tty
import termios

def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

from pathlib import Path

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Mayavi visualization of nuscenes dataset')
    parser.add_argument('-d', '--dataroot', type=str, default="/home/lixiang/datasets/nuscenes/", metavar='N',
                        help='data directory path  (default: ./data/nuScenes/)')
    #parser.add_argument('--scene', type=str, default=None, metavar='N', help='scene token')
    #parser.add_argument('--sample', type=str, default=None, metavar='N', help='sample token')
    parser.add_argument('--version', type=str, default="v1.0-mini", metavar='N', help='v1.0-trainval or v1.0-mini')
    args = parser.parse_args()
    #dataroot = Path(args.dataroot)
    #json_path = dataroot / 'data/'
    print('Loading dataset with nuScenes SDK ...')
    #root_folder = "/home/zhou/nuscenes/"
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    multisweep = False
    sweep_num  = 3
    token = 'e6e877f31dd447199b56cae07f86daad'
    plot_one_sample_nuScenes(nusc, token, multisweep, sweep_num, None ,None, None)
