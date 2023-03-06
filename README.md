# LWSIS
An official implementation of AAAI2023 paper "LWSIS: LiDAR-guided Weakly Supervised Instance Segmentation for Autonomous Driving"

# Models
|Model|Backbone|Annotations|Lr_schedule|Mask_AP|Download|
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|[BoxInst](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html)|R-50|box|1x |33.65|[link](https://cloud.189.cn/t/viI3EbQNrAvu)(访问码：pmw0)|
|BoxInst|R-101|box|1x|34.39|link|
|[PointSup](https://arxiv.org/abs/2104.06404)|R-50|box+point|1x  |43.80|link|
|PointSup|R-101|box+point|1x |44.72|link|
|LWSIS+BoxInst|R-50|3dbox+pc|1x   |35.65|[link](https://cloud.189.cn/t/U7juMz6vi2Af)(访问码：hy6a)|
|LWSIS+BoxInst|R-101|3dbox+pc|1x  |36,22|link|
|LWSIS+PointSup|R-50|3dbox+pc|1x   |45.46|link|
|LWSIS+PointSup|R-101|3dbox+pc|1x  |46.17|link|

Here we explain different annotations used in the exp. 'box' means only using the 2D bounding box annotation for each instance, 'point' means using a specific number of points with human annotation indicating the background/foreground, '3dbox' means using the 3D bounding box annotations for each instance and 'pc' means the original point cloud.

# Install
First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build LWSIS with:

```
git clone git@github.com:Serenos/LWSIS.git
cd LWSIS
python setup.py build develop
```


# Quick Start
- Download the nuscenes origin datasets to ${HOME}/datasets/. The folder structure shall be like this:
    - nuscenes
        - annotations
        - lidarseg
        - maps
        - samples
        - sweeps
        - v1.0-trainval
        - v1.0-mini
- Download [nuInsSeg3d_train](https://cloud.189.cn/t/QZRzqiiEJ7ri)(访问码：4aml), [nuInsSeg3d_val](https://cloud.189.cn/t/FRzYNvQbIbui)(访问码：luw8) and put it into the nuscenes/annotations folder.

- Training
    ```
   bash tools/train.sh configs/BoxInst/MS_R_50_1x_nuscenes.yaml Boxinst_LWSIS 000
    ```

- Evaluation
    ```
    bash tools/test.sh configs/BoxInst/MS_R_50_1x_nuscenes.yaml output/Boxinst_LWSIS/000/model_final.pth 
    ```

# nuInsSeg Dataset and devkit
We supplement instance mask annotation for nuScenes dataset. For more detail, please follow the [nuinsseg-devkit](https://github.com/Serenos/nuInsSeg).

## Acknowledgements

The authors are grateful to School of Computer Science, Beijing Institute of Technology, Shanghai AI Laboratory, Inceptio, 
4SKL-IOTSC, CIS, University of Macau.

The code is based on [Adlaidet](https://github.com/aim-uofa/AdelaiDet/).