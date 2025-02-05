# Introduction

This is a project to train on 3D Vision, The datasets used will mainly be Teeth3Ds and ShapeNet

## usage

## Implemented Models Results

[Trained models can be found here](https://drive.google.com/drive/folders/1HkosNZrBiGC8gFM38o_f9LQ0fARovoV5?usp=drive_link)

| Model  | Accuracy | IoU |
| ------ | -------- | --- |
| PointNet| ![Accuracy](./images/accuracy_plot.png) | ![IoU](./images/IoU_plot.png) |

## papers that is planned to be implemented :
- [x] PointNet
- [ ] PointNet++
- [ ] DGCNN
- [ ] Point-MAE
- [ ] Shape As Points

PointNet and PointNet++ structre was inspired by [yanx27 repo](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master)


#TODO:
- [ ] Make backend
- [ ] Make a simple viewer
- [ ] use [fpsample](https://pypi.org/project/fpsample/#:~:text=Python%20efficient%20farthest%20point%20sampling,in%20single%2Dthreaded%20CPU%20environment.) for deployment
