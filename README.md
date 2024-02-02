# Waypoint Predictor Training for Discrete-Continuous-VLN

## Prerequisites

1. Please follow [Discrete-Continuous-VLN](https://github.com/YicongHong/Discrete-Continuous-VLN) to set up your environments, prepare scene dataset of MP3D, download the adapted mp3d connectivity graphs, and the pretrained ddppo ResNet encoder. Data and model path should be similar to Discrete-Continuous VLN. Download the adapted mp3d graphs from [here](https://drive.google.com/drive/folders/1wpuGAO-rRalPKt8m1-QIvlb_Pv1rYJ4x?usp=sharing).

2. Change the data path `/home/vlnce/vln-ce/data/` in the codes to your above data path.
3. Change the `RAW_GRAPH_PATH` in the codes to your unzipped adapted mp3d connectivity graphs.

## Preparing Training Data

1. Run `gen_training_data/get_images_inputs.py` to get the RGBD inputs of the waypoint predictor, which will be saved at `training_data/rgbd_fov90`.
2. Run `gen_training_data/get_nav_dict.py` to get the computed navigability dict of each node, which will be saved at `gen_training_data/nav_dicts`.
3. Run `gen_training_data/test_twm0.2_obstacle_first.py` to get the direct training data for training waypoint predictor, which will be saved at `training_data`.

## Running

### Training and Evaluation

Please run `bash run_waypoint.bash` to train the waypoint predictor. If you only want to evaluate trained model, change `--TRAINEVAL` to `eval`. Modify the `checkpoint_load_path` in `waypoint_predictor.py` to evaluate different models.


## Citation
Please cite our paper:
```
@InProceedings{Hong_2022_CVPR,
    author    = {Hong, Yicong and Wang, Zun and Wu, Qi and Gould, Stephen},
    title     = {Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```
