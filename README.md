# Multi-V2X | CoRTSG 
This repository contains the official PyTorch implementations of training and testing of:
- **Multi-V2X**: A large scale, multi-modal, multi-penetration-rate dataset for operative perception. Learn more [here](./docs/md_files/README_multiv2x.md).
- **CoRTSG**: The first driving safety-oriented testing scenario generation framework for cooperative perception in V2X environment. The results cover 11 risky functional scenarios and 17,490 concrete scenarios. Learn more [here](https://github.com/RadetzkyLi/CoRTSG). 

## Features

- Dataset Support
    - [x] OPV2V
    - [x] V2XSet
    - [x] Multi-V2X
    - [x] CoRTSG
    - [x] V2V4Real
    - [ ] DAIR-V2X

- SOTA cooperative perception methods support
    - [x] [Where2comm [NeurIPS2022]](https://arxiv.org/abs/2209.12836)
    - [x] [V2X-ViT [ECCV2020]](https://arxiv.org/abs/2008.07519)
    - [x] Late Fusion
    - [x] Early Fusion

- Intensity Simulation
    - [x] CARLA's default point cloud intensity simulation (so as to directly apply models trained with xyzi-channel point cloud to xyz-channel point cloud)


## Quick Start

### Install
Please refer to the [installation.md](./docs/md_files/installation.md) for detailed documentations.

### Download datasets
Download one or more of the following datasets:
- **OPV2V** in [google drive](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu)
- **V2XSet** in [google drive](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6?usp=sharing)
- **Multi-V2X** in [OpenDataLab](https://opendatalab.com/Rongsong/Multi-V2X) (search "Multi-V2X" in datasets plate)
- **CoRTSG** in [OpenDataLab](https://opendatalab.com/Rongsong/CoRTSG) (search "CoRTSG" in datasets plate)

### Train your model
We adopt the similar setting as [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) which uses yaml file to configure all the parameters for training. To train your own model from scratch or a continued checkpoint, run the following commands:
```bash
cd OpenCOOD
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```
Arguments explanation:
- `hypes_yaml`: the path of the training configuration file, e.g., `opencood/hypes_yaml/early_fusion.yaml`. 
    - To train models on [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/), [V2XSet](https://github.com/DerrickXuNu/v2x-vit) and [V2V4Real](https://github.com/ucla-mobility/V2V4Real), see [Tutorial 1: Config System](./docs/md_files/config_tutorial.md) to learn more. 
    - To train models on [Multi-V2X](./docs/md_files/README_multiv2x.md), see [Tutorial 1: Config System (Multi-V2X)](./docs/md_files/README_multiv2x.md) to learn more.
- `model_dir` (optional): the path of checkpoints. This is used for fine-tuning the trained models. When the `model_dir` is given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.

To train on **multiple gpus**, run:
```bash
cd OpenCOOD
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```

### Test your model

run:
```bash
cd OpenCOOD
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --dataset_format ${DATASET_FORMAT} [--dataset_root ${DATASET_ROOT}]
```
Arguments explanation:
- `model_dir`: the path of the checkpoints.
- `fusion_method`: `"no"`, `"late"`, `"early"` and `"intermediate"` supported.
- `dataset_format`: `"test"`, `"opv2v"` and `"multi-v2x"` supported. 
    - `"opv2v"`: used for **OPV2V**, **V2XSet** and **V2V4Real**
    - `"multi-v2x"`: used for **Multi-V2X**
    - `"test"`: used for **CoRTSG**.
- `dataset_root` (optional): the folder of your dataset. If set, `root_dir` in `config.yaml` would be overwrited. For testing on **CoRTSG**, you should specify the directory of a functional scenario as `dataset_root`.


## Acknowledgement
Thanks for the excellent cooperative perception codebase [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD).

## Contact
If you have any problem with this code, feel free to open an issue.

## Citation
If you find the [Multi-V2X](./docs/md_files/README_multiv2x.md) dataset useful in your research, feel free to cite:
```Bibtex
@article{rongsong2024multiv2x,
      title={Multi-V2X: A Large Scale Multi-modal Multi-penetration-rate Dataset for Cooperative Perception}, 
      author={Rongsong Li and Xin Pei},
      year={2024},
      eprint={2409.04980},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.04980}, 
}
```

If you find the [CoRTSG](https://github.com/RadetzkyLi/CoRTSG) useful in your research, feel free to cite:
```Bibtex
@article{rongsong2024cortsg,
    title={CoRTSG: A general and effective framework of risky testing scenario generation for cooperative perception in mixed traffic}, 
    author={Rongsong Li and Xin Pei and Lu Xing},
    year={2024}
}
```