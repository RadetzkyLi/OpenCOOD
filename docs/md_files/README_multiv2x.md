# Multi-V2X Dataset

More details for Multi-V2X in https://github.com/RadetzkyLi/Multi-V2X .

Multi-V2X is a **large-scale**, **multi-modal**, **multi-penetration-rate** dataset for **cooperative perception** under vehicle-to-everything (**V2X**) environment. Multi-V2X is gathered by SUMO-CARLA co-simulation and supports tasks including 3D object detection and tracking. Multi-V2X provides RGB images, point clouds from CAVs and RSUs with various CAV penetration rates (up to 86.21%).

**Features**:
- **Multiple Penetration Rates**: nearly all cars in a town are equipped with sensors (thus can be CAV). By masking some equipped cars as normal vehicles, datasets of various penetration rates can be generated.
- **Multiple Categories**: **6** categories: car, van, truck, cycle, motorcycle, pedestrian, covering the common traffic participants. As comparison, the well-know OPV2V, V2XSet and V2X-Sim contain only car. 
- **Multiple CAV Shapes**: all kinds of cars in CARLA can be CAVs, whereas only lincoln in OPV2V and V2XSet.
- **Multiple Modalities**: RGB images and point clouds from CAVs and RSUs are provided.
- **Dynamic Connections**: CAVs are spawned and running in the whole town and thus connections would be lost and created over time. This is more realistic compared to spawning and running vehicles around a site.


**Comparison with other datasets**
| Dataset              | Year | Type | V2X   | RGB Images | LiDAR | 3D boxes | Classes | Locations                                  | connections |
| -------------------- | ---- | ---- | ----- | ---------- | ----- | -------- | ------- | ------------------------------------------ | ----- |
| DAIR-V2X             | 2022 | Real | V2I   | 71k        | 71k   | 1200k    | 10      | Beijing, China                             | 1     |
| V2V4Real             | 2023 | Real | V2V   | 40k        | 20k   | 240k     | 5       | Ohio, USA                                  | 1     |
| RCooper              | 2024 | Real | I2I   | 50k        | 30k   | -        | 10      | -                                          | -     |
| OPV2V                | 2022 | Sim  | V2V   | 132k       | 33k   | 230k     | 1       | CARLA Town01, 02, 03, 04, 05, 06, 07, 10HD | 1-6   |
| V2XSet               | 2022 | Sim  | V2V&I | 132K       | 33K   | 230K     | 1       | Same as OPV2V                              | 1-4   |
| V2X-Sim              | 2022 | Sim  | V2V&I | 283K       | 47K   | 26.6K    | 1       | CARLA Town03, 04, 05                       | 1-4   |
| **Multi-V2X** (ours) | 2024 | Sim  | V2V&I | 549k       | 146k  | 4219k    | 6       | CARLA Town01,  03, 05, 06, 07,  10HD       | 0-31  |

**Note**: the data was counted per agent.

---

## Usage
### Downloading
Download **Multi-V2X** from [OpenDataLab](https://opendatalab.org.cn/Rongsong/Multi-V2X) and unzip.

### Annotation
Refer to this [data annotation tutorial](./data_annotation_tutorial_multiv2x.md) to learn about data annotation. **Multi-V2X** holds similar format as [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/).

### Config
There are two configs:
- `yaml` config: In this codebase, some adjustments were applied to [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) so that training on **Multi-V2X** is supported.
You need to do minor change to `yaml` config file referring to [config tutorial](config_tutorial_multiv2x.md).
- **pr** config: to fully utilize **Multi-V2X**, an masking-based algorithm is developed to extract data from Multi-V2X to obtain a sub dataset with specified CAV **p**enetration **r**ate and ego frames. This **pr config**, a `.json` file, specifies CAVs and RSUs that can be accessed in each town for given penetration rate. The egos are also specified in each town, and for each ego, **along time axis**, its data is splited into **train : val : test** by **6 : 2 : 2**. The default pr config locates in `OpenCOOD/opencood/hypes_pr/`. To customize your pr config, see https://RadetzkyLi/Multi-V2X .


## Benchmarks

For lack of cooperative perception algorithms targeted for high CAV penetration rate, by now, we just conducted experiments on $\mathcal{D}^{\text{Multi-V2X}}_{\text{10\%}}$, a V2X dataset with 10% CAV penetration rate and 14932 frames (counted by 48 ego cars).

Cooperative 3D object detection benchmarks on $\mathcal{D}^{\text{Multi-V2X}}_{\text{10\%}}$
| Method        | AP@0.3 | AP@0.5 | AP@0.7 |
| ------------- | ----- | ----- | ----- |
| No Fusion     | 0.307 | 0.237 | 0.117 |
| Late Fusion   | 0.346 | 0.270 | 0.141 |
| Early Fusion  | 0.510 | 0.408 | 0.235 |
| V2X-ViT       | 0.440 | 0.350 | 0.228 |
| Where2comm    | 0.452 | 0.348 | 0.213 |

## Contact

If you have any questiones, feel free to open an issue or contact the author by [email](lirs17@tsinghua.org.cn). 

## Citation
If you find our work useful in your research, feel free to give us a cite:

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