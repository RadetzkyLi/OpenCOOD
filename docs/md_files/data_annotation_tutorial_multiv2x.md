## Tutorial 2:  Data Annotation Introduction

----

We save all groundtruth annotations per agent per timestamp in the yaml files. For instance,
`Town01__2023_11_10_22_23_28/cav_208/000068.yaml` refers to the data annotations with the perspective of te
agent 208 at timestamp 68 in the scenario database `Town01__2023_11_10_22_23_28`. 

The annotation format inherites from that of OPV2V and V2XSet (see [here](https://github.com/DerrickXuNu/OpenCOOD/blob/main/docs/md_files/data_annotation_tutorial.md)) and the minor modifications are as follows:
- changed field name: "vehicles" --> "objects"

- changed field content: 

  - scalar speed (e.g., 6.3) --> vector speed, i.e., \[longitudinal speed, lateral speed] (e.g., \[5.6, 0.9])

  - the object's category is added

- new field: "conn_agents"

Here we go through an example: 

```yaml
cameraFront: # parameters for frontal camera (only cav)
  coords: # the x,y,z,roll,yaw,pitch under CARLA map coordinate
  - 141.35067749023438
  - -388.642578125
  - 1.0410505533218384
  - 0.07589337974786758
  - 174.18048095703125
  - 0.20690691471099854
  extrinsic: # extrinsic matrix from camera to LiDAR
  - - 0.9999999999999999
    - -5.1230071481984265e-18
    - 9.322129061605055e-20
    - -2.999993025731527
  - - -2.5011383190939924e-18
    - 1.0
    - 1.1458579204685086e-19
    - -3.934422863949294e-06
  - - 2.7713237218713775e-20
    - 3.7310309839064755e-20
    - 1.0
    - 0.8999999040861146
  - - 0.0
    - 0.0
    - 0.0
    - 1.0
  intrinsic: # camera intrinsic matrix
  - - 335.639852470912
    - 0.0
    - 400.0
  - - 0.0
    - 335.639852470912
    - 300.0
  - - 0.0
    - 0.0
    - 1.0
cameraRear: ... # params of rear camera (only cav)
cameraLeft: ... # params of left camera (only cav)
cameraRight: ... # params of right camera (only cav)
cameraForward: ... # params of forward camera (only rsu)
cameraBackward: ... # params of backward camera (only rsu)
ego_speed: 
- 8.13 # agent's current longitudinal speed, m/s
- 0.1 # agent's current lateral speed, m/s
lidar_pose: # LiDAR pose under CARLA map coordinate system
- 144.33
- -388.94
- 1.93
- 0.078
- 174.18
- 0.21
geo_location: # lon, lat, alt from GNSS
- 0.0030083171957802513
- -0.00018716769929483235
- 1.300939917564392
true_ego_pos: # agent's true localization
- 143.83
- -388.89
- 0.032
- 0.075
- 174.18
- 0.21
conn_agents:  # connected agents with ego (including ego agent), COMM_RANGE=70m
- 1324
- 1332
objects: # the surrounding objects that lie in the agent's rectangle evaluation range
  1332: # the object id
    angle: # roll, yaw, pitch under CARLA map coordinate system
    - 0.096
    - -177.86
    - 0.197
    center: # the relative position from bounding box center to the frontal axis of this object
    - 0.0004
    - 0.0005
    - 0.71
    extent: # half length, width and height of the object in meters
    - 2.45
    - 1.06
    - 0.75
    location: # x, y ,z position of the center in the frontal axis of the vehicle under CARLA map coordinate system
    - 158.55
    - -385.75
    - 0.032
    speed: 
    - 9.47 # object's longitudinal speed in m/s
    - 0.1 # object's lateral speed in m/s
    category: car # object's category
  4880: ...
```

By default, detection performance is evaluated in ego's range of x ∈ \[−140, 140]m, y ∈ \[−40, 40]m. However, there may exist object observed by no agent, so, it's impossible to detect it utilizing observations from just one frame or just one agent. When developing algorithms without cooperation, one can eliminate such objects using file `Town01__2023_11_10_22_23_28/cav_208/00068_view.yaml` which records **visible information** of an agent. An example is like:

```yaml
visible_objects:
- object_id: 472
  visible_points: 3 # hit points from semantic lidar
  category: pedestrian
- object_id: 475
  visible_points: 1
  category: pedestrian
- ...
```

To facilitate log replay and various evaluation range, **all objects' moving information** at each frame are stored `scenario_name/map`, in which one `.yaml` file records info of a frame. Here, we go through an example of ``Town01__2023_11_10_22_23_28/map/00068.yaml`` :

```yaml
objects:
  733:
    angle:
    - 0.0
    - -89.99999237060547
    - 0.0
    location:
    - 346.1300048828125
    - 332.7997131347656
    - 0.9300000071525574
    center:
    - 0.0
    - 0.0
    - 0.0
    extent:
    - 0.18767888844013214
    - 0.18767888844013214
    - 0.9300000071525574
    speed:
    - 0.0
    - 0
    category: pedestrian
  732: ...
```
