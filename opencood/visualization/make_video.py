import os
import numpy as np
import argparse

import cv2
from glob import glob
from tqdm import tqdm


def save_concat_video(img_paths_list, save_path, fps):
    array = cv2.imread(img_paths_list[0][0])
    size = (array.shape[1], array.shape[0] * len(img_paths_list))
    num_frame = len(img_paths_list[0])

    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size, isColor=True)
    for i in tqdm(range(num_frame)):
        path_list = [paths[i] for paths in img_paths_list]
        array_list = [cv2.imread(path) for path in path_list]
        array = np.concatenate(array_list, axis=0)
        video.write(array)

    video.release()
    cv2.destroyAllWindows()
    print(f'Video has been save to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--num_frame', type=int, required=False)
    parser.add_argument('--fps', type=int, default=10, required=False)
    parser.add_argument('--vis_type', default=['frame', 'final', 'lidar'], nargs='+', type=str, required=False)
    parser.add_argument('--save_path', type=str, required=False)
    parser.add_argument('--start_idx', type=int, default=0, required=False)
    args = parser.parse_args()

    model_dir = args.model_dir
    if not args.save_path:
        save_path = os.path.join(model_dir, 'vis', 'bev_vis_concat.avi')
    else:
        save_path = args.save_path

    assert len(args.vis_type) > 0
    vis_dirs = [os.path.join(model_dir, 'vis', type_, '*.png') for type_ in args.vis_type]
    vis_dirs = [sorted(glob(dir_)) for dir_ in vis_dirs]
    max_frame = min([len(dir_) for dir_ in vis_dirs])
    if args.num_frame:
        max_frame = min(max_frame, args.num_frame)
    # vis_dirs = [dir_[:max_frame] for dir_ in vis_dirs]
    start_idx = args.start_idx
    vis_dirs = [dir_[start_idx:start_idx+max_frame] for dir_ in vis_dirs]

    save_concat_video(vis_dirs, save_path, args.fps)