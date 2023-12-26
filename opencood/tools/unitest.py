import os

from opencood.tools.multi_gpu_utils import get_dist_info



if __name__ == '__main__':
    rank, world_size = get_dist_info()
    print("rank={0}, world_size={1}".format(rank, world_size))