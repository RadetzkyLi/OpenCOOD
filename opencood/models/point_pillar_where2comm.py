import torch.nn as nn
import numpy as np
import torch

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarWhere2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=args.get('num_point_features', 4),
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.fusion_net = Where2comm(args['where2comm_fusion'])
        self.multi_scale = args['where2comm_fusion']['multi_scale']

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()


        # evaluation metrics
        self.eval_metric_dict = {}
        # one full communication map's (C, H, W) volume in bytes
        self.eval_metric_dict['one_feat_map_comm_vol'] = None
        self.eval_metric_dict['one_req_map_comm_vol'] = None

    def compute_one_map_comm_vol(self, feature_map):
        """
        Compute one feature/request map's communication volume.
        """
        if self.eval_metric_dict['one_feat_map_comm_vol'] is None:
            _,C,H,W = feature_map.shape
            # 32(float32); 8: bit to byte
            self.eval_metric_dict['one_feat_map_comm_vol'] = C*H*W*32/8
            # 1(bool); 8: bit to byte
            self.eval_metric_dict['one_req_map_comm_vol'] = H*W*1/8

    def compute_communication_volume(self, record_len, communication_rates:float):
        """Compute the total communication volume in bytes.
        Transmitted: feature map + request map.

        """
        B,N = len(record_len),record_len[0]
        # now, only useful for batch size = 1
        if B > 1:
            return 0.0
        
        feature_comm_vol = communication_rates * self.eval_metric_dict['one_feat_map_comm_vol']
        # receive + send
        request_comm_vol = ((N-1) + (N-1)) * self.eval_metric_dict['one_req_map_comm_vol']
        
        comm_vol = feature_comm_vol + request_comm_vol
        if torch.is_tensor(comm_vol):
            comm_vol = comm_vol.cpu().numpy()

        return comm_vol

        
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if self.multi_scale:
            self.compute_one_map_comm_vol(batch_dict['spatial_features'])
            
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 self.backbone)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            self.compute_one_map_comm_vol(spatial_features_2d)

            fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        comm_vol = self.compute_communication_volume(record_len, communication_rates)
    
        output_dict = {'psm': psm, 'rm': rm, 
                       'com': communication_rates, 'comm_vol': comm_vol}
        return output_dict
