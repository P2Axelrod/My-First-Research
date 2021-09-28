import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule, PointnetSAModule


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0, NUM_CLASSES = 6):
        super().__init__()

        # local feature extraction
        self.local_feature = nn.Sequential(
            nn.Conv1d(160, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
        )
        # local predction
        self.local_predictor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(256, NUM_CLASSES, kernel_size=1),
        )
        # global feature extraction
        self.sa1 = PointnetSAModuleVotes(
            npoint=8192,
            radius=1,
            nsample=16,
            mlp=[3 + input_feature_dim, 256, 256, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=4096,
            radius=5,
            nsample=64,
            mlp=[256, 256, 256, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=2048,
            radius=15,
            nsample=64,
            mlp=[256, 256, 256, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        # self.sa4 = PointnetSAModuleVotes(
        #     npoint=128,
        #     radius=1.2,
        #     nsample=16,
        #     mlp=[256, 128, 128, 256],
        #     use_xyz=True,
        #     normalize_xyz=True
        # )

        # self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp4 = PointnetFPModule(mlp=[256 + 3+ input_feature_dim, 256, 256])

        self.global_feature = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
        )

        self.global_predictor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(256, NUM_CLASSES, kernel_size=1),
        )
        # local and global predictor
        self.local_and_global_predictor = nn.Sequential(
            nn.Conv1d(256*2 , 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(256, NUM_CLASSES, kernel_size=1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
#            pc[..., 3:].contiguous()
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]
#        med = torch.median(pointcloud[:,:,2])
#        device = torch.device('cuda:0')
#        c = torch.ones([10000], dtype=torch.float64, device=device)
#        print(c*0.5)
#        print(pointcloud[:,:,2])
#        islow=torch.le(pointcloud[:,:,2], c)
#        print(islow)
#        pointcloud_low  = pointcloud[:,islow[0],:]
#        print(pointcloud_low.size())
        xyz, features = self._break_up_pc(pointcloud)
        # print(pointcloud.size())
        # print(xyz.size())
        # print(features.size())
        end_points['input_ind'] = None
        end_points['input_xyz'] = xyz
        end_points['input_features'] = features

        # ------- LOCAL FEATURE------------
        # print(features.size())
        local_features = self.local_feature(features)
        local_prob = self.local_predictor(local_features)
        local_prob = F.log_softmax(local_prob, dim=1)
        local_prob = local_prob.transpose(1, 2)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features
        # print(xyz.size())
        # print(features.size())

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features
        # print(xyz.size())
        # print(features.size())

        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features
        # print(xyz.size())
        # print(features.size())

        # xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        # end_points['sa4_xyz'] = xyz
        # end_points['sa4_features'] = features
        # print(xyz.size())
        # print(features.size())

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        # features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
        #                     end_points['sa4_features'])
        # print(features.size())
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        # print(features.size())
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds
        features = self.fp3(end_points['sa1_xyz'], end_points['sa2_xyz'], end_points['sa1_features'], features)
#        print(features.size())
        features = self.fp4(end_points['input_xyz'], end_points['sa1_xyz'], end_points['input_features'], features)
        # print(features.size())

        # -------GLOBAL FEATURE -----------
        # print(features.size())
        global_features = self.global_feature(features)
        global_prob = self.global_predictor(global_features)
        global_prob = F.log_softmax(global_prob, dim=1)
        global_prob = global_prob.transpose(1, 2)

        # -----LOCAL and GLOBAL  ------------
        # print(features.size())
        # print(local_features.size())
        # print(global_features.size())
        features = torch.cat((local_features, global_features), 1)
        # print(features.size())
        pred = self.local_and_global_predictor(features)
        pred = F.log_softmax(pred, dim=1)
        pred = pred.transpose(1, 2)
        # print(pred.size())



        return pred , local_prob , global_prob


if __name__ == '__main__':

    backbone_net = Pointnet2Backbone(input_feature_dim=160-3, NUM_CLASSES = 6).cuda()

    print(backbone_net)
    backbone_net.eval()
    out, out_local, out_global = backbone_net(torch.rand(1, 10000, 163).cuda())
    check = out.cpu().data.numpy()
    check = np.argmax(check, axis=2)
    print(check)
    print (out)
    # for key in sorted(out.keys()):
    #     print(key, '\t', out[key].shape)
