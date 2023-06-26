import torch
import torch.nn as nn
import torchvision
import numpy as np

from ddppo_resnet.resnet_policy import PNResnetDepthEncoder

class RGBEncoder(nn.Module):
    def __init__(self, resnet_pretrain=True, trainable=False):
        super(RGBEncoder, self).__init__()
        if resnet_pretrain:
            print('\nLoading Torchvision pre-trained Resnet50 for RGB ...')
        rgb_resnet = torchvision.models.resnet50(pretrained=resnet_pretrain)
        rgb_modules = list(rgb_resnet.children())[:-2]
        rgb_net = torch.nn.Sequential(*rgb_modules)
        self.rgb_net = rgb_net
        for param in self.rgb_net.parameters():
            param.requires_grad_(trainable)

        # self.scale = 0.5

    def forward(self, rgb_imgs):
        rgb_shape = rgb_imgs.size()
        rgb_imgs = rgb_imgs.reshape(rgb_shape[0]*rgb_shape[1],
                                    rgb_shape[2], rgb_shape[3], rgb_shape[4])
        rgb_feats = self.rgb_net(rgb_imgs)  # * self.scale

        # print('rgb_imgs', rgb_imgs.shape)
        # print('rgb_feats', rgb_feats.shape)

        return rgb_feats.squeeze()


class DepthEncoder(nn.Module):
    def __init__(self, resnet_pretrain=True, trainable=False):
        super(DepthEncoder, self).__init__()

        self.depth_net = PNResnetDepthEncoder()
        if resnet_pretrain:
            print('Loading PointNav pre-trained Resnet50 for Depth ...')
            ddppo_pn_depth_encoder_weights = torch.load('/home/vlnce/vln-ce/data/ddppo-models/gibson-2plus-resnet50.pth')
            weights_dict = {}
            for k, v in ddppo_pn_depth_encoder_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue
                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v
            del ddppo_pn_depth_encoder_weights
            self.depth_net.load_state_dict(weights_dict, strict=True)
        for param in self.depth_net.parameters():
            param.requires_grad_(trainable)

    def forward(self, depth_imgs):
        depth_shape = depth_imgs.size()
        depth_imgs = depth_imgs.reshape(depth_shape[0]*depth_shape[1],
                                    depth_shape[2], depth_shape[3], depth_shape[4])
        depth_feats = self.depth_net(depth_imgs)

        # print('depth_imgs', depth_imgs.shape)
        # print('depth_feats', depth_feats.shape)
        #
        # import pdb; pdb.set_trace()

        return depth_feats
