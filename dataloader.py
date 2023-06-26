
import glob
import numpy as np
from PIL import Image
import pickle as pkl

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# dataloader and transforms
class RGBDepthPano(Dataset):
    def __init__(self, args, img_dir, navigability_dict):
        # self.IMG_WIDTH = 256
        # self.IMG_HEIGHT = 256
        self.RGB_INPUT_DIM = 224
        self.DEPTH_INPUT_DIM = 256
        self.NUM_IMGS = args.NUM_IMGS
        self.navigability_dict = navigability_dict

        self.rgb_transform = torch.nn.Sequential(
            # [transforms.Resize((256,341)),
            #  transforms.CenterCrop(self.RGB_INPUT_DIM),
            #  transforms.ToTensor(),]
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            )
        # self.depth_transform = transforms.Compose(
        #     # [transforms.Resize((self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)),
        #     [transforms.ToTensor(),
        #     ])

        self.img_dirs = glob.glob(img_dir)

        for img_dir in glob.glob(img_dir):
            scan_id = img_dir.split('/')[-1][:11]
            waypoint_id = img_dir.split('/')[-1][12:-14]
            if waypoint_id not in self.navigability_dict[scan_id]:
                self.img_dirs.remove(img_dir)

    def __len__(self): # default name when writing class
        return len(self.img_dirs)

    def __getitem__(self, idx): # default name when writing class

        img_dir = self.img_dirs[idx]
        sample_id = str(idx)
        scan_id = img_dir.split('/')[-1][:11]
        waypoint_id = img_dir.split('/')[-1][12:-14]

        ''' rgb and depth images '''
        rgb_depth_img = pkl.load(open(img_dir, "rb"))
        rgb_img = torch.from_numpy(rgb_depth_img['rgb']).permute(0, 3, 1, 2)
        depth_img = torch.from_numpy(rgb_depth_img['depth']).permute(0, 3, 1, 2)

        # 3 should be the last channel
        trans_rgb_imgs = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM)
        trans_depth_imgs = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        no_trans_rgb = torch.zeros(self.NUM_IMGS, 3, self.RGB_INPUT_DIM, self.RGB_INPUT_DIM, dtype=torch.uint8)
        no_trans_depth = torch.zeros(self.NUM_IMGS, self.DEPTH_INPUT_DIM, self.DEPTH_INPUT_DIM)

        for ix in range(self.NUM_IMGS):
            trans_rgb_imgs[ix] = self.rgb_transform(rgb_img[ix])
            # no_trans_rgb[ix] = rgb_img[ix]
            trans_depth_imgs[ix] = depth_img[ix][0]
            # no_trans_depth[ix] = depth_img[ix][0]

        sample = {'sample_id': sample_id,
                  'scan_id': scan_id,
                  'waypoint_id': waypoint_id,
                  'rgb': trans_rgb_imgs,
                  'depth': trans_depth_imgs.unsqueeze(-1),
                #   'no_trans_rgb': no_trans_rgb,
                #   'no_trans_depth': no_trans_depth,
                  }

        # print('------------------------')
        # print(trans_rgb_imgs[0][0])
        # print(rgb_img[0].shape, rgb_img[0])
        # anivlrb

        return sample
