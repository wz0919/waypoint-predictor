
import torch
import argparse
from dataloader import RGBDepthPano

from image_encoders import RGBEncoder, DepthEncoder
from TRM_net import BinaryDistPredictor_TRM, TRM_predict

from eval import waypoint_eval

import os
import glob
import utils
import random
from utils import nms
from utils import print_progress
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(args):
    torch.manual_seed(0)
    random.seed(0)
    exp_log_path = './checkpoints/%s/'%(args.EXP_ID)
    os.makedirs(exp_log_path, exist_ok=True)
    exp_log_path = './checkpoints/%s/snap/'%(args.EXP_ID)
    os.makedirs(exp_log_path, exist_ok=True)

class Param():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train waypoint predictor')

        self.parser.add_argument('--EXP_ID', type=str, default='test_0')
        self.parser.add_argument('--TRAINEVAL', type=str, default='train', help='trian or eval mode')
        self.parser.add_argument('--VIS', type=int, default=0, help='visualize predicted hearmaps')
        # self.parser.add_argument('--LOAD_EPOCH', type=int, default=None, help='specific an epoch to load for eval')

        self.parser.add_argument('--ANGLES', type=int, default=24)
        self.parser.add_argument('--NUM_IMGS', type=int, default=24)
        self.parser.add_argument('--NUM_CLASSES', type=int, default=12)
        self.parser.add_argument('--MAX_NUM_CANDIDATES', type=int, default=5)

        self.parser.add_argument('--PREDICTOR_NET', type=str, default='TRM', help='TRM only')

        self.parser.add_argument('--EPOCH', type=int, default=10)
        self.parser.add_argument('--BATCH_SIZE', type=int, default=2)
        self.parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
        self.parser.add_argument('--WEIGHT', type=int, default=0, help='weight the target map')

        self.parser.add_argument('--TRM_LAYER', default=2, type=int, help='number of TRM hidden layers')
        self.parser.add_argument('--TRM_NEIGHBOR', default=2, type=int, help='number of attention mask neighbor')
        self.parser.add_argument('--HEATMAP_OFFSET', default=2, type=int, help='an offset determined by image FoV and number of images')
        self.parser.add_argument('--HIDDEN_DIM', default=768, type=int)

        self.args = self.parser.parse_args()

def predict_waypoints(args):

    print('\nArguments', args)
    log_dir = './checkpoints/%s/tensorboard/'%(args.EXP_ID)
    writer = SummaryWriter(log_dir=log_dir)

    ''' networks '''
    rgb_encoder = RGBEncoder(resnet_pretrain=True, trainable=False).to(device)
    depth_encoder = DepthEncoder(resnet_pretrain=True, trainable=False).to(device)
    if args.PREDICTOR_NET == 'TRM':
        print('\nUsing TRM predictor')
        print('HIDDEN_DIM default to 768')
        args.HIDDEN_DIM = 768
        predictor = BinaryDistPredictor_TRM(args=args,
            hidden_dim=args.HIDDEN_DIM, n_classes=args.NUM_CLASSES).to(device)

    ''' load navigability (gt waypoints, obstacles and weights) '''
    navigability_dict = utils.load_gt_navigability(
        './training_data/%s_*_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(args.ANGLES))

    ''' dataloader for rgb and depth images '''
    train_img_dir = './gen_training_data/rgbd_fov90/train/*/*.pkl'
    traindataloader = RGBDepthPano(args, train_img_dir, navigability_dict)
    eval_img_dir = './gen_training_data/rgbd_fov90/val_unseen/*/*.pkl'
    evaldataloader = RGBDepthPano(args, eval_img_dir, navigability_dict)
    if args.TRAINEVAL == 'train':
        trainloader = torch.utils.data.DataLoader(traindataloader, 
        batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4)
    evalloader = torch.utils.data.DataLoader(evaldataloader, 
        batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4)

    ''' optimization '''
    criterion_bcel = torch.nn.BCEWithLogitsLoss(reduction='none')
    criterion_mse = torch.nn.MSELoss(reduction='none')

    params = list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.LEARNING_RATE)

    ''' training loop '''
    if args.TRAINEVAL == 'train':
        print('\nTraining starts')
        best_val_1 = {"avg_wayscore": 0.0, "log_string": '', "update":False}
        best_val_2 = {"avg_pred_distance": 10.0, "log_string": '', "update":False}

        for epoch in range(args.EPOCH):  # loop over the dataset multiple times
            sum_loss = 0.0

            rgb_encoder.eval()
            depth_encoder.eval()
            predictor.train()

            for i, data in enumerate(trainloader):
                scan_ids = data['scan_id']
                waypoint_ids = data['waypoint_id']
                rgb_imgs = data['rgb'].to(device)
                depth_imgs = data['depth'].to(device)

                ''' checking image orientation '''
                # from PIL import Image
                # from matplotlib import pyplot
                # import numpy as np
                # # import pdb; pdb.set_trace()
                # out_img = np.swapaxes(
                #     np.swapaxes(
                #         data['no_trans_rgb'][0].cpu().numpy(), 1,2),
                #     2, 3)
                # for kk, out_img_i in enumerate(out_img):
                #     im = Image.fromarray(out_img_i)
                #     im.save("./play/%s.png"%(kk))
                #     pyplot.imsave("./play/mpl_%s.png"%(kk), out_img_i)
                # out_depth = data['no_trans_depth'][0].cpu().numpy() * 255
                # out_depth = out_depth.astype(np.uint8)
                # for kk, out_depth_i in enumerate(out_depth):
                #     im = Image.fromarray(out_depth_i)
                #     im.save("./play/depth_%s.png"%(kk))

                ''' processing observations '''
                rgb_feats = rgb_encoder(rgb_imgs)        # (BATCH_SIZE*ANGLES, 2048)
                depth_feats = depth_encoder(depth_imgs)  # (BATCH_SIZE*ANGLES, 128, 4, 4)

                ''' learning objectives '''
                target, obstacle, weight, _, _ = utils.get_gt_nav_map(
                    args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
                target = target.to(device)
                obstacle = obstacle.to(device)
                weight = weight.to(device)

                if args.PREDICTOR_NET == 'TRM':
                    vis_logits = TRM_predict('train', args,
                        predictor, rgb_feats, depth_feats)

                    loss_vis = criterion_mse(vis_logits, target)
                    if args.WEIGHT:
                        loss_vis = loss_vis * weight
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES

                total_loss.backward()
                optimizer.step()
                sum_loss += total_loss.item()

                print_progress(i+1, len(trainloader), prefix='Epoch: %d/%d'%((epoch+1),args.EPOCH))
            writer.add_scalar("Train/Loss", sum_loss/(i+1), epoch)
            print('Train Loss: %.5f' % (sum_loss/(i+1)))  # (epoch+1),args.EPOCH

            ''' evaluation - inference '''
            # print('Evaluation ...')
            sum_loss = 0.0
            predictions = {'sample_id': [], 
                'source_pos': [], 'target_pos': [],
                'probs': [], 'logits': [],
                'target': [], 'obstacle': [], 'sample_loss': []}

            rgb_encoder.eval()
            depth_encoder.eval()
            predictor.eval()

            for i, data in enumerate(evalloader):
                scan_ids = data['scan_id']
                waypoint_ids = data['waypoint_id']
                sample_id = data['sample_id']
                rgb_imgs = data['rgb'].to(device)
                depth_imgs = data['depth'].to(device)

                target, obstacle, weight, \
                source_pos, target_pos = utils.get_gt_nav_map(
                    args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
                target = target.to(device)
                obstacle = obstacle.to(device)
                weight = weight.to(device)

                ''' processing observations '''
                rgb_feats = rgb_encoder(rgb_imgs)        # (BATCH_SIZE*ANGLES, 2048)
                depth_feats = depth_encoder(depth_imgs)  # (BATCH_SIZE*ANGLES, 128, 4, 4)

                if args.PREDICTOR_NET == 'TRM':
                    vis_probs, vis_logits = TRM_predict('eval', args,
                        predictor, rgb_feats, depth_feats)
                    overall_probs = vis_probs
                    overall_logits = vis_logits
                    loss_vis = criterion_mse(vis_logits, target)
                    if args.WEIGHT:
                        loss_vis = loss_vis * weight
                    sample_loss = loss_vis.sum(-1).sum(-1) / args.ANGLES
                    total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES

                sum_loss += total_loss.item()
                predictions['sample_id'].append(sample_id)
                predictions['source_pos'].append(source_pos)
                predictions['target_pos'].append(target_pos)
                predictions['probs'].append(overall_probs.tolist())
                predictions['logits'].append((overall_logits.tolist()))
                predictions['target'].append(target.tolist())
                predictions['obstacle'].append(obstacle.tolist())
                predictions['sample_loss'].append(target.tolist())

            print('Eval Loss: %.5f' % (sum_loss/(i+1)))
            results = waypoint_eval(args, predictions)
            writer.add_scalar("Evaluation/Loss", sum_loss/(i+1), epoch)
            writer.add_scalar("Evaluation/p_waypoint_openspace", results['p_waypoint_openspace'], epoch)
            writer.add_scalar("Evaluation/p_waypoint_obstacle", results['p_waypoint_obstacle'], epoch)
            writer.add_scalar("Evaluation/avg_wayscore", results['avg_wayscore'], epoch)
            writer.add_scalar("Evaluation/avg_pred_distance", results['avg_pred_distance'], epoch)
            log_string = 'Epoch %s '%(epoch)
            for key, value in results.items():
                if key != 'candidates':
                    log_string += '{} {:.5f} | '.format(str(key), value)
            print(log_string)

            # save checkpoint
            if results['avg_wayscore'] > best_val_1['avg_wayscore']:
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_wayscore'%(args.EXP_ID) #, epoch+1
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path)
                print('New best avg_wayscore result found, checkpoint saved to %s'%(checkpoint_save_path))
                best_val_1['avg_wayscore'] = results['avg_wayscore']
                best_val_1['log_string'] = log_string
            checkpoint_reg_save_path = './checkpoints/%s/snap/check_latest'%(args.EXP_ID) #, epoch+1
            utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_reg_save_path)
            print('Best avg_wayscore result til now: ', best_val_1['log_string'])

            if results['avg_pred_distance'] < best_val_2['avg_pred_distance']:
                checkpoint_save_path = './checkpoints/%s/snap/check_val_best_avg_pred_distance'%(args.EXP_ID) #, epoch+1
                utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_save_path)
                print('New best avg_pred_distance result found, checkpoint saved to %s'%(checkpoint_save_path))
                best_val_2['avg_pred_distance'] = results['avg_pred_distance']
                best_val_2['log_string'] = log_string
            checkpoint_reg_save_path = './checkpoints/%s/snap/check_latest'%(args.EXP_ID) #, epoch+1
            utils.save_checkpoint(epoch+1, predictor, optimizer, checkpoint_reg_save_path)
            print('Best avg_pred_distance result til now: ', best_val_2['log_string'])

    elif args.TRAINEVAL == 'eval':
        ''' evaluation - inference (with a bit mixture-of-experts) '''
        print('\nEvaluation mode, please doublecheck EXP_ID and LOAD_EPOCH')
        checkpoint_load_path = './checkpoints/%s/snap/check_val_best_avg_wayscore'%(args.EXP_ID)  #args.LOAD_EPOCH
        epoch, predictor, optimizer = utils.load_checkpoint(
                        predictor, optimizer, checkpoint_load_path)

        sum_loss = 0.0
        predictions = {'sample_id': [], 
            'source_pos': [], 'target_pos': [],
            'probs': [], 'logits': [],
            'target': [], 'obstacle': [], 'sample_loss': []}

        rgb_encoder.eval()
        depth_encoder.eval()
        predictor.eval()

        for i, data in enumerate(evalloader):
            if args.VIS and i == 5:
                break

            scan_ids = data['scan_id']
            waypoint_ids = data['waypoint_id']
            sample_id = data['sample_id']
            rgb_imgs = data['rgb'].to(device)
            depth_imgs = data['depth'].to(device)

            target, obstacle, weight, \
            source_pos, target_pos = utils.get_gt_nav_map(
                args.ANGLES, navigability_dict, scan_ids, waypoint_ids)
            target = target.to(device)
            obstacle = obstacle.to(device)
            weight = weight.to(device)

            ''' processing observations '''
            rgb_feats = rgb_encoder(rgb_imgs)        # (BATCH_SIZE*ANGLES, 2048)
            depth_feats = depth_encoder(depth_imgs)  # (BATCH_SIZE*ANGLES, 128, 4, 4)

            ''' predicting the waypoint probabilities '''
            if args.PREDICTOR_NET == 'TRM':
                vis_probs, vis_logits = TRM_predict('eval', args,
                    predictor, rgb_feats, depth_feats)
                overall_probs = vis_probs
                overall_logits = vis_logits
                loss_vis = criterion_mse(vis_logits, target)

                if args.WEIGHT:
                    loss_vis = loss_vis * weight
                sample_loss = loss_vis.sum(-1).sum(-1) / args.ANGLES
                total_loss = loss_vis.sum() / vis_logits.size(0) / args.ANGLES

            sum_loss += total_loss.item()
            predictions['sample_id'].append(sample_id)
            predictions['source_pos'].append(source_pos)
            predictions['target_pos'].append(target_pos)
            predictions['probs'].append(overall_probs.tolist())
            predictions['logits'].append(overall_logits.tolist())
            predictions['target'].append(target.tolist())
            predictions['obstacle'].append(obstacle.tolist())
            predictions['sample_loss'].append(target.tolist())

        print('Eval Loss: %.5f' % (sum_loss/(i+1)))
        results = waypoint_eval(args, predictions)
        log_string = 'Epoch %s '%(epoch)
        for key, value in results.items():
            if key != 'candidates':
                log_string += '{} {:.5f} | '.format(str(key), value)
        print(log_string)
        print('Evaluation Done')

    else:
        RunningModeError

if __name__ == "__main__":
    param = Param()
    args = param.args
    setup(args)

    if args.VIS:
        assert args.TRAINEVAL == 'eval'

    predict_waypoints(args)
