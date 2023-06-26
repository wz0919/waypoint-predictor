import torch
import numpy as np
import math
import utils
import copy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.transform import resize
import os

def waypoint_eval(args, predictions):
    ''' Evaluation of the predicted waypoint map,
        notice that the number of candidates is cap at args.MAX_NUM_CANDIDATES,
        but the number of GT waypoints could be any value in range [1,args.ANGLES].

        The preprocessed data is constraining each angle sector has at most
        one GT waypoint.
    '''

    sample_id = predictions['sample_id']
    source_pos = predictions['source_pos']
    target_pos = predictions['target_pos']
    probs = predictions['probs']
    logits = predictions['logits']
    target = predictions['target']
    obstacle = predictions['obstacle']
    sample_loss = predictions['sample_loss']

    results = {
        'candidates': {},
        'p_waypoint_openspace': 0.0,
        'p_waypoint_obstacle': 0.0,
        'avg_wayscore': 0.0,
        'avg_pred_distance': 0.0,
        'avg_chamfer_distance': 0.0,
        'avg_hausdorff_distance': 0.0,
        'avg_num_delta': 0.0,
    }

    num_candidate = []          # cap at args.MAX_NUM_CANDIDATES
    num_waypoint_openspace = [] # % waypoint in open space
    num_waypoint_obstacle = []  # % waypoint in obstacle
    waypoint_score = []         # scores on target map collected by predictions
    pred_distance = []          # distance from targets to predictions
    chamfer_distance_all = []
    hausdorff_distance_all = []
    num_delta_all = []

    ''' output prediction '''
    for i, batch_x in enumerate(logits):
        batch_sample_id = sample_id[i]
        batch_source_pos = source_pos[i]
        batch_target_pos = target_pos[i]
        batch_target = target[i]
        batch_obstacle = obstacle[i]
        batch_sample_loss = sample_loss[i]

        batch_x = torch.tensor(batch_x)
        batch_x_norm = torch.softmax(
            batch_x.reshape(
                batch_x.size(0), args.ANGLES*args.NUM_CLASSES
                ), dim=1
            )
        batch_x_norm = batch_x_norm.reshape(batch_x.size(0), args.ANGLES, args.NUM_CLASSES)
        # batch_x_norm = torch.sigmoid(batch_x)

        # batch_output_map = utils.nms(
        #     batch_x_norm.unsqueeze(1), max_predictions=args.MAX_NUM_CANDIDATES,
        #     sigma=(7.0,5.0))
        # batch_output_map = batch_output_map.squeeze()

        batch_x_norm_wrap = torch.cat(
            (batch_x_norm[:,-1:,:], batch_x_norm, batch_x_norm[:,:1,:]), 
            dim=1)
        batch_output_map = utils.nms(
            batch_x_norm_wrap.unsqueeze(1), max_predictions=5,
            sigma=(7.0,5.0))
        batch_output_map = batch_output_map.squeeze()[:,1:-1,:]

        if args.VIS:
            # # nms without different sigma
            batch_output_map_sig4 = utils.nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=args.MAX_NUM_CANDIDATES,
                sigma=(4.0,4.0))
            batch_output_map_sig4 = batch_output_map_sig4.squeeze()[:,1:-1,:]
            batch_output_map_sig5 = utils.nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=args.MAX_NUM_CANDIDATES,
                sigma=(5.0,5.0))
            batch_output_map_sig5 = batch_output_map_sig5.squeeze()[:,1:-1,:]
            batch_output_map_sig7_5 = utils.nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=args.MAX_NUM_CANDIDATES,
                sigma=(7.0,5.0))
            batch_output_map_sig7_5 = batch_output_map_sig7_5.squeeze()[:,1:-1,:]

        for j, id in enumerate(batch_sample_id):
            # pick one distance from each non-zeros column
            candidates = {}
            c_openspace = 0
            c_obstacle = 0
            candidates_pos = []

            ''' gather predicted candidates and check if candidates are in openspace '''
            for jdx, angle_view in enumerate(batch_output_map[j]):
                if angle_view.sum() != 0:
                    candidates[jdx] = angle_view.argmax().item()
                    candidates_pos.append(
                        [jdx * 2 * math.pi / args.ANGLES, 
                        (candidates[jdx]+1) * 0.25])
                    # opensapce / obstacle
                    if batch_obstacle[j][jdx][candidates[jdx]] == 0:
                        c_openspace += 1
                    else:
                        c_obstacle += 1

            # the inferene ouput
            results['candidates'][id] = {
                # 'loss': batch_sample_loss[j],
                'angle_dist': candidates,
            }
            num_candidate.append(len(candidates))
            num_waypoint_openspace.append(c_openspace)
            num_waypoint_obstacle.append(c_obstacle)

            ''' score collected over the target heatmap by predictions '''
            # score = (torch.tensor(batch_target[j])[batch_output_map[j] != 0]).sum()
            # waypoint_score.append(score.item())
            score_map = torch.tensor(batch_target[j])
            # using binary selection here doesn't conflict with
            # the candidates due to the large sigmas for NMS
            score = (score_map[batch_output_map[j] != 0]
                ).sum() / (len(candidates))
            waypoint_score.append(score.item())

            ''' measure target to prediction distance '''
            bsp = np.array(batch_source_pos[j])
            btp = np.array(batch_target_pos[j])
            cp = np.array(candidates_pos)
            cp_x = np.sin(cp[:,0]) * cp[:,1] + bsp[0]
            cp_y = np.cos(cp[:,0]) * cp[:,1] + bsp[1]
            cp = np.concatenate(
                (np.expand_dims(cp_x, axis=1),
                np.expand_dims(cp_y, axis=1)), axis=1)
            # take the minimal distance from each target
            # to all predictions
            tp_dists = cdist(btp, cp)
            tp_dist_min = tp_dists.min(1).mean()
            pred_distance.append(tp_dist_min)

            # Chamfer distance
            predict_to_gt_0 = tp_dists.min(0).mean()
            gt_to_predict_0 = tp_dists.min(1).mean()
            chamfer_distance = 0.5 * (
                predict_to_gt_0 + gt_to_predict_0)
            chamfer_distance_all.append(chamfer_distance)

            # Hausdorff distance
            predict_to_gt_1 = tp_dists.min(0).max()
            gt_to_predict_1 = tp_dists.min(1).max()
            hausdorff_distance = max(
                predict_to_gt_1, gt_to_predict_1)
            hausdorff_distance_all.append(hausdorff_distance)

            # prediction-target delta
            num_target = len(batch_target_pos[j])
            num_predict = len(candidates_pos)
            num_delta = num_predict - num_target
            num_delta_all.append(num_delta)

            if args.VIS:
                import pdb; pdb.set_trace()
                save_img_dir = './visualize/%s-best_avg_wayscore'%(args.EXP_ID.split('-')[1])
                if not os.path.exists(save_img_dir):
                    os.makedirs(save_img_dir)
                
                im1 = (np.array(batch_target[j])/np.array(batch_target[j]).max()*255).astype('uint8')
                batch_x_pos = copy.deepcopy(batch_x[j].numpy())
                batch_x_pos[batch_x_pos<0]=0.0
                im2 = (batch_x_pos/batch_x_pos.max()*255).astype('uint8')
                im6 = (batch_output_map_sig4[j].numpy()/batch_output_map_sig4[j].numpy().max()*255).astype('uint8')
                im7 = (batch_output_map_sig5[j].numpy()/batch_output_map_sig5[j].numpy().max()*255).astype('uint8')
                im8 = (batch_output_map_sig7_5[j].numpy()/batch_output_map_sig7_5[j].numpy().max()*255).astype('uint8')
                fig = plt.figure(figsize=(10,14))
                fig.add_subplot(1, 5, 1); plt.imshow(im6); plt.axis('off')
                fig.add_subplot(1, 5, 2); plt.imshow(im7); plt.axis('off')
                fig.add_subplot(1, 5, 3); plt.imshow(im8); plt.axis('off')
                fig.add_subplot(1, 5, 4); plt.imshow(im2); plt.axis('off')
                fig.add_subplot(1, 5, 5); plt.imshow(im1); plt.axis('off')
                plt.savefig(save_img_dir+'/predict-target-%s-%s.jpeg'%(i,j),
                    bbox_inches='tight')
                plt.close()

    p_waypoint_openspace = sum(num_waypoint_openspace) / sum(num_candidate)
    p_waypoint_obstacle = sum(num_waypoint_obstacle) / sum(num_candidate)
    avg_wayscore = np.mean(waypoint_score).item()
    avg_pred_distance = np.mean(pred_distance).item()
    avg_chamfer_distance = np.mean(chamfer_distance_all).item()
    avg_hausdorff_distance = np.mean(hausdorff_distance_all).item()
    avg_num_delta = np.mean(num_delta_all).item()

    results['p_waypoint_openspace'] = p_waypoint_openspace
    results['p_waypoint_obstacle'] = p_waypoint_obstacle
    results['avg_wayscore'] = avg_wayscore
    results['avg_pred_distance'] = avg_pred_distance
    results['avg_chamfer_distance'] = avg_chamfer_distance
    results['avg_hausdorff_distance'] = avg_hausdorff_distance
    results['avg_num_delta'] = avg_num_delta

    return results
