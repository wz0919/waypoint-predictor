import json
import math
import numpy as np
import copy
import torch
import os
import utils
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

ANGLES = 120
DISTANCES = 12
RAW_GRAPH_PATH = '/home/vlnce/habitat_connectivity_graph/%s.json'

RADIUS = 3.25  # corresponding to max forward distance of 2 meters

print('Running TRM-0.2 !!!!!!!!!!')

print('\nProcessing navigability and connectivity to GT maps')
print('Using %s ANGLES and %s DISTANCES'%(ANGLES, DISTANCES))
print('Maximum radius for each waypoint: %s'%(RADIUS))
print('\nConstraining each angle sector has at most one GT waypoint')
print('For all sectors with edges greater than %s, create a virtual waypoint at %s'%(RADIUS, 2.50))
print('\nThis script will return the target map, the obstacle map and the weigh map for each environment')

np.random.seed(7)

splits = ['train', 'val_unseen']
for split in splits:
    print('\nProcessing %s split data'%(split))

    with open(RAW_GRAPH_PATH%split, 'r') as f:
        data = json.load(f)
    if os.path.exists('./gen_training_data/nav_dicts/navigability_dict_%s.json'%split):
        with open('./gen_training_data/nav_dicts/navigability_dict_%s.json'%split) as f:
            nav_dict = json.load(f)
    raw_nav_dict = {}
    nodes = {}
    edges = {}
    obstacles = {}
    for k, v in data.items():
        nodes[k] = data[k]['nodes']
        edges[k] = data[k]['edges']
        obstacles[k] = nav_dict[k]
    raw_nav_dict['nodes'], raw_nav_dict['edges'], raw_nav_dict['obstacles'] = nodes, edges, obstacles
    data_scans = {
        'nodes': raw_nav_dict['nodes'],
        'edges': raw_nav_dict['edges'],
    }
    obstacle_dict_scans = raw_nav_dict['obstacles']
    scans = list(raw_nav_dict['nodes'].keys())


    overall_nav_dict = {}
    del_nodes = 0
    count_nodes = 0
    target_count = 0 # not count target because it is Gaussian
    openspace_count = 0; obstacle_count = 0
    rawedges_count = 0; postedges_count = 0

    for scan in scans:
        ''' connectivity dictionary '''
        obstacle_dict = obstacle_dict_scans[scan]

        connect_dict = {}
        for edge_id, edge_info in data_scans['edges'][scan].items():
            node_a = edge_info['nodes'][0]
            node_b = edge_info['nodes'][1]

            if node_a not in connect_dict:
                connect_dict[node_a] = [node_b]
            else:
                connect_dict[node_a].append(node_b)
            if node_b not in connect_dict:
                connect_dict[node_b] = [node_a]
            else:
                connect_dict[node_b].append(node_a)

        ''' process each node to standard data format '''
        navigability_dict = {}
        groundtruth_dict = {}
        count_nodes_i = 0
        del_nodes_i = 0
        for node_a, neighbors in connect_dict.items():
            count_nodes += 1; count_nodes_i += 1
            navigability_dict[node_a] = utils.init_node_nav_dict(ANGLES)
            groundtruth_dict[node_a] = utils.init_node_gt_dict(ANGLES)

            node_a_pos = np.array(data_scans['nodes'][scan][node_a])[[0,2]]
            groundtruth_dict[node_a]['source_pos'] = node_a_pos.tolist()

            for node_b in neighbors:
                node_b_pos = np.array(data_scans['nodes'][scan][node_b])[[0,2]]

                edge_vec = (node_b_pos - node_a_pos)
                angle, angleIndex, \
                distance, \
                distanceIndex = utils.edge_vec_to_indexes(
                    edge_vec, ANGLES)

                # remove too far or too close viewpoints
                if distanceIndex == -1:
                    continue
                # keep the further keypoint in the same sector
                if navigability_dict[node_a][str(angleIndex)]['has_waypoint']:
                    if distance < navigability_dict[node_a][str(angleIndex)]['waypoint']['distance']:
                        continue

                # if distance <= RADIUS:
                navigability_dict[node_a][str(angleIndex)]['has_waypoint'] = True
                navigability_dict[node_a][str(angleIndex)]['waypoint'] = {
                        'node_id': node_b,
                        'position': node_b_pos,
                        'angle': angle,
                        'angleIndex': angleIndex,
                        'distance': distance,
                        'distanceIndex': distanceIndex,
                    }
                ''' set target map '''
                groundtruth_dict[node_a]['target'][angleIndex, distanceIndex] = 1
                groundtruth_dict[node_a]['target_pos'].append(node_b_pos.tolist())

            # record the number of raw targets
            raw_target_count = groundtruth_dict[node_a]['target'].sum()

            if raw_target_count == 0:
                del(groundtruth_dict[node_a])
                del_nodes += 1; del_nodes_i += 1                
                continue

            ''' a Gaussian target map '''
            gau_peak = 10
            gau_sig_angle = 1.0
            gau_sig_dist = 2.0
            groundtruth_dict[node_a]['target'] = groundtruth_dict[node_a]['target'].astype(np.float32)

            gau_temp_in = np.concatenate(
                (
                    np.zeros((ANGLES,10)),
                    groundtruth_dict[node_a]['target'],
                    np.zeros((ANGLES,10)),
                ), axis=1)

            gau_target = gaussian_filter(
                gau_temp_in,
                sigma=(1,2),
                mode='wrap',
            )
            gau_target = gau_target[:, 10:10+DISTANCES]

            gau_target_maxnorm = gau_target / gau_target.max()
            groundtruth_dict[node_a]['target'] = gau_peak * gau_target_maxnorm

            for k in range(ANGLES):
                k_dist = obstacle_dict[node_a][str(k)]['obstacle_distance']
                if k_dist is None:
                    k_dist = 100
                navigability_dict[node_a][str(k)]['obstacle_distance'] = k_dist

                k_dindex = utils.get_obstacle_distanceIndex12(k_dist)
                navigability_dict[node_a][str(k)]['obstacle_index'] = k_dindex

                ''' deal with obstacle '''
                if k_dindex != -1:
                    groundtruth_dict[node_a]['obstacle'][k][:k_dindex] = np.zeros(k_dindex)
                else:
                    groundtruth_dict[node_a]['obstacle'][k] = np.zeros(12)


            ''' ********** very important ********** '''
            ''' adjust target map '''
            ''' obstacle comes first '''

            rawt = copy.deepcopy(groundtruth_dict[node_a]['target'])

            groundtruth_dict[node_a]['target'] = \
                groundtruth_dict[node_a]['target'] * (groundtruth_dict[node_a]['obstacle'] == 0)

            # a confidence thresholding
            if groundtruth_dict[node_a]['target'].max() < 0.75*gau_peak:
                del(groundtruth_dict[node_a])
                del_nodes += 1; del_nodes_i += 1
                continue

            postt = copy.deepcopy(groundtruth_dict[node_a]['target'])
            rawedges_count += (rawt==gau_peak).sum()
            postedges_count += (postt==gau_peak).sum()

            ''' ********** very important ********** '''

            openspace_count += (groundtruth_dict[node_a]['obstacle'] == 0).sum()
            obstacle_count += (groundtruth_dict[node_a]['obstacle'] == 1).sum()

            groundtruth_dict[node_a]['target'] = groundtruth_dict[node_a]['target'].tolist()
            groundtruth_dict[node_a]['weight'] = groundtruth_dict[node_a]['weight'].tolist()
            groundtruth_dict[node_a]['obstacle'] = groundtruth_dict[node_a]['obstacle'].tolist()

        overall_nav_dict[scan] = groundtruth_dict

    print('Obstacle comes before target !!!')
    print('Number of deleted nodes: %s/%s'%(del_nodes, count_nodes))
    print('Ratio of obstacle behind target: %s/%s'%(postedges_count,rawedges_count))

    print('Ratio of openspace %.5f'%(openspace_count/(openspace_count+obstacle_count)))
    print('Ratio of obstacle %.5f'%(obstacle_count/(openspace_count+obstacle_count)))

    with open('./training_data/%s_%s_mp3d_waypoint_twm0.2_obstacle_first_withpos.json'%(ANGLES, split), 'w') as f:
        json.dump(overall_nav_dict, f)
    print('Done')

# import pdb; pdb.set_trace()
