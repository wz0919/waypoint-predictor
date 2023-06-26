import json
import numpy as np
import utils
import habitat
import os
import pickle
from habitat.sims import make_sim


config_path = './gen_training_data/config.yaml'
scene_path = '/home/vlnce/vln-ce/data/scene_datasets/mp3d/{scan}/{scan}.glb'
image_path = './training_data/rgbd_fov90/'
save_path = os.path.join(image_path,'{split}/{scan}/{scan}_{node}_mp3d_imgs.pkl')
RAW_GRAPH_PATH= '/home/vlnce/habitat_connectivity_graph/%s.json'
NUMBER = 120

SPLIT = 'train'

with open(RAW_GRAPH_PATH%SPLIT, 'r') as f:
    raw_graph_data = json.load(f)

nav_dict = {}
total_invalids = 0
total = 0

for scene, data in raw_graph_data.items():
    ''' connectivity dictionary '''
    connect_dict = {}
    for edge_id, edge_info in data['edges'].items():
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

    '''make sim for obstacle checking'''
    config = habitat.get_config(config_path)
    config.defrost()
    # config.TASK.POSSIBLE_ACTIONS = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'FORWARD_BY_DIS']
    config.TASK.SENSORS = []
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    config.SIMULATOR.SCENE = scene_path.format(scan=scene)
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    '''save images'''
    if not os.path.exists(image_path+'{split}/{scan}'.format(split=SPLIT,scan=scene)):
        os.makedirs(image_path+'{split}/{scan}'.format(split=SPLIT,scan=scene))
    navigability_dict = {}
    
    i = 0
    for node_a, neighbors in connect_dict.items():
        navigability_dict[node_a] = utils.init_single_node_dict(number=NUMBER)
        rgbs = []
        depths = []
        node_a_pos = np.array(data['nodes'][node_a])[[0, 2]]

        habitat_pos = np.array(data['nodes'][node_a])
        for info in navigability_dict[node_a].values():
            position, heading = habitat_pos, info['heading']
            theta = -(heading - np.pi) / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            obs = sim.get_observations_at(position, rotation)
            rgbs.append(obs['rgb'])
            depths.append(obs['depth'])
        with open(save_path.format(split=SPLIT, scan=scene, node=node_a), 'wb') as f:
            pickle.dump({'rgb': np.array(rgbs),
                         'depth': np.array(depths, dtype=np.float16)}, f)
        utils.print_progress(i+1,total)
        i+=1

    sim.close()
