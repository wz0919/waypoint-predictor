import json
import numpy as np
import utils
import habitat
from habitat.sims import make_sim
from utils import Simulator

config_path = 'gen_training_data/config.yaml'
scene_path = '/home/vlnce/vln-ce/data/scene_datasets/mp3d/{scan}/{scan}.glb'
RAW_GRAPH_PATH= '/home/vlnce/habitat_connectivity_graph/%s.json'
NUMBER = 120

SPLIT = 'val_unseen'

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
    # config.SIMULATOR.AGENT_0.SENSORS = []
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    config.SIMULATOR.TYPE = 'Sim-v1'
    config.SIMULATOR.SCENE = scene_path.format(scan=scene)
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    ''' process each node to standard data format '''
    navigability_dict = {}
    total = len(connect_dict)
    for i, pair in enumerate(connect_dict.items()):
        node_a, neighbors = pair
        navigability_dict[node_a] = utils.init_single_node_dict(number=NUMBER)
        node_a_pos = np.array(data['nodes'][node_a])[[0,2]]
    
        habitat_pos = np.array(data['nodes'][node_a])
        for id, info in navigability_dict[node_a].items():
            obstacle_distance, obstacle_index = utils.get_obstacle_info(habitat_pos,info['heading'],sim)
            info['obstacle_distance'] = obstacle_distance
            info['obstacle_index'] = obstacle_index
    
        for node_b in neighbors:
            node_b_pos = np.array(data['nodes'][node_b])[[0,2]]
    
            edge_vec = (node_b_pos - node_a_pos)
            angle, angleIndex, distance, distanceIndex = utils.edge_vec_to_indexes(edge_vec,number=NUMBER)
    
            navigability_dict[node_a][str(angleIndex)]['has_waypoint'] = True
            navigability_dict[node_a][str(angleIndex)]['waypoint'].append(
                {
                    'node_id': node_b,
                    'position': node_b_pos.tolist(),
                    'angle': angle,
                    'angleIndex': angleIndex,
                    'distance': distance,
                    'distanceIndex': distanceIndex,
                })
        utils.print_progress(i+1,total)
    
    nav_dict[scene] = navigability_dict
    sim.close()

output_path = './gen_training_data/nav_dicts/navigability_dict_%s.json'%SPLIT
with open(output_path, 'w') as fo:
    json.dump(nav_dict, fo, ensure_ascii=False, indent=4)
