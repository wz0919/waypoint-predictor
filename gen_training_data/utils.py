import math
import numpy as np
import sys
import torch

def init_single_node_dict(number=24):
    init_dict = {}
    for k in range(number):
        init_dict[str(k)] = {
            'heading': k * 2 * math.pi / number,
            'has_waypoint': False,
            'waypoint': [],             # could be multiple waypoints in a direction
            'obstacle_distance': None,  # maximum 2 meters
            'obstacle_index': None,
        }
    return init_dict

def horizontal_distance(start, end):
    return np.linalg.norm(np.array(start)[[0,2]] - np.array(end)[[0,2]])

def get_viewIndex15(heading,number=24):
    viewIndex = heading // (2*math.pi/number)
    b = heading % (2*math.pi/number)
    if (viewIndex == number-1) and (b >= (math.pi/number)):
        viewIndex = 0
    elif b >= (math.pi/number):
        viewIndex += 1
    return int(viewIndex)

def get_distanceIndex12(dist):
    distanceIndex = int(dist // 0.25)
    # >12 means greater than 3.25m, <1 means shorter than 0.25m
    if distanceIndex > 12 or distanceIndex < 1:
        distanceIndex = int(-1)
    return distanceIndex - 1

def get_obstacle_distanceIndex12(dist):
    # the obstacle distance is measured as the maximum distance
    # agent can travel before collision
    distanceIndex = int((dist) // 0.25)
    if distanceIndex > 11:
        distanceIndex = int(-1)
    return distanceIndex

def get_obstacle_info(position, heading, sim):
    theta = -(heading - np.pi)/2
    rotation = np.quaternion(np.cos(theta),0,np.sin(theta),0)
    sim.set_agent_state(position,rotation)
    for i in range(12):
        sim.step_without_obs(1)
        if sim.previous_step_collided:
            break
    if not sim.previous_step_collided:
        return None, None
    collided_at = sim.get_agent_state().position
    distance = horizontal_distance(position,collided_at)
    index = get_obstacle_distanceIndex12(distance)

    return distance, index

def edge_vec_to_indexes(edge_vec,number=24):
    ''' angle index
        {0, 1, ..., 23} for 24 angles, 15 degrees separation
    '''


    angle = -np.arctan2(1.0, 0.0) + np.arctan2(edge_vec[1], edge_vec[0])
    if angle < 0.0:
        angle += 2 * math.pi

    angleIndex = get_viewIndex15(angle,number=number)

    ''' distance index
        {0, 1, ..., 7} for 8 distances, 0.25 meters separation
        {-1} denotes the target waypoint is not in 2 meters range
    '''
    distance = np.linalg.norm(edge_vec)
    distanceIndex = get_distanceIndex12(distance)

    return angle, angleIndex, distance, distanceIndex

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '_' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def init_node_nav_dict(angles):
    init_dict = {}
    for k in range(angles):
        init_dict[str(k)] = {
            'heading': k * math.pi / (angles/2),
            'has_waypoint': False,
            'waypoint': None,           # could be multiple waypoints in a direction, but we only consider one
            'obstacle_distance': None,  # maximum 2 meters
            'obstacle_index': None,
        }
    return init_dict


def init_node_gt_dict(angles):
    init_dict = {
        # 'target': np.zeros((24, 8), dtype=np.int8),
        # 'weight': np.ones((24, 8)),
        'target': np.zeros((angles, 12), dtype=np.int8),
        'obstacle': np.ones((angles, 12), dtype=np.int8),
        'weight': np.ones((angles, 12)),
        'source_pos': None,
        'target_pos': [],
    }
    return init_dict


def init_node_gt_dict_twm03():
    init_dict = {
        # 'target': np.zeros((24, 8), dtype=np.int8),
        # 'weight': np.ones((24, 8)),
        'target': np.zeros((24, 12), dtype=np.int8),
        'obstacle': np.ones((24, 12), dtype=np.int8),
        'weight': np.zeros((24, 12)),
    }
    return init_dict


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    """ Generate masks centered at mu of the given x and y range with the
        origin in the centre of the output
    Inputs:
        mu: tensor (N, 2)
    Outputs:
        tensor (N, y_range, s_range)
    """
    x_mu = mu[:,0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:,1].unsqueeze(1).unsqueeze(1)

    # Generate bivariate Gaussians centered at position mu
    x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)

    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    if gaussian:
        output = torch.exp(-0.5 * ((x_diff/sigma)**2 + (y_diff/sigma)**2 ))
    else:
        output = torch.logical_and(torch.abs(x_diff) <= sigma, torch.abs(y_diff) <= sigma).type(mu.dtype)

    return output


def nms(pred, max_predictions=10, sigma=1.0, gaussian=False):
    ''' Input (batch_size, 1, height, width) '''

    shape = pred.shape

    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0],-1))  # (BATCH_SIZE, 24*48)
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0],-1))  # (BATCH_SIZE, 24*48)

    for i in range(max_predictions):
        # Find and save max over the entire map
        flat_supp_pred = supp_pred.reshape((shape[0],-1))
        val, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0,shape[0])
        flat_output[indices,ix] = flat_pred[indices,ix]

        # Suppression
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x,y], dim=1).float()

        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)

        supp_pred *= (1-g.unsqueeze(1))

    output[output < 0] = 0
    return output

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from gym import spaces
from gym.spaces.box import Box
from numpy import ndarray

if TYPE_CHECKING:
    from torch import Tensor

from habitat_sim.simulator import MutableMapping, MutableMapping_T
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    VisualObservation,
)
from habitat.core.spaces import Space

@registry.register_simulator(name="Sim-v1")
class Simulator(HabitatSim):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def step_without_obs(self,
        action: Union[str, int, MutableMapping_T[int, Union[str, int]]],
        dt: float = 1.0 / 60.0,):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
            return_single = True
        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self.__last_state[agent_id] = agent.get_state()

        # # step physics by dt
        # step_start_Time = time.time()
        # super().step_world(dt)
        # self._previous_step_time = time.time() - step_start_Time

        multi_observations = {}
        for agent_id in action.keys():
            agent_observation = {}
            agent_observation["collided"] = collided_dict[agent_id]
            multi_observations[agent_id] = agent_observation


        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
        else:
            sim_obs = multi_observations

        self._prev_sim_obs = sim_obs