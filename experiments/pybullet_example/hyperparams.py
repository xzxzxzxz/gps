""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.pybullet.agent_pybullet import AgentPyBullet
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.proto.gps_pb2 import ALL_STATES, ACTION, JOINT_ANGLES
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    ALL_STATES: 9,
    ACTION: 3
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/pybullet_example/'

common = {
    'experiment_name': 'pybullet_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentPyBullet,
    "taskname" : 'KukaBulletEnv-v0',
    'render' : False,
    'x0': np.array([ 5.32117332e-01, -1.10671468e-03,  4.52298438e-01,  3.14129037e+00,
                     1.43835457e-03, -3.14137140e+00, -1.04033291e-01,  6.11723252e-02,
                     -1.32389020e+00]),
    'target_state': np.array([ 5.32117332e-01, -1.10671468e-03,  4.52298438e-01,  3.14129037e+00,
                               1.43835457e-03, -3.14137140e+00, -1.04033291e-01,  6.11723252e-02,
                               -1.32389020e+00]),
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [ALL_STATES],
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 5.0,
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([5e-5, 5e-5, 5e-5])
}

state_cost = {
    'type': CostState,
    'data_types': {
        ALL_STATES: {
            'wp': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'target_state': agent["target_state"],
        }
    }
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': 10,
    'num_samples': 5,
    'common': common,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'dQ': algorithm['init_traj_distr']['dQ'],
}

common['info'] = generate_experiment_info(config)
