""" This file defines an agent for the Carla simulator environment. """
import copy

import numpy as np

import gym
import gym_carla
import carla

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_CARLA
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ALL_STATES, TRACKING, OBS_AVOI


class AgentCarla(Agent):
    """
    All communication between the algorithms and Carla env is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_CARLA)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        params = {
            'number_of_vehicles': 0,
            'number_of_walkers': 0,
            'display_size': 256,  # screen size of bird-eye render
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 0.1,  # time interval between two frames
            'discrete': False,  # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
            'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
            'port': 2000,  # connection port
            'town': 'Town01',  # which town to simulate
            'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
            'max_time_episode': 1000,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'obs_range': 32,  # observation range (meter)
            'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
            'd_behind': 12,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 2.0,  # threshold for out of lane
            'desired_speed': 8,  # desired speed (m/s)
            'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
            'target_waypt_index': 1,  # index of the target way point
        }
        self._setup_world(params)

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, params):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self.x0 = self._hyperparams["x0"]
        self._world = [gym.make('carla-v0', params=params)
                       for _ in range(self._hyperparams['conditions'])]

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        obs_carla = self._world[condition].reset()
        CL_X = {TRACKING: obs_carla['tracking'],
                OBS_AVOI: obs_carla['obs_avoi']}
        new_sample = self._init_sample(CL_X)
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    obs_carla, _, _, _ = self._world[condition].step(U[t, :])
                CL_X = {TRACKING: obs_carla['tracking'],
                        OBS_AVOI: obs_carla['obs_avoi']}
                self._set_sample(new_sample, CL_X, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, CL_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, CL_X, -1)
        return sample

    def _set_sample(self, sample, CL_X, t):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            CL_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
        for sensor in CL_X.keys():
            sample.set(sensor, np.array(CL_X[sensor]), t=t+1)