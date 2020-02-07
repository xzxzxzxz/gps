""" This file defines an agent for the PyBullet simulator environment. """
import copy

import numpy as np

import gym
import pybullet_envs

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_PYBULLET
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ALL_STATES


class AgentPyBullet(Agent):
    """
    All communication between the algorithms and PyBullet is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_PYBULLET)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        self._setup_world(hyperparams['taskname'])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'taskname'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, taskname):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self.x0 = self._hyperparams["x0"]
        self._world = [gym.make(taskname)
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
        x = self._world[condition].reset()
        PB_X = {ALL_STATES: x}
        new_sample = self._init_sample(PB_X)
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
                    x, _, _, _ = self._world[condition].step(U[t, :])
                PB_X = {ALL_STATES: x}
                self._set_sample(new_sample, PB_X, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, PB_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, PB_X, -1)
        return sample

    def _set_sample(self, sample, PB_X, t):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
        for sensor in PB_X.keys():
            sample.set(sensor, np.array(PB_X[sensor]), t=t+1)