
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:26:05 2020
@author: Youssef
"""

# !/usr/bin/env python
# coding: utf-8


# from jobGenerator import create_workload

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, MultiBinary, Discrete, Dict, flatten, flatten_space
from env.job_generator import create_workload


class MCVBEnv(gym.Env):
    def __init__(self, env_config={'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4, 'buffer_length': 10}):
        #add here description of each parameter
        self.seed()
        self.time = 0
        self.buffer_length = env_config['buffer_length']
        self.job_num = np.random.randint(low=3, high=3 * self.buffer_length)
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=1, high=self.job_num)  # env_config['job_density']
        self.speed = 1
        self.buffer_length = env_config['buffer_length']
        self.dummy_jobs = None
        #self.observation_buffer = np.zeros((self.buffer_length, 7))
        self.job_num = np.random.uniform(low=3, high=3*self.buffer_length)
        workload = np.zeros((self.job_num, 7))
        workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1]-workload_raw[:, 2])]
        self.workload = np.abs(workload)

        if self.buffer_length > self.job_num:
            self.dummy_jobs = np.zeros((self.buffer_length-self.job_num, 7))
            self.dummy_jobs[:, :4] = create_workload(self.buffer_length-self.job_num, self.total_load,
                                                     self.lo_per, self.job_density)
            self.dummy_jobs[:, [5, 6]] = 1
            if len(self.dummy_jobs.shape) == 1:
                self.dummy_jobs = self.dummy_jobs.reshape(1, -1)

        self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
        self.action_space = Discrete(self.buffer_length)
        self.observation_space = Dict({
            #'action_mask': Box(0, 1, shape=(self.buffer_length,)),
            #'avail_actions': Box(-10, 10, shape=(self.buffer_length, 2)),
            #'MCenv': Dict({
            'RDP_jobs': Box(low=0, high=np.inf, shape=(self.buffer_length, 3)),
            'CRSE_jobs': MultiBinary(self.buffer_length*4),
            'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
            })
        #})
        #self.observation_space flatten_space(self.observation_space_dict)
        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1 #jobs that can't be done anyways
        #TODO: handle cases of multiple switches between degradation and normal execution
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.random.uniform(low=self.total_load)
        self.action_mask = np.ones(self.buffer_length)
        self._update_available()
        thetas = np.arange(0, 360, 360 / self.buffer_length)[..., None]
        self.action_assignments = np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):

        done = self._done()
        reward = 0
        action = self.obs_idx[act] #convert action to workload space
        #if not self.action_mask[action]:
        #    raise ValueError(
        #        "Chosen action was not one of the non-zero action embeddings",
        #        action, self.action_assignments, self.action_mask)
        time = max(self.time, self.workload[action, 0])

        if time >= self.degradation_schedule:
            self.speed = self.degradation_speed
            time += self.workload[action, 2] / self.speed
        elif self.workload[action, 2] + time < self.degradation_schedule:
            time += self.workload[action, 2]
        else:
            time_in_norm = self.degradation_schedule-time
            self.speed = self.degradation_speed
            time_in_deg = (self.workload[action][2]-time_in_norm)/self.speed
            time += time_in_norm + time_in_deg
        # double check, as in case of degradation, time will not increment properly which might lead to the
        # starvation a job unexpectedly
        if time <= self.workload[action, 1]:
            self.time = time
            self.workload[action, 6] = 1
            self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
            starved_condition = (self.time >= self.workload[:, 1]) * (1-self.workload[:, 6]).astype(bool)
            self.workload[:, 5][starved_condition] = 1
            will_starve_condition = (self.time + self.workload[:, 2]/self.speed > self.workload[:, 1])\
                                    *(1-self.workload[:, 6]).astype(bool)
            self.workload[:, 5][will_starve_condition] = 1
            done = self._done()
            # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)

            if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
                reward += np.sum(self.workload[:, 6])

        return self._get_obs(), reward, done, {}

    def _update_available(self):
        #TODO(Karim): How to add Criticality
        to_evict = np.argwhere(
            self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)).squeeze(1)
        to_load = np.setdiff1d(np.argwhere(
            np.invert(self.workload[:, 5].astype(bool) & self.workload[:, 6].astype(bool))).squeeze(1),
                               self.obs_idx)

        if to_evict.shape[0] != 0 and to_load.shape[0] != 0:
            min_shape = min(to_evict.shape[0], to_load.shape[0])
            to_load = to_load.sort() #make sure that we load llf
            self.obs_idx[to_evict[:min_shape]] = self.workload[to_load[:min_shape]]

        self.action_mask[self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)] = 0

    def _get_obs(self):
        self._update_available()
        if self.dummy_jobs is not None:
            buffer = np.concatenate([self.workload[self.obs_idx], self.dummy_jobs], axis = 0)

        obs_dict = dict({#'action_mask': self.action_mask,
                         #'avail_actions': self.action_assignments,  # *self.action_mask,
                         #'MCenv': dict({
                         'RDP_jobs': np.array(buffer[:, :3]),
                         'CRSE_jobs': np.array(buffer[:, 3:]).flatten(),
                         'Processor': np.array([1, 0]).flatten()
                                        })
                         #})
        return obs_dict #flatten(self.observation_space_dict, obs_dict)

    def reset(self):
        self.time = 0

        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=1, high=self.job_num)  # env_config['job_density']
        self.speed = 1
        #self.buffer_length = env_config['buffer_length']
        self.dummy_jobs = None
        # self.observation_buffer = np.zeros((self.buffer_length, 7))
        self.job_num = np.random.uniform(low=3, high=3 * self.buffer_length)
        workload = np.zeros((self.job_num, 7))
        workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        workload[:, :5] = workload_raw[np.argsort(workload_raw[:, 1] - workload_raw[:, 2])]
        self.workload = np.abs(workload)

        if self.buffer_length > self.job_num:
            self.dummy_jobs = np.zeros((self.buffer_length - self.job_num, 7))
            self.dummy_jobs[:, :5] = create_workload(self.buffer_length - self.job_num, self.total_load,
                                                     self.lo_per, self.job_density)
            self.dummy_jobs[:, [5, 6]] = 1
            if len(self.dummy_jobs.shape) == 1:
                self.dummy_jobs = self.dummy_jobs.reshape(1, -1)

        self.obs_idx = np.arange(min(self.buffer_length, self.job_num))

        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.random.uniform(low=self.total_load)
        self.action_mask = np.ones(self.job_num)
        self._update_available()
        return self._get_obs()

    def _done(self):
        return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())

env = MCVBEnv()
observation=env.reset()
done = env._done()
while not done:
    action = np.random.randint(0, 9)
    print("Action: ", action)
    observation, reward, done, empty = env.step(action)
    print(observation)