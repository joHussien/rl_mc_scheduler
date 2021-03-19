

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:26:05 2020
@author: Youssef
"""

#!/usr/bin/env python
# coding: utf-8


#from jobGenerator import create_workload

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, MultiBinary, Discrete, Dict, flatten, flatten_space,tuple
from env.job_generator import create_workload
from scipy.stats import loguniform
import sys
import getopt


# class MCEnv(gym.Env):
#     def __init__(self, env_config= {'job_num': 10, 'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4}):
#         #add here description of each parameter
#         self.time = 0
#         self.job_num = env_config['job_num']
#         self.total_load = env_config['total_load']#np.random.uniform(low=0.2, high=1)
#         self.lo_per = env_config['lo_per']#np.random.uniform(low=0, high=1-2/self.job_num)
#         self.job_density = env_config['job_density']#np.random.randint(low=1, high=self.job_num)
#         self.speed = 1
#
#         workload = np.zeros((self.job_num, 7))
#         workload[:, :4] = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
#         self.workload = np.abs(workload) #negative processing time
#
#         self.action_space = Discrete(self.job_num)
#         self.observation_space = Dict({
#             #'action_mask': Box(0, 1, shape=(self.job_num,)),
#             #'avail_actions': Box(-10, 10, shape=(self.job_num, 2)),
#             #'MCenv': Dict({
#                 'RDP_jobs': Box(low=0, high=np.inf, shape=(self.job_num, 3)),
#                 'CRSE_jobs': MultiBinary(self.job_num * 4),
#                 'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
#             })
#         #})
#         #self.observation_space flatten_space(self.observation_space_dict)
#         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#         self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1 #jobs that can't be done anyways
#         #TODO: handle cases of multiple switches between degradation and normal execution
#         self.seed()
#         self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
#         self.degradation_speed =  np.random.uniform(low=self.total_load)#np.around(loguniform.rvs(self.total_load, 1e0), decimals=2)
#         self.action_mask = np.ones(self.job_num)
#         self._update_available()
#         thetas = np.arange(0, 360, 360 / self.job_num)[..., None]
#         self.action_assignments = np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def step(self, action):
#         #define step logic
#         #if not self.action_mask[action]:
#         #    raise ValueError(
#         #        "Chosen action was not one of the non-zero action embeddings",
#         #        action, self.action_assignments, self.action_mask)
#
#         done = self._done()
#         prev_workload = np.copy(self.workload)
#         reward = 0
#
#         if self.workload[action, 5].astype(bool) or self.workload[action, 6].astype(bool):
#             return self._get_obs(), -10, done, {}
#
#         time = max(self.time, self.workload[action, 0])
#
#         if time >= self.degradation_schedule:
#             self.speed = self.degradation_speed
#             time += self.workload[action, 2] / self.speed
#         elif self.workload[action, 2] + time < self.degradation_schedule:
#             time += self.workload[action, 2]
#         else:
#             time_in_norm = self.degradation_schedule-time
#             self.speed = self.degradation_speed
#             time_in_deg = (self.workload[action][2]-time_in_norm)/self.speed
#             time += time_in_norm + time_in_deg
#         # double check, as in case of degradation, time will not increment properly which might lead to the
#         # starvation a job unexpectedly
#         if time <= self.workload[action, 1]:
#             self.time = time
#             self.workload[action, 6] = 1
#             self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#             starved_condition = (self.time >= self.workload[:, 1]) * (1-self.workload[:, 6]).astype(bool)
#             self.workload[:, 5][starved_condition] = 1
#             will_starve_condition = (self.time + self.workload[:, 2]/self.speed > self.workload[:, 1])\
#                                     *(1-self.workload[:, 6]).astype(bool)
#             self.workload[:, 5][will_starve_condition] = 1
#             done = self._done()
#             # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
#             if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
#                 reward += np.sum(self.workload[:, 6])
#
#         return self._get_obs(), reward, done, {}
#
#     def _update_available(self):
#         self.action_mask[self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)] = 0
#
#     def _get_obs(self):
#         assert self.action_mask.any()
#         self._update_available()
#         obs_dict = dict({#'action_mask': self.action_mask,
#                          #'avail_actions': self.action_assignments, #*self.action_mask,
#                          #'MCenv': dict({
#                          'RDP_jobs': np.array(self.workload[:, :3]),
#                          'CRSE_jobs': np.array(self.workload[:, 3:]).flatten(),
#                          'Processor': np.array([1, 0]).flatten()
#                               })
#                          #})
#         return obs_dict #flatten(self.observation_space_dict, obs_dict)
#
#     def reset(self):
#         #self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
#         #self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
#         #self.job_density = np.random.randint(low=1, high=self.job_num)  # env_config['job_density']
#         self.speed = 1
#         self.time = 0
#
#         workload = np.zeros((self.job_num, 7))
#         workload[:, :4] = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
#         self.workload = np.abs(workload)  # negative processing time
#         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#         self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1
#         self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
#         self.degradation_speed = np.random.uniform(low=self.total_load)#np.around(loguniform.rvs(self.total_load, 1e0), decimals=2)
#         self.action_mask = np.ones(self.job_num)
#         self._update_available()
#         return self._get_obs()
#
#     def _done(self):
#         return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())
#
#
# class MCVBEnv(gym.Env):
#     def __init__(self, env_config={'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4, 'buffer_length': 5}):
#         #add here description of each parameter
#         self.seed()
#         self.time = 0
#         self.buffer_length = env_config['buffer_length']
#         self.job_num = np.random.randint(low=3, high=3 * self.buffer_length)
#         self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
#         self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
#         self.job_density = np.random.randint(low=1, high=self.job_num)  # env_config['job_density']
#         self.speed = 1
#         workload = np.zeros((self.job_num, 7))
#         workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
#         workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1]-workload_raw[:, 2])]
#         self.workload = np.abs(workload)
#         self.dummy_jobs = None
#         if self.buffer_length > self.job_num:
#             dummy_len = self.buffer_length - self.job_num
#             self.dummy_jobs = np.zeros((dummy_len, 7))
#             min_feat, max_feat = np.amin(self.workload[:, :3], axis=0), np.amax(self.workload[:, :3], axis=0)
#             self.dummy_jobs[:, :3] = np.around(
#                 np.random.uniform(size=(dummy_len, 3)) * (max_feat - min_feat) + min_feat, decimals=3)
#             self.dummy_jobs[:, [5, 6]] = 1
#
#         self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
#         print(self.obs_idx)
#         self.action_space = Discrete(self.buffer_length)
#         self.observation_space = Dict({
#             #'action_mask': Box(0, 1, shape=(self.buffer_length,)),
#             #'avail_actions': Box(-10, 10, shape=(self.buffer_length, 2)),
#             #'MCenv': Dict({
#             'RDP_jobs': Box(low=0, high=np.inf, shape=(self.buffer_length, 3)),
#             'CRSE_jobs': MultiBinary(self.buffer_length*4),
#             'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
#             })
#         #})
#         #self.observation_space flatten_space(self.observation_space_dict)
#         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#         self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1 #jobs that can't be done anyways
#         #TODO: handle cases of multiple switches between degradation and normal execution
#         self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
#         self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2)
#         self.action_mask = np.zeros(self.buffer_length)
#         self._update_available()
#         thetas = np.arange(0, 360, 360 / self.buffer_length)[..., None]
#         self.action_assignments = np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)
#         print("Length of Generated Workload: ", len(self.workload))
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     # def step(self, act_buffer):
#     #
#     #     done = self._done()
#     #     reward = 0
#     #     if act_buffer >= self.obs_idx.shape[0]:
#     #         print("Action selected not in the buffer currently")
#     #         return self._get_obs(), -10, done, {}
#     #     action = self.obs_idx[act_buffer] #convert action to workload space
#     #     #if not self.action_mask[action]:
#     #     #    raise ValueError(
#     #     #        "Chosen action was not one of the non-zero action embeddings",
#     #     #        action, self.action_assignments, self.action_mask)
#     #
#     #     if self.workload[action, 5].astype(bool) or self.workload[action, 6].astype(bool):
#     #         return self._get_obs(), -10, done, {}
#     #     time = max(self.time, self.workload[action, 0])
#     #
#     #     if time >= self.degradation_schedule:
#     #         self.speed = self.degradation_speed
#     #         time += self.workload[action, 2] / self.speed
#     #     elif self.workload[action, 2] + time < self.degradation_schedule:
#     #         time += self.workload[action, 2]
#     #     else:
#     #         time_in_norm = self.degradation_schedule-time
#     #         self.speed = self.degradation_speed
#     #         time_in_deg = (self.workload[action][2]-time_in_norm)/self.speed
#     #         time += time_in_norm + time_in_deg
#     #     # double check, as in case of degradation, time will not increment properly which might lead to the
#     #     # starvation a job unexpectedly
#     #     if time <= self.workload[action, 1]:
#     #         self.time = time
#     #         self.workload[action, 6] = 1
#     #         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#     #         starved_condition = (self.time >= self.workload[:, 1]) * (1-self.workload[:, 6]).astype(bool)
#     #         self.workload[:, 5][starved_condition] = 1
#     #         will_starve_condition = (self.time + self.workload[:, 2]/self.speed > self.workload[:, 1])\
#     #                                 *(1-self.workload[:, 6]).astype(bool)
#     #         self.workload[:, 5][will_starve_condition] = 1
#     #         done = self._done()
#     #         # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
#     #
#     #         if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
#     #             reward += np.sum(self.workload[:, 6])
#     #
#     #     return self._get_obs(), reward, done, {}
#
#     # def _update_available(self):
#     #     #TODO(Karim): How to add Criticality
#     #     print("Initial Map: ", self.obs_idx)
#     #     to_evict = np.argwhere(
#     #         self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)).squeeze(1)
#     #     to_load = np.setdiff1d(np.argwhere(np.invert(self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool))).squeeze(1),
#     #                            self.obs_idx)
#     #     # we find the values which are in workload and not in mapping_obs and those to be loaded
#     #     print("To be evicted from Buffer: ", to_evict)
#     #     print("To be Loaded To Buffer from Workload: ", to_load)
#     #     #if to_evict.shape[0] != 0 and to_load.shape[0] == 0:
#     #     #    self.obs_idx = np.delete(self.obs_idx, to_evict)
#     #     #    self.dummy_jobs = np.zeros((self.buffer_length-self.obs_idx.shape[0], 7))
#     #     #el
#     #     if to_evict.shape[0] != 0 and to_load.shape[0] != 0:
#     #         min_shape = min(to_evict.shape[0], to_load.shape[0])
#     #         to_load = np.sort(to_load) #make sure that we load llf
#     #         self.obs_idx[to_evict[:min_shape]] = to_load[:min_shape]
#     #         #self.obs_idx[to_evict[:min_shape]] = self.workload[to_load[:min_shape]]
#     #         #if to_load.shape[0] < to_evict.shape[0]:
#     #         #    self.obs_idx[to_evict[:to_load.shape[0]]] = to_load[:to_load.shape[0]]
#     #         #    remaining = to_evict[to_evict.shape[0] - to_load.shape[0] - 1:]
#     #         #    self.obs_idx = np.delete(self.obs_idx, remaining)
#     #         #    self.dummy_jobs = np.zeros((self.buffer_length-self.obs_idx.shape[0], 7))
#     #         #   print("Current Map: ", self.obs_idx)
#     #         #else:
#     #         #    self.obs_idx[to_evict[:min_shape]] = to_load[:min_shape]
#     #         #    print("Current Map: ", self.obs_idx)
#     #
#     #     self.action_mask[:self.obs_idx.shape[0]][np.invert(self.workload[self.obs_idx, 5].astype(bool) |
#     #                                                        self.workload[self.obs_idx, 6].astype(bool))] = 1
#     def step(self, act):
#         done = self._done()
#         reward = 0
#         action = self.obs_idx[act]  # convert action to workload space
#
#         #if not self.action_mask[action]:
#         #    raise ValueError(
#         #        "Chosen action was not one of the non-zero action embeddings",
#         #        action, self.action_assignments, self.action_mask)
#
#         if act == len(self.obs_idx):
#
#             #this means wait till the next release
#             time = np.amin(self.workload[np.invert(self.workload[:,4].astype(bool)), 0])
#             if time >= self.degradation_schedule:
#                 self.speed = self.degradation_speed
#             self.time = time
#             return self._get_obs(), reward, done, {}
#
#         time = max(self.time, self.workload[action, 0])
#
#         if time >= self.degradation_schedule:
#             self.speed = self.degradation_speed
#             time += self.workload[action, 2] / self.speed
#         elif self.workload[action, 2] + time < self.degradation_schedule:
#             time += self.workload[action, 2]
#         else:
#             time_in_norm = self.degradation_schedule-time
#             self.speed = self.degradation_speed
#             time_in_deg = (self.workload[action][2]-time_in_norm)/self.speed
#             time += time_in_norm + time_in_deg
#         # double check, as in case of degradation, time will not increment properly which might lead to the
#         # starvation a job unexpectedly
#         if time <= self.workload[action, 1]:
#             self.time = time
#             self.workload[action, 6] = 1
#             self._update_workload()
#             done = self._done()
#             # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
#
#             if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
#                 reward += np.sum(self.workload[:, 6])
#
#         return self._get_obs(), reward, done, {}
#
#     def _update_available(self):
#
#         to_evict = np.argwhere(
#             self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)
#         ).squeeze(1)
#         to_load = np.setdiff1d(np.argwhere(
#             np.invert(self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)) &
#             self.workload[:, 4].astype(bool)
#         ).squeeze(1), self.obs_idx)
#
#         if to_load.shape[0] != 0:
#             to_load = np.sort(to_load)
#             #how much do we need to evict to add all
#             num_evict = min(min(0, self.obs_idx.shape[0]+to_load.shape[0]-self.buffer_length),
#                             to_evict.shape[0])
#             if num_evict != 0:
#                 self.obs_idx[to_evict[:num_evict]] = to_load[:num_evict]
#             # to_evict = np.array(to_evict[:num_evict]) if num_evict ==1 else to_evict[:num_evict]
#             # self.obs_idx = self.obs_idx[(self.obs_idx[..., None] != to_evict[None, ...]).all(1)]
#
#             if to_load.shape[0]-num_evict > 0:
#                 to_load = to_load[num_evict:]
#                 self.obs_idx = np.concatenate([self.obs_idx, to_load[:min(self.buffer_length-self.obs_idx,
#                                                  to_load.shape[0])]], axis=None)
#             self.obs_idx = np.sort(self.obs_idx)
#
#         #if released and starved or executed it's overwritten to 0
#         self.action_mask[:self.obs_idx.shape[0]] = np.invert(
#             self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)) #\
#         # self.workload[self.obs_idx, 4].astype(bool) |
#     def _get_obs(self):
#         self._update_available()
#         buffer = self.workload[self.obs_idx]
#         if self.dummy_jobs is not None:
#             buffer = np.concatenate([buffer, self.dummy_jobs], axis=0)
#
#         print("Current Time: ", self.time, "Speed: ", self.speed)
#         print("Current Workload: ", self.workload)
#
#         obs_dict = dict({#'action_mask': self.action_mask,
#                          #'avail_actions': self.action_assignments,  # *self.action_mask,
#                          #'MCenv': dict({
#                          'RDP_jobs': np.array(buffer[:, :3]),
#                          'CRSE_jobs': np.array(buffer[:, 3:]).flatten(),
#                          'Processor': np.array([1, 0]).flatten()
#                                         })
#                          #})
#         return obs_dict #flatten(self.observation_space_dict, obs_dict)
#
#     def reset(self):
#         self.time = 0
#         self.job_num = np.random.randint(low=3, high=3 * self.buffer_length)
#         self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
#         self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
#         self.job_density = np.random.randint(low=1, high=self.job_num)  # env_config['job_density']
#         self.speed = 1
#         workload = np.zeros((self.job_num, 7))
#         workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
#         workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1] - workload_raw[:, 2])]
#         self.workload = np.abs(workload)
#
#         self.dummy_jobs = None
#         if self.buffer_length > self.job_num:
#             dummy_len = self.buffer_length - self.job_num
#             self.dummy_jobs = np.zeros((dummy_len, 7))
#             min_feat, max_feat = np.amin(self.workload[:, :3], axis=0), np.amax(self.workload[:, :3], axis=0)
#             self.dummy_jobs[:, :3] = np.around(
#                 np.random.uniform(size=(dummy_len, 3)) * (max_feat - min_feat) + min_feat, decimals=3)
#             self.dummy_jobs[:, [5, 6]] = 1
#
#         self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
#
#         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#         self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1
#         self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
#         self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2)
#         self.action_mask = np.zeros(self.buffer_length)
#         self._update_available()
#         return self._get_obs()
#
#     def _done(self):
#         return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())
#
#     def final(self):
#         if self._done():
#             print("Final Workload after done:",self.workload)
#
#
# class MCOEnv(gym.Env):
#     def __init__(self, env_config= {'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4, 'buffer_length':10}):
#         #add here description of each parameter
#         self.seed()
#         self.time = 0
#         self.buffer_length = env_config['buffer_length']
#         self.job_num = np.random.randint(low=3, high=3 * self.buffer_length)
#         self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
#         self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
#         self.job_density = np.random.randint(low=1, high=self.job_num)  # env_config['job_density']
#         self.speed = 1
#         workload = np.zeros((self.job_num, 7))
#         workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
#         workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1]-workload_raw[:, 2])]
#         self.workload = np.abs(workload)
#         self.action_space = Discrete(self.buffer_length+1)
#         self.observation_space = Dict({
#             #'action_mask': Box(0, 1, shape=(self.buffer_length,)),
#             #'avail_actions': Box(-10, 10, shape=(self.buffer_length, 2)),
#             #'MCenv': Dict({
#             'RDP_jobs': Box(low=0, high=np.inf, shape=(self.buffer_length, 3)),
#             'CRSE_jobs': MultiBinary(self.buffer_length*4),
#             'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
#             })
#         #})
#         #self.observation_space flatten_space(self.observation_space_dict)
#         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#         self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1 #jobs that can't be done anyways
#         #TODO: handle cases of multiple switches between degradation and normal execution
#         self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
#         self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2)
#         self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
#         self.obs_idx = self.obs_idx[self.workload[self.obs_idx, 4].astype(bool)] #remove unreleased jobs (must be removed)
#         self.action_mask = np.zeros(self.buffer_length)
#         self._update_available()
#         thetas = np.arange(0, 360, 360 / self.job_num)[..., None]
#         self.action_assignments = np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)
#         print("Workload:" , " its length is : ", len(self.workload))
#         print(self.workload)
#         print("Initial Map:")
#         print(self.obs_idx)
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def step(self, act):
#         done = self._done()
#         reward = 0
#         if act == self.buffer_length:
#
#             #this means wait till the next release
#             if (self.workload[:,4].all()): #i.e all jobs have been released but the workload is not done yet so can't wait till next rlease
#                 print("Chosen to wait till next release, however all jobs are released: ")
#                 return self._get_obs(), 0, done, {} #-10
#             time = np.amin(self.workload[np.invert(self.workload[:, 4].astype(bool)), 0])
#             if time >= self.degradation_schedule:
#                 self.speed = self.degradation_speed
#             self.time = time
#             self._update_workload()
#             return self._get_obs(), reward, done, {}
#
#         elif act >= len(self.obs_idx):
#             print("Chosen action Is not currently in buffer")
#             return self._get_obs(), -10, done, {}
#         else:
#             action = self.obs_idx[act]  # convert action to workload space
#
#         # if not self.action_mask[action]:
#         #     raise ValueError(
#         #         "Chosen action was not one of the non-zero action embeddings",
#         #         action, self.action_assignments, self.action_mask)
#
#
#         # we reached here so the job chosen is already in the buffer so we have to make sure it is not rusty
#         if (self.workload[self.obs_idx[act],5].astype(bool) or self.workload[self.obs_idx[act],6].astype(bool)):
#             print("Chosen job is a rusty job: ")
#             return self._get_obs(), -10, done, {}
#         time = max(self.time, self.workload[action, 0])
#
#         if time >= self.degradation_schedule:
#             self.speed = self.degradation_speed
#             time += self.workload[action, 2] / self.speed
#         elif self.workload[action, 2] + time < self.degradation_schedule:
#             time += self.workload[action, 2]
#         else:
#             time_in_norm = self.degradation_schedule-time
#             self.speed = self.degradation_speed
#             time_in_deg = (self.workload[action][2]-time_in_norm)/self.speed
#             time += time_in_norm + time_in_deg
#         # double check, as in case of degradation, time will not increment properly which might lead to the
#         # starvation a job unexpectedly
#         if time <= self.workload[action, 1]:
#             self.time = time
#             self.workload[action, 6] = 1
#             self._update_workload()
#             done = self._done()
#             # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
#
#             if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
#                 reward += np.sum(self.workload[:, 6])
#
#         return self._get_obs(), reward, done, {}
#
#     def _update_available(self):
#
#         to_evict = np.argwhere(
#             self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)
#         ).squeeze(1)
#         to_load = np.setdiff1d(np.argwhere(
#             np.invert(self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)) &
#             self.workload[:, 4].astype(bool)
#         ).squeeze(1), self.obs_idx)
#
#         if to_load.shape[0] != 0:
#             to_load = np.sort(to_load)
#             #how much do we need to evict to add all
#             num_evict = min(max(0, self.obs_idx.shape[0]+to_load.shape[0]-self.buffer_length), to_evict.shape[0])
#             #num_evict = min(to_load.shape[0], to_evict.shape[0])
#             if num_evict != 0:
#                 self.obs_idx[to_evict[:num_evict]] = to_load[:num_evict]
#             # to_evict = np.array(to_evict[:num_evict]) if num_evict ==1 else to_evict[:num_evict]
#             # self.obs_idx = self.obs_idx[(self.obs_idx[..., None] != to_evict[None, ...]).all(1)]
#
#             if to_load.shape[0]-num_evict > 0:
#                 minimum=min(self.buffer_length-len(self.obs_idx),to_load.shape[0])
#                 to_load = to_load[num_evict:]
#                 self.obs_idx = np.concatenate([self.obs_idx, to_load[:minimum]], axis=None)
#             self.obs_idx = np.sort(self.obs_idx)
#         #self.obs_idx=np.delete(self.obs_idx,to_evict)
#         print("To evict and To Load: ", to_evict," , " , to_load)
#         #if released and starved or executed it's overwritten to 0
#         self.action_mask[:self.obs_idx.shape[0]] = np.invert(
#             self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)) #\
#         # self.workload[self.obs_idx, 4].astype(bool) |
#
#     def _update_workload(self):
#         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#         starved_condition = (self.time >= self.workload[:, 1]) * (1 - self.workload[:, 6]).astype(bool)
#         self.workload[:, 5][starved_condition] = 1
#         will_starve_condition = (self.time + self.workload[:, 2] / self.speed > self.workload[:, 1]) \
#                                 *(1 - self.workload[:, 6]).astype(bool)
#         self.workload[:, 5][will_starve_condition] = 1
#
#     def _get_obs(self):
#         self._update_available()
#         buffer = self.workload[self.obs_idx, :]
#         if self.obs_idx.shape[0] < self.buffer_length:
#             #self.dummy_jobs = np.zeros((self.buffer_length-self.obs_idx.shape[0], 7))
#             self.dummy_jobs[:, :4] = create_workload(self.buffer_length-self.obs_idx.shape[0], self.total_load,self.lo_per, self.job_density)
#             self.dummy_jobs[:, [5, 6]] = 1
#             #TODO(): does buffer ever change in this scope
#             buffer = np.concatenate([buffer, self.dummy_jobs], axis = 0)
#         print("Current time step and speed: ", self.time, self.speed)
#         print("This Buffer-Map")
#         print(self.obs_idx)
#         print("Current Buffer")
#         print(buffer)
#         obs_dict = dict({#'action_mask': self.action_mask,
#                      #'avail_actions': self.action_assignments,  # *self.action_mask,
#                      #'MCenv': dict({
#                      'RDP_jobs': np.array(buffer[:, :3]),
#                      'CRSE_jobs': np.array(buffer[:, 3:]).flatten(),
#                      'Processor': np.array([1, 0]).flatten()
#                                     })
#                      #})
#
#         return obs_dict
#
#     def reset(self):
#         self.time = 0
#         self.job_num = np.random.randint(low=3, high=3 * self.buffer_length)
#         self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
#         self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
#         self.job_density = np.random.randint(low=1, high=self.job_num)  # env_config['job_density']
#         self.speed = 1
#         workload = np.zeros((self.job_num, 7))
#         workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
#         workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1] - workload_raw[:, 2])]
#         self.workload = np.abs(workload)
#         self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
#         self.workload[:, 5][self.time + self.workload[:, 2] / self.speed > self.workload[:, 1]] = 1  # jobs that can't be done anyways
#         # TODO: handle cases of multiple switches between degradation and normal execution
#         self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
#         self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2)
#         self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
#         self.obs_idx = self.obs_idx[
#             self.workload[self.obs_idx, 4].astype(bool)]  # remove unreleased jobs (must be removed)
#         self.action_mask = np.zeros(self.buffer_length)
#         self._update_available()
#         return self._get_obs()
#
#     def _done(self):
#         return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())
#
#     def final(self):
#         if self._done():
#             print("Final Workload after done:",self.workload)
# env = MCOEnv()
#  #done = env._done()
# done=False
# state=0
# while not done:
# #for i in range(20):
#     state=state+1
#     action = np.random.randint(0, 6)
#     print("Action: ", action)
#     observation, reward, done, empty = env.step(action)
#     #print(observation)
#     if done:
#         env.final()
#         break
#
# print("Finished this Eposide after ", state, " States")
def arg_parser():
    sys.argv[1]=sys.argv[1] #Name of the Experiment
    sys.argv[2]=int(sys.argv[2]) #Checkpoints
    sys.argv[3]=int(sys.argv[3]) #Iterations
    sys.argv[4]=int(sys.argv[4]) #Workers
    sys.argv[5]=float(sys.argv[5]) #Cpu/worker
    sys.argv[6]=float(sys.argv[6]) #Gamma
    sys.argv[7] = sys.argv[7]  # Gamma
# arg_parser()
# print("Name of the environment is ", sys.argv[1])
# print( "gamma ", sys.argv[6])
# print("Param: ", sys.argv[7])

def get_opt():

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]
    sum = 0

    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(argv, 'a:b:c:d:e:f:g:')
        # Check if the options' length is 2 (can be enhanced)
        if len(opts) == 0 or len(opts) <7 :
            print('usage: rllib_train.py -a <Exp_Name> -b<Checks_Freq> <Iterations_Num> <Worker_Num> <CPUs/Worker> <Gamma> <Parametric_Actions>')
        else:
            # opts=np.array(opts)
            # args= opts[:,[1]]
            # args = args.squeeze(1)
            # Iterate the options and get the corresponding values
            for opt, arg in opts:
                args.append(arg)
            args[1]=  int(args[1])
            args[2] = int(args[2])
            args[3] = int(args[3])
            args[4] = float(args[4])
            args[5] = float(args[5])
            args[6]= bool(args[6])
            return args
    except getopt.GetoptError:
        # Print something useful
        print('usage: rllib_train.py -a <Exp_Name> -b<Checks_Freq> <Iterations_Num> <Worker_Num> <CPUs/Worker> <Gamma> <Parametric_Actions>')
        sys.exit(2)
arguments = get_opt()
