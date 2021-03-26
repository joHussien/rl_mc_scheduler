

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
from gym.spaces import Box, MultiBinary, Discrete, Dict
from env.job_generator import create_workload
from scipy.stats import loguniform
from gym.spaces.utils import flatten, flatten_space
from env.Filtering_EDF import filtering_workload
from gym import spaces
#Modifications needed to add preemption to the system
#1. We will apply preemption currently to the offline environement
#2.     Will add a flag if a job is started or not. CS"Chosen Status"
#3.     Work using remingin processing time which will be a column in itself RP "Remaining Processing"
#4.     Edit the rewarding system by the following modifications:
#if no degradation:
#           Penalty: -5 if preempted a lo-job
#           Penalty: -10 if preempted a Hi-ob
#           Reward: 2 scheduled a job that already started
#if degradation:
    # if preempted a low job give a reward
# Experiment with these rewarding values
class MCEnv(gym.Env):
    def __init__(self, env_config= {'job_num': 5, 'total_load': 1, 'lo_per': 0.3, 'job_density': 4}):
        #add here description of each parameter
        self.time = 0

        #Here should filter the workload and pass the size of the filtered workload to the self.job_num

        self.job_num = env_config['job_num']
        self.total_load = np.random.uniform(low=0.2, high=1)#env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1-2/self.job_num) #env_config['lo_per']
        self.job_density = np.random.randint(low=int(self.job_num*(1/4)), high=int(self.job_num*(1/2)))##env_config['job_density']
        self.speed = 1

        workload = np.zeros((self.job_num, 9))
        #[R,D,P,X,RS,SS,ES,CS,RP]
        # # -----added----- filling the rest of the workload with dummy jobs starved#
        # print(self.job_num)

        workload_filtered= filtering_workload(self.job_num, self.total_load, self.lo_per, self.job_density, self.time)
        workload[:workload_filtered.size//4, :4]=workload_filtered
        workload[workload_filtered.size//4:, 6]=1
        # print("Final Filtered-Scheduble instance using EDF on 1-speed: ")
        # print(workload)
        #hence the remaingin jobs in the initial workload will be filled with zeros
        # ----commented----#
        # workload[:, :4] = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        self.workload = np.abs(workload) #negative processing time

        # self.action_space = Discrete(self.job_num)
        #ToDo() Is the box upper boundary is 1 or what
        self.action_space = spaces.Tuple(( spaces.Discrete(self.job_num), spaces.Box(low=0, high=np.inf, shape=(1,))))
        self.observation_space_dict = Dict({
            # 'action_mask': Box(0, 1, shape=(self.job_num,)),
            # 'avail_actions': Box(-np.inf, np.inf, shape=(self.job_num, 4)),
            # 'MCenv': Dict({
               'RDP_jobs': Box(low=0, high=np.inf, shape=(self.job_num, 3)),
               'CRSEC_jobs': MultiBinary(self.job_num * 5),
                'RP_jobs': Box(low=0,high=np.inf, shape=(self.job_num, 1)),
               'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
            })
        # })
        self.observation_space = flatten_space(self.observation_space_dict)
        # ToDo(Youssef): added the RP column logic
        # RP =  procesing - update_qt for now update_qt=0
        self.update_qt = 0.0
        self.workload[:,8]=self.workload[:,2] - self.update_qt
        #print("Init Workload: \n ")
        #print(self.workload)
        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 8]/self.speed > self.workload[:, 1]] = 1 #jobs that can't be done anyways
         # self.workload[:, 7][self.time >= self.workload[:, 0]] = 1
        #TODO: handle cases of multiple switches between degradation and normal execution
        self.seed()
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2) #np.random.uniform(low=self.total_load)
        #ToDo this need to be edited
        self.action_mask = np.ones(self.job_num)
        #self.action_assignments = self.workload[:, :4]
        #thetas = np.arange(0, 360, 360 / self.job_num)[..., None]
        #self.thetas = np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)
        #self.action_assignments = np.zeros_like(self.thetas)
        self.action_assignments = np.zeros_like(self.workload[:, :4])
        self.action_assignments[self.action_mask.astype(bool)] = self.workload[self.action_mask.astype(bool), :4]
        #self.action_assignments[self.action_mask.astype(bool)] = self.thetas[self.action_mask.astype(bool)]
        self._update_available()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print("Current workload and time before stepping: ", self.time, "self.speed,",self.speed,"\n", self.workload)
        #print("action taken: ",action)

        #TODO : Very Important I have a current problem, now I am forcing the system to preempt always by always subtracting 1
        #define step logic
        # if not self.action_mask[action]:
        #    raise ValueError(
        #        "Chosen action was not one of the non-zero action embeddings",
        #        action, self.action_assignments, self.action_mask, self.workload)

        done = self._done()
        prev_workload = np.copy(self.workload)
        reward = 0
        # ToDO(Youssef): Rewarding system edits here
        #added here a test if the selected job has RP = 0
        #new checking if chosen preemption time is greater than processing time of the job
        # or you decided that you will not execute it then you should be penalized for selecting it
        if self.workload[action[0],2]< action[1][0] or action[1][0] == 0 :
            obs = self._get_obs()
            done = self._done()
            return obs, -10, done, {}
        if self.workload[action[0], 5].astype(bool) or self.workload[action[0], 6].astype(bool) :
            obs = self._get_obs()
            done = self._done()
            return obs, -10, done, {}
        # print("Reached here")
        #label the job as chosen if not chosen before
        self.update_qt = action[1][0]

        if (self.workload[action[0],7]==0):
            #label this job as preempted
            self.workload[action[0], 7]=1
        else:
            #Then this job is preempted and have to incur a penalty according to its criticality level
            if(self.workload[action[0],3]==0):
                reward-=5
            elif (self.workload[action[0], 3] == 1):
                reward -= 10
        #ToDO(Youssef): see here
        # ToDo(Youssef): added the RP column logic
        #compute RP for each job
        # self.workload[:, 8] = np.abs(self.workload[:,1] - self.workload[:, 2] - self.time)
        #I think that this shouldnt be changed
        #If the job is not chosen before
        # if self.workload[action,7]==0:
        #ToDo this needs to be checked
        time = max(self.time, self.workload[action[0], 0])
        #print("Before update time: ",time)
        # else:
        #     #As we preempted this job before then for sure self.time must be equal than the release
        #     time=self.time
        #edit here
        # ToDO(Youssef): see here
        if time >= self.degradation_schedule:
            self.speed = 1.0 #no degradation test
            #self.speed = self.degradation_speed
            # Think aboutttttttttttttttt thissssssssssssss this what dr nouri was talking about, should I update time
            # with only one time unit
            # time += self.workload[action, 2] / self.speed
            self.update_qt = self.update_qt / self.speed
            time += self.update_qt
            #ToDo Put the reward of preempting a low job in degraded mode here
        # elif self.workload[action, 2] + time < self.degradation_schedule:
        #ToDo should it be -processing or -reminaingprocessing "the last parameter downbelow"
        elif (self.workload[action[0],1] - time - self.workload[action[0], 2] )   < self.degradation_schedule:
            # Think aboutttttttttttttttt thissssssssssssss this what dr nouri was talking about, should I update time
            # with only one time unit
            # time += self.workload[action, 2]
            self.update_qt = self.update_qt
            time += self.update_qt
        else:
            time_in_norm = self.degradation_schedule-time
            self.speed = 1  # no degradation test
            # self.speed = self.degradation_speed
            # Think aboutttttttttttttttt thissssssssssssss this what dr nouri was talking about, should I update time
            # with only one time unit
            #ToDo(youssef) There will be a problem here if I subtracted only one as it can be negative think abt it
            time_in_deg = (self.workload[action[0]][2]-time_in_norm)/self.speed
            # time_in_deg = (1 - time_in_norm) / self.speed
            time += time_in_norm + time_in_deg
        # double check, as in case of degradation, time will not increment properly which might lead to the
        # starvation a job unexpectedly
        #print("After time: ",time)
        if time <= self.workload[action[0], 1]:
            self.time = time
            #ToDo Think here about jobs with P < 1.0
            #print("Update Quantity",self.update_qt)
            self.workload[action[0], 8] = self.workload[action[0], 8] - self.update_qt
            #print("After Update", self.workload)
            if(self.workload[action[0],8]<=0 and self.workload[action[0],7]==1):
                self.workload[action[0], 6] = 1
            self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
            starved_condition = (self.time >= self.workload[:, 1]) * (1-self.workload[:, 6]).astype(bool)
            self.workload[:, 5][starved_condition] = 1
            will_starve_condition = ((self.time + self.workload[:, 8])/self.speed > self.workload[:, 1])\
                                    *(1-self.workload[:, 6]).astype(bool)
            self.workload[:, 5][will_starve_condition] = 1
            #print(self.workload)
            done = self._done()
            #print(done)
            # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
            if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
                reward += np.sum(self.workload[:, 6])
        obs = self._get_obs()
        done = self._done()
        return obs, reward, done, {}

    def _update_available(self):
        self.action_mask[self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)] = 0
        #self.action_assignments = np.zeros_like(self.thetas)
        #self.action_assignments[self.action_mask.astype(bool)] = self.thetas[self.action_mask.astype(bool)]
        self.action_assignments = np.zeros_like(self.workload[:, :4])
        self.action_assignments[self.action_mask.astype(bool)] = self.workload[self.action_mask.astype(bool), :4]

    def _get_obs(self):
        assert self.action_mask.any()
        self._update_available()
        obs_dict = dict({
                         # 'action_mask': self.action_mask,
                         # 'avail_actions': self.action_assignments, #*self.action_mask,
                         # 'MCenv': dict({
                             'RDP_jobs': np.array(self.workload[:, :3]),
                             'CRSEC_jobs': np.array(self.workload[:, 3:8]).flatten(),
                             'RP_jobs':np.array(self.workload[:, 8]).flatten(),
                             'Processor': np.array([self.degradation_speed, self.time]).flatten()
                              })
                         # })
        return flatten(self.observation_space_dict, obs_dict)
        #return obs_dict

    def reset(self):
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=int(self.job_num * (1 / 4)),
                                             high=int(self.job_num * (1 / 2)))  ##env_config['job_density']

        self.speed = 1
        self.time = 0

        workload = np.zeros((self.job_num, 9))
        # # -----added----- filling the rest of the workload with dummy jobs starved#
        # print(self.job_num)
        #print("Hey reset: job_num",self.job_num)
        workload_filtered = filtering_workload(self.job_num, self.total_load, self.lo_per, self.job_density, self.time)
        workload[:workload_filtered.size // 4, :4] = workload_filtered
        workload[workload_filtered.size // 4:, 6] = 1
        # print("Final Filtered-Scheduble instance using EDF on 1-speed: ")
        # print(workload)
        # hence the remaingin jobs in the initial workload will be filled with zeros
       #------------------------Commented---#
        # workload[:, :4] = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        #-----------------------------------#
        self.update_qt = 0.0
        self.workload[:,8] = self.workload[:,2] - self.update_qt
        self.workload = np.abs(workload)  # negative processing time
        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 8]/self.speed > self.workload[:, 1]] = 1
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2) #
        self.action_mask = np.ones(self.job_num)
        self.action_assignments = np.zeros_like(self.workload[:, :4])
        self.action_assignments[self.action_mask.astype(bool)] = self.workload[self.action_mask.astype(bool), :4]
        self._update_available()
        #thetas = np.arange(0, 360, 360 / self.job_num)[..., None]
        #self.thetas = np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)
        #self.action_assignments = np.zeros_like(self.thetas)
        #self.action_assignments[self.action_mask.astype(bool)] = self.thetas[self.action_mask.astype(bool)]
        return self._get_obs()

    def _done(self):
        return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())


class MCVBEnv(gym.Env):
    def __init__(self, env_config={'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4, 'buffer_length': 10}):
        #add here description of each parameter
        self.seed()
        self.time = 0
        self.buffer_length = env_config['buffer_length']
        self.job_num = np.random.randint(low=3, high=2 * self.buffer_length)
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))  # env_config['job_density']
        self.speed = 1
        workload = np.zeros((self.job_num, 7))
        workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1]-workload_raw[:, 2])]
        self.workload = np.abs(workload)
        self.dummy_jobs = None
        if self.buffer_length > self.job_num:
            dummy_len = self.buffer_length - self.job_num
            self.dummy_jobs = np.zeros((dummy_len, 7))
            min_feat, max_feat = np.amin(self.workload[:, :3], axis=0), np.amax(self.workload[:, :3], axis=0)
            self.dummy_jobs[:, :3] = np.around(
                np.random.uniform(size=(dummy_len, 3)) * (max_feat - min_feat) + min_feat, decimals=3)
            self.dummy_jobs[:, [5, 6]] = 1

        self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
        self.action_space = Discrete(self.buffer_length)
        self.observation_space_dict = Dict({
            'action_mask': Box(0, 1, shape=(self.buffer_length,)),
            'avail_actions': Box(-10, 10, shape=(self.buffer_length, 2)),
            'MCenv': Dict({
            'RDP_jobs': Box(low=0, high=np.inf, shape=(self.buffer_length, 3)),
            'CRSE_jobs': MultiBinary(self.buffer_length*4),
            'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
            })
        })
        self.observation_space = flatten_space(self.observation_space_dict)
        #print("Hola",(self.observation_space_dict))
        #print(str(self.observation_space_dict['MCenv']))
        #print(str(flatten_space(self.observation_space_dict)))
        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1 #jobs that can't be done anyways
        #TODO: handle cases of multiple switches between degradation and normal execution
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2) #np.random.uniform(low=self.total_load)
        self.action_mask = np.zeros(self.buffer_length)
        self._update_available()
        thetas = np.arange(0, 360, 360 / self.buffer_length)[..., None]
        self.action_assignments = np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)
        #print("Length of Generated Workload: ", len(self.workload))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act_buffer):
        if not self.action_mask[act_buffer]:
           raise ValueError(
               "Chosen action was not one of the non-zero action embeddings",
               act_buffer, self.action_assignments, self.action_mask)
        done = self._done()
        reward = 0
        if act_buffer >= self.obs_idx.shape[0]:
            #print("Action selected not in the buffer currently") #Youssef: I guess that this is handled by the parametric actions
            return self._get_obs(), 0, done, {}
        action = self.obs_idx[act_buffer] #convert action to workload space


        if self.workload[action, 5].astype(bool) or self.workload[action, 6].astype(bool):
            return self._get_obs(), -10, done, {}
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
        #print("Initial Map: ", self.obs_idx)
        to_evict = np.argwhere(
            self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)).squeeze(1)
        to_load = np.setdiff1d(np.argwhere(np.invert(self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool))).squeeze(1),
                               self.obs_idx)
        # we find the values which are in workload and not in mapping_obs and those to be loaded
        # print("To be evicted from Buffer: ", to_evict)
        # print("To be Loaded To Buffer from Workload: ", to_load)
        #if to_evict.shape[0] != 0 and to_load.shape[0] == 0:
        #    self.obs_idx = np.delete(self.obs_idx, to_evict)
        #    self.dummy_jobs = np.zeros((self.buffer_length-self.obs_idx.shape[0], 7))
        #el
        if to_evict.shape[0] != 0 and to_load.shape[0] != 0:
            min_shape = min(to_evict.shape[0], to_load.shape[0])
            to_load = np.sort(to_load) #make sure that we load llf
            self.obs_idx[to_evict[:min_shape]] = to_load[:min_shape]
            #self.obs_idx[to_evict[:min_shape]] = self.workload[to_load[:min_shape]]
            #if to_load.shape[0] < to_evict.shape[0]:
            #    self.obs_idx[to_evict[:to_load.shape[0]]] = to_load[:to_load.shape[0]]
            #    remaining = to_evict[to_evict.shape[0] - to_load.shape[0] - 1:]
            #    self.obs_idx = np.delete(self.obs_idx, remaining)
            #    self.dummy_jobs = np.zeros((self.buffer_length-self.obs_idx.shape[0], 7))
            #   print("Current Map: ", self.obs_idx)
            #else:
            #    self.obs_idx[to_evict[:min_shape]] = to_load[:min_shape]
            #    print("Current Map: ", self.obs_idx)

        self.action_mask[:self.obs_idx.shape[0]]=np.invert(self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool))

    def _get_obs(self):
        self._update_available()
        buffer = self.workload[self.obs_idx]
        if self.dummy_jobs is not None:
            buffer = np.concatenate([buffer, self.dummy_jobs], axis=0)

        # print("Current Time: ", self.time, "Speed: ", self.speed)
        # print("Current Workload: ", self.workload)

        obs_dict = dict({
                         'action_mask': self.action_mask,
                         'avail_actions': self.action_assignments,  # *self.action_mask,
                         'MCenv': dict({
                         'RDP_jobs': np.array(buffer[:, :3]),
                         'CRSE_jobs': np.array(buffer[:, 3:]).flatten(),
                         'Processor': np.array([1, 0]).flatten()
                                        })
                         })

        #return obs_dict
        #print("Hi",len(flatten(self.observation_space_dict, obs_dict)))
        #print(len(obs_dict))
        return flatten(self.observation_space, obs_dict)

    def reset(self):
        self.time = 0
        self.job_num = np.random.randint(low=3, high=2 * self.buffer_length)
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))  # env_config['job_density']
        self.speed = 1
        workload = np.zeros((self.job_num, 7))
        workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1] - workload_raw[:, 2])]
        self.workload = np.abs(workload)

        self.dummy_jobs = None
        if self.buffer_length > self.job_num:
            dummy_len = self.buffer_length - self.job_num
            self.dummy_jobs = np.zeros((dummy_len, 7))
            min_feat, max_feat = np.amin(self.workload[:, :3], axis=0), np.amax(self.workload[:, :3], axis=0)
            self.dummy_jobs[:, :3] = np.around(
                np.random.uniform(size=(dummy_len, 3)) * (max_feat - min_feat) + min_feat, decimals=3)
            self.dummy_jobs[:, [5, 6]] = 1

        self.obs_idx = np.arange(min(self.buffer_length, self.job_num))

        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2)#np.random.uniform(low=self.total_load)
        self.action_mask = np.zeros(self.buffer_length)
        self._update_available()

        return self._get_obs()

    def _done(self):
        return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())

    # def final(self):
    #     if self._done():
    #         print("Final Workload after done:",self.workload)


class MCOEnv(gym.Env):
    def __init__(self, env_config= {'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4, 'buffer_length':10}):
        #add here description of each parameter
        self.seed()
        self.time = 0
        self.buffer_length = env_config['buffer_length']
        self.job_num = np.random.randint(low=3, high=2 * self.buffer_length)
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))  # env_config['job_density']
        self.speed = 1
        workload = np.zeros((self.job_num, 7))
        #--------added
        # # -----added----- filling the rest of the workload with dummy jobs starved#
        workload_filtered= filtering_workload(self.job_num, self.total_load, self.lo_per, self.job_density, self.time)
        workload[:workload_filtered.size//4, :4]=workload_filtered
        # this line for lacity ordering
        #workload[:, :4] = workload_filtered[np.argsort(workload_filtered[:, 1] - workload_filtered[:, 2])]
        workload[workload_filtered.size//4:, 6]=1
        # hence the remaingin jobs in the initial workload will be filled with zeros
        # ----commented----#
        #commented-----#
        # workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        # workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1]-workload_raw[:, 2])]
        #------------------#
        self.workload = np.abs(workload)
        self.action_space = Discrete(self.buffer_length+1)
        self.observation_space = Dict({
            # 'action_mask': Box(0, 1, shape=(self.buffer_length+1,)),
            # 'avail_actions': Box(-np.inf, np.inf, shape=(self.buffer_length+1, 4)),
            # 'MCenv': Dict({
            'RDP_jobs': Box(low=0, high=np.inf, shape=(self.buffer_length, 3)),
            'CRSE_jobs': MultiBinary(self.buffer_length*4),
            'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
            })
        # })
        #self.observation_space= flatten_space(self.observation_space_dict)
        #self.workload[:, 4][self.time >= self.workload[:, 0]] = 1 (KF)
        #self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1 (KF)#jobs that can't be done anyways
        #TODO: handle cases of multiple switches between degradation and normal execution
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0), decimals=2) #np.random.uniform(low=self.total_load)
        self._update_workload()
        self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
        self.obs_idx = self.obs_idx[self.workload[self.obs_idx, 4].astype(bool)] #remove unreleased jobs (must be removed)
        self.action_mask = np.zeros(self.buffer_length + 1)
        self.action_mask[self.buffer_length] = 1
        #self.action_mask[:self.obs_idx.shape[0]] = self.workload[self.obs_idx, 4]
        #self.max_dim = max(self.buffer_length+1, self.job_num)
        #thetas = np.arange(0, 360, 360 / self.max_dim)[..., None]
        self.action_assignments = np.zeros((self.buffer_length + 1, 4))
        #self.action_assignments[self.action_mask.astype(bool)] = self.workload[self.action_mask.astype(bool), :4]
        self._update_available()
        #self.action_assignments = np.zeros() np.concatenate([np.sin(thetas), np.cos(thetas)], axis=-1)
        # #----#
        # print(self.workload)
        # #----#


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        done = self._done()
        # if not self.action_mask[act]:
        #
        #      #print("Chosen action was not one of the non-zero action embeddings")
        #      raise ValueError(
        #     "Chosen action was not one of the non-zero action embeddings",
        #           act, self.action_assignments, self.action_mask, )
        #      #return self._get_obs(), 0, done, {}
        reward = 0
        # if not self.action_mask.any():
        #     if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
        #         reward += np.sum(self.workload[:, 6])
        #         return self._get_obs(), reward, done, {}
        if act == self.buffer_length:

            #this means wait till the next release
            if (self.workload[:,4].all()): #i.e all jobs have been released but the workload is not done yet so can't wait till next rlease
                #print("Chosen to wait till next release, however all jobs are released: ")
                return self._get_obs(), 0, done, {} #-10
            time = np.amin(self.workload[np.invert(self.workload[:, 4].astype(bool)), 0])
            if time >= self.degradation_schedule:
                self.speed = self.degradation_speed
            self.time = time
            self._update_workload()
            obs = self._get_obs()
            done = self._done()
            return obs, reward, done, {}

        elif act >= len(self.obs_idx):
            #print("Chosen action Is not currently in buffer")
            obs = self._get_obs()
            done = self._done()
            return obs, 0, done, {}
        else:
            action = self.obs_idx[act]  # convert action to workload space

        # we reached here so the job chosen is already in the buffer so we have to make sure it is not rusty
        if (self.workload[self.obs_idx[act],5].astype(bool) or self.workload[self.obs_idx[act],6].astype(bool)):
            #print("Chosen job is a rusty job: ")
            obs = self._get_obs()
            done = self._done()
            return obs, -10, done, {}

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
            self._update_workload()
            done = self._done()
            # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)

            if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
                reward += np.sum(self.workload[:, 6])
        obs = self._get_obs()
        done = self._done()
        return obs, reward, done, {}

    def _update_available(self):

        to_evict = np.argwhere(
            self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)
        ).squeeze(1)
        to_load = np.setdiff1d(np.argwhere(
            np.invert(self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)) &
            self.workload[:, 4].astype(bool)
        ).squeeze(1), self.obs_idx)

        if to_load.shape[0] != 0:
            to_load = np.sort(to_load)
            #how much do we need to evict to add all
            num_evict = min(max(0, self.obs_idx.shape[0]+to_load.shape[0]-self.buffer_length), to_evict.shape[0])
            #num_evict = min(to_load.shape[0], to_evict.shape[0])
            if num_evict != 0:
                self.obs_idx[to_evict[:num_evict]] = to_load[:num_evict]
            # to_evict = np.array(to_evict[:num_evict]) if num_evict ==1 else to_evict[:num_evict]
            # self.obs_idx = self.obs_idx[(self.obs_idx[..., None] != to_evict[None, ...]).all(1)]

            if to_load.shape[0]-num_evict > 0:
                minimum = min(self.buffer_length-len(self.obs_idx),to_load.shape[0])
                to_load = to_load[num_evict:]
                self.obs_idx = np.concatenate([self.obs_idx, to_load[:minimum]], axis=None)
            self.obs_idx = np.sort(self.obs_idx)
        #self.obs_idx=np.delete(self.obs_idx,to_evict)
        #print("To evict and To Load: ", to_evict," , " , to_load)
        #if released and starved or executed it's overwritten to 0
        #self.action_mask[:self.obs_idx.shape[0]] = 1
        avail_released = np.invert((
            self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool)))
        self.action_mask[:self.obs_idx.shape[0]] = avail_released
        #print(self.action_mask.shape, self.action_assignments.shape)
        #print(self.action_assignments[:-1][self.action_mask[:-1].astype(bool), :].shape)
        #print(self.workload[self.obs_idx][avail_released].shape)#, self.workload[avail_released].shape)
        self.action_assignments[:-1][self.action_mask[:-1].astype(bool), :] = self.workload[self.obs_idx][avail_released][:,:4] #self.workload[self.obs_idx, 5].astype(bool) | self.workload[self.obs_idx, 6].astype(bool))
        self.action_assignments[self.buffer_length] = np.array([self.time+1,
                                                                self.time+1+np.mean(self.action_assignments[:, 1])-np.mean(self.action_assignments[:, 0]),
                                                                np.mean(self.time + self.action_assignments[:, 1]),
                                                                1] )

        if (self.workload[:,4].all()):
            self.action_mask[self.buffer_length] = 0
            self.action_assignments[self.buffer_length] = 0
            # print("No More Waiting, Current Workload: ", self.workload)

    def _update_workload(self):
        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        starved_condition = (self.time >= self.workload[:, 1]) * (1 - self.workload[:, 6]).astype(bool)
        self.workload[:, 5][starved_condition] = 1
        will_starve_condition = (self.time + self.workload[:, 2] / self.speed > self.workload[:, 1]) \
                                *(1 - self.workload[:, 6]).astype(bool)
        self.workload[:, 5][will_starve_condition] = 1

    def _get_obs(self):
        #assert self.action_mask.any()
        self._update_available()
        buffer = self.workload[self.obs_idx, :]
        if self.obs_idx.shape[0] < self.buffer_length:
            dummy_len = self.buffer_length-self.obs_idx.shape[0]
            self.dummy_jobs = np.zeros((dummy_len, 7))
            min_feat, max_feat = np.amin(self.workload[:, :3], axis=0), np.amax(self.workload[:, :3], axis=0)
            self.dummy_jobs[:, :3] = np.around(
                np.random.uniform(size=(dummy_len, 3)) * (max_feat - min_feat) + min_feat, decimals=3)
            # self.dummy_jobs[:, :4] = create_workload(self.buffer_length-self.obs_idx.shape[0], self.total_load,self.lo_per, self.job_density)
            self.dummy_jobs[:, [5, 6]] = 1
            #TODO(): does buffer ever change in this scope
            buffer = np.concatenate([buffer, self.dummy_jobs], axis=0)
        # print("Current time step and speed: ", self.time, self.speed)
        # print("This Buffer-Map")
        # print(self.obs_idx)
        # print("Current Buffer")
        # print(buffer)
        # temp=np.concatenate([self.action_assignments[:self.buffer_length+1],self.action_assignments[self.obs_idx],
        #                      self.action_assignments[self.buffer_length]],axis=0)
        obs_dict = dict({
                     # 'action_mask': self.action_mask,
                     # 'avail_actions': self.action_assignments[:self.buffer_length+1],  # *self.action_mask,
                     # 'MCenv': dict({
                     'RDP_jobs': np.array(buffer[:, :3]),
                     'CRSE_jobs': np.array(buffer[:, 3:]).flatten(),
                     'Processor': np.array([self.speed, self.time]).flatten()
                                    })
                     # })
        # #---#
        # print("Cureent Map, ", self.obs_idx)
        # print("Action_masking ", self.action_mask)
        # print("T: ",self.time,"S: ", self.speed)
        # #---#
        #return flatten(self.observation_space_dict,obs_dict)
        return obs_dict

    def reset(self):
        self.time = 0
        self.job_num = np.random.randint(low=3, high=2 * self.buffer_length)
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))  # env_config['job_density']
        self.speed = 1
        workload = np.zeros((self.job_num, 7))
        # --------added
        # # -----added----- filling the rest of the workload with dummy jobs starved#
        workload_filtered = filtering_workload(self.job_num, self.total_load, self.lo_per, self.job_density, self.time)
        workload[:workload_filtered.size // 4, :4] = workload_filtered
        #this line for lacity ordering
        # workload[:, :4] = workload_filtered[np.argsort(workload_filtered[:, 1] - workload_filtered[:, 2])]
        workload[workload_filtered.size // 4:, 6] = 1
        # hence the remaingin jobs in the initial workload will be filled with zeros
        # ----commented----#
        # commented-----#
        # workload_raw = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        # workload[:, :4] = workload_raw[np.argsort(workload_raw[:, 1]-workload_raw[:, 2])]
        # ------------------#
        self.workload = np.abs(workload)
        #self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        #self.workload[:, 5][self.time + self.workload[:, 2] / self.speed > self.workload[:, 1]] = 1  # jobs that can't be done anyways
        # TODO: handle cases of multiple switches between degradation and normal execution
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.around(loguniform.rvs(self.total_load, 1e0),
                                           decimals=2)  # np.random.uniform(low=self.total_load)
        self._update_workload()
        self.obs_idx = np.arange(min(self.buffer_length, self.job_num))
        self.obs_idx = self.obs_idx[
            self.workload[self.obs_idx, 4].astype(bool)]  # remove unreleased jobs (must be removed)
        self.action_mask = np.zeros(self.buffer_length + 1)
        self.action_mask[self.buffer_length] = 1
        self.action_assignments = np.zeros((self.buffer_length + 1, 4))

        self._update_available()
        return self._get_obs()

    def _done(self):
        return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())
    def final(self):
            if self._done():
                print("Final Workload after done:",self.workload)

# print("hello")
