

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
from gym.spaces import Box, MultiBinary, Discrete, Dict, flatten, flatten_space
from env.job_generator import create_workload


class MCEnv(object):
    pass


class MCEnv(gym.Env):
    def __init__(self, env_config= {'job_num': 10, 'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4}):
        #add here description of each parameter
        self.time = 0
        self.job_num = env_config['job_num']
        self.total_load = np.random.uniform(low=0.1, high=0.9)#env_config['total_load']
        self.lo_per = np.random.uniform(high=1-2/(self.job_num)) #env_config['lo_per']
        self.job_density = env_config['job_density']
        self.speed = 1

        workload = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        workload = np.insert(workload, 4, [0] * self.job_num, axis=1)
        workload = np.insert(workload, 5, [0] * self.job_num, axis=1)
        workload = np.insert(workload, 6, [0] * self.job_num, axis=1)
        self.workload = np.abs(workload) #negative processing time

        self.action_space = Discrete(self.job_num)
        self.observation_space_dict = Dict({
         'action_mask': Box(0, 1, shape=(self.job_num,)),
         'MCenv': Dict({
         'RDP_jobs': Box(low=0, high=np.inf, shape=(self.job_num, 3)),
         'CRSE_jobs': MultiBinary(self.job_num*4),
         'Processor': Box(low=np.array([0., 0.]), high=np.array([1, np.inf])),
            })
        })
        self.observation_space = flatten_space(self.observation_space_dict)
        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1 #jobs that can't be done anyway
        #TODO: handle cases of multiple switches between degradation and normal execution
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.random.uniform(low=self.total_load)
        self.action_mask = np.ones(self.job_num)
        self.seed()
    def fill_buffer(self,workload):
        #Logic each time we enter here we append the released jobs only and the size of this buffer currently is 5 as the jobs we generate
        #are 10, also wehn we fill we pass this to the step where the job to be sleected must be from the buffer not the workload
        #Then we update the time based on the chosen job and afterwards we refill the buffer
        #Ofcourse the job we selected before won't be selected again as it will be starved and the choice of a specific job will reflected
        #on the next time we try to select a job from the buffer
        #regarding the reward we should see which starved from the self.workload and what was excuted as if a job was selected it must
        #be added to another buffer "Chosen" in order to know at the final which was selected and which ws starved
        #Will make the size of the buffer 5
        last_deadline=self.workload[self.job_num-1][1]
        nop=np.array([last_deadline,last_deadline+1,2,0,0,0,0])
        added=0
        for job in workload:
            if((job[0]>=self.time) and (job[5]==0) and (len(self.buffer)<5)): #Released and not starved and still there is a space
                self.buffer.append(job)
                added = added + 1
        for el in range(5-len(self.buffer)):
            self.buffer.append(nop)
        return self.buffer




    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #define step logic
        done = self._done()
        prev_workload = np.copy(self.workload)
        reward = 0
        buffer=self.fill_buffer(self.workload) #Now we should replace each self.workload with buffer
        self.chosen.append(buffer[action]) #To know at the end what was scheduled
        self.buffer.pop
        if self.workload[action, 5] or self.workload[action, 6]:
            return self._get_obs(), -10, done, {}
            #reward = -10
        else:
            time = max(self.time, self.workload[action, 0])

            if time >= self.degradation_schedule:
                self.speed = self.degradation_speed
                time += self.workload[action, 2] / self.speed
            elif self.workload[action, 2] + time < self.degradation_schedule:
                time += self.workload[action, 2]
            else:
                time_in_norm = self.degradation_schedule - time
                self.speed = self.degradation_speed
                time_in_deg = (self.workload[action][2] - time_in_norm) / self.speed
                time += time_in_norm + time_in_deg
            # double check, as in case of degradation, time will not increment properly which might lead to the starvation a job unexpectedly
            if time <= self.workload[action, 1]:
                self.time = time
                self.workload[action, 6] = 1
                self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
                starved_condition = (self.time >= self.workload[:, 1]) * (1 - self.workload[:, 6]).astype(bool)
                self.workload[:, 5][starved_condition] = 1
                will_starve_condition = (self.time + self.workload[:, 2] / self.speed > self.workload[:, 1]) \
                                        * (1 - self.workload[:, 6]).astype(bool)
                self.workload[:, 5][will_starve_condition] = 1
                done = self._done()
                # reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
                if done and self.workload[self.workload[:, 3].astype(bool), 6].all():
                    reward += np.sum(self.workload[:, 6])

        return self._get_obs(), reward, done, {}

    def _update_available(self):
        self.action_mask[self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)] = 0

    def _get_obs(self):

        self._update_available()
        obs_dict = dict({'action_mask': self.action_mask, 'MCenv': dict(
                        {'RDP_jobs': np.array(self.workload[:, :3]),
                        'CRSE_jobs': np.array(self.workload[:, 3:]).flatten(),
                         'Processor': np.array([1, 0]).flatten()})
                         })
        return obs_dict #flatten(self.observation_space_dict, obs_dict)

    def reset(self):
        self.time = 0
        self.speed = 1
        self.total_load = np.random.uniform(low=0.2, high=0.9)  # env_config['total_load']
        self.lo_per = np.random.uniform(high=1 - 2 / (self.job_num))  # env_config['lo_per']
        workload = create_workload(self.job_num, self.total_load, self.lo_per, self.job_density)
        workload = np.insert(workload, 4, [0] * self.job_num, axis=1) #Released
        workload = np.insert(workload, 5, [0] * self.job_num, axis=1)  #Starved
        workload = np.insert(workload, 6, [0] * self.job_num, axis=1)  #Excuted
        self.workload = np.abs(workload)
        self.workload[:, 4][self.time >= self.workload[:, 0]] = 1
        self.workload[:, 5][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1
        self.degradation_schedule = np.random.uniform(high=np.sum(workload[:, 2]))
        self.degradation_speed = np.random.uniform(low=self.total_load)
        self.action_mask = np.ones(self.job_num)
        return self._get_obs()

    def _done(self):
        return bool((self.workload[:, 5].astype(bool) | self.workload[:, 6].astype(bool)).all())

    def get_workload(self):
        print(self.workload)

