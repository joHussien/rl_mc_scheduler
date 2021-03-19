# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:26:05 2020

@author: Youssef
"""

#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import math,  random
from typing import Tuple
import gym 


# In[8]:


#iports of the Environment
from gym import spaces
#from gym.utils import flatten, flatten_space 
#from gym.utils import seeding
from JobGenerator_Python_Version import create_workload

# from os import path


# # Notes Section

# Every environment comes with an action_space and an observation_space. These attributes are of type Space, and they describe the format of valid actions and observations:
# The most common are Discrete and Box"Continous space".
# ##### So what will we do?
# First I thought that we should have two action spaces: select_job {0: not_selected, 1:select} and update_job_state {0:not_released,1:released,2:starved}. 
# Then I thought that we should have one action space select_job and based on it the state of each job shuld chaneg and hene the job_state should be in the observation_space.

# ### Passing the Parameters to the Workload-Generator

# =============================================================================
# # In[9]:

# In[11]:


class MC_agent(gym.Env):
    
    def __init__(self,job_num,total_load,lo_per,job_density,time, add_bonus = True):
        
        self.time=time # I think should be always zero and starts from there
        self.job_num=job_num
        self.total_load=total_load
        self.lo_per=lo_per
        self.job_density=job_density
        self.add_bonus = add_bonus
        #----Just Prinitng the workload as dataframe for easier reading only---#
        #print("Workload Generated ")
        workload=create_workload(self.job_num,self.total_load,self.lo_per,self.job_density)
        #workload = np.concatenate([workload, np.zeros(job_num, 3)], axis = 1)
        workload=np.insert(workload,4, [0] * job_num, axis=1) #Released
        workload=np.insert(workload,5,[0] * job_num, axis=1)  #Starved  
        workload=np.insert(workload,6,[0] * job_num, axis=1)  #Excuted
        #print(workload)
        #No Need for Pandas Dataframe
        #workload_df = pd.DataFrame({'Releases': workload[:, 0], 'Deadlines': workload[:, 1],'Processing': workload[:, 2], 'Criticality': workload[:, 3],"Released":workload[:,4],"Starved":workload[:,5],"Excuted":workload[:,6] })
        self.workload=workload
        #print(workload_df)
        #Action: outputs the index of the job to be selected
        self.action_space = spaces.Discrete(job_num) 
        #self.observation_space = spaces.Box() 
        
        #potential observation_space
        self.observation_space = spaces.Dict({
         'RDP_jobs'  : spaces.Box(low=0, high=np.inf, shape=(job_num,4)),
         'CRSE_jobs' : spaces.MultiBinary(4),
         'Processor' : spaces.Box(low=np.array([0. , 0.]) ,high = np.array([1, np.inf])),
            })
         
        self.reward_weights = np.where(self.workload[:,3], 1, 1/self.job_num)
        #(Karim):why
        self.workload[0][4]=1 #setting first job as released
        self.speed = 1
        #TODO: handle cases of multiple switches between degradation and normal execution, in this case degradation happens
        # in a random time step during the execution 
        self.degradation_schedule = np.random.uniform(high = np.sum(workload[:, 2]))#None
        self.degradation_speed =  np.random.uniform(low = 0.4)

    def step(self, action):
        # the step updates 
        # execute the job selected if released and update all the other jobs states according to the execution time of 
        # that job taken into account degradation schedule. if not released leave everything as it is with no change
        # TODO: add termination criteria 
        done = False
        prev_workload = self.workload
        if ((self.workload[action, 4]) and (self.workload[action,6]==0)):
            # update exceution state of job if released
            self.workload[action, 6] = 1
            
            if self.time > self.degradation_schedule:
                self.time += self.workload[action,2] / self.speed
            #update time in case of degradation or not
            elif self.workload[action, 2] + self.time < self.degradation_schedule:
                self.time += self.workload[action, 2] 
            
            else:
                time_in_norm = np.max(0, self.time - self.degradation_schedule)
                time_in_deg = (self.workload[action][2]-time_in_norm)/self.speed
                self.time += self.time + time_in_norm + time_in_deg
                self.speed = self.degradation_speed
            
            #update starving, release time for all jobs
            self.workload[:, 4][self.time >= self.workload[:, 0]] = 1 
            self.workload[:, 5][self.time >= self.workload[:, 1]] = 1
            
            #incur cost on newly starved jobs
            reward = -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
            done = self._done()
            if done and not self.workload[:, 5].all():
                # scheduling all in time is the ultimate goal (max possible reward) 
                # TODO: add bonus for superpassing EDF
                reward += self.job_num
            
        else:
            #took an unreleased job which is considered a violation
            reward = -100
        
        #I don't know if this is true, but will assume that for the agent to be correct
        #Then it has to select a job at any time step, it cannot pass a time step and 
        #it is free of processing a job
        
        return self.get_obs(), reward, done, {}
    
    def compute_reward(self, prev_workload =None):
        if prev_workload is None:
            return -np.sum(self.workload[:, 5]*self.reward_weights)
        return -np.sum((self.workload[:, 5] - prev_workload[:, 5])*self.reward_weights)
            
        
        
    def get_obs(self):
        print('time: ', self.time)
        return dict({'RDP_jobs': self.workload[:,:3], 'CRSE': self.workload[:,3:], 'Processor': np.array([self.speed, self.time])})
        
    def reset(self):
        self.time = 0
        self.speed = 1
        workload=create_workload(self.job_num,self.total_load,self.lo_per,self.job_density)
        #workload = np.concatenate([workload, np.zeros(job_num, 3)], axis = 1)
        workload=np.insert(workload,4, [0] * self.job_num, axis=1) #Released
        workload=np.insert(workload,5,[0] * self.job_num, axis=1)  #Starved  
        workload=np.insert(workload,6,[0] * self.job_num, axis=1)  #Excuted
        self.wokload=self.workload
        self.degradation_speed =  np.random.uniform(low = 0.4)
    
    def _done(self):
         #if (np.sum(self.workload[:,6])==self.job_num):
        #    return True
        #elif (np.sum(self.workload[:,5])==self.job_num):
        #    return True
        return self.workload[:,6].all()
       
#Main
agent= MC_agent(5,0.3,0.5,5,0)
obs= agent.get_obs()
done=agent.done()
trial=0
#while(not done):
for _ in range(100):
     trial=trial+1
     action= agent.action_space.sample()
     print('action: ',action)
     agent.step(action)
     #print(obs)
     done=agent.done()
     print(obs)

    

# print('Done with the eposide after: ',trial,' Steps')
# for key,value in obs.items():
#     if key=='CRSE':
#         print (sum(value[2]))


# #### Tasks couldn't finish
# passing parameters better,
# rendering,
# observation space defintion
