import gym

from stable_baselines import DQN, PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from env.job_gen_env import MCEnv
from gym.envs.registration import register
import numpy as np
import gym
from stable_baselines.gail import generate_expert_traj


def suboptim_policy(_obs):

    workload = env.workload
    speed = env.speed
    avail_jobs = np.bitwise_and((1 - workload[:, 6]).astype(bool), (1 - workload[:, 5]).astype(bool))
    avail_jobs = workload[avail_jobs]
    if speed == 1:
        return (np.argwhere((workload == avail_jobs[np.argmin(avail_jobs[:, 1])]).all(1))).squeeze()
    else:
        if avail_jobs[avail_jobs[:, 3] == 1].shape[0] != 0:
            min_HI = np.argmin(avail_jobs[avail_jobs[:, 3] == 1, 1])
            return (np.argwhere((workload == avail_jobs[min_HI]).all(1))).squeeze()
        else:
            return (np.argwhere((workload == avail_jobs[np.argmin(avail_jobs[:, 1])]).all(1))).squeeze()


register(
    id='MC-v0',
    entry_point='env.job_gen_env:MCEnv',
    max_episode_steps=50,

)
env = 'MC-v0'
env = gym.make(env)
#accum = []
generate_expert_traj(suboptim_policy, 'edf_expert_mc', env, n_episodes=50000)
#for _ in range(5000):
#    env.reset()
#    total = 0
##    for i in range(10):
#        #action,_states= model.predict(obs deterministic=True)
#        #print("action: ", suboptim_policy(env))
#        action = suboptim_policy(env)
#        obs, reward, done, empty = env.step(action)
#        total += reward
#        if done:
#            #print(env.workload[env.workload[:,3].astype(bool),6])
#            break
#    accum.append(total)
#accum= np.array(accum)
#print(np.mean(accum))