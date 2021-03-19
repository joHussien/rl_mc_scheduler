import gym

from stable_baselines import DQN, PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from env.job_gen_env import MCEnv
from gym.envs.registration import register
import numpy as np
import gym
from stable_baselines.gail import generate_expert_traj


def suboptim_policy(env):

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
    max_episode_steps=20,

)
env = 'MC-v0'
env = gym.make(env)
HC_jobs_completed = np.zeros((1000, 10))
total_jobs_withHC = np.zeros((1000, 10))

for i in range(10):
    deg_speed = 0.1*(i+1)
    for j in range(10000):
        obs = env.reset()
        env.degradation_schedule = 0
        env.degradation_speed = deg_speed
        total = 0
        for i in range(15):
            action = suboptim_policy(env)
            #print(env.time)
            obs, reward, done, empty = env.step(action[0][0])
            total += reward
            if done:
                HC_jobs_completed[j, i] = env.workload[env.workload[:, 3].astype(bool), 6].astype(bool).all()
                total_jobs_withHC[j, i] = total
                break


print(np.mean(HC_jobs_completed, axis=0), np.mean(total_jobs_withHC, axis=0))
#env = MCEnv()
