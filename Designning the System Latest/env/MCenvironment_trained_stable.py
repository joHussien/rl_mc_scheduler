#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 19:43:15 2020

@author: Yussef & Karim
"""

from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy


#To train an environment we have ertain steps, 
# Instantiate the agent
# Train the agent
# Save the agent
# Load the trained agent
# Evaluate the agent
# Observing the results of the trained agent


# Instantiate the agent
env = MCEnv(10, 0.3, 0., 5, 0)
obs=env.get_obs()
print(obs)
#env = env([lambda: env])  # The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=25000)

#model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1).learn(total_timesteps=int(2e5))
#model = ACKTR('MlpPolicy', env, verbose=1).learn(5000)

# It will check your custom environment and output additional warnings if needed
check_env(env,warn=True)

#Train the agent


# Save the agent
model.save("MC_env_trained")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("MC_env_trained",tensorboard_log="./DQN_MC_env_tensorboard/")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
env.reset()

    
trial=0
done=False
total_reward=0      
while(not done):
    trial=trial+1
    action,_states= model.predict(obs) 
    #action,_states= model.predict(obs deterministic=True) 
    print("action: " ,action)
    obs,reward,done,empty=env.step(action)
    total_reward+=reward
print("Finished after : ",trial,"trials ")
print("Total reward of this eposide: ", total_reward)
print(obs)