import gym

from stable_baselines import DQN, PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from env.job_gen_env import MCEnv
from gym.envs.registration import register
from stable_baselines.gail import ExpertDataset
# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset


dataset = ExpertDataset(expert_path='edf_expert_mc.npz',
                        traj_limitation=-1, batch_size=128)

# Pretrain the PPO2 model
register(
    id='MC-v0',
    entry_point='env.job_gen_env:MCEnv',
    max_episode_steps=30,

)
env = 'MC-v0'
env = gym.make(env)
env.get_workload()
# It will check your custom environment and output additional warnings if needed
check_env(env)
env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=2, tensorboard_log='./log/', exploration_fraction=0.3,
            exploration_final_eps=0.08)
## Train the agent
model.pretrain(dataset, n_epochs=100)
model.learn(total_timesteps=int(1e5))