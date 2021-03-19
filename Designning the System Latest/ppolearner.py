import gym

from stable_baselines import DQN, PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import SubprocVecEnv
from env.job_gen_env import MCEnv
from gym.envs.registration import register
from stable_baselines.gail import ExpertDataset
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines.bench import Monitor

def make_env(env_id, rank, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

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
if __name__ == '__main__':
    num_cpu = 4  # Number of processes to use
    env_id = 'MC-v0'
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i, log_dir) for i in range(num_cpu)])

    #env = 'MC-v0'
    #env = gym.make(env)

    #env = Monitor(env, log_dir)
    #env.get_workload()
    # It will check your custom environment and output additional warnings if needed
    #check_env(env)
    #env = DummyVecEnv([lambda: env])
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)
    model = PPO2('MlpPolicy', env, learning_rate=5e-5, verbose=2, tensorboard_log='./log/')
    ## Train the agent
    model.pretrain(dataset, n_epochs=100)
    model.learn(total_timesteps=int(1e6), callback=callback)
    #model.save('ppo')
    #model = PPO2.load('ppo')
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2000)
    #print(mean_reward, std_reward)

