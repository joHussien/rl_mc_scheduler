import gym, ray
from env.job_gen_env_youssef import MCOEnv,MCEnv,MCVBEnv
from ray import tune
from ray.rllib.agents.dqn import ApexTrainer
from gym.spaces import unflatten
from gym.envs.registration import register
from ray.tune.registry import register_env
import numpy as np

register(
    id='MC-v0',
    entry_point='env.job_gen_env:MCEnv',
    max_episode_steps=20,

)
#checkpoint_path = 'APEX/APEX_my_env_0_2020-07-08_20-02-02lxf9cfz8/checkpoint_100/checkpoint-100'
#checkpoint_path ='/home/ml2/ray_results/APEX/APEX_OfflineParametricAction_env_0_2020-08-12_19-08-178l0hppoa/checkpoint_100/checkpoint-100'

# checkpoint_path='/home/ml2/ray_results/APEX/APEX_filtered_Offline_random_0_2021-02-10_19-45-45xj0l6xju/checkpoint_50/checkpoint-50'
checkpoint_path='/home/ml2/ray_results/APEX/APEX_filtered_Offline_Nodegradation_rand_0_2021-02-11_02-16-53kvahvzg7/checkpoint_50/checkpoint-50'
ray.init()
env = MCEnv()
#gym.make('MC-v0')
register_env("Filtered-Offline-SpeedupFirstTest", lambda env_config: env)
agent = ApexTrainer(config={"env": "Filtered-Offline-SpeedupFirstTest", "num_workers": 5, "num_gpus": 1})

agent.restore(checkpoint_path)
policy = agent.workers.local_worker().get_policy()
#env = gym.vector.make('MC-v0', num_envs=10, asynchronous=False)
#obs = env.reset()
accum = []
HC_jobs_completed = np.zeros((10000, 10))
total_jobs_withHC = np.zeros((100,10))
#First we will remove degradation
#-commented
# for i in range(10):
#     deg_speed = 0.1*(i+1)
#added
deg_speed=1.0
#original
for j in range(100):

    obs = env.reset()
    print(obs)
    env.degradation_schedule = 0 #will start degradaing from t = 0
    env.degradation_speed = deg_speed
    total = 0
    for i in range(10):
        action = policy.compute_actions([obs])#(env)
        #print(env.time)
        print(action)
        obs, reward, done, empty = env.step(action[0][0])
        total += reward
        if done:
            #commented
            # HC_jobs_completed[j, i] = env.workload[env.workload[:, 3].astype(bool), 6].astype(bool).all()
            print(total)
            total_jobs_withHC[j,i] = total
            break
def plot(x,y):
    import matplotlib.pyplot as plt

    degradation=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # axi=[10,20,30,40,50,60,70,80,90,100]
    axi=list(range(0,100))
    # plt.plot(degradation, axi, 'white')
    # plt.plot(degradation, x, 'red', label='HC_jobs_completed %')

    plt.plot(axi, y, 'black', label='total_jobs_withHC %')
    plt.ylabel("Percentage of Jobs Completed")
    plt.xlabel("Number of Trial")
    plt.title("Filterd offline First Test")
    plt.legend(title="Checkpoint 350")
    plt.savefig("Inshallah.png")
    plt.show()
# print(np.mean(HC_jobs_completed, axis=0), np.mean(total_jobs_withHC, axis=0))
# HC_jobs_completed=10*np.array((np.mean(HC_jobs_completed, axis=0)))
print(total_jobs_withHC)
print(np.mean(total_jobs_withHC, axis=1))
total_jobs_withHC=100*np.array((np.mean(total_jobs_withHC,axis=1)))
print(total_jobs_withHC)
plot(HC_jobs_completed, total_jobs_withHC)