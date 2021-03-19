import gym, ray
from env.job_gen_env import MCOEnv,MCEnv,MCVBEnv
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

checkpoint_path='/home/ml2/ray_results/APEX/APEX_Offline_Uniform_new_0_2020-08-29_16-37-19ci7u4x4r/checkpoint_300/checkpoint-300'
ray.init()
env = MCEnv()
#gym.make('MC-v0')
register_env("Offline_logUniform_new_check_300_env", lambda env_config: env)
agent = ApexTrainer(config={"env": "Offline_logUniform_new_check_300_env", "num_workers": 5, "num_gpus": 1})

agent.restore(checkpoint_path)
policy = agent.workers.local_worker().get_policy()
#env = gym.vector.make('MC-v0', num_envs=10, asynchronous=False)
#obs = env.reset()
accum = []
HC_jobs_completed = np.zeros((10, 10))
total_jobs_withHC = np.zeros((10, 10))

for i in range(10):
    deg_speed = 0.1*(i+1)
    for j in range(10):
        obs = env.reset()
        print(obs)
        print("Obse was printed above: \n")
        env.degradation_schedule = 0
        env.degradation_speed = deg_speed
        total = 0
        for m in range(10):
            action = policy.compute_actions([obs])#(env)
            #print(env.time)
            obs, reward, done, empty = env.step(action[0][0])
            print("m : ",m,"action: ",action,"\n",obs)
            total += reward
            if done:
                HC_jobs_completed[j, i] = env.workload[env.workload[:, 3].astype(bool), 6].astype(bool).all()
                total_jobs_withHC[j, i] = total
                break
import numpy as np
def plot(x,y):
    import matplotlib.pyplot as plt

    degradation=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    axi=[10,20,30,40,50,60,70,80,90,100]
    plt.plot(degradation, axi, 'white')
    plt.plot(degradation, x, 'red', label='HC_jobs_completed %')
    plt.plot(degradation,  y, 'black', label='total_jobs_withHC %')
    plt.ylabel("Number of Jobs Completed")
    plt.xlabel("Degradation")
    plt.title("Online Allrandom  Log-Distribution")
    plt.legend(title="Checkpoint-300")
    plt.savefig("graphde300a.png")
    plt.show()
print(np.mean(HC_jobs_completed, axis=0), np.mean(total_jobs_withHC, axis=0))
HC_jobs_completed=10*np.array((np.mean(HC_jobs_completed, axis=0)))
total_jobs_withHC=10*np.array((np.mean(total_jobs_withHC,axis=0)))
#plot(HC_jobs_completed,total_jobs_withHC)