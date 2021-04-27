import gym, ray
from env.job_gen_env_filtered import MCOEnv,MCEnv,MCVBEnv
from ray import tune
from ray.rllib.agents.dqn import ApexTrainer
from gym.spaces import unflatten
from gym.envs.registration import register
from ray.tune.registry import register_env
import numpy as np

register(
    id='MC-v0',
    entry_point='env.job_gen_env:MCOEnv',
    max_episode_steps=20,

)
#checkpoint_path = 'APEX/APEX_my_env_0_2020-07-08_20-02-02lxf9cfz8/checkpoint_100/checkpoint-100'
#checkpoint_path ='/home/ml2/ray_results/APEX/APEX_OfflineParametricAction_env_0_2020-08-12_19-08-178l0hppoa/checkpoint_100/checkpoint-100'

checkpoint_path='/home/youssefhussien/ray_results/APEX_2021-04-26_19-03-05/APEX_online_newdummy_thetaOne_07f35_00000_0_2021-04-26_19-03-05/checkpoint_300/checkpoint-300'
#checkpoint_path = '/home/youssefhussien/ray_results/APEX_2021-04-25_19-02-11/APEX_no_speed_offline_bd828_00000_0_2021-04-25_19-02-11/checkpoint_400/checkpoint-400'
#checkpoint_path='/home/youssefhussien/ray_results/APEX_2021-04-26_02-08-53/APEX_VB_newdummy_5997f_00000_0_2021-04-26_02-08-53/checkpoint_400/checkpoint-400'
ray.init()
env = MCOEnv()
#gym.make('MC-v0')
register_env("Online_Theta_evaluation", lambda env_config: env)
agent = ApexTrainer(config={"env": "Online_Theta_evaluation", "num_workers": 8, "num_gpus": 0})

agent.restore(checkpoint_path)
policy = agent.workers.local_worker().get_policy()
#env = gym.vector.make('MC-v0', num_envs=10, asynchronous=False)
#obs = env.reset()
accum = []
HC_jobs_completed = np.zeros((100, 10))
total_jobs_withHC = np.zeros((100, 10))
print("DSpeed ",env.degradation_speed)
print("My theta ",env.theta)
for i in range(10):
    deg_speed = 0.1*(i+1)
    theta = 0.1*(i+1)
    for j in range(100):
        obs = env.reset()
        #print(obs)
        #print(obs)
        #print("Obse was printed above: \n")
        env.degradation_schedule = 0
        env.degradation_speed = 1
        env.theta = theta
        total = 0
        for m in range(10):
            action = policy.compute_actions([obs])#(env)
            #print(env.time)
            obs, reward, done, empty = env.step(action[0][0])
            #print("m : ",m,"action: ",action,"\n",obs)
            total += reward
            if done:
                HC_jobs_completed[j, i] = env.workload[env.workload[:, 3].astype(bool), 6].astype(bool).all()
                total_jobs_withHC[j, i] = total
                break
import numpy as np
def plot(x,y):
    import matplotlib.pyplot as plt

    thet=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # axi=[10,20,30,40,50,60,70,80,90,100]
    axi=list(range(0,100))
    # plt.plot(degradation, axi, 'white')
    # plt.plot(degradation, x, 'red', label='HC_jobs_completed %')

    plt.plot(thet, y, 'black', label='Total Jobs Completed %')
    plt.plot(thet, x, 'red', label='Hi-critical Jobs Completed%')
    plt.ylabel("Percentage of Jobs Completed")
    plt.xlabel("Theta Value")
    plt.title("Online Theta Evaluation")
    plt.legend(title="Not Degraded")
    plt.savefig("online_ev_theta.png")
    plt.show()
# print(np.mean(HC_jobs_completed, axis=0), np.mean(total_jobs_withHC, axis=0))
# HC_jobs_completed=10*np.array((np.mean(HC_jobs_completed, axis=0)))
print(total_jobs_withHC)
print()
print(np.mean(total_jobs_withHC, axis=0))
total_jobs_withHC=100*np.array((np.mean(total_jobs_withHC,axis=0)))
print("HI ",HC_jobs_completed)
HC_jobs_completed = 100*np.array((np.mean(HC_jobs_completed,axis=0)))
print("Hi2 ",HC_jobs_completed)
plot(HC_jobs_completed, total_jobs_withHC)
