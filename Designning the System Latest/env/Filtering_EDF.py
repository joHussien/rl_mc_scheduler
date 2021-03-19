# in this file we will implement a new evaluation method for the system in both the offline and the online context
# the new method is the speedup matrid defined by "The minimum speed up the system need to schedule optimally an instance that
# is already schedulable by --an offline algorithm as EDF-VD,EDF,OBCP---
#ToDo:
    #Update the logic error, s.t it chooses the earliest released if the earliest deadline is not released yet, althought therer
    #might be a job that is not earliest in release and not earliest in deadline but its the earliest deadline in the released ones

# algorithm: TODO:
# 1. Generate an instancce
# 2. Test the instance offline-schedulabilty using EDF it should be fine assuming the speed to be one if not then filter it to be good and schedulable
# 3. Shcedule this instnace using the agent assuming the speed to be one
# 4. Compare the performance and decide if any speedup needed
import numpy as np
from job_generator import create_workload
#important note here about this function the parameter time has no use, I assume internally that the first time step is 0 as there will be a
#a job of release data = 0 and this will be the first job selected
def filtering_workload(num_jobs, total_load, lo_per, job_density, time):
    #print(num_jobs)
    workload = np.zeros((num_jobs, 6))
    workload[:, :4] = create_workload(num_jobs, total_load, lo_per, job_density)

    #print(workload.shape)
    # print("Initial Workload generated from before filteration")
    # print(workload[:,:4])
    index=np.argmin(workload[:,0])
    workload[index,4]=1
    t= workload[index,2]
    workload[np.where( (workload[:,2]+t) > workload[:,1]),5]=1
    new_array=workload[index,:]
    workload = np.delete(workload, np.where(workload[:,4]==1), axis=0)


    i=0
    while(len(workload) > 1):

        i=i+1
        # print("Num of iterations: ",i)
        #print("t: ",t)
        workload[np.where( (workload[:,2]+t) > workload[:,1]),5]=1
        workload = np.delete(workload, np.where(workload[:,4]==1), axis=0)
        workload = np.delete(workload, np.where(workload[:,5] == 1), axis=0)
        if (len(workload) >1):
            if (workload[np.argmin(workload[:,1]),0]<= t and workload[np.argmin(workload[:,1]),4] == 0):
                # print("First State: ",np.argmin(workload[:,1]))
                workload[np.argmin(workload[:,1]),4]=1
                t+= workload[np.argmin(workload[:,1]),2]
                new_array = np.concatenate((new_array,workload[np.argmin(workload[:,1]),:]))


            else:
                released = np.where(workload[:, 0] <= t)

                if (len(released[0])>1):

                    index = np.min(workload[released, 1])
                    earliest_dead_released = np.where(workload[:,1]==index)
                    index = earliest_dead_released
                    ind = index[0]
                    ind2=ind[0]
                    t += workload[ind2, 2]
                    workload[ind2, 4] = 1
                    new_array = np.concatenate((new_array, workload[ind2, :]))
                else :
                    index = np.argmin(workload[:, 0])
                    if(workload[index,0]>t):
                        #print("There was a gap here and had to update time with rlease and processing of earliest job")
                        workload[index,4]=1
                        t= workload[index,2] + workload[index,0]
                        new_array = np.concatenate((new_array,workload[index,:]))
                    else:
                        t+=workload[index,2]
                        workload[index,4]=1
                        new_array = np.concatenate((new_array,workload[index,:]))

    # print("Final Scheduble instance using EDF on 1-speed: ")
    #print("size ", len(new_array)//6)
    new_array.resize((len(new_array)//6,6))
    new_array=new_array[:,:4]
    # print(new_array)
    return new_array

# workload =  filtering_workload(10,0.5,.3,4,0)
# print(workload)