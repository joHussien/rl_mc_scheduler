#!/usr/bin/env python
# coding: utf-8

# In[28]:


# We have one main function job_creator
#Parameters: # of jobs, total load (<=1), percentage of LO jobs(<=1), Average density (active job number at any instant,0 to JobNum, suggested to be set as 5)
#output is a JobNum by 4 matrix, R-D-P-C
#total Load is sum_of_all_processings/Time_Horizon and is <=1, if >1 so not feasible, hence itis better to be small


# In[29]:


#imports 
import numpy as np
import pandas as pd
import math
import random
#import matplotlib as lp


# ### Definning the parameters' values
# 

# In[30]:
def create_workload(job_num, total_load, lo_per, job_density):
# =============================================================================
#     self.job_num=x
#     self.total_load=y
#     self.lo_per=z
#     self.job_density=g
# =============================================================================
# job_num=10
# =============================================================================
#     get_ipython().run_line_magic('store', '-r job_num')
#     get_ipython().run_line_magic('store', '-r total_load')
#     get_ipython().run_line_magic('store', '-r lo_per')
#     get_ipython().run_line_magic('store', '-r job_density')
# =============================================================================
    #If we want to access specific row/col in the final matrix
   
    
    
    # ### Release Time calculation part 
    # 
    
    # In[31]:
    
    
    UB=3.00000000 #Upper_bound for exponential distrbution
    mean=1
    exp_dist=np.random.exponential(mean,(1,job_num-1))#array of Jobnum-1 size each entry is a exponeneital distribution of the entry itself
    exp_dist=np.array(exp_dist)
    exp_dist=exp_dist[0]
    exp_dist[exp_dist>UB]=UB #setting the Upper Bound of the exponential distribution
    
    release_array =[]
    for i in range (job_num):
        if(i==0):
            release_array.append(0)
        else:
            curr_val=exp_dist[i-1]+release_array[i-1]
            curr_val=round(curr_val,4) #roudning the release value to 4 digits
            release_array.append(curr_val)
    release_array=np.array(release_array)
    # print(release_array)
    
    
    # ### Criticality level calculation part, we have a probelm here 
    # 
    
    # In[32]:
    
    
    #in the random number distribution
    size=(1,job_num)
    criticality_array=np.random.uniform(0,1,size)
    # criticality_array=np.random.randn(1,job_num)# This gives an error in the final number of lo and hi
    criticality_array[criticality_array<lo_per]=0
    criticality_array[criticality_array>=lo_per]=1
    criticality_array=np.array(criticality_array)
    criticality_array=criticality_array[0]
    #------------Trying to accomodate the error generated by the rand----#
    # count=0
    # for i in range(job_num):
    #     if (critcality_array[i]==0):
    #         count=count+1
    #         if ((count/job_num)<lo_per):
    #             critcality_array[i]=
    #         else:
    #             break
    
        
    
    #print(criticality_array)
    
    
    # ### Deadlines Calculaition part
    
    # In[33]:
    
    
    
    def f(x):
        return (math.exp(x) -1 -x*x)
    def f1(x):
        return (math.exp(x)- x)
    def equation_solver(a):
        x0=a
        f_out=f(x0)
        f1_out=f1(x0)
        x1=x0-f_out/f1_out
        x1=round(x1,4)
    #     while(abs(x1-x0)>10^(-3)):
    #         print("Here")
    #         x0=x1
    #         f_out=f(x0)
    #         f1_out=f1(x0)
    #         x1=x0-f_out/f1_out
        return x1
    b= equation_solver(job_density)   
    relative_deadline=[]
    deadline_array=[]
    for i in range(job_num):
        unidis=np.random.uniform(0,b)
        relative_deadline.append(math.exp(unidis))
    
    relative_deadline=np.array(relative_deadline)
    relative_deadline[1]
    for el in range(job_num):
         deadline_array.append(release_array[el]+relative_deadline[el])
    deadline_array=np.array(deadline_array)
    # print(deadline_array)
    
    
    # ### Processing Time; the complicated part
    #  
    
    # In[34]:
    
    
    Sum_C=max(deadline_array)*total_load
    b1=max(0,Sum_C-sum(relative_deadline)+relative_deadline[0])
    b2=min(relative_deadline[0],Sum_C)
    the_mean=relative_deadline[0]*total_load*max(deadline_array)/sum(relative_deadline)
    beta = 2*(b2-b1)/(the_mean-b1)-2
    excution_Lo=[1]*job_num #creating an empty array 
    excution_Lo[0] = random.betavariate(2,beta)*(b2-b1)+b1 #using Beta random distribution
    #IF there is an error in this code most probably it will be here
    for i in range(job_num):
        if(i!=0):
            sum1 = 0
            sum2 = 0
            for  j in range(i-1):
                sum1 = sum1+excution_Lo[j];
            k=i+1    
            for  k in range(job_num): 
                    sum1 = sum1+relative_deadline[k]
                
            for  m in range(i-1):
                sum2 = sum2+excution_Lo[m]
                
            b1=max(0,Sum_C-sum1) #lowerbound 
            b2=min(relative_deadline[i],Sum_C-sum2); #upperbound
            if (b1>b2):
                excution_Lo[i] = 0
                break
        #     %T_excution_Lo(i)=unifrnd(b1,b2);
            the_mean=relative_deadline[i]*total_load*max(deadline_array)/sum(relative_deadline)
            if (the_mean<b1) or (the_mean>b2):
                excution_Lo[i]= 0
                break
        #    %T_excution_Lo(i)=unifrnd(themean-min(b2-themean,themean-b1),themean+min(b2-themean,themean-b1));
            beta = 2*(b2-b1)/(the_mean-b1)-2
            excution_Lo[i] = random.betavariate(2,beta)*(b2-b1)+b1
    
    #excution_Lo[job_num-1] = min(Sum_C - sum(excution_Lo),relative_deadline[job_num-1]) # can't be larger than reletive deadline
    processing_array=np.array(excution_Lo)
    for _ in range(job_num):
        deadline_array[_]=round(deadline_array[_],4)
        processing_array[_]=round(processing_array[_],4)
    
    
    # ### Printing Final Results
    
    # In[35]:
    
    
    
    release_array=release_array.reshape(job_num,1)
    deadline_array=deadline_array.reshape(job_num,1)
    processing_array=processing_array.reshape(job_num,1)
    criticality_array=criticality_array.reshape(job_num,1)
    final=np.concatenate((release_array, deadline_array, processing_array, criticality_array),axis=1)
    final=final[np.argsort(final[:, 1])] #Sorting by Deadline
    #print("#------Shape of final matrix: ", final.shape)
    return final
# =============================================================================
#     print("#------Example of indexing: row= ",row," col= ",col)
#     print(final[row][col])
# =============================================================================
    

# ### Saving the Generated Workload to .xlsx file 

# In[36]:


# =============================================================================
# writer = pd.ExcelWriter(r'C:\Users\Youssef\Desktop\pythonjobgenerator\workload.xlsx', engine='xlsxwriter')
# #Converting the matrix into a datframe to save it in an excel file
# dataset = pd.DataFrame({'Releases': final[:, 0], 'Deadlines': final[:, 1],'Processing': final[:, 2], 'Criticality': final[:, 3]})
# 
# dataset.to_excel(writer, sheet_name='Sheet1')
# writer.save()
# dataset
# 
# =============================================================================

# ## Problems  still have to be reviewed

# #### 1) In the processing time part, check why one value is negaitve, and make sure that the for loops are correct by comparing this part to the main generator and test if this part only behaves the same or not
# #### 2) In the Deadlines part, check the while loop if it's of importance or not, I gues not but we have to make sure of that

# In[ ]:


#create_workload(3,0.3,0.3,5)

