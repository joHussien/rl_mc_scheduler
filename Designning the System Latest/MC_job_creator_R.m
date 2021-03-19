% Mixed Cricality Job Creator (release version for RTSS2013)
% Sanjoy Baruah and Zhishan Guo. Mixed-criticality scheduling upon varying-speed processors. Proceedings of the IEEE Real-Time Systems Symposium (RTSS 2013), Vancouver, BC. December 2013. IEEE Computer Society Press.
% Code By Zhishan Guo
% 
% Update Log 
% 2013/04/28 - 1.0 version, use for submission
% 2014/03/31 - Added some comments for easier reading
%
function JOB = MC_job_creator_R (JobNum, E_LOADRATE, LOHI, alpha)  
% Parameters: # of jobs, total load (<=1), percentage of LO jobs(<=1), Average density (active job number at any instant,0 to JobNum, suggested to be set as 5) 
% JOB: a JobNum by 4 matrix, with 1st column = release date, 2nd column = deadline, 3rd column = WCET,  4th column = criticality level

% IMPORTANT: total load != utilization, total load is simply "the sum of all WCETs devided by length of whole duration"
% total load < 1 may still lead to utilization > 1 = not feasible
% When setting E_LOADRATE close to 1, it's LIKELY that the generated job set is NOT FEASIBLE (setting alpha small will help).

%==============================//===============================
% 1 release time
UB=3; %setting UB large may cause the set to be "seperated" into jobs with no overlapping scheduling window
expdis = min(exprnd(1,1,JobNum-1),UB); %exponential distribution, with a controlled upperbound
T_release(1)=0;
for i=2:JobNum
    T_release(i)=expdis(i-1)+T_release(i-1);
end

% 2 deadline
b=equationsolver0b(alpha); % solves the equation using Newton's method
for i=1:JobNum
    unidis = unifrnd(0,b); %uniform distribution U[0,b]
    T_relativedealine(i)=exp(unidis);%log-uniform distribution
    T_deadline(i)=T_release(i)+T_relativedealine(i);
end                                                                                            

%3 WCET
% This part might be a little complicated.
% It follows directly from the appendix of our paper, where serval
% bounds are derived and used from the contructed inequalities.
Sum_C = max(T_deadline)*E_LOADRATE;
b1=max(0,Sum_C-sum(T_relativedealine)+T_relativedealine(1));
b2=min(T_relativedealine(1),Sum_C);
themean=T_relativedealine(1)*E_LOADRATE*max(T_deadline)/sum(T_relativedealine);
beta = 2*(b2-b1)/(themean-b1)-2;
T_excution_Lo(1) = betarnd(2,beta)*(b2-b1)+b1;
for i=2:JobNum-1
    sum1 = 0;
    sum2 = 0;
    for  j=1:i-1 
        sum1 = sum1+T_excution_Lo(j);
    end
    for  j=i+1:JobNum 
        sum1 = sum1+T_relativedealine(j);
    end
    for  j=1:i-1 
        sum2 = sum2+T_excution_Lo(j);
    end
    
    b1=max(0,Sum_C-sum1); %lowerbound 
    b2=min(T_relativedealine(i),Sum_C-sum2); %upperbound
    if b1>b2
        T_excution_Lo(i) = 0;
        break;
    end
    %T_excution_Lo(i)=unifrnd(b1,b2);
    themean=T_relativedealine(i)*E_LOADRATE*max(T_deadline)/sum(T_relativedealine);
    if themean<b1 || themean>b2
        T_excution_Lo(i) = 0;
        break;
    end
    %T_excution_Lo(i)=unifrnd(themean-min(b2-themean,themean-b1),themean+min(b2-themean,themean-b1));
    beta = 2*(b2-b1)/(themean-b1)-2;
    T_excution_Lo(i) = betarnd(2,beta)*(b2-b1)+b1;
end
T_excution_Lo(JobNum) = min(Sum_C - sum(T_excution_Lo),T_relativedealine(JobNum)); % can't be larger than reletive deadline

% 4 - criticality level
T_cricality = rand(1,JobNum); % 0 for LO, 1 for HI
for i=1:JobNum
    if T_cricality(i)< LOHI
        T_cricality(i)=0;
    else
        T_cricality(i)=1;
    end
end

JOB_release=[T_release;T_deadline;T_excution_Lo;T_cricality]';
%JOB=JOB_release;
JOB=sortrows(JOB_release,2); %jobs sorted by deadlines
JOB = 0.0001*round(10000*JOB); % round up a little, 4 digits after "." should be enough for most experiments


