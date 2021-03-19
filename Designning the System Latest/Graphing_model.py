import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,z):
    import matplotlib.pyplot as plt

    degradation= [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    degradation=100*np.array(degradation)
    degradation=np.array(degradation)
    #degradation = np.flip(degradation)
    plt.figure(dpi=200)
    axi=np.arange(0,100,10)
    plt.plot(degradation, axi, 'white')
    plt.plot(degradation,  x, 'red', label='Offline')
    plt.plot(degradation,  y, 'black', label='Varying Buffer',linestyle='--')
    plt.plot(degradation,  z, 'blue', label='Online',linestyle='--')
    plt.ylabel("Percentage of Hi Critical Jobs Completed")
    plt.xlabel("Performance Percentage (100 - degradation)")
    plt.title('Hi Critical Jobs Completion Percentage')
    plt.legend(title="Environment Type")
    plt.savefig("sensitivity_analysis_result_new_offline.png")
    #plt.xlim(-1, 11)
    #plt.ylim(-1.5, 1.5)
    plt.show()
#Offlinea=[1, 6.0000e-04, 6.9000e-03, 3.4800e-02, 1.2800e-01, 4.0860e-01, 1.1284e+00, 2.6080e+00, 5.2083e+00, 9.5180e+00]
Offlinea=[1, 1.8000e-03, 9.8000e-03, 5.1900e-02, 2.1040e-01, 6.2800e-01, 1.4828e+00, 2.8394e+00, 4.9796e+00, 9.6270e+00]
VBa= [0.0029, 0.0606, 0.5322, 0.7488, 0.887,  1.1058, 1.2859, 1.5304, 1.6452, 1.891 ]
Onlinea=[8.0000e-04, 1.4300e-02, 1.0900e-01, 3.4660e-01, 5.0050e-01, 5.4110e-01, 6.2950e-01, 7.5700e-01, 9.8210e-01, 1.0132e+00]
#Offlineb= [0.000e+00, 3.000e-04,2.300e-03, 8.700e-03, 2.560e-02, 6.810e-02, 1.612e-01, 3.260e-01, 5.787e-01, 9.518e-01]
Offlineb=[1, 0.001,  0.0033, 0.0131, 0.0422, 0.1048, 0.212,  0.355,  0.5533, 0.9627]
VBb=[0.0029, 0.0303, 0.1774, 0.1872, 0.1774, 0.1843, 0.1837, 0.1913, 0.1828, 0.1891]
Onlineb=[0.0008, 0.0079, 0.0403, 0.1119, 0.1439, 0.1246, 0.1282, 0.1306, 0.1509, 0.138 ]

Offline = np.divide(Offlineb,Offlinea)
VB=np.divide(VBb, VBa)
Online =np.divide(Onlineb, Onlinea)

Online= 100*np.array(Online)
Offline=100*np.array(Offline)
VB=100*np.array(VB)

plot(Offline,VB,Online)

plt.show()