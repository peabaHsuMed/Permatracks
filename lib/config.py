import numpy as np

"""
Defined in the early phase, but not used. 
"""


# Data input area
#Br = 1.30                           #T in paper, 1.2 is used
r_mag = 2.5                         #mm
N_sensor = 16
#m_j = (Br/3)*(r_mag**3)               
m_j = -4.395964957156119e-10
G1 = -1.3286580823462076e-05
G2 = 5.065216230155552e-06
G3 = -1.1821015352061691e-05

plot_limit_min = -10
plot_limit_max = 100


filename1 = 'data.txt'
#sensor_B = np.loadtxt(filename1)
#sensor_B = sensor_B -  ambient
#sensor_B = np.ones([3*N_sensor,1])

#filename2 = 'sensor_position.txt'
filename2 = 'sensor_position_40.txt'
#sensor_pos =  np.ones([3*N_sensor,1])
sensor_pos_3 =  np.loadtxt(filename2)*1e-3 #1e-3
sensor_pos = sensor_pos_3.reshape(-1,1)


#G = np.array([0,1,2])

class configuration ():
    
    def __init__(self):
        self.plot_limit_max = -10
        self.plot_limit_min = 100
        self.filename1 = 'data.txt'
        self.filename2 = 'sensor_position_30.txt'
        self.sensor_pos_3 = np.loadtxt(filename2) * 1e-3
        self.sensor_pos = sensor_pos_3.reshape(-1,1)
        
