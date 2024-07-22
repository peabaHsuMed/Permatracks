import numpy as np


class CalibrationSimple:
    '''
    Simple Calibration by using 6 points
    '''
    def __init__(self,N_data,N_sensor):
        self.N_data = N_data
        self.N_sensor = N_sensor
        self.cal_data = np.zeros
        self.cal_data = np.zeros([N_data,N_sensor,3])
        self.cal_offset = np.zeros([N_sensor,3]) #offset value correspond to each sensor, axis
        self.cal_scale = np.zeros([N_sensor,3])  #scale value correspond to each sensor, axis
     
        self.max_offset = np.zeros([N_sensor,3])
        self.min_offset = np.zeros([N_sensor,3])
        self.avg_delta = np.zeros([N_sensor,3])
        self.avg_delta_scalar = np.zeros([N_sensor,1])
        
    
    def retrieve(self,i,data):
        '''
        Input: Iterator, data of the iteration
        '''
        self.cal_data[i,:,:] = data
        
    def formation(self):
        '''
        Form calibration variables from cal_data
        '''
    
        for j in range (self.N_sensor):
            for i in range(3):
                self.max_offset[j,i] = np.max(self.cal_data[:,j,i])
                self.min_offset[j,i] = np.min(self.cal_data[:,j,i])
                self.cal_offset[j,i] = (self.max_offset[j,i] + self.min_offset[j,i])/2
                self.avg_delta[j,i] = (self.max_offset[j,i] - self.min_offset[j,i])/2
        
            self.avg_delta_scalar[j] = (self.avg_delta[j,0] + self.avg_delta[j,1] + self.avg_delta[j,2])/3
            
        for i in range(3):
            self.cal_scale[:,i] = np.divide(self.avg_delta_scalar[:,0],self.avg_delta[:,i])
    
        print("Offset is: ", self.cal_offset)
        print("Scale is: ", self.cal_scale)
        
    def apply(self,data):
        '''
        Apply calibration variables to data
        
        '''
        post_process = np.zeros([self.N_sensor,3])
        for j in range (self.N_sensor):
            for i in range(3):
                post_process[j,i] = (data[j][i] - self.cal_offset[j,i]) *self.cal_scale[j,i]
        
        return post_process