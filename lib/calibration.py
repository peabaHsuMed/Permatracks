import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from .config import *
from .func import read_serial,update_serial_read
import time



class calibration:
    """
    Initial implementation of ellipsoid calibration with referenced to https://www.media.mit.edu/projects/magnetomicrometry/overview/
    However, when getting W matrix, under the square root there was also negative. Need to re-check everything.
    
    """
    def __init__(self, ser,k_data):
        self.ser = ser
        self.k_data = k_data
        self.cal_data = np.zeros([self.k_data, N_sensor,3])
        self.W = np.zeros([N_sensor,3,3])
        self.Sensitivity = 1.0
        self.g = np.zeros([N_sensor,3,1])
        self.v = np.zeros([N_sensor,3,1])
        self.r = np.zeros([N_sensor,1])
        self.S = 1.0
        
    #def record_dummy_data(self):
        
    
    def dummy_data(self):
        self.cal_data = np.load("cal_data.npy")
        
        
    def record_data(self):
        '''
        Record data according to k_data, then transfer to microTesla (uT)
        '''
        for i in range(self.k_data):
            self.cal_data[i,:,:] = update_serial_read(self.ser, self.cal_data[i])
            #if self.cal_data[i,:,:].all() == 0:
                #self.cal_data[i,:,:] = self.cal_data[i-1,:,:]
                #print("Cal_data reached 0 ")
                
            print("Calibration step: ", i+1)
            print("Calibration data: ", self.cal_data[i,:,:])
            time.sleep(0.5)
            
        self.cal_data = self.cal_data * 1e-1
        
        for k in range(self.k_data):
            for j in range(N_sensor):
                if self.cal_data[k,j,:].all() == 0:
                    self.cal_data[k,j,:] = self.cal_data[k-1,j,:]
                    print("Cal_data reached 0 ")
            
    def get_sls_i (self, i_sensor):
        '''
        Perform least square fitting to solve Bi_L*sls = Bi_R
        get the sls for i_th sensor
        
        Bi_L is assemble each column
        '''
        
        Bi_L = np.zeros([self.k_data, 9])
        Bi_R = np.zeros([self.k_data,1])
        
        Bix = self.cal_data[:,i_sensor,0]
        Biy = self.cal_data[:,i_sensor,1]
        Biz = self.cal_data[:,i_sensor,2]
        
        
        Bi_L[:, 0] = Bix**2 + Biy**2 - 2*Biz**2
        Bi_L[:, 1] = Bix**2 - 2*Biy**2 + Biz**2
        Bi_L[:, 2] = 4 * Bix * Biy
        Bi_L[:, 3] = 2 * Bix * Biz
        Bi_L[:, 4] = 2 * Biy * Biz
        Bi_L[:, 5] = Bix
        Bi_L[:, 6] = Biy
        Bi_L[:, 7] = Biz
        Bi_L[:, 8] = 1.0
        
        Bi_R = (Bix**2) + (Biy**2) + Biz**2
        
        result = scipy.linalg.lstsq(Bi_L,Bi_R)
        return result[0]
    
    def slsmaptoA_i (self, sls_i):
        '''
        Transform back from sls to A
        '''
        mtx1 = np.array([[1,-1,-1],[1,-1,2],[1,2,-1]])
        vct1 = np.array([-4,-2,-2])
        
        Ui, Vi, Mi, Ni, Pi, Qi, Ri, Si, Ti = sls_i[0:9]
        
        
        Ai, Bi, Ci = np.dot(mtx1, np.array([1,sls_i[0],sls_i[1]]).T)
        Di, Ei, Fi = np.multiply(vct1, sls_i[2:5])
        
        A_i = np.array([[1-Ui-Vi, -2*Mi, -Ni], [-2*Mi, 1-Ui+2*Vi, -Pi], [-Ni, -Pi, 1+2*Ui-Vi]]) ##change the negatives
        #A_i = np.array([[Ai, Di/2, Ei/2],[Di/2, Bi, Fi/2], [Ei/2, Fi/2, Ci]])
        
        return A_i
    
    def get_v_i (self, A_i, sls_i):
        v_i = 0.5 * np.matmul(np.linalg.inv(A_i), sls_i[5:8])
        return v_i
    
    def get_r_i (self, v_i, A_i, sls_i):
        c_i = sls_i[8] + np.matmul(np.matmul(v_i.T, A_i), v_i) 
        r_i = np.sqrt(c_i)
        return r_i
    
    def assemble_mtx (self):
        '''
        Assemble submatrix of each sensor into global
        Could be optimized
        '''
        
        r_sum = 0.0
        
        for i in range(N_sensor):
            sls_i = self.get_sls_i(i)
            A_i = self.slsmaptoA_i(sls_i)
            v_i = self.get_v_i(A_i, sls_i)
            r_i = self.get_r_i(v_i, A_i, sls_i)
            
            self.W[i,:,:] = np.sqrt(np.abs(A_i)) ####### Manually added absolute value, as it keep on having negative value
            self.v[i,:,:] = v_i.reshape([3,1])
            self.r[i,:] = r_i
            
        r_avg = np.mean(self.r)
        ##need to asssemble g
        self.g = r_avg/(self.r)
        
    def apply_calibration(self, data):
        '''
        apply the calibration according: B__i = giSiWi-1(B_i-vi)
        '''
        post = np.zeros_like(data)
        
        for i in range(N_sensor):
            #post[i,:] = np.matmul(np.matmul(self.g[i],self.W[i]), (data[i,:]-self.v[i]))
            post[i,:] = (self.g[i] * (np.matmul(self.W[i], (data[i,:].reshape([3,1])-self.v[i])))).T
        
        post = post * self.S
        return post
'''
    def assemble_mtx_optimized (self):
'''
        #Assemble submatrix of each sensor into global
        #stacked version
'''
        
        r_sum = 0.0
        
        for i in range(N_sensor):
            sls_i = self.get_sls_i(i)
            A_i = self.slsmaptoA_i(sls_i)
            v_i = self.get_v_i(A_i, sls_i)
            r_i = self.get_r_i(v_i, A_i, sls_i)
            
            self.W[i,:,:] = np.sqrt(A_i)
            self.v[i,:,:] = v_i
            self.r[i,:] = r_i
            
        r_avg = np.mean(r)
        ##need to asssemble g
        self.g = r_avg/(self.r)
'''
        
        