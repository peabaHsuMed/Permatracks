import sys
import numpy as np
#from scipy import linalg
import scipy as sp

class EllipsoidCalibrator():
    ''' Class for ellipsoid calibration
        
        Based on https://teslabs.com/articles/magnetometer-calibration/
        Referenced https://github.com/nliaudat/magnetometer_calibration/blob/main/calibrate.py

        Get F constant from http://www.ngdc.noaa.gov/geomag-web (tab Magnetic Field), transform into microtesla(uT)
        It is assumed there will be an outer loop and putting data as arguments. 


        Parameters
        ----------
        N_sensor : number of sensors
        N_data : The default is 60
        
    '''


    def __init__(self, N_sensor, N_data=60):
        # initialize values
        
        self.N_sensor = N_sensor
        self.N_data = N_data
        
        self.F   = 48.7162 #47.6965 #48.7162                  ###Expected earth magnetic field intensity, 48.7162 microtesla
        self.b   = np.zeros([N_sensor, 3, 1])
        self.A_1 = np.tile(np.eye(3)[np.newaxis, :, :], (N_sensor, 1, 1)) ### (N_sensor,3,3)
        self.cal_data = np.zeros([N_data,N_sensor,3])
        
        self.M_ = np.zeros([N_sensor,3,3])
        self.n = np.zeros([N_sensor,3,1])
        self.d= np.zeros([N_sensor,1])

        
        
    def retrieve_data(self,i,data):
        '''
        Input: i (Iterator), data (data of the iteration)
        '''
        self.cal_data[i,:,:] = data
        
    def get_dummy_data(self):
        '''
        Get pre-recorded data for debugging
        '''
        self.cal_data = np.load('cal_data.npy')

    def assemble_coefficient(self):
        '''
        Assemble all calibration coefficients b, A_1 for ALL sensor
        Also save the coefficient into "calibration_b", "calibration_A_1"
        '''
        print('Calculating calibration coefficients...')
        
        for i_sensor in range(self.N_sensor):
            
            # ellipsoid fit
            s = np.array(self.cal_data[:,i_sensor,:]).T ### Should be row vector of x,y,z
            M, n, d = self.__ellipsoid_fit(s)
    
            # calibration parameters
            # note: some implementations of sqrtm return complex type, taking real
            M_1 = sp.linalg.inv(M)
            self.b[i_sensor] = -np.dot(M_1, n)
            self.A_1[i_sensor] = np.real(self.F / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) *
                               sp.linalg.sqrtm(M))
            
        np.save("calibration_b", self.b)
        np.save("calibration_A_1", self.A_1)

    
    def load_coefficient(self):
        '''
        Load calibration coefficient from previously calibration
        '''
        self.b = np.load("calibration_b.npy")            
        self.A_1 =np.load("calibration_A_1.npy")
            
        
    def apply(self, data):
        '''
        Apply the calibration coefficient to the data and calibrate ellipsoid data to circular dat
        '''

        post = np.zeros([self.N_sensor,3])
        
        for i_sensor in range(self.N_sensor):
            post[i_sensor,:] = np.matmul(self.A_1[i_sensor], (data[i_sensor,:].reshape(3,1) - self.b[i_sensor])).ravel()
            
        return post
        
    def __ellipsoid_fit(self, s):
        ''' Estimate ellipsoid parameters from a set of points for EACH sensor.

            Parameters
            ----------
            s : array_like
              The samples (M,N) where M=3 (x,y,z) and N=number of samples.

            Returns
            -------
            M, n, d : array_like, array_like, float
              The ellipsoid parameters M, n, d.

            References
            ----------
            .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
               fitting," in Geometric Modeling and Processing, 2004.
               Proceedings, vol., no., pp.335-340, 2004
        '''

        # D (samples)
        D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                      2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                      2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6,:6]
        S_12 = S[:6,6:]
        S_21 = S[6:,:6]
        S_22 = S[6:,6:]

        # C (Eq. 8, k=4)
        C = np.array([[-1,  1,  1,  0,  0,  0],
                      [ 1, -1,  1,  0,  0,  0],
                      [ 1,  1, -1,  0,  0,  0],
                      [ 0,  0,  0, -4,  0,  0],
                      [ 0,  0,  0,  0, -4,  0],
                      [ 0,  0,  0,  0,  0, -4]])

        # v_1 (eq. 15, solution)
        E = np.dot(sp.linalg.inv(C),
                   S_11 - np.dot(S_12, np.dot(sp.linalg.inv(S_22), S_21)))

        E_w, E_v = np.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0: v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

        # quadric-form parameters
        M = np.array([[v_1[0], v_1[3], v_1[4]],
                      [v_1[3], v_1[1], v_1[5]],
                      [v_1[4], v_1[5], v_1[2]]])
        n = np.array([[v_2[0]],
                      [v_2[1]],
                      [v_2[2]]])
        d = v_2[3]

        return M, n, d

