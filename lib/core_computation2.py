import numpy as np
import math


class OptimizationProcess:

    """
    Class for optimization process

    Each optimization process, cost function regarding to one instance of magnetic data is needed to optimize the parameters to fit the magnetic data solution.
    Inside optimization process, each iteration in the algorithm will change the parameters feed and compute the cost value. Therefore, it is required to wrap the cost function to each data.
    Magnetic model, jacobian based on https://ieeexplore.ieee.org/document/8809206

    9DOF of jacobian is already implemented. However, cost function is not. But only make the other 3DOF as arguments should be enough to modify the cost function.
    """
    def __init__(self,N_sensor,N_magnets):
        self.N_magnets = N_magnets
        self.params = np.zeros((N_magnets, 6))  # Each magnet has 6 parameters: x, y, z, theta, phi, m_j
        self.Gx = 0.0
        self.Gy = 0.0
        self.Gz = 0.0
        self.N_sensor = N_sensor
        
        ###Parameters is mostly only useful when multi-step optimization is required, then parameters could be stored here
        
        self.sensor_pos = np.loadtxt('sensor_position_30.txt').reshape(-1,1)*1e-3 #1e-3 mm -> m
        
    def Bx_model(self,x_i, y_i, z_i, theta, phi, m_j):
        a = (3*x_i*(np.sin(theta)*np.cos(phi)*x_i + np.sin(theta)*np.sin(phi)*y_i + np.cos(theta)*z_i))/((x_i**2 + y_i **2 + z_i**2)**(5/2))
        b = (np.sin(theta)*np.cos(phi))/((x_i**2 + y_i **2 + z_i**2)**(3/2))
        result = m_j*(a - b)
        return result

    def By_model(self,x_i, y_i, z_i, theta, phi, m_j):
        a = (3*y_i*(np.sin(theta)*np.cos(phi)*x_i + np.sin(theta)*np.sin(phi)*y_i + np.cos(theta)*z_i))/((x_i**2 + y_i **2 + z_i**2)**(5/2))
        b = (np.sin(theta)*np.sin(phi))/((x_i**2 + y_i **2 + z_i**2)**(3/2))
        result = m_j*(a - b)
        return result

    def Bz_model(self,x_i, y_i, z_i, theta, phi, m_j):
        a = (3*z_i*(np.sin(theta)*np.cos(phi)*x_i + np.sin(theta)*np.sin(phi)*y_i + np.cos(theta)*z_i))/((x_i**2 + y_i **2 + z_i**2)**(5/2))
        b = (np.cos(theta))/((x_i**2 + y_i **2 + z_i**2)**(3/2))
        result = m_j*(a - b)
        return result
    
    def cost_6DOF(self, data):
        '''
        Parameters
        ----------
        ambient : Array N_sensor*3
            G
        data : Array N_sensor*3
            sensor_B

        Returns
        -------
        function: cost_funct(param)
            cost_func_factory will modify cost_funct base on ambient and data

        '''
        ##print(data) # 16*3
        #print("Initialized cost_func_factory")
        def cost_func (param):
            '''
            Parameters
            ----------
            param: Array
                Vector of 5 DOF, x,y,z,theta,phi
            
            Returns
            ----------
            result: Array
                Vector of cost [x1,y1,z1,x2...] 3xN
            '''
            #print(param) # at first equal to x0(6 elements), and later change while optimizing
            sensor_B = data
             
            #print("size of dataaaaaaaaa: ",np.size(sensor_B)) ## 48
            cost_value = np.zeros([3*self.N_sensor], dtype=np.float64)
            
            for i in range(0, np.size(sensor_B), 3):
                for j in range(self.N_magnets):
                    idx = j*6
                    x_i = self.sensor_pos[i] - param[idx]
                    y_i = self.sensor_pos[i+1] - param[idx+1]
                    z_i = self.sensor_pos[i+2] - param[idx+2]
                    theta = param[idx+3]
                    phi = param[idx+4]
                    m_j = param[idx+5]

                    cost_value[i] += self.Bx_model(x_i, y_i, z_i, theta, phi, m_j) 
                    cost_value[i+1] += self.By_model(x_i, y_i, z_i, theta, phi, m_j) 
                    cost_value[i+2] += self.Bz_model(x_i, y_i, z_i, theta, phi, m_j) 
                
                cost_value[i] += self.Gx
                cost_value[i+1] += self.Gy
                cost_value[i+2] += self.Gz

                cost_value[i] -= sensor_B[i//3,i%3]
                cost_value[i+1] -= sensor_B[i//3,i%3+1] 
                cost_value[i+2] -= sensor_B[i//3,i%3+2]
                
            #print("Parameters inside cost: ", param[0:3])
            
            return cost_value
        return cost_func
    
    def cost_9DOF(self, data):
        '''
        Parameters
        ----------
        ambient : Array N_sensor*3
            G
        data : Array N_sensor*3
            sensor_B

        Returns
        -------
        function: cost_funct(param)
            cost_func_factory will modify cost_funct base on ambient and data

        '''
        #print("Initialized cost_func_factory")
        def cost_func (param):
            '''
            Parameters
            ----------
            param: Array
                Vector of 9 DOF, x,y,z,theta,phi,m_j,Gx,Gy,Gz
            
            Returns
            ----------
            result: Array
                Vector of cost [x1,y1,z1,x2...] 3xN
            '''
            
            sensor_B = data
             
            cost_value = np.zeros([3*self.N_sensor], dtype=np.float64)
            
            for i in range(0, np.size(sensor_B), 3):
                for j in range(self.N_magnets):
                    idx = j*9
                    x_i = self.sensor_pos[i] - param[idx]
                    y_i = self.sensor_pos[i+1] - param[idx+1]
                    z_i = self.sensor_pos[i+2] - param[idx+2]
                    theta = param[idx+3]
                    phi = param[idx+4]
                    m_j = param[idx+5]
                    Gx = param[idx+6]
                    Gy = param[idx+7]
                    Gz = param[idx+8]

                    cost_value[i] += (self.Bx_model(x_i, y_i, z_i, theta, phi, m_j) + Gx)
                    cost_value[i+1] += (self.By_model(x_i, y_i, z_i, theta, phi, m_j) + Gy) 
                    cost_value[i+2] += (self.Bz_model(x_i, y_i, z_i, theta, phi, m_j) + Gz)
                
                #cost_value[i] += Gx
                #cost_value[i+1] += Gy
                #cost_value[i+2] += Gz

                cost_value[i] -= sensor_B[i//3,i%3]
                cost_value[i+1] -= sensor_B[i//3,i%3+1] 
                cost_value[i+2] -= sensor_B[i//3,i%3+2]
            
            #print("Parameters inside cost: ", param[0:3])
            
            return cost_value
        return cost_func
    
    
    def jacobian_6DOF (self,param):
        
        '''
        Parameters
        ----------
        param : Array
            Vector of 9 DOF, x,y,z,theta,phi, mj, Gx, Gy, Gz (constant)
        
        Returns
        -------
        result : Array
            Combonents of jacobian, [3*N,9*M]

        '''
        J_mtx = np.zeros([3*self.N_sensor,6*self.N_magnets])
     
        for i in range(0, 3*self.N_sensor, 3):
            for j in range(self.N_magnets):
                idx = j*6
                x_i = self.sensor_pos[i] - param[idx]   ### Should not combine with cost function xi, yi, zi, because iteration of jacobian and cost function are different
                y_i = self.sensor_pos[i+1] - param[idx+1]
                z_i = self.sensor_pos[i+2] - param[idx+2]
                theta = param[idx+3]
                phi = param[idx+4]
                m_j = param[idx+5]

                sub_J = np.zeros([3,6])
                ma_7 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-7/2)
                ma_5 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-5/2)
                a_5 = (x_i**2 + y_i **2 + z_i**2)**(-5/2)
            
                sub_J[0,0] = (ma_7*(3*x_i*(2*x_i**2 - 3*y_i**2 - 3*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.cos(theta)))
                sub_J[1,0] = (ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta)))
                sub_J[2,0] = (ma_7*(3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 15*x_i*y_i*z_i*np.sin(theta)*np.sin(phi) + 3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta)))
                sub_J[0,1] = (ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta)))
                sub_J[1,1] = (ma_7*(3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-3*x_i**2 + 2*y_i**2 - 3*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.cos(theta)))
                sub_J[2,1] = (ma_7*(15*x_i*y_i*z_i*np.sin(theta)*np.cos(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta)))
                sub_J[0,2] = (sub_J[2,0])
                sub_J[1,2] = (sub_J[2,1])
                sub_J[2,2] = (ma_7*(3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-3*x_i**2 - 3*y_i**2 + 2*z_i**2)*np.cos(theta)))
                sub_J[0,3] = ma_5*((2*x_i**2 - y_i**2 - z_i**2)*np.cos(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(phi)*np.cos(theta) - 3*x_i*z_i*np.sin(theta))
                sub_J[1,3] = ma_5*(3*x_i*y_i*np.cos(phi)*np.cos(theta) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(phi)*np.cos(theta) - 3*y_i*z_i*np.sin(theta))
                sub_J[2,3] = ma_5*(3*x_i*z_i*np.cos(phi)*np.cos(theta) + 3*y_i*z_i*np.sin(phi)*np.cos(theta) - (-x_i**2 - y_i**2 + 2*z_i**2)*np.sin(theta))
                sub_J[0,4] = ma_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) - (2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi))
                sub_J[1,4] = ma_5*((-x_i**2 + 2*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) - 3*x_i*y_i*np.sin(theta)*np.sin(phi))
                sub_J[2,4] = ma_5*(3*y_i*z_i*np.sin(theta)*np.cos(phi) - 3*x_i*z_i*np.sin(theta)*np.sin(phi))
                sub_J[0,5] = a_5*((2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(theta)*np.sin(phi) + 3*x_i*z_i*np.cos(theta))
                sub_J[1,5] = a_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*z_i*np.cos(theta))
                sub_J[2,5] = a_5*(3*x_i*z_i*np.sin(theta)*np.cos(phi) + 3*y_i*z_i*np.sin(theta)*np.sin(phi) + (-x_i**2 - y_i**2 + 2*z_i**2)*np.cos(theta))
            
                J_mtx[i:i+3,idx:idx+6] = sub_J

        return J_mtx
    

    def jacobian_9DOF (self,param):
        
        '''
        Parameters
        ----------
        param : Array
            Vector of 9 DOF, x,y,z,theta,phi, mj, Gx, Gy, Gz (constant)
        
        Returns
        -------
        result : Array
            Combonents of jacobian, [3*N,9*M]

        '''
        J_mtx = np.zeros([3*self.N_sensor,9*self.N_magnets])
        D = np.identity(3)
        for i in range(0, 3*self.N_sensor, 3):
            for j in range(self.N_magnets):
                idx = j*9
                x_i = self.sensor_pos[i] - param[idx]   ### Should not combine with cost function xi, yi, zi, because iteration of jacobian and cost function are different
                y_i = self.sensor_pos[i+1] - param[idx+1]
                z_i = self.sensor_pos[i+2] - param[idx+2]
                theta = param[idx+3]
                phi = param[idx+4]
                m_j = param[idx+5]
            
                sub_J = np.zeros([3,6])
                ma_7 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-7/2)
                ma_5 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-5/2)
                a_5 = (x_i**2 + y_i **2 + z_i**2)**(-5/2)
                
                sub_J[0,0] = (ma_7*(3*x_i*(2*x_i**2 - 3*y_i**2 - 3*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.cos(theta)))
                sub_J[1,0] = (ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta)))
                sub_J[2,0] = (ma_7*(3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 15*x_i*y_i*z_i*np.sin(theta)*np.sin(phi) + 3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta)))
                sub_J[0,1] = (ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta)))
                sub_J[1,1] = (ma_7*(3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-3*x_i**2 + 2*y_i**2 - 3*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.cos(theta)))
                sub_J[2,1] = (ma_7*(15*x_i*y_i*z_i*np.sin(theta)*np.cos(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta)))
                sub_J[0,2] = (sub_J[2,0])
                sub_J[1,2] = (sub_J[2,1])
                sub_J[2,2] = (ma_7*(3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-3*x_i**2 - 3*y_i**2 + 2*z_i**2)*np.cos(theta)))
                sub_J[0,3] = ma_5*((2*x_i**2 - y_i**2 - z_i**2)*np.cos(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(phi)*np.cos(theta) - 3*x_i*z_i*np.sin(theta))
                sub_J[1,3] = ma_5*(3*x_i*y_i*np.cos(phi)*np.cos(theta) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(phi)*np.cos(theta) - 3*y_i*z_i*np.sin(theta))
                sub_J[2,3] = ma_5*(3*x_i*z_i*np.cos(phi)*np.cos(theta) + 3*y_i*z_i*np.sin(phi)*np.cos(theta) - (-x_i**2 - y_i**2 + 2*z_i**2)*np.sin(theta))
                sub_J[0,4] = ma_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) - (2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi))
                sub_J[1,4] = ma_5*((-x_i**2 + 2*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) - 3*x_i*y_i*np.sin(theta)*np.sin(phi))
                sub_J[2,4] = ma_5*(3*y_i*z_i*np.sin(theta)*np.cos(phi) - 3*x_i*z_i*np.sin(theta)*np.sin(phi))
                sub_J[0,5] = a_5*((2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(theta)*np.sin(phi) + 3*x_i*z_i*np.cos(theta))
                sub_J[1,5] = a_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*z_i*np.cos(theta))
                sub_J[2,5] = a_5*(3*x_i*z_i*np.sin(theta)*np.cos(phi) + 3*y_i*z_i*np.sin(theta)*np.sin(phi) + (-x_i**2 - y_i**2 + 2*z_i**2)*np.cos(theta))
            
                J_mtx[i:i+3,idx:idx+6] = sub_J
                J_mtx[i:i+3,idx+6:idx+9] = D    
        
        return J_mtx