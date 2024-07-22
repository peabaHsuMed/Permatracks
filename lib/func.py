import numpy as np
import math
import matplotlib.pyplot as plt
from .config import *
import serial
import re
import time

class sensor:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
class magnet:
    def __init__(self, x, y, z, theta, phi):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.phi = phi

def Bx_model(x_i, y_i, z_i, theta, phi, m_j):
    a = (3*x_i*(np.sin(theta)*np.cos(phi)*x_i + np.sin(theta)*np.sin(phi)*y_i + np.cos(theta)*z_i))/((x_i**2 + y_i **2 + z_i**2)**(5/2))
    b = (np.sin(theta)*np.cos(phi))/((x_i**2 + y_i **2 + z_i**2)**(3/2))
    result = m_j*(a - b)
    return result

def By_model(x_i, y_i, z_i, theta, phi, m_j):
    a = (3*y_i*(np.sin(theta)*np.cos(phi)*x_i + np.sin(theta)*np.sin(phi)*y_i + np.cos(theta)*z_i))/((x_i**2 + y_i **2 + z_i**2)**(5/2))
    b = (np.sin(theta)*np.sin(phi))/((x_i**2 + y_i **2 + z_i**2)**(3/2))
    result = m_j*(a - b)
    return result

def Bz_model(x_i, y_i, z_i, theta, phi, m_j):
    a = (3*z_i*(np.sin(theta)*np.cos(phi)*x_i + np.sin(theta)*np.sin(phi)*y_i + np.cos(theta)*z_i))/((x_i**2 + y_i **2 + z_i**2)**(5/2))
    b = (np.cos(theta))/((x_i**2 + y_i **2 + z_i**2)**(3/2))
    result = m_j*(a - b)
    return result

def cost_func_factory(data):
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
    print("Initialized cost_func_factory")
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
        #Gx = param[6]
        #Gy = param[7]
        #Gz = param[8]
        Gx = G1
        Gy = G2
        Gz = G3
        
        sensor_B = data
        #m_j = param[5]/3*(r_mag**3)
        #m_j = param[5]
         
        
        cost_value = np.zeros([3*N_sensor])
        
        #print("G and Data are: ")
        #print(G)
        #print(sensor_B)
        
        for i in range(0, np.size(sensor_B), 3):
            x_i = sensor_pos[i] - param[0]
            y_i = sensor_pos[i+1] - param[1]
            z_i = sensor_pos[i+2] - param[2]
        
            cost_value[i] = Bx_model(x_i, y_i, z_i, param[3], param[4], m_j) - sensor_B[i//3,i%3] + Gx
            cost_value[i+1] =By_model(x_i, y_i, z_i, param[3], param[4], m_j) - sensor_B[i//3,i%3+1] + Gy
            cost_value[i+2] = Bz_model(x_i, y_i, z_i, param[3], param[4], m_j) - sensor_B[i//3,i%3+2] + Gz
        
        #print("Parameters inside cost: ", param[0:3])
        
        return cost_value
    return cost_func

def jacobian_9DOF (param):
    
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
    J_mtx = np.zeros([3*N_sensor,9])
    D = np.identity(3)
    theta = param[3]
    phi = param[4]
    m_j = param[5]
    for i in range(0, 3*N_sensor, 3):
        x_i = sensor_pos[i] - param[0]
        y_i = sensor_pos[i+1] - param[1]
        z_i = sensor_pos[i+2] - param[2]
        
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
        
        J_mtx[i:i+3,0:6] = sub_J
        J_mtx[i:i+3,6:10] = D
        
    
    return J_mtx

def jacobian_6DOF (param):
    
    '''
    Parameters
    ----------
    param : Array
        Vector of 6 DOF, x,y,z,theta,phi, mj
    
    Returns
    -------
    result : Array
        Combonents of jacobian, [3*N,6*M]

    '''
    J_mtx = np.zeros([3*N_sensor,6])
    theta = param[3]
    phi = param[4]
    m_j = param[5]
    for i in range(0, 3*N_sensor, 3):
        x_i = sensor_pos[i] - param[0]
        y_i = sensor_pos[i+1] - param[1]
        z_i = sensor_pos[i+2] - param[2]
        
        sub_J = np.zeros([3,6])
        ma_7 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-7/2)
        ma_5 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-5/2)
        a_5 = (x_i**2 + y_i **2 + z_i**2)**(-5/2)
        
        sub_J[0,0] = ma_7*(3*x_i*(2*x_i**2 - 3*y_i**2 - 3*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.cos(theta))
        sub_J[1,0] = ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta))
        sub_J[2,0] = ma_7*(3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 15*x_i*y_i*z_i*np.sin(theta)*np.sin(phi) + 3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta))
        sub_J[0,1] = ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta))
        sub_J[1,1] = ma_7*(3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-3*x_i**2 + 2*y_i**2 - 3*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.cos(theta))
        sub_J[2,1] = ma_7*(15*x_i*y_i*z_i*np.sin(theta)*np.cos(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta))
        sub_J[0,2] = sub_J[2,0]
        sub_J[1,2] = sub_J[2,1]
        sub_J[2,2] = ma_7*(3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-3*x_i**2 - 3*y_i**2 + 2*z_i**2)*np.cos(theta))
        sub_J[0,3] = ma_5*((2*x_i**2 - y_i**2 - z_i**2)*np.cos(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(phi)*np.cos(theta) - 3*x_i*z_i*np.sin(theta))
        sub_J[1,3] = ma_5*(3*x_i*y_i*np.cos(phi)*np.cos(theta) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(phi)*np.cos(theta) - 3*y_i*z_i*np.sin(theta))
        sub_J[2,3] = ma_5*(3*x_i*z_i*np.cos(phi)*np.cos(theta) + 3*y_i*z_i*np.sin(phi)*np.cos(theta) - (-x_i**2 - y_i**2 + 2*z_i**2)*np.sin(theta))
        sub_J[0,4] = ma_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) - (2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi))
        sub_J[1,4] = ma_5*((-x_i**2 + 2*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) - 3*x_i*y_i*np.sin(theta)*np.sin(phi))
        sub_J[2,4] = ma_5*(3*y_i*z_i*np.sin(theta)*np.cos(phi) - 3*x_i*z_i*np.sin(theta)*np.sin(phi))
        sub_J[0,5] = a_5*((2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(theta)*np.sin(phi) + 3*x_i*z_i*np.cos(theta))
        sub_J[1,5] = a_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*z_i*np.cos(theta))
        sub_J[2,5] = a_5*(3*x_i*z_i*np.sin(theta)*np.cos(phi) + 3*y_i*z_i*np.sin(theta)*np.sin(phi) + (-x_i**2 - y_i**2 + 2*z_i**2)*np.cos(theta))
        
        J_mtx[i:i+3,0:6] = sub_J
    
    return J_mtx

def jacobian_5DOF (param):
    
    '''
    Parameters
    ----------
    param : Array
        Vector of 5 DOF, x,y,z,theta,phi
    
    Returns
    -------
    result : Array
        Combonents of jacobian, [3*N,5*M]

    '''
    J_mtx = np.zeros([3*N_sensor,5])
    theta = param[3]
    phi = param[4]
    for i in range(0, 3*N_sensor, 3):
        x_i = sensor_pos[i] - param[0]
        y_i = sensor_pos[i+1] - param[1]
        z_i = sensor_pos[i+2] - param[2]

        
        sub_J = np.zeros([3,5])
        ma_7 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-7/2)
        ma_5 = m_j*(x_i**2 + y_i **2 + z_i**2)**(-5/2)
        a_5 = (x_i**2 + y_i **2 + z_i**2)**(-5/2)
        
        sub_J[0,0] = ma_7*(3*x_i*(2*x_i**2 - 3*y_i**2 - 3*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.cos(theta))
        sub_J[1,0] = ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta))
        sub_J[2,0] = ma_7*(3*z_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 15*x_i*y_i*z_i*np.sin(theta)*np.sin(phi) + 3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta))
        sub_J[0,1] = ma_7*(3*y_i*(4*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 15*x_i*y_i*z_i*np.cos(theta))
        sub_J[1,1] = ma_7*(3*x_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-3*x_i**2 + 2*y_i**2 - 3*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.cos(theta))
        sub_J[2,1] = ma_7*(15*x_i*y_i*z_i*np.sin(theta)*np.cos(phi) + 3*z_i*(-x_i**2 + 4*y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.cos(theta))
        sub_J[0,2] = sub_J[2,0]
        sub_J[1,2] = sub_J[2,1]
        sub_J[2,2] = ma_7*(3*x_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.cos(phi) + 3*y_i*(-x_i**2 - y_i**2 + 4*z_i**2)*np.sin(theta)*np.sin(phi) + 3*z_i*(-3*x_i**2 - 3*y_i**2 + 2*z_i**2)*np.cos(theta))
        sub_J[0,3] = ma_5*((2*x_i**2 - y_i**2 - z_i**2)*np.cos(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(phi)*np.cos(theta) - 3*x_i*z_i*np.sin(theta))
        sub_J[1,3] = ma_5*(3*x_i*y_i*np.cos(phi)*np.cos(theta) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(phi)*np.cos(theta) - 3*y_i*z_i*np.sin(theta))
        sub_J[2,3] = ma_5*(3*x_i*z_i*np.cos(phi)*np.cos(theta) + 3*y_i*z_i*np.sin(phi)*np.cos(theta) - (-x_i**2 - y_i**2 + 2*z_i**2)*np.sin(theta))
        sub_J[0,4] = ma_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) - (2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.sin(phi))
        sub_J[1,4] = ma_5*((-x_i**2 + 2*y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) - 3*x_i*y_i*np.sin(theta)*np.sin(phi))
        sub_J[2,4] = ma_5*(3*y_i*z_i*np.sin(theta)*np.cos(phi) - 3*x_i*z_i*np.sin(theta)*np.sin(phi))
        #sub_J[0,5] = a_5*((2*x_i**2 - y_i**2 - z_i**2)*np.sin(theta)*np.cos(phi) + 3*x_i*y_i*np.sin(theta)*np.sin(phi) + 3*x_i*z_i*np.cos(theta))
        #sub_J[1,5] = a_5*(3*x_i*y_i*np.sin(theta)*np.cos(phi) + (-x_i**2 + 2*y_i**2 -z_i**2)*np.sin(theta)*np.sin(phi) + 3*y_i*z_i*np.cos(theta))
        #sub_J[2,5] = a_5*(3*x_i*z_i*np.sin(theta)*np.cos(phi) + 3*y_i*z_i*np.sin(theta)*np.sin(phi) + (-x_i**2 - y_i**2 + 2*z_i**2)*np.cos(theta))
        
        J_mtx[i:i+3,0:5] = sub_J
    
    return J_mtx

def sensor_position (x, y):
    '''

    Parameters
    ----------
    x : int
        Separation in x-dir
    y : int
        Separation in y-dir

    Returns
    -------
    sensor_pos : Vector
        xyz of sensor [x1,y1,z1,x2,y2,z2,...].

    '''
    result =  np.zeros([3*N_sensor,1])
    
    for i in range(N_sensor):
            result[i*3] = (i%4)*50
            result[i*3+1] = (i//4)*50
    return result




def serial_setup():
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM4'
    ser.open()
    return ser

def read_serial(ser):
    '''
    

    Parameters
    ----------
    ser : serial connection object
        DESCRIPTION.

    Returns
    -------
    Array of signal

    '''
    ser.reset_input_buffer()
    data = np.zeros([N_sensor,3])
    
    for i in range (0, N_sensor):
        line = str(ser.readline(),'UTF-8')
        #line = str(ser.readline(),'UTF-8')
        #print(line)

    
        match = re.search(r'Sensor(\d+): (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)', line)
        #match = re.search(r'(-?\d+.\d+) (-?\d+.\d+) (-?\d+.\d+)', line)
        #print(match)
        if match:
            value = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            #values = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            data[int(match.group(1))-1,0:3] = value
                
    #print("Number of byte in waiting at serial: " + str(ser.in_waiting))
    ser.reset_input_buffer()
    #print("Number of byte in waiting at serial after flush: " +  str(ser.in_waiting))
    #print(data)
    return data

def update_serial_read(ser, data):
    '''
    Refresh data read to corresponding
    
    Parameters
    ----------
    ser : serial connection object
        DESCRIPTION.

    Returns
    -------
    Array of signal
    '''
    ser.reset_input_buffer()
    temp_str = np.empty([N_sensor,1], dtype='object')
    
    for i in range (0, N_sensor):
        temp_str[i] = str(ser.readline(),'UTF-8')
        #print(temp_str[i])
        
    for i in range (0,N_sensor):
        match = re.search(r'Sensor(\d+): (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)', str(temp_str[i]))
        
        if match:
            value = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            data[int(match.group(1))-1,0:3] = value
                
    #print("Number of byte in waiting at serial: " + str(ser.in_waiting))
    ser.reset_input_buffer()
    #print("Number of byte in waiting at serial after flush: " +  str(ser.in_waiting))
    #print(data)
    return data

def calibration_scale (ser):
    '''
    Retrieve the parameters to perform calibration
    
    Parameters
    ----------
    ser : i2c conection
        

    Returns
    -------
    offset : Array of (16,3)
        Hard iron offset
        
    scale : Array of (16,3)
        Soft iron offset

    '''
    N_data = 60
    cal_data = np.zeros([N_data,N_sensor,3])
    offset = np.zeros([N_sensor,3]) #offset value correspond to each sensor, axis
    scale = np.zeros([N_sensor,3])  #scale value correspond to each sensor, axis
    empty = read_serial(ser)
    max_offset = np.zeros([N_sensor,3])
    min_offset = np.zeros([N_sensor,3])
    avg_delta = np.zeros([N_sensor,3])
    avg_delta_scalar = np.zeros([N_sensor,1])
    
    for i in range(N_data):
        cal_data[i,:,:] = update_serial_read(ser, cal_data[i])
        print("Calibration step: ", i+1)
        print("Calibration data: ", cal_data[i,:,:])
        
                    
        time.sleep(0.5)
        
    for k in range(N_data):
        for j in range(N_sensor):
            if cal_data[k,j,:].all() == 0:
                cal_data[k,j,:] = cal_data[k-1,j,:]
                print("Cal_data reached 0 ")

        
    for j in range (N_sensor):
        for i in range(3):
            max_offset[j,i] = np.max(cal_data[:,j,i])
            min_offset[j,i] = np.min(cal_data[:,j,i])
            offset[j,i] = (max_offset[j,i] + min_offset[j,i])/2
            avg_delta[j,i] = (max_offset[j,i] - min_offset[j,i])/2
    
        avg_delta_scalar[j] = (avg_delta[j,0] + avg_delta[j,1] + avg_delta[j,2])/3
        
    for i in range(3):
        scale[:,i] = np.divide(avg_delta_scalar[:,0],avg_delta[:,i])
        
    return offset, scale

def apply_calibration(offset,scale,data):
    '''
    Apply the calibration parameters to the data, need to be same dimension

    Parameters
    ----------
    offset : Array of (16,3)
        Hard iron offset
        
    scale : Array of (16,3)
        Soft iron scaling
        
    data : Array of (16,3)
        Raw data from i2c, in milligauss

    Returns
    -------
    post_process : Array of (16,3)
        Data calibrated

    '''
    
    post_process = np.zeros([N_sensor,3])
    for j in range (N_sensor):
        for i in range(3):
            post_process[j,i] = (data[j][i] - offset[j,i])*scale[j,i]
    
    return post_process

def get_num_gradient_factory(cost):
    '''
    Cost function as input to generate .get_num_gradient
    
    '''
    
    def get_num_gradient (params):
        '''
        Get numerical gradient through central difference methods
    
        Parameters
        ----------
        params : vector of 5/6 DOF (if 6 dof, comment the parameters in config.py)
            DESCRIPTION.
            
        Returns
        -------
        gradient : matrix of [3*Nth sensor, 5/6 DOF]
            gradient w.r.t each DOF and Bix/y/z
    
        '''
        h = 1e-5
        gradient = np.zeros([3*N_sensor,len(params)])
        for j in range(np.shape([gradient])[1]):
            for i in range(np.shape([gradient])[2]):
                temp = np.copy(params)
                temp[i] = params[i] + h
                fr = cost(temp)[j]
                
                temp = np.copy(params)
                temp[i] = temp[i] - h
                fl = cost(temp)[j]
                gradient[j,i] = (fr - fl)/(2*h)
            
        return gradient
    return get_num_gradient