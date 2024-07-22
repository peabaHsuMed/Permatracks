from .config import *
import serial
import re
import json


class SERIAL:   
    """
    Original use for UART/SERIAL communicate for LIS3MDL, but not maintained
    """
    def __init__(self, name):
        self.obj = name
        self.ser = serial.Serial()
        
        
        try:
            if self.obj == "QMC5883L":    
                self.ser.baudrate = 115200
                self.ser.port = 'COM4'
                self.ser.open()
            
            elif self.obj == "LIS3MDL":
                self.ser.baudrate = 115200
                self.ser.port = 'COM3'
                self.ser.open()
                
            else:
                raise ValueError("The sensor name is wrong!")
                    
        except ValueError as e:
            print (e)
    
        class ReadLine:
            #Snatched from internet, still not integrated 
            def __init__(self, s):
                self.buf = bytearray()
                self.s = s
        
            def readline(self):
                i = self.buf.find(b"\n")
                if i >= 0:
                    r = self.buf[:i+1]
                    self.buf = self.buf[i+1:]
                    return r
                while True:
                    i = max(1, min(2048, self.s.in_waiting))
                    data = self.s.read(i)
                    i = data.find(b"\n")
                    if i >= 0:
                        r = self.buf + data[:i+1]
                        self.buf[0:] = data[i+1:]
                        return r
                    else:
                        self.buf.extend(data)
        
        self.r1 = ReadLine(self.ser)
        
    def read(self):
        if self.obj == "LIS3MDL":
            self.ser.reset_input_buffer()
            
            #line2 = self.r1.readline()
            #print(f"Buffer waiting: {self.ser.in_waiting}")
            raw = self.ser.readline()
            line = str(raw,'UTF-8')
            print(line)
            return np.array(json.loads(line),dtype=float)   #Read serial, convert to str, to list by json, then to numpy
            
        if self.obj == "QMC5883L":
            self.ser.reset_input_buffer()
            data = np.zeros([N_sensor,3])
            
            for i in range (0, N_sensor):
                line = str(self.ser.readline(),'UTF-8')
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
            self.ser.reset_input_buffer()
            #print("Number of byte in waiting at serial after flush: " +  str(ser.in_waiting))
            #print(data)
            return data
    
    def write(self, data):
        '''
        Get string from data, then encode, then write to serial
        '''
        self.ser.write(data.encode('utf-8'))

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