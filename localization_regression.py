#import asyncio
#from bleak import BleakScanner
#
#async def run():
#    devices = await BleakScanner.discover()
#    for d in devices:
#        print(d)
#
#loop = asyncio.get_event_loop()
#loop.run_until_complete(run())

# mpy-magnetometer: 68:B6:B3:3E:45:2E
# UUID: 00002aa1-0000-1000-8000-00805f9b34fb

"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Perform magnet localization
This script directly performs 6DOF optimization to retrieve 3 position, 3 orientation and magnetic strength
Optimization algorithm could be chosen 'trf' & 'lm' by commenting different lines, and rolling median filter, direct deduct of static background could be chosen by commenting the line

"""

import argparse
import time
import asyncio
import logging
import struct
from collections import deque 
from operator import sub

#from lib import *
from lib.calibration_simple import *
from lib.calibration_ellipsoid_fit import *
from lib.core_computation import *
from lib.utility import *
from lib.plot import *
import scipy as sp
import time

from bleak import BleakClient, BleakScanner

import numpy as np
import cv2 
import keyboard
import csv

from keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler

import joblib


# Import libraries
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading



logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
#"2aa10001-88b8-11ee-8501-0800200c9a66",
#"2aa10002-88b8-11ee-8501-0800200c9a66",
#"2aa10003-88b8-11ee-8501-0800200c9a66",
#"2aa10004-88b8-11ee-8501-0800200c9a66",
#"2aa10005-88b8-11ee-8501-0800200c9a66",
#"2aa10006-88b8-11ee-8501-0800200c9a66",
#"2aa10007-88b8-11ee-8501-0800200c9a66",
#"2aa10008-88b8-11ee-8501-0800200c9a66",
#"2aa10009-88b8-11ee-8501-0800200c9a66",
#"2aa1000a-88b8-11ee-8501-0800200c9a66",
#"2aa1000b-88b8-11ee-8501-0800200c9a66",
#"2aa1000c-88b8-11ee-8501-0800200c9a66",
#"2aa1000d-88b8-11ee-8501-0800200c9a66",
#"2aa1000e-88b8-11ee-8501-0800200c9a66",
#"2aa1000f-88b8-11ee-8501-0800200c9a66"
)

first_run = True
background = []
values_to_save = []
matthew_position_to_save = []
count = 0
results_to_save = []

class DeviceNotFoundError(Exception):
    pass

def save_to_csv(filename,data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['x_s', 'y_s', 'z_s'])
        for i in range(len(data)):
            writer.writerow(data[i])
        #print(data)    
    print("Values saved to sensor_data.csv")

"""
plt.ion()  # Turn on interactive mode
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
"""

# Initialize empty arrays for data storage
x_data, y_data, z_data = [], [], []



async def run_ble_client(args: argparse.Namespace, queue: asyncio.Queue):
    logger.info("starting scan...")

    if args.address:
        logger.info("searching for device with address '%s'", args.address)
        device = await BleakScanner.find_device_by_address(
            args.address, cb=dict(use_bdaddr=args.macos_use_bdaddr)
        )
        if device is None:
            logger.error("could not find device with address '%s'", args.address)
            raise DeviceNotFoundError
    else:
        device = await BleakScanner.find_device_by_name(
            args.name, cb=dict(use_bdaddr=args.macos_use_bdaddr)
        )
        if device is None:
            logger.error("could not find device with name '%s'", args.name)
            raise DeviceNotFoundError
    
    logger.info("connecting to device...")

    async def callback_handler(_, d):
        #print("Callback called.")
        global first_run
        global background

        while not queue.empty():
            queue.get_nowait()

        #logger.info("received data...")
        values = []
        for uuid in ALL_SENSOR_CHAR_UUIDS:
            data = await client.read_gatt_char(uuid)
            #print (data.hex())
            n = int(len(data)/2)   
            # Unpack data to flat list
            v = struct.unpack("<"+str(n)+"h", data)
            if not first_run:
                v = list(map(sub, v, background))
            if first_run:
                background = v
                first_run = False
            # Rearrange into list of 3-value lists
            vl = [v[i:i+3] for i in range(0, len(v), 3)]
            for item in vl:
                x,y,z = item
                x_s = float(x)/6842*100#*100
                y_s = float(y)/6842*100#*100
                z_s = float(z)/6842*100#*100
                values.append((x_s,y_s,z_s))
            
            #print("vvvvvvvvvvvvvvvvvalue: ", values) # raw data [(1,2,3),(4,5,6),...,(46,47,48)]

            # Save values when space is pressed
            if keyboard.is_pressed('space'):
                global values_to_save
                print("Space pressed! Saving values to CSV...")
                values_to_save.append(values)
                #print(values_to_save)
                #save_to_csv('data_acquisition.csv',values_to_save)
                print('value list length beforeeeeeeeeeeeeee: ',len(values_to_save))
                if len(values_to_save) == 5:
                    save_to_csv('data_acquisition.csv',values_to_save)
                    print('5 times this point alreadyyyyyyyyyyyyy!!!!!!')
                    values_to_save = []
                    print('value list length afterrrrrrrrrrrrrrr: ',len(values_to_save))
                    time.sleep(0.5)


                #values.append(item)
            #print ("data received:"+data.hex())
            #print("length: "+str(len(data)))
            #x,y,z = struct.unpack("<hhh", data)
            #values.append((x,y,z))
           
        #for val in values:
        #    x,y,z = struct.unpack("<hhh", val)
        #    q.put_nowait((time.time(), (x,y,z)))
        #logger.info("Data: %r", data)
        #print(values) 
        
        await queue.put((time.time(), values)) #* 1e-2)) ### e-3 Adjusted for the *1000 from esp32, then times 10, it shd be in microTesla
        #await queue.put((time.time(), d))
    
    async with BleakClient(device) as client:
        logger.info("connected")
        #await client.start_notify(args.characteristic, callback_handler)
        # invoke callback_handler when receiving notify on first sensor characteristic
        characteristic_sensor1 = ALL_SENSOR_CHAR_UUIDS[0]
        #print (characteristic_sensor1)
        await client.start_notify(characteristic_sensor1, callback_handler)
        await asyncio.sleep(1.0)
        #paired = await client.pair(protection_level=2)
        #print(f"Paired: {paired}")
         
        while True:
            try:
                for uuid in ALL_SENSOR_CHAR_UUIDS:
                    print (uuid)
                    data = await client.read_gatt_char(uuid)
                    print (data)
                await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                break
                # data will be empty on EOF (e.g. CTRL+D on *nix)
                #if not data:
                #    break

        await client.stop_notify(characteristic_sensor1)

    # Send an "exit command to the consumer"
    #await queue.put((time.time(), None))
    await queue.put((time.time(), None))

    logger.info("disconnected")
   

async def run_queue_consumer(queue: asyncio.Queue):
    model = load_model('z_39_to_79_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()})
    #model = load_model('5points_z_84_to_144_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()}) ## bad, large vibrating
    #model = load_model('z_39_to_79_plus_109_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()}) ## good from 39 to 79, above 79 bad in 3 directions, even can't detect 109
    # Load the scaler
    scaler = joblib.load('scaler_z3979.joblib')
    #scaler = joblib.load('scaler_84_144.joblib')
    #scaler = joblib.load('scaler_39_79_109.joblib')

    logger.info("Starting queue consumer")
    
    global x_data, y_data, z_data

    grid_canvas = create_grid_canvas_reg()
    
###Get static background noise
    epoch, data = await queue.get()
    
###Countdown
    countdown=10
    for i in range(countdown):
        print("Localization start in ",countdown-i)
        time.sleep(0.5)

###Initialization for rolling median
    data_queue = deque([])
    for i in range(3):
        epoch,data = await queue.get()
        data_queue.append(data)
        time.sleep(0.1)
    
    #print(f"Fininished initializing data_queue, it is: {data_queue}")
    while True:
        epoch, data = await queue.get()
        #print("dataaaaaaaaaaaaaa: ", data) # raw data [(1,2,3),(4,5,6),...,(46,47,48)]
        flattened_data = [item for sublist in data for item in sublist] # 1d [1,2,...,48]
        #print("data ffffffffff: ", flattened_data)
        flattened_data_2d = [flattened_data] # 2d [[1,2,...,48]]
        #print("data flattened 22222222222222d: ", flattened_data_2d)
        X = np.array(flattened_data_2d) # turn to numpy array for model.predict() # [[1 2 ... 48]]
        #print("X np arrayyyyyyyyyyy: ", X)
        X = scaler.transform(X)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXX: ",X)
        result = model.predict(X) # [[1 2 3]]
        result = result[0] # [1 2 3]
        #print(result)
    
        print(f"x: {result[0]}, y: {result[1]}, z: {result[2]}") # already in mm 


        # Save values when space is pressed
        if keyboard.is_pressed('s'):
            global results_to_save
            print("Space pressed! Saving values to CSV...")
            plot_3d(result)
            #results_to_save.append(result)
            #print(values_to_save)
            #save_to_csv('data_acquisition.csv',values_to_save)
            #print('result list length beforeeeeeeeeeeeeee: ',len(results_to_save))
            #if len(results_to_save) == 5:
            #    save_to_csv('real_time_results_record.csv',results_to_save)
            #    print('5 times this point alreadyyyyyyyyyyyyy!!!!!!')
            #    results_to_save = []
            #    print('result list length afterrrrrrrrrrrrrrr: ',len(results_to_save))
            #    time.sleep(0.5)
        
        # Append new data to arrays
        x_data.append(result[0])
        y_data.append(result[1])
        z_data.append(result[2])
        """
        # Clear previous plot
        ax.cla()
        
        # Plot new data
        ax.scatter(x_data, y_data, z_data, c='b', marker='o')
        
        # Update labels if needed
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        # Adjust plot limits if needed
        ax.set_xlim([min(x_data), max(x_data)])
        ax.set_ylim([min(y_data), max(y_data)])
        ax.set_zlim([min(z_data), max(z_data)])
        
        # Draw plot
        plt.draw()
        plt.pause(0.001)  # Pause for a short time to allow plot to update
        """
        #plot_cv2grey(data)
        plot_cv2localization_30_reg(result[0:3],grid_canvas) 
        #plot_3d(results_to_save)
        #plot3d()
        #time.sleep(1)

def plot3d():
    ax = plt.axes(projection='3d')
    x = []
    y = []
    z = []
    #print(result)
    def animate(i):
        x.append(x_data)
        y.append(y_data)
        z.append(z_data)
        #print(x, y, z)
        #x = result[0]
        #y = result[1]
        #z = result[2]
        plt.cla()
        #ax.plot3D(x, y, z, 'red')
        ax.scatter3D(x, y, z, c='b', s=5)  # 's' specifies the size of the points
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim([-17, 105])
        ax.set_ylim([-7, 115])

    ani = FuncAnimation(plt.gcf(), animate, interval=500)

    plt.tight_layout()
    plt.show()

async def main(args: argparse.Namespace):
    queue = asyncio.Queue(maxsize=100)
    client_task = run_ble_client(args, queue)
    consumer_task = run_queue_consumer(queue)
    #plot_task = plot_3d()
    print("main is called.")

    try:
        #await client_task 
        await asyncio.gather(client_task, consumer_task)
    except DeviceNotFoundError:
        pass
    logger.info("Main method done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    device_group = parser.add_mutually_exclusive_group(required=False)

    device_group.add_argument(
        "--name",
        metavar="<name>",
        help="the name of the bluetooth device to connect to",
    )
    device_group.add_argument(
        "--address",
        default="68:B6:B3:3E:45:2E",
        metavar="<address>",
        help="the address of the bluetooth device to connect to",
    )

    parser.add_argument(
        "--macos-use-bdaddr",
        action="store_true",
        help="when true use Bluetooth address instead of UUID on macOS",
    )

    #parser.add_argument(
    #    "characteristic",
    #    metavar="<notify uuid>",
    #    help="UUID of a characteristic that supports notifications",
    #)

    parser.add_argument(
        "-d",
        "--debug",  
        action="store_true",
        help="sets the logging level to debug",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)-15s %(name)-8s %(levelname)s: %(message)s",
    )

    #plot_thread = threading.Thread(target=start_plot)
    #plot_thread.start()
    asyncio.run(main(args))
    
    """
    # Set up the plot and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update_plot, interval=100)

    asyncio.run(main(args))
    plt.show()
    """
