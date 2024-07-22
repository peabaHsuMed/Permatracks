import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .config import *
import serial
import re


def localization_plot(result):
    #plt.scatter(result[0], result[1])
    #plt.close('all')
    canvas_width = 500
    canvas_height = 500
    fig1, ax = plt.subplots(figsize=(canvas_width / 100, canvas_height / 100))
    fig1.canvas.manager.window.move(1250,800)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
    
    
    
    # Optional: Add labels or titles
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("150x150 of x: "+ str(int(result[0]*1000)) + " y: " + str(int(result[1]*1000)) + " z: " + str(int(result[2]*1000)))
    plt.ion()
    #fig1.show()
    return fig1, ax

def update_localization_plot(fig1,ax,result):
    # Plot the point
    #ax.set_xlim(0, 150)
    #ax.set_ylim(0, 150)
    #plt.plot()  # 'ro' means red circle marker
    #plt.figure(fig1)
    ax.scatter(result[0]*1000, result[1]*1000)
    ax.set_title("150x150 of x: "+ str(int(result[0]*1000)) + " y: " + str(int(result[1]*1000)) + " z: " + str(int(result[2]*1000)))
    #.show()
    plt.pause(0.01)
    return

def localization_3D_plot(result):
    
    canvas_width = 500
    canvas_height = 500
    #fig1, ax = plt.subplots(figsize=(canvas_width / 100, canvas_height / 100))
    fig1 = plt.figure(figsize=(canvas_width / 100, canvas_height / 100))
    ax = fig1.add_subplot(projection='3d')

    fig1.canvas.manager.window.move(1250,800)
    ax.set_xlim(-50, 200)
    ax.set_ylim(-50, 200)
    ax.set_ylim(-50, 200)
    ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
    
    
    
    # Optional: Add labels or titles
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("150x150 of x: "+ str(int(result[0]*1000)) + " y: " + str(int(result[1]*1000)) + " z: " + str(int(result[2]*1000)))
    plt.ion()
    #fig1.show()
    return fig1, ax

def update_localization_3D_plot(fig1,ax,result):
    # Plot the point
    #ax.set_xlim(0, 150)
    #ax.set_ylim(0, 150)
    #plt.plot()  # 'ro' means red circle marker
    #plt.figure(fig1)
    ax.scatter(result[0], result[1], result[2])
    ax.set_title("150x150 of x: "+ str(int(result[0]*1000)) + " y: " + str(int(result[1]*1000)) + " z: " + str(int(result[2]*1000)))
    #.show()
    plt.pause(0.01)
    return

def raw_axis_plot(data):
    
    #plt.close('all')
    fig2, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30, 7))
    #fig2.canvas.manager.window.move(100,100)
    cm = plt.colormaps['plasma']
    x = np.array([0,40,80,120])
    y = np.array([0,40,80,120])
    X, Y = np.meshgrid(x,y)
    vmin = -400
    vmax = 400
    
    
    limit_min = 0
    limit_max = 120
    ax1.set_xlim(limit_min, limit_max)
    ax1.set_ylim(limit_min, limit_max)
    ax1.set_aspect('equal')  # Ensure equal scaling for x and y axes
    ax1.set_title("x-axis result")
    cf1=ax1.pcolormesh(X,Y,np.reshape(data[:,0],[4,4]),cmap=cm,vmin=vmin, vmax=vmax)
    #cbar1 = fig2.colorbar(cf1,ax=ax1)
    
    ax2.set_xlim(limit_min, limit_max)
    ax2.set_ylim(limit_min, limit_max)
    ax2.set_aspect('equal')  # Ensure equal scaling for x and y axes
    ax2.set_title("y-axis result")
    cf2=ax2.pcolormesh(X,Y,np.reshape(data[:,1],[4,4]),cmap=cm,vmin=vmin, vmax=vmax)
    #cbar2 = fig2.colorbar(cf2,ax=ax2)

    ax3.set_xlim(limit_min, limit_max)
    ax3.set_ylim(limit_min, limit_max)
    ax3.set_aspect('equal')  # Ensure equal scaling for x and y axes
    ax3.set_title("z-axis result")
    cf3=ax3.pcolormesh(X,Y,np.reshape(data[:,2],[4,4]),cmap=cm,vmin=vmin, vmax=vmax)
    #cbar3 = fig2.colorbar(cf3,ax=ax3)
    fig2.colorbar(cf3,ax=ax3)


    # Optional: Add labels or titles
    plt.ion()
    #fig2.show()
    return fig2, (ax1,ax2,ax3), (cf1,cf2,cf3)

def update_raw_axis(fig2,ax1,ax2,ax3,cbar1,cbar2,cbar3,data):
    #plt.figure(fig2)
    x = np.array([0,40,80,120])
    y = np.array([0,40,80,120])
    X, Y = np.meshgrid(x,y)
    cm = plt.colormaps['plasma']
    vmin = -400
    vmax = 400
    
    cf1 = ax1.pcolormesh(X,Y,np.reshape(data[:,0],[4,4]),cmap=cm,vmin=vmin, vmax=vmax)
    ax2.pcolormesh(X,Y,np.reshape(data[:,1],[4,4]),cmap=cm,vmin=vmin, vmax=vmax)
    ax3.pcolormesh(X,Y,np.reshape(data[:,2],[4,4]),cmap=cm,vmin=vmin, vmax=vmax)
    '''
    for i in range(4):
        for j in range(4):
            ax1.text(x[j], y[i], f'{data[i * 4 + j, 0]:.2f}', color='black', ha='center', va='center')
            ax2.text(x[j], y[i], f'{data[i * 4 + j, 1]:.2f}', color='black', ha='center', va='center')
            ax3.text(x[j], y[i], f'{data[i * 4 + j, 2]:.2f}', color='black', ha='center', va='center')
    '''
    #cbar1.update_normal(cf1)
    #plt.show()
    #fig.colorbar(cf,ax=ax3)
    plt.pause(0.01)
    return

def plot_2axis_cal_check(title,data):
    '''
    Plotting data from only one sensor, shape: (N_data,3)
    

    '''
    fig2, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30, 7))
    
    #n_sensor = 3
    ax1.scatter(data[:,0],data[:,1])
    ax1.set_title("xy")
    
    if title == "Calibrated":
        ax1.set_xlim(-2, 2) 
        ax1.set_ylim(-2, 2)
    ax2.scatter(data[:,1],data[:,2])
    ax2.set_title("yz")
    if title == "Calibrated":
        ax2.set_xlim(-2, 2) 
        ax2.set_ylim(-2, 2)
    ax3.scatter(data[:,0],data[:,2])
    ax3.set_title("xz")
    if title == "Calibrated":
        ax3.set_xlim(-2, 2) 
        ax3.set_ylim(-2, 2)
    fig2.suptitle(f"{title}", fontsize=16)
    
    #plt.ion()
    plt.show()
    return fig2,(ax1,ax2,ax3)

def update_plot_2axis_cal_check(fig2,ax1,ax2,ax3,data):
    #fig2, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30, 7))
    
    n_sensor = 3
    ax1.scatter(data[n_sensor,0],data[n_sensor,1])
    ax1.set_title("xy")
    ax2.scatter(data[n_sensor,1],data[n_sensor,2])
    ax2.set_title("yz")
    ax3.scatter(data[n_sensor,0],data[n_sensor,2])
    ax3.set_title("xz")
    
    #plt.ion()
    plt.show()
    return  

def plot_3D_scatter(data):
    fig1, ax1 = plt.subplots(1,1,figsize=(20, 20), subplot_kw={"projection": "3d"})    
    n_sensor = 3
    ax1.scatter(data[n_sensor,0], data[n_sensor,1], data[n_sensor,2])
    ax1.set_title("xyz")
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    #ax1.set_zlabel('Z Label')
    
    plt.ion()
    plt.show()
    return fig1, ax1

def update_plot_3D_scatter(fig1, ax1, data):    
    n_sensor = 3
    ax1.scatter(data[n_sensor,0], data[n_sensor,1], data[n_sensor,2])
    ax1.set_title("xyz")
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    #ax1.set_zlabel('Z Label')
    
    #plt.ion()
    #plt.show()
    return