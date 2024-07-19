#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:14:12 2024

@author: usuario
"""



import numpy as np
import os.path
import pyvista as pv
from matplotlib import pyplot as plt


def Droplet_RaioVariável_quina_P001(I, J, K, angulo, raio_int):
    if angulo != np.pi or angulo != 0:
        raio = raio_int/np.sin(angulo)
    else:
        raio_int = 0
    D=np.ones((I,J,K),dtype="uint8")
    a: int = 12
    cx = I/2
    cy = J/2
    cz = a + 0.5 + raio*np.cos(angulo)
    if angulo != np.pi:            
        for i in range(0,I):
            for j in range (0, J):
                for k in range (0, K):
                    dist = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz))
                    dist_int = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy))
                    if (dist <= raio) or (k<=a and dist_int <= raio_int):
                        D[i,j,k] = 2
                    if k <= a and dist_int >= raio_int:
                        D[i,j,k] = 0
    else:
        for i in range(0,I):
            for j in range (0, J):
                for k in range (0, K):
                    dist_int = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy))
                    if (k<=a and dist_int <= raio_int):
                        D[i,j,k] = 2
                    if k <= a and dist_int >= raio_int:
                        D[i,j,k] = 0        
    return D;

def Droplet_RaioFixo_quina_P001(I, J, K, angulo, raio):
    raio_int = raio*np.sin(angulo)
    D=np.ones((I,J,K),dtype="uint8")
    a: int = K/5
    cx = I/2
    cy = J/2
    cz = a + 0.5 + raio*np.cos(angulo)
    for i in range(0,I):
        for j in range (0, J):
            for k in range (0, K):
                dist = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz))
                dist_int = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy))
                if (dist <= raio) or (k<=a and dist_int <= raio_int):
                    D[i,j,k] = 2
                if k <= a and dist_int >= raio_int:
                    D[i,j,k] = 0
    return D;


def Tubo_RaioVariável_quina_P001(I, J, K, angulo, raio_int, b):
    try:
        raio = abs(raio_int/np.cos(angulo))
    except:
        raio = 1000000
    D=np.ones((I,J,K),dtype="uint8")
    a: int = K/2
    cx = I/2
    cy = J/2
    if angulo == np.pi/2:
        for i in range(0,I):
            for j in range (0, J):
                for k in range (0, K):
                    dist_int = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy))
                    if k <= a-b:
                        D[i,j,k] = 2
                    if k <= a and dist_int >= raio_int:
                        D[i,j,k] = 0
    elif angulo < np.pi/2:        
        cz = a - 0.5 - b - raio*np.sin(angulo)
        for i in range(0,I):
            for j in range (0, J):
                for k in range (0, K):
                    dist = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz))
                    dist_int = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy))
                    if (dist <= raio) or (k<=a-b  and dist_int <= raio_int):
                        D[i,j,k] = 2
                    if k <= a and dist_int >= raio_int:
                        D[i,j,k] = 0
    else:
        cz = a - 0.5 - b + raio*np.sin(angulo)
        for i in range(0,I):
            for j in range (0, J):
                for k in range (0, K):
                    dist = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz))
                    dist_int = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy))
                    if (k<=a-b and dist_int <= raio_int):
                        D[i,j,k] = 2
                    if (dist <= raio):
                        D[i,j,k] = 1
                    if k <= a and dist_int >= raio_int:
                        D[i,j,k] = 0
    return D;


def Droplet_bastão(I, J, K, angulo, raio):
    raio_int = raio*np.sin(angulo)
    D=np.ones((I,J,K),dtype="uint8")
    a: int = 12
    cx = I/2
    cy = J/2
    cz = a + 0.5 + raio*np.cos(angulo)
    for i in range(0,I):
        for j in range (0, J):
            for k in range (0, K):
                dist = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy) + (k-cz)*(k-cz))
                dist_int = np.sqrt((i-cx)*(i-cx) + (j-cy)*(j-cy))
                if (dist < raio) and k>a:
                    D[i,j,k] = 2
                if k <= a and dist_int <= raio_int:
                    D[i,j,k] = 0
    return D;


def SaveImage(P, I, z, Filename):
    grid = pv.ImageData()
    grid.dimensions = (I+1),(I+1),(I+1)
    grid.origin = (0, 0, 0)  # Adjust the origin if needed
    grid.spacing = (1, 1, 1)  # Adjust the spacing if needed
    grid.cell_data["P"] = P.ravel(order="F")  
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid.outline(), color="k")
    cmap = plt.cm.get_cmap("viridis", 3)
    slices = grid.slice_orthogonal(x = (I+1)/2, y = (J+1)/2, z = z)
    plotter.add_mesh(slices, cmap=cmap)
    plotter.remove_scalar_bar()
    plotter.view_vector([-0.5,2,1])
    plotter.show(screenshot=f'{Filename}')
    return

#Radius = [7]
#Size   = [38]
#Types  = [3]
#Corner = [0]

Radius = [  7,  14,  28,  56]
Size   = [ 38,  52,  80, 136]
Types  = [  0,   1,   2,   3]
Corner = [  0]
Angle  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
directory = '/home/usuario/ENCIT_2024/Benchmark'
z = 8
for i in range(len(Radius)):
    radius = Radius[i]
    I = J = K = Size[i]
    for types in Types:
        for angle in Angle:
            angle_rad = angle*np.pi/180
            if types == 0:
                P = Droplet_RaioFixo_quina_P001(I, J, K, angle*np.pi/180, radius)
                if not os.path.exists(f'{directory}/T{types}R{radius}x{I}'):
                    os.mkdir(f'{directory}/T{types}R{radius}x{I}')
                filename = f'{directory}/T{types}R{radius}x{I}/T{types}_R{radius}_A{angle}_x{I}.raw'
#                SaveImage(P, I, z, filename)
                P.tofile(f'{filename}')
            if types == 1:
                P = Droplet_RaioVariável_quina_P001(I, J, K, angle*np.pi/180, radius)
                if not os.path.exists(f'{directory}/T{types}R{radius}x{I}'):
                    os.mkdir(f'{directory}/T{types}R{radius}x{I}')
                filename = f'{directory}/T{types}R{radius}x{I}/T{types}_R{radius}_A{angle}_x{I}.raw'
#                SaveImage(P, I, z, filename)
                P.tofile(f'{filename}')
            if types == 2:
                for corner in Corner:
                    P = Tubo_RaioVariável_quina_P001(I, J, K, angle*np.pi/180, radius, corner)
                    if not os.path.exists(f'{directory}/T{types}R{radius}x{I}'):
                        os.mkdir(f'{directory}/T{types}R{radius}x{I}')
                    filename = f'{directory}/T{types}R{radius}x{I}/T{types}_R{radius}_A{angle}_x{I}.raw'
#                    SaveImage(P, I, z, filename)
                P.tofile(f'{filename}')
            if types == 3:
                P = Droplet_bastão(I, J, K, angle*np.pi/180, radius)
                if not os.path.exists(f'{directory}/T{types}R{radius}x{I}'):
                    os.mkdir(f'{directory}/T{types}R{radius}x{I}')
                filename = f'{directory}/T{types}R{radius}x{I}/T{types}_R{radius}_A{angle}_x{I}.raw'
#                SaveImage(P, I, z, filename)
                P.tofile(f'{filename}')



