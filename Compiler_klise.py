#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:12:49 2024

@author: usuario
"""

#Klise code for contact angle measurement
import numpy as np
import Library_Klise as K
from argparse import ArgumentParser
import os.path

Radius = [  7,  14,  28,  56]
Size   = [ 38,  52,  80, 136]
Types  = [  0,   1,   2,   3]
Corner = [  0]
Angle  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
Directory = '/home/usuario/ENCIT_2024/Benchmark'
Directory_out = '/home/usuario/ENCIT_2024/Medições/Klise'

for i in range(len(Radius)):
    radius = Radius[i]
    size = Size[i]
    for types in Types:
        for angle in Angle:
            input_file = f'{Directory}/T{types}R{radius}x{size}/T{types}_R{radius}_A{angle}_x{size}.raw'
            if not os.path.exists(f'{Directory_out}/T{types}R{radius}x{size}'):
                os.mkdir(f'{Directory_out}/T{types}R{radius}x{size}')
            output_file_name = f'Klise_{types}_R{radius}_A{angle}_x{size}'
            npimg = np.fromfile(input_file, dtype=np.uint8)
            imageSize = (size, size, size)
            npimg = npimg.reshape(imageSize)
            #Result in radians
            theta, angles_position = K.Klise(npimg, d = 5)
            #Convert to degree
            theta = theta*180/np.pi
            np.save(f"{Directory_out}/T{types}R{radius}x{size}/{output_file_name}_Angle", theta)
            np.save(f"{Directory_out}/T{types}R{radius}x{size}/{output_file_name}_Position", angles_position)


            if not os.path.exists(f'{Directory_out}/T{types}R{radius}x{size}_inv'):
                os.mkdir(f'{Directory_out}/T{types}R{radius}x{size}_inv')
            npimg = np.where(npimg == 1, 2, np.where(npimg == 2, 1, npimg))
            theta, angles_position = K.Klise(npimg, d = 5)
            #Convert to degree
            theta = theta*180/np.pi
            np.save(f"{Directory_out}/T{types}R{radius}x{size}_inv/{output_file_name}_Angle", theta)
            np.save(f"{Directory_out}/T{types}R{radius}x{size}_inv/{output_file_name}_Position", angles_position)


