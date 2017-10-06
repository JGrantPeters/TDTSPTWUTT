#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:53:25 2017

@author: user
"""

import DTDTSPSTW
import LTDTSPSTW


import numpy as np

#%%
reload(DTDTSPSTW)
reload(LTDTSPSTW)


#%%

servicetime = 5    
toy_nnodes = 10;

#number of linear time windows
nltw = 5;

#number of discrete time windows
ndtw = 30

#hour is 60 minutes. 
hour = 60;

#generate delivery windows for each customer.
toy_customer_bounds = np.random.uniform(low=0,high=8*hour, size=toy_nnodes-1)


toy_TimeWindows = np.empty([toy_nnodes, 2])
toy_TimeWindows[0] = np.array([0, 8.5*hour])
toy_TimeWindows[1:,0] = toy_customer_bounds;
toy_TimeWindows[1:, 1] = toy_customer_bounds+hour;

#randomly generate the vertices of the linear travel times tensor
linear_toy_travel_times = np.array([[[0 if i==j else np.random.uniform(low = 5, high = hour) for k in range(nltw+1)] for i in range(toy_nnodes) ]for j in range(toy_nnodes)] )

#randomly generate the piecewise constant discrete travel times tensor
discrete_toy_travel_times = np.array([[[float(0) for k in range(ndtw+1)] for i in range(toy_nnodes)] for j in range(toy_nnodes)])

#force the discrete travel times to match the linear travel times to an extent
discrete_toy_travel_times[:, :, 0] = linear_toy_travel_times[:, :, 0]
discrete_toy_travel_times[:, :, -1] = linear_toy_travel_times[:, :, -1]

#Define the duration of the working day
LTW = np.linspace(0, 9.5*hour, nltw+1)
DTW = np.linspace(0, 9.5*hour, ndtw+1)

#again, force teh discrete travel times to match the linear travel times
for i in range(1,ndtw):
   interval = None
   for k in range(nltw):
        if LTW[k] <= DTW[i] and DTW[i] < LTW[k+1]:
            interval = k
    
    #Now we know the interval i lies in.
   dt = DTW[i] - LTW[interval]
   b = linear_toy_travel_times[:, :, interval]
   slope = np.divide(linear_toy_travel_times[:, :, interval+1] - linear_toy_travel_times[:, :, interval] , 1.0*(LTW[interval+1]- LTW[interval]) )
    
   discrete_toy_travel_times[:, :, i] = slope*dt + b;
    
    
#the penalty parameters. regret >= 5 * lateness
#                        regret >0 100*lateness -950
# This results in a double hockeystick regret function
toy_pen_params = [5, 100, -950]

#%%
#construct both problems
linear = LTDTSPSTW.LTDTSPSTW(toy_nnodes,toy_TimeWindows, linear_toy_travel_times, LTW, servicetime, toy_pen_params)
discrete = DTDTSPSTW.DTDTSPSTW(toy_nnodes, toy_TimeWindows, discrete_toy_travel_times, DTW, servicetime, toy_pen_params)
#%%
a=linear.solve()

linear.summary()
#%%
discrete.solve()
discrete.summary()