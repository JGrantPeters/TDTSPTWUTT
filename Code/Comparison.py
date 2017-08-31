#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:00:19 2017

@author: user
"""

import TSPTW
import TSPTW2



import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import cplex
import scipy.sparse as ss
import sets
import time

#%%
reload(TSPTW)
reload(TSPTW2)

#%%

toy_nnodes = 10;

hour = 60;

toy_customer_bounds = np.random.uniform(low=0,high=8*hour, size=toy_nnodes-1)

toy_TimeWindows = np.empty([toy_nnodes, 2])
toy_TimeWindows[0] = np.array([0, 8.5*hour])
toy_TimeWindows[1:,0] = toy_customer_bounds;
toy_TimeWindows[1:, 1] = toy_customer_bounds+hour;

toy_travel_times_min = np.array([[100000 if i==j else np.random.uniform(low = 5, high = hour) for i in range(toy_nnodes) ]for j in range(toy_nnodes)])

toy_travel_times_max = np.array([[toy_travel_times_min[i,j] + np.random.gamma(shape = hour/4, scale = np.random.uniform(low=0.5, high =2 )) if i != j else 0 for i in range(toy_nnodes)] for j in range(toy_nnodes) ])

#%%
method1 = TSPTW.TSPTW(toy_nnodes, toy_TimeWindows, toy_travel_times_min)
method2 = TSPTW2.TSPTW2(toy_nnodes, toy_TimeWindows, toy_travel_times_min)

formulation1 = method1.formulate()
formulation1.write('form1.lp')
formulation2 =method2.formulate()
formulation2.write('form2.lp')

#%%
tic = time.time();
formulation1.solve()
toc = time.time()
t1 = toc-tic

#%%
tic = time.time()
formulation2.solve()
toc = time.time()
t2 = toc-tic


print(t1, t2, t2/t1)
#%%
opt1 = formulation1.solution.get_objective_value()
x1 = formulation1.solution.get_values()
args1 = [int(i) for i in x1[:method1.nedges]]

print(opt1)
print('\n')
print(x1)

for i in range(method1.nedges):
    if args1[i]:
        print(method1.edges[i])


#%%

opt2 = formulation2.solution.get_objective_value()
x2 = formulation2.solution.get_values()
args2 = [int(i) for i in x2[:method2.nedges]]


print(opt2)

print('\n')
print(x2)


for i in range(method2.nedges):
    if args2[i]:
        print(method2.edges[i])