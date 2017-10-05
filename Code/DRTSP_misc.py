#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:07:35 2017

@author: Jonathan Grant-Peters
"""


import numpy as np

def generate(ncustomers = 16, nchangetimes = 4):
    hour = 60
    TW = np.array([hour*i for i in range(nchangetimes+1)])
    
    inds = [i for i in range(1,ncustomers+1)]
    np.random.shuffle(inds)
    DW = np.empty([ncustomers+1, 2])
    
    const = (nchangetimes)/(ncustomers+1.0)
    for i in range(ncustomers):
        DW[inds[i]] = np.array([TW[int((i+1)*const)], TW[int((i+1)*const)+1]])
    DW[0] = [0, TW[-1]+hour/2]
    
    tt = np.array([[0 if i==j else np.random.uniform(low = 5, high = 10) for i in range(ncustomers+1) ]for j in range(ncustomers+1)] )
    
    scale1 = np.random.uniform(1, 2, [ncustomers+1, ncustomers+1])
    scale2 = np.random.uniform(1, 2, [ncustomers+1, ncustomers+1])

    #tt2 = np.array([[[tt[i,j,k]*np.random.uniform(1,2) for k in range(nchangetimes+1)] for i in range(ncustomers+1)] for j in range(ncustomers+1)]) 
    #tt3 = np.array([[[tt2[i,j,k]*np.random.uniform(1,2) for k in range(nchangetimes+1)] for i in range(ncustomers+1)] for j in range(ncustomers+1)]) 

    tt2 = np.multiply(tt, scale1)
    tt3 = np.multiply(tt2, scale2);
    
    utt = np.array([[[tt[i,j], tt2[i,j], tt3[i,j]] for i in range(ncustomers+1)] for j in range(ncustomers+1)])
    
    return TW, DW, utt

class subRoute(object):
    def __init__(self, start_node, end_node, time_taken, route, regret):
        self.end = end_node
        self.start = start_node
        self.time = time_taken
        self.route = route
        self.regret = regret
