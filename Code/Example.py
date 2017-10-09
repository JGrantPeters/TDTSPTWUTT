#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:50:59 2017

@author: Jonathan Grant-Peters
"""

import numpy as np
import datetime


import DRTSP

reload(DRTSP)

#Read from the precomputed distance matrix
big_mat_utt = np.load('big_mat.npy')
#Read from the corresponding list of postcodes
postcodes = np.load('codes.npy')


nnodes = 17
#choose which indices from the postcodes to use
inds = np.random.choice(len(postcodes), nnodes)
codes = postcodes[inds]

nbins = 4;
#Divide the customers across 4 hours. 
day = datetime.datetime(2017, 9, 29)
hours = np.zeros([nnodes], dtype = int)
inds = np.random.permutation(nnodes)

#assign each customer to an hour. The index for which hours has the value of -1 
#refers to the random node chosen to be the depot
for i in range(1,nnodes):
    hours[inds[i]] = int((i-1)/(nnodes/nbins))
hours[inds[0]]= -1

#select the submatrix
utt = np.array([[big_mat_utt[i, j, :] for i in inds] for j in inds])

#increase the travel times, so as to focus on situations with a unique solution.
utt = np.multiply(utt, 1.3)

    #%%
ex = DRTSP.DRTSP(codes, hours, day,utt)
ex.solve()
#s4 = ex.SC[4]
#s5 = ex.SC[5]
#s10 = ex.SC[10]



#for s in ex.SC:
#    print(ex.solvePrecient(s))


'''
for s in range(11):
    #print(ex.SC[s])
    print('--------------------')
    print('New Scenario')    
    print('--------------------')

    s1 = ex.SC[s]
    a1 = ex.solvePrecient(s1)
    regret = a1[0]
    route = a1[1]
    regret2 = DRTSP.Regret(nnodes, route, utt, s1, hours, 5)
    
    

    
    #print('FLAG')
    #print(arrivals)
    print(regret, regret2, regret-regret2)
    #print(a1[1])
    #print('FLAG')
    #print([ex.reg[i] for i in a1[1]])
'''
    
#route = [ 6, 15,  2 ,16,  5]





ex.solve()

for s in range(ex.nSC):
    print('-------------------')
    print('Scenario')
    print('-------------------')
    
    base = ex.solvePrecient(ex.SC[s])
    base_reg = base[0]
    print(base[1])
    for scen in ex.SC:
        sol = ex.solvePrecient(scen)
        #scen_regret = sol[0];
        scen_regret = DRTSP.Regret(nnodes, sol[1], utt, ex.SC[s], hours, 5)
        print(scen_regret/base_reg)
        
    print('+++++++++++++++++++')
    print('Recourse')
    rec = ex.path[s]
    print(rec)
    Rrec = DRTSP.Regret(ex.n, rec, utt, ex.SC[s], hours, 5)
    print(Rrec/base_reg)
    print('+++++++++++++++++++')


#a10 = ex.solvePrecient(s10)

#a4 = ex.solvePrecient(s4)
#a5 = ex.solvePrecient(s5)

#print(Regret(17,route, utt, ex.SC[0], hours, 5))
#print(a0[0])
#%%
print('The 11 routes corresponding to the precient solutions')
for i in ex.pRoutes:
    print([j for j in i])
    
print('-------------------------------------------------------------')
print('-------------------------------------------------------------')
print('The 11 routes corresponding to the recourse solutions. Note that differ on in correspondance to the shape of the scenario branching tree.')

for i in ex.path:
    print(i)


