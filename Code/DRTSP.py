#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:16:00 2017

@author: Jonathan Grant-Peters
"""

import numpy as np
import DataGenerate as dat_misc
import pandas as pd
import itertools
import datetime
import scipy as sp
import random

'''
class Tree(object):
    def __init__(self, children, data):
        self.children = children
    def add_data(self, data):
        self.data = data

class ScenarioTree(object):
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.nstages = np.size(scenarios,1)
        self.n = np.size(scenarios,0)
    
    def buildTree(self):
        trees = [Tree([], i) for i in range(self.n)]
        
        
'''
class RobustSolution(object):
    def __init(self, Precient_routes, Precient_regret, Precient_arrivals, robust_route, robust_regret, robust_arrivals):
        self.Precient_routes = Precient_routes
        self.Precient_regret = Precient_regret
        self.Precient_arrivals = Precient_arrivals
        self.robust_route = robust_route
        self.robust_regret = robust_regret
        self.robust_arrivals = robust_arrivals

class SubSolution(object):
    def __init__(self, rel_arrivals, rel_regret, route, delta, inds):
        self.rel_arrival = rel_arrivals
        self.rel_regret = rel_regret
        self.route = route
        self.delta = delta
        self.w = 60 - (self.rel_arrival[inds.index(self.route[-1])])
        self.phi = np.maximum(self.rel_arrival[inds.index(self.route[-2])], -self.w)
        self.indMap= inds
        

def Subproblem(tt, service_time, inds):
    hour = 60;
    #finds the shortest tour from the first node to the last node, travelling through
    #all nodes in between. The first node corresponds to the first row of the travel
    #time matrix, and similarly for the last node.
    nnodes = np.size(tt,0)
    routes = [[0] + [j+1 for j in i] + [nnodes-1] for i in itertools.permutations(range(nnodes-2))]
    
    bestRoute = 0;
    bestTimes = 0;
    bestMaxRegret = np.inf;
    for route in routes:
        travelTimes = Measure(route,tt, service_time);
        MaxRegret = np.maximum(travelTimes[route[-2]], travelTimes[route[-1]] - hour)
        
        if MaxRegret < bestMaxRegret:
            bestMaxRegret = MaxRegret
            bestRoute = route
            bestTimes = travelTimes
    
    delta = bestTimes[bestRoute[-1]]-hour
    
    return SubSolution(bestTimes, bestMaxRegret, [inds[i] for i in bestRoute], delta, inds)
    
    
def Measure(route, tt, service_time):
    arrival_times = np.zeros([len(route)]); 
    for i in range((len(route)-1)):
        #print(arrival_times[i], service_time, tt[i,i+1])
        arrival_times[i+1] = arrival_times[i] + service_time + tt[i,i+1]
    return arrival_times
#%%

class DRTSP(object):
    def __init__(self, Locations, Hours, start_day_time, utt):
        self.n = len(Locations)
        self.Hours = Hours
        self.indDepot = np.where(Hours==-1)[0][0]
        self.indCustomers = np.array([i for i in range(self.n) if Hours[i] !=-1], dtype = int)
        
        self.start_day_time = start_day_time
        self.utt = utt
        self.locs = Locations
        
        self.nsc=3
        #self.DW = np.array([[(Hours[i])*60, (1+Hours[i])*60] for i in self.indCustomers])
        
        #self.CT = np.array([(i)*60 for i in np.unique(Hours[self.indCustomers])])
        
        #self.tt = dat_misc.build_tt_matrix(Locations, [(i if i>-1 else 0) for i in Hours], start_day_time)

        self.service_time  =5
        #self.nCT = len(self.CT) 
        #self.n = np.size(self.tt, 0)
        self.K = np.max(Hours)+1
        self.N = self.getCategories(Hours)
        self.Lambda = np.array([60*i for i in range(self.K)])

        self.buildScenarios()
        #self.cSizes = [len()]
        
        
    def getCategories(self, Hours):
        N = [[i for i in self.indCustomers if Hours[i] == j] for j in range(self.K)]
        return N
    
    def buildScenarios(self):
        self.nSC = int(sp.misc.comb(self.K+1, 2)) + 1
        #print(self.nSC)
        self.SC = np.zeros([self.nSC, self.K], dtype = int)
        self.SC[0,:] = np.array([1,2,2,2])
        self.SC[1,:] = np.array([1,1,2,2])
        self.SC[2,:] = np.array([1,1,1,2])
        self.SC[3,:] = np.array([1,1,1,1])
        self.SC[4,:] = np.array([0,1,2,2])
        self.SC[5,:] = np.array([0,1,1,2])
        self.SC[6,:] = np.array([0,1,1,1])
        self.SC[7,:] = np.array([0,0,1,2])
        self.SC[8,:] = np.array([0,0,1,1])
        self.SC[9,:] = np.array([0,0,0,1])
        self.SC[10,:] = np.array([0,0,0,0])
        
        
        self.tree = [[[i for i in range(11)]], 
                     [[0,1,2,3], [4,5,6,7,8,9,10]],
                     [[0], [1,2,3], [4,5,6],[7,8,9,10]],
                     [[0], [1], [2,3], [4], [5,6], [7,8], [9,10]],
                     [[i] for i in range(11)]]
        
        #self.tree = np.array([  [0,0,0,0,0,0,0,1,1,1,1],
        #                        [0,0,0,0,1,1,1,2,2,2,3],
        #                        [0,0,1,1,2,2,3,4,4,5,6],
        #                        [0,1,2,3,4,5,6,7,8,9,10]]).T
        
        self.nbranches = np.array([2, 4, 7, 11], dtype = int)
        #self.SC[11,:] = np.array([0,0,0,0,0])
        #self.SC[1,:] = np.array([0,0,0,0,0])
        
        #self.SCTree = ScenarioTree(range(self.nSC), [i for i in range(self.nSC) if self.SC[i,0]], [i for i in range(self.nSC) if not self.SC[i,0]])
        
        
    
    def solvePrecient(self, s, param = 0):
        #Store regret values here
        regret = np.zeros([self.n])
        routes = [ None for _ in range(self.n)]
        mymap = np.zeros([self.n])
        
        #treat the first iteration separately
        k = self.K-1
        j0 = self.indDepot
        for i0 in self.N[k]:
            Set = np.array([i for i in self.N[k] if i != i0])

            min_regret = np.inf;
            for sigma in itertools.permutations(Set):
                route = [i0]+[i for i in sigma]+[j0];
                tt = np.array([[utt[ii,jj,s[k]] for ii in route] for jj in route])
                
                arrivals = Measure(route, tt, self.service_time)
                aj0 = arrivals[-1]
                aim = arrivals[-2]
                
                local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + param)
                if local_regret < min_regret:
                    min_regret = local_regret
                    min_sigma = sigma
            #print(i0, min_regret, min_sigma)
            routes[i0] = [i for i in min_sigma]+[self.indDepot]
            regret[i0] = min_regret
            mymap[i0] = j0
        
        #print(routes)
        #Now iterate through the main body
        for k in range(self.K-2, -1,-1):
            for i0 in self.N[k]:
                min_regret = np.inf
                for j0 in self.N[k+1]:
                    Set = np.array([i for i in self.N[k] if i != i0])
                    #print(i0, j0, Set)
                    best_route_regret = np.inf
                    for sigma in itertools.permutations(Set):
                        route = [i0]+[i for i in sigma]+[j0];
                        tt = np.array([[utt[ii,jj,s[k]] for ii in route] for jj in route])
                        
                        arrivals = Measure(route, tt, self.service_time)
                        aj0 = arrivals[-1]
                        aim = arrivals[-2]
                        local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + regret[j0])
                        
                        #print(aim, aj0, local_regret, regret[j0],i0, j0)
                        if local_regret < best_route_regret:
                            best_route_regret = local_regret
                            min_sigma = sigma
                    
                    if best_route_regret < min_regret:
                        min_regret = best_route_regret;
                        jstar = j0
                        sigma_j = min_sigma
                
                #print('FLAG', i0, sigma_j, jstar)
                routes[i0] = [i for i in sigma_j]+[jstar]+routes[jstar]
                regret[i0] = min_regret
                mymap[i0] = jstar
        
        #print(routes)
            
        #Now for the final iteration
        min_regret = np.inf
        i0=  self.indDepot
        sigma_j = 0

        #for i in self.N:
        #    print(regret[i])
        for j0 in self.N[0]:
            #print(j0, Set)
            
            #tt = np.array([[utt[ii,jj,s[k]] for ii in route] for jj in route])
            
            aim = self.service_time
            aj0 = aim + utt[i0,j0,0]
            local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + regret[j0])
            #print(j0, local_regret, regret[j0])    
            if local_regret < min_regret:
                min_regret = local_regret;
                jstar = j0
                mymap[i0] = jstar
        routes[i0] = [i0]+ [jstar]+routes[jstar]
        regret[i0] = min_regret
        
        #print(routes[i0])
        
        route = np.zeros([self.K+1], dtype = int)
        route[0] = self.indDepot
        for k in range(self.K):
            route[k+1] = mymap[route[k]]
        
        #self.reg = regret
        #self.path = routes
        
        return regret[i0], route, routes[self.indDepot]
    
    
    def solve(self):
        param = 30
        
        #precient regret
        #tmp = np.array([[self.solvePrecient(s, param)] for s in self.SC])
        
        #pR = tmp[:,0]
        #pRoute = tmp[:,1]
        
        
        pR = np.zeros([self.nSC])
        pRoute = np.zeros([self.nSC, self.K+1])
        for s in range(self.nSC):
            pR[s], pRoute[s,:], _ = self.solvePrecient(self.SC[s],param)
        
        mymap = np.zeros([self.n, self.nSC])
        routes = [ [None for _ in range(self.nSC)] for _ in range(self.n)]
        regret = np.zeros([self.n, self.nSC])
        
        print(pRoute)
        print(pR)
        
        
        self.alpha = np.zeros([self.n, self.n, self.nSC])
        self.Beta = np.array([[np.inf for _ in range(self.nSC)] for _ in range(self.n)])
        
        #print(pR)
        #treat the first iteration separately
        k = self.K-1
        j0 = self.indDepot
        for Branch in self.tree[k+1]:
            for i0 in self.N[k]:

                Set = np.array([i for i in self.N[k] if i != i0])
                
                
                
                min_regret = np.inf;
                perms = itertools.permutations(Set)
                
                
                for permIndex in range(len(perms)):
                    sigma = perms[permIndex]
                    
                    robustness = 0
                    for s in Branch:
                        sk = self.SC[s,k]
                        route = [i0]+[i for i in sigma]+[j0];

                        tt = np.array([[utt[ii,jj,sk] for ii in route] for jj in route])
                        arrivals = Measure(route, tt, self.service_time)
                        aj0 = arrivals[-1]
                        aim = arrivals[-2]
                        
                        local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + param)
                        
                        if local_regret < self.Beta[i0, s]:
                            self.Beta[i0, s] = local_regret
                        
                        
                        local_robustness = local_regret/pR[s]
                        
                        if robustness<local_robustness:
                            robustness = local_robustness;
                            min_sigma = sigma
                        
                    if robustness < min_regret:
                        min_regret = robustness
                        sigma_s = min_sigma
                
                for s in Branch:
                    routes[i0][s] = sigma_s
                    self.alpha[i0, j0, s] = min_regret
                    mymap[i0,s] = j0
                    regret[i0,s] 
        #for i in self.alpha:
        #    print(i)
        #print('\n\n')
        #print(self.Beta)
        
        for k in range(self.K-2, -1, -1):
            for Branch in self.tree[k]:
                for i0 in self.N[k]:
                    for j0 in self.N[k+1]:
                        Set = np.array([i for i in self.N[k] if i != i0])
                        for sigma in itertools.permutations(Set):
                            robustness = 0
                            for s in Branch:
                                sk = self.SC[s,k]
                                route = [i0]+[i for i in sigma]+[j0];
                                #print('FLAG')
                                #print(route)
                                #print(np.shape(self.utt))
                                #print(s, k,sk)
                                tt = np.array([[utt[ii,jj,sk] for ii in route] for jj in route])
                                #print(tt)
                                arrivals = Measure(route, tt, self.service_time)
                                aj0 = arrivals[-1]
                                aim = arrivals[-2]
                                
                                local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + self.Beta[j0,s])
                                
                                if local_regret < self.Beta[i0, s]:
                                    self.Beta[i0, s] = local_regret
                                
                                
                                local_robustness = local_regret/pR[s]
                                
                                if robustness<local_robustness:
                                    robustness = local_robustness;
                                
                            if robustness < min_regret:
                                min_regret = robustness
                        for s in Branch:
                            self.alpha[i0, j0, s] = min_regret
                            routes[i0][s] = sigma_s
                            mymap[i0,s] = j0   
                            
        i0 = self.indDepot
        for j0 in self.N[0]:
            
            robustness = 0;
            for s in range(self.nSC):
                #sk = 0;
                #route = [i0]+[j0]
                #tt = np.array([[utt[ii,jj,sk] for ii in route] for jj in route])
                #9arrivals = Measure(route, tt, self.service_time)
                aim = self.service_time
                aj0 = aim + utt[i0,j0,0]
                local_regret =  np.maximum(aj0,0)+ self.Beta[j0,s]
                if local_regret < self.Beta[i0, s]:
                    self.Beta[i0, s] = local_regret
                    
                local_robustness = local_regret/pR[s]
                
                if robustness<local_robustness:
                    robustness = local_robustness;
                
            for s in range(self.nSC):
                self.alpha[i0, j0, s] = robustness
                mymap[i0,s] = j0

        print(routes)
        print(mymap)

        
        
        #for i in self.alpha:
        #    print(i)
        #print('\n\n')
        #print(self.Beta)
        
        
        route = np.zeros([self.K+1, self.nSC], dtype=int)
        for i in range(self.nSC):
            route[0,i] = self.indDepot
        
        minAlpha = np.inf
        for j in self.N[0]:
            tmp = self.alpha[self.indDepot, j, 0]
            if tmp < minAlpha:
                minAlpha = tmp
                jstar = j
        
        for s in range(self.nSC):
            route[1, s] = jstar
            
        #print(route)
            
        
        for k in range(0,self.K):
            #print(route[k-1,:])
            
            #print(istar)
            for Branch in self.tree[k-1]:
                istar = route[k,self.tree[k][0][0]]   
                
                #print(self.N[k-1])
                tmp = np.array([self.alpha[istar, jj, Branch[0]] for jj in self.N[k]])
                jstar = np.argmin(tmp)
                #print(istar, [self.alpha[istar, i, Branch[0]] for i in self.N[k-1]])
                #print(istar,self.N[k])
                #print(tmp)
                for s in Branch:
                    route[k+1, s] = self.N[k][jstar]
                    
        print(route)
        

#%%
ex = DRTSP(codes, hours, day,utt)
#s0 = ex.SC[0]
#s4 = ex.SC[4]
#s5 = ex.SC[5]
#s10 = ex.SC[10]



#for s in ex.SC:
#    print(ex.solvePrecient(s))

#a0 = ex.solvePrecient(s0)
#a10 = ex.solvePrecient(s10)
#a4 = ex.solvePrecient(s4)
#a5 = ex.solvePrecient(s5)


ex.solve()
#print(a4, s4)
#print(a5, s5)
#print(a0)
#print(a10)




#%%
for i in ex.N[0]:
    print(ex.path[i], ex.reg[i])

print(ex.path[ex.indDepot])
print('\n\n')
print(ex.N)

#%%
#example.solveSubproblems()
#example.buildScenarios()
#tmp = example.OptimizeRoute()
#example.solve()
#for i in example.SC:
#    example.solvePrecient(i)

example.solve()
print(example.N)

    
#%%
data = pd.read_csv('randomOxDeliveries.csv')
postcodes = np.unique(data['postcode'].values)

nnodes = 17
#inds = np.random.choice(len(postcodes), nnodes)
codes = postcodes[inds]



nbins = 4;

day = datetime.datetime(2017, 9, 29)
hours = np.zeros([nnodes], dtype = int)
inds = np.random.permutation(nnodes)

for i in range(1,nnodes):
    hours[inds[i]] = int((i-1)/(nnodes/nbins))
hours[inds[0]]= -1
    

#tt = dat_misc.build_tt_matrix(codes, [(i if i>-1 else 0) for i in hours], day)
#tt2 = np.array([[tt[i,j]*np.random.uniform(1,2) if np.random.randint(0,2) else tt[i,j] for j in range(nnodes)] for i in range(nnodes)])
#tt3 = np.array([[tt2[i,j]*np.random.uniform(1,2) if np.random.randint(0,2) else tt2[i,j] for j in range(nnodes)] for i in range(nnodes)])


#tt2 = np.multiply(tt, np.random.uniform(1,2,size=[nnodes, nnodes])) 
#tt3 = np.multiply(tt2, np.random.uniform(1,2,size=[nnodes, nnodes])) 
#utt = np.array([[[tt[i,j], tt2[i,j], tt3[i,j]] for i in range(nnodes)] for j in range(nnodes)])

utt = np.load('ex2_dist_mat.npy')

#%%
np.save('ex3_dist_mat.npy', utt)


#%%#%%
data = pd.read_csv('randomOxDeliveries.csv')
postcodes = np.unique(data['postcode'].values)

nnodes = 17
inds = np.random.choice(len(postcodes), nnodes)
codes = postcodes[inds]



nbins = 4;

day = datetime.datetime(2017, 9, 29)
hours = np.zeros([nnodes], dtype = int)
inds = np.random.permutation(nnodes)

for i in range(1,nnodes):
    hours[inds[i]] = int((i-1)/(nnodes/nbins))
hours[inds[0]]= -1
    

tt = dat_misc.build_tt_matrix(codes, [(i if i>-1 else 0) for i in hours], day)
tt2 = np.array([[tt[i,j]*np.random.uniform(1,2) if np.random.randint(0,2) else tt[i,j] for j in range(nnodes)] for i in range(nnodes)])
tt3 = np.array([[tt2[i,j]*np.random.uniform(1,2) if np.random.randint(0,2) else tt2[i,j] for j in range(nnodes)] for i in range(nnodes)])

'''
#tt2 = np.multiply(tt, np.random.uniform(1,2,size=[nnodes, nnodes])) 
#tt3 = np.multiply(tt2, np.random.uniform(1,2,size=[nnodes, nnodes])) 
'''
utt = np.array([[[tt[i,j], tt2[i,j], tt3[i,j]] for i in range(nnodes)] for j in range(nnodes)])

#utt = np.load('ex2_dist_mat.npy')


