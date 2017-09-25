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
    arrival_times = np.empty([len(route)]); 
    for i in range((len(route)-1)):
        arrival_times[route[i+1]] = arrival_times[route[i]] + service_time + tt[route[i],route[i+1]]
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
        self.ncategories = np.max(Hours)+1
        self.categories = self.getCategories(Hours)
        
        #self.cSizes = [len()]
        
        
    def getCategories(self, Hours):
        categories = [[i for i in self.indCustomers if Hours[i] == j] for j in range(self.ncategories)]
        return categories
    
    def buildScenarios(self):
        self.nSC = int(sp.misc.comb(self.ncategories+1, 2)) + 1
        self.SC = np.zeros([self.nSC, self.ncategories], dtype = int)
        self.SC[0,:] = np.array([0,0,0,0])
        self.SC[1,:] = np.array([0,0,0,1])
        self.SC[2,:] = np.array([0,0,1,1])
        self.SC[3,:] = np.array([0,0,1,2])
        self.SC[4,:] = np.array([0,1,1,1])
        self.SC[5,:] = np.array([0,1,1,2])
        self.SC[6,:] = np.array([0,1,2,2])
        self.SC[7,:] = np.array([1,1,1,1])
        self.SC[8,:] = np.array([1,1,1,2])
        self.SC[9,:] = np.array([1,1,2,2])
        self.SC[10,:] = np.array([1,2,2,2])
        
        
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
        
            
    def solveSubproblems(self):
        #print(self.utt)
        category_size = len(self.categories[0])
        self.StartProblems = np.empty([category_size,3], dtype = SubSolution)
        for s in range(self.nsc):
            for i in range(category_size):
                inds = [self.indDepot] + [self.categories[0][i]]
                #print(inds)
                subutt = np.array([[self.utt[k,j,s] for k in inds] for j in inds])
                #print(subutt)
                self.StartProblems[i,s]= Subproblem(subutt, self.service_time, inds)
                
        category_size = len(self.categories[-1])
        self.EndProblems = np.empty([category_size, 3], dtype = SubSolution)
        for s in range(self.nsc):
            for i in range(category_size):
                #print(self.categories[-1], self.indDepot)
                inds = [self.categories[-1][i]] + [j for j in self.categories[-1] if j!= self.categories[-1][i]]+[self.indDepot]
                #print(inds)
                subutt = np.array([[self.utt[k,j,s] for k in inds] for j in inds])
                self.EndProblems[i,s]= Subproblem(subutt, self.service_time, inds)

        self.MiddleProblems = [[[[None  for s in range(self.nsc)] for end in self.categories[i+1]]for start in self.categories[i]] for i in range(self.ncategories-1)]
        
        for stage in range(self.ncategories-1):
            
            for start in range(len(self.categories[stage])):
                for end in range(len(self.categories[stage+1])):
                    #print(self.categories[stage][start], self.categories[stage+1][end])
                    inds = [self.categories[stage][start]] + [j for j in self.categories[stage] if j!= self.categories[stage][start]]+[self.categories[stage+1][end]]
                    #print(inds)
                    for s in range(self.nsc):
                        subutt = np.array([[self.utt[k,j,s] for k in inds] for j in inds])
                        tmp = Subproblem(subutt, self.service_time, inds)
                        self.MiddleProblems[stage][start][end][s] = tmp
                        #print(tmp.route, self.categories[stage+1][start], self.categories[stage+2][end])
                        
    def OptimizeRoute(self):
        a = np.zeros([self.n, self.nSC])
        atilde = np.zeros([self.n, self.n, self.nSC])
        
        
        for s in range(self.nSC):
            for i in self.EndProblems:
                #print(self.SC[s,-1], i[self.SC[s, -1]].route)
                a[i[self.SC[s,-1]].route[0], s] = i[self.SC[s,-1]].phi
        
                
        for s in range(self.nSC):
            for stage in range(self.ncategories-2,-1, -1):
                for i in self.MiddleProblems[stage]:
                    smallest = np.inf
                    
                    for j in i:
                        start = j[self.SC[s,stage]].route[0]
                        end = j[self.SC[s,stage]].route[-1]
                        #print(start, end)
                        tmp = np.maximum(j[self.SC[s,stage]].phi, a[end, s] + np.maximum(0, -j[self.SC[s,stage]].w))
                        atilde[start, end, s] = tmp
                        if tmp<smallest:
                            smallest = tmp
                    a[i[0][self.SC[s,stage]].route[0], s ] =smallest 
        
        smallest = np.inf
        start = self.indDepot
        for s in range(self.nSC):
            for j in self.StartProblems:
                smallest = np.inf
                end = j[0].route[-1]
                #print(a[end,s])
                tmp = np.maximum(j[0].phi, a[end, s] + np.maximum(0, -j[0].w))
                atilde[start, end, s] = tmp
                if tmp<smallest :
                    smallest = tmp
            a[self.indDepot, s] = smallest
            
        
        
        
        
        Delta = np.array([0.0 for _ in range(self.nSC)])
        Precient_arrivals = np.zeros([self.n, self.nSC])
        Precient_routes = [None for _ in self.SC]
        Precient_regret = np.zeros([self.n, self.nSC])
        for s in range(self.nSC):
            istar = np.argmin([a[i,s] for i in self.categories[0]])
            
            Precient_arrivals[self.categories[0][istar], s] = self.StartProblems[istar][self.SC[s,stage]].rel_arrival[-1]
            Precient_regret[self.categories[0][istar],s] = self.StartProblems[istar][self.SC[s,stage]].rel_arrival[-1]
            
            Precient_routes[s] = self.StartProblems[istar, 0].route
            for stage in range(self.ncategories-1):
                jstar = np.argmin([atilde[self.categories[-1][istar], jj, s] for jj in self.categories[stage]])
                
                tmp = self.MiddleProblems[stage][istar][jstar][self.SC[s, stage]]
                for jj in range(len(tmp.route[1:])):
                    Precient_arrivals[tmp.route[jj+1], s] = tmp.rel_arrival[jj+1] + Delta[s]+stage*60
                    Precient_regret[tmp.route[jj+1],s] = tmp.rel_arrival[jj+1] + Delta[s]
                
                
                Precient_routes[s] = Precient_routes[s] + self.MiddleProblems[stage][istar][jstar][self.SC[s, stage]].route[1:]
                Delta[s] += np.maximum(0, -self.MiddleProblems[stage][istar][jstar][self.SC[s,stage]].w)
                #print(self.MiddleProblems[stage][istar][jstar][self.SC[s,stage]].w)
                #print(Delta)
                istar = jstar
            Precient_routes[s] = Precient_routes[s]+self.EndProblems[istar, self.SC[s,-1]].route[1:]
            Delta[s] += np.maximum(0, -self.EndProblems[istar][self.SC[s,-1]].w)
        
        
        Precient = np.array([i+60 for i in Delta])
        
        #Now we have our precient solutions, and need to find the robust solution;
        
        recoursePath = np.zeros([self.nSC, self.ncategories])
        #print(self.nSC, self.ncategories)
        #print(np.shape(self.tree))
        istar = np.argmin([np.min([a[i,s]/Precient[s] for s in range(self.nSC)]) for i in self.categories[0]])
        #print(istar)
        recoursePath[:,0] = [istar for _ in self.SC]
        
        for stage in range(0, self.ncategories):
            for branch in range(self.nbranches[stage]):
                scens = [i for i in range(self.nSC) if self.tree[i,stage] ==branch] 
                istar = recoursePath[scens[0], stage-1]
                
                #print(scens)
                
                jstar = np.argmin([float(atilde[self.categories[stage][istar],jj,s])/Precient[s] for s in scens] for jj in self.categories[stage])
                #print(jstar, 'flag')
                recoursePath[scens, stage] = [self.categories[stage][jstar] for _ in scens]
                
        
        print(Precient_arrivals)
        print('\n')
        print(Precient_regret)
        
        print(Delta)
        for i in Precient_routes:
            print(i)
        print(recoursePath)
        '''
        return route, Delta
        '''
#%%
example = DRTSP(codes, hours, day,utt)

#%%
example.solveSubproblems()
example.buildScenarios()
tmp = example.OptimizeRoute()
print(example.categories)

    
#%%
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


#tt2 = np.multiply(tt, np.random.uniform(1,2,size=[nnodes, nnodes])) 
#tt3 = np.multiply(tt2, np.random.uniform(1,2,size=[nnodes, nnodes])) 
utt = np.array([[[tt[i,j], tt2[i,j], tt3[i,j]] for i in range(nnodes)] for j in range(nnodes)])

#%%
 np.save('ex3_dist_mat', utt)





