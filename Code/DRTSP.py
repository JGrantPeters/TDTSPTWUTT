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

def Regret(n,route, utt, scenario, DW, servicetime):
    reg = 0
    arrivals = np.zeros([n+1])
    #print(servicetime)
    arrivals[route[2]] = servicetime + utt[route[1], route[2], 0]
    for i in range(2,n-1):
        #print(route[i], (DW[route[i]]*60)) 
        #if route[i]==0:
        #    print(route[i],route[i+1], arrivals[route[i]],servicetime + utt[route[i], route[i+1], scenario[DW[route[i]]]],  arrivals[route[i]]+servicetime + utt[route[i], route[i+1], scenario[DW[route[i]]]])
        
        if (DW[route[i]] != DW[route[i+1]]):
            arrivals[route[i+1]] = np.maximum(arrivals[route[i]] + servicetime + utt[route[i], route[i+1], scenario[DW[route[i]] ]], (DW[route[i+1]])*60)
            
        else:
            arrivals[route[i+1]] = arrivals[route[i]]+servicetime + utt[route[i], route[i+1], scenario[DW[route[i]]]]
        
        loc_reg = arrivals[route[i+1]] - (DW[route[i+1]])*60
        
        
        
        if loc_reg>reg:
            #print(route[i], route[i+1], utt[route[i], route[i+1], scenario[DW[route[i]]]], arrivals[route[i]], arrivals[route[i+1]], DW[route[i+1]], loc_reg)
            reg = loc_reg
    
    return reg

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
        ARRIVALS = [None for _ in range(self.n)]
        mymap = np.zeros([self.n])
        
        #treat the first iteration separately
        k = self.K-1
        j0 = self.indDepot
        for i0 in self.N[k]:
            Set = np.array([i for i in self.N[k] if i != i0])

            min_regret = np.inf;
            for sigma in itertools.permutations(Set):
                route = [i0]+[i for i in sigma]+[j0];
                tt = np.array([[utt[ii,jj,s[k]] for jj in route]for ii in route] )
                
                arrivals = Measure(route, tt, self.service_time)
                aj0 = arrivals[-1]
                aim = arrivals[-2]
                
                local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + param)
                if local_regret < min_regret:
                    min_regret = local_regret
                    min_sigma = sigma
                    min_arrive = arrivals
            #print(i0, min_regret, min_sigma)
            routes[i0] = [i for i in min_sigma]+[self.indDepot]
            regret[i0] = min_regret
            mymap[i0] = j0
            ARRIVALS[i0] = min_arrive
        
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
                        tt = np.array([[utt[ii,jj,s[k]]  for jj in route]for ii in route])
                        
                        
                        
                        arrivals = Measure(route, tt, self.service_time)
                        aj0 = arrivals[-1]
                        aim = arrivals[-2]
                        local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + regret[j0])
                        
                        
                
                        #print(aim, aj0, local_regret, regret[j0],i0, j0)
                        if local_regret < best_route_regret:
                            best_route_regret = local_regret
                            min_sigma = sigma
                            min_arrive = arrivals
                    
                    if best_route_regret < min_regret:
                        min_regret = best_route_regret;
                        jstar = j0
                        sigma_j = min_sigma
                        arrive_j = min_arrive
                
                #print('FLAG', i0, sigma_j, jstar)
                routes[i0] = [i for i in sigma_j]+[jstar]+routes[jstar]
                regret[i0] = min_regret
                mymap[i0] = jstar
                #print('here', arrive_j, ARRIVALS[jstar])
                ARRIVALS[i0] = [m for m in arrive_j] +[np.maximum(arrive_j[-1], 60) + m for m in ARRIVALS[jstar][1:]]
                
        #print(routes)
            
        #Now for the final iteration
        min_regret = np.inf
        i0=  self.indDepot
        sigma_j = 0

        for j0 in self.N[0]:
            #print(j0, Set)
            
            #tt = np.array([[utt[ii,jj,s[k]] for ii in route] for jj in route])
            
            aim = 0
            
            aj0 = aim + self.service_time+utt[i0,j0,0]
            local_regret = aj0 
            #print(j0, local_regret, regret[j0])    
            if local_regret < min_regret:
                min_regret = regret[j0];
                jstar = j0
                mymap[i0] = jstar
                arrivals = [aim, aj0]
                ajstar = aj0
                
        routes[i0] = [i0]+ [jstar]+routes[jstar]
        regret[i0] = min_regret
        #print(routes[i0])
        ARRIVALS[i0] = [i for i in arrivals] + [ajstar+  m for m in ARRIVALS[jstar][1:]]
        
        route = np.zeros([self.K+1], dtype = int)
        route[0] = self.indDepot
        
        for k in range(self.K):
            route[k+1] = mymap[route[k]]
            
        
        self.debug = ARRIVALS
        
        
        self.arrival_times = ARRIVALS[i0]
        
        self.reg = regret
        #self.path = routes
        
        return regret[i0], route, routes[self.indDepot]
    
    
    def solve(self):
        param = 30
        
        #precient regret
        #tmp = np.array([[self.solvePrecient(s, param)] for s in self.SC])
        
        #pR = tmp[:,0]
        #pRoute = tmp[:,1]
        
        
        self.pR = np.zeros([self.nSC])
        self.pRoute = np.zeros([self.nSC, self.K+1], dtype = int)
        self.pROUTES = np.zeros([self.nSC, self.n+1], dtype = int)
        for s in range(self.nSC):
            self.pR[s], self.pRoute[s,:], self.pROUTES[s,:] = self.solvePrecient(self.SC[s],param)
            
        #pR2 = np.array([ Regret(self.n, ROUTES[s,:], utt, self.SC[s], self.Hours, self.service_time) for s in range(self.nSC)])
        
        #pR = pR2
        
        
        full_route = np.empty([self.n, self.nSC], dtype = list)
        
        
        #mymap = np.zeros([self.n, self.nSC])
        #routes = [ [None for _ in range(self.nSC)] for _ in range(self.n)]
        #Regret = np.zeros([self.n, self.nSC])
        #Robustness = np.zeros([self.n])
        
        SIGijs = np.empty([self.n, self.n, self.nSC], dtype = list)
        #SOGij = np.empty([self.n, self.n, self.nSC], dtype = list)
        PHIijs = np.empty([self.n, self.n, self.nSC])
        ALPijs = np.empty([self.n, self.n, self.nSC])
        PHIis = np.empty([self.n, self.nSC])
        #ALPis = np.empty([self.n, self.nSC])
        
        alpha = np.zeros([self.n, self.nSC])
        
        print(self.pRoute)
        #print(pR)
        
        #treat the first iteration separately
        k = self.K-1
        j0 = self.indDepot
        for Branch in self.tree[k+1]:
            for i0 in self.N[k]:
                maxAlph = 0
                Set = np.array([i for i in self.N[k] if i != i0])
                for s in Branch:
                    sk = self.SC[s,k]
                    
                    min_regret = np.inf;
                    for sigma in itertools.permutations(Set):
                        
                        #calculate
                        route = [i0]+[i for i in sigma]+[j0];
                        tt = np.array([[utt[ii,jj,sk] for jj in route] for ii in route])
                        arrivals = Measure(route, tt, self.service_time)
                        aj0 = arrivals[-1]
                        aim = arrivals[-2]
                        
                        #determine regret
                        local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + param)
                        
                        if local_regret < min_regret:
                            min_regret = local_regret
                            min_sigma = sigma
                    
                    PHIijs[i0, j0, s] = min_regret
                    ALPijs[i0, j0, s] = PHIijs[i0, j0, s]/self.pR[s]
                    PHIis[i0, s] = min_regret
                    SIGijs[i0, j0, s] = min_sigma
                    
                    #ALPis[i0, s] = ALPijs[i0, j0, s]
                    #mymap[i0,s] = j0
                
                
                ss = np.argmax([ALPijs[i0, j0, sss] for sss in Branch])
                for s in Branch:
                    ALPijs[i0, j0,s] = ALPijs[i0,j0,ss]
                    #print(s)
                    alpha[i0, s] = ALPijs[i0,j0,  s]
                    
                    tmp =  [l for l in SIGijs[i0, j0, s]] + [j0]
                 
                    full_route[i0,s] =tmp
                
        print(SIGijs)
        
        
        for k in range(self.K-2, -1, -1):
            for Branch in self.tree[k+1]:
                for i0 in self.N[k]:
                    for j0 in self.N[k+1]:
                        Set = np.array([i for i in self.N[k] if i != i0])
                        
                        for s in Branch:
                            sk = self.SC[s,k]
                            min_regret  = np.inf
                            for sigma in itertools.permutations(Set):
                                
                                route = [i0]+[i for i in sigma]+[j0];
                                tt = np.array([[utt[ii,jj,sk]  for jj in route]for ii in route])
                                arrivals = Measure(route, tt, self.service_time)
                                aj0 = arrivals[-1]
                                aim = arrivals[-2]
                                local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + PHIis[j0,s])
                                
                                if local_regret < min_regret:
                                    min_regret = local_regret
                                    min_sigma = sigma
                            
                            PHIijs[i0, j0, s] = min_regret
                            ALPijs[i0, j0, s] = PHIijs[i0, j0, s]/self.pR[s] 
                            SIGijs[i0, j0, s] = min_sigma
                        
                        ss = np.argmax([ALPijs[i0, j0, sss] for sss in Branch])
                        for s in Branch:
                            SIGijs[i0,j0,s] = SIGijs[i0, j0, ss]
                            ALPijs[i0, j0,s] = ALPijs[i0,j0,ss]
                            
                    
                    for s in Branch:
                        vals =np.array([ PHIijs[i0, jj, s] for jj in self.N[k+1]])
                        PHIis[i0, s] = np.min(vals)
                        
                        
                        jstar = self.N[k+1][np.argmin([ALPijs[i0, jj, s] for jj in self.N[k+1]])]
                        
                        
                        
                        full_route[i0, s] = [l for l in SIGijs[i0, j0, s]]+[jstar]+full_route[jstar,s]
                        alpha[i0,s] = ALPijs[i0, jstar, s]

                    #tmp = np.array([[ ALPijs[i0, jj, s] for jj in self.N[k+1]] for s in Branch])
                    #jstar = self.N[k+1][np.argmin(np.max(tmp,0))]
                    #mymap[i0,s] = jstar
                    

                            
        i0 = self.indDepot
        for j0 in self.N[0]:
            for s in range(self.nSC):
                #sk = 0;
                #route = [i0]+[j0]
                #tt = np.array([[utt[ii,jj,sk] for ii in route] for jj in route])
                #9arrivals = Measure(route, tt, self.service_time)
                aim = self.service_time
                aj0 = aim + utt[i0,j0,0]
                local_regret =  np.maximum(aj0,0)+ PHIis[j0,s]
                
                
                PHIijs[i0,j0,s] = local_regret
                ALPijs[i0,j0,s] = local_regret/self.pR[s]
                
                vals =np.array([ PHIijs[i0, jj, s] for jj in self.N[k+1]])
                PHIis[i0, s] = np.min(vals)
                vals =np.array([ ALPijs[i0, jj, s] for jj in self.N[k+1]])
                
            ss = np.argmax([ALPijs[i0, j0, sss] for sss in Branch])
            for s in self.tree[0]:
                ALPijs[i0, j0,s] = ALPijs[i0,j0,ss]
                #ALPis[i0, s] = np.min(vals)
        
        for s in range(self.nSC):                        
            jstar = self.N[0][np.argmin([ALPijs[i0, jj, s] for jj in self.N[0]])]
            full_route[i0,s] = [i0] +[jstar] +full_route[jstar,s]

        #full_route[i0, s] = [i0]+ full_route

        
        #vals =np.array([ PHIijs[i0, jj, 0] for jj in self.N[0]])
        #tmp = np.array([[ ALPijs[i0, jj, s] for jj in self.N[k+1]] for s in Branch])
        #jstar = self.N[0][np.argmin(np.max(tmp,0))]
        
        #for s in self.SC:
        #    mymap[i0,s] = jstar
        #print(routes)
        #print(mymap)

        #print(mymap)
        
        #for i in self.alpha:
        #    print(i)
        #print('\n\n')
        #print(self.Beta)
        
        #for i in full_route:
        #    print(i)

        
        
        route = np.zeros([self.nSC,self.K+1], dtype=int)
        for s in range(self.nSC):
            route[s,0] = self.indDepot
        
        
        for k in range(self.K):
            for Branch in self.tree[k+1]:
                i0 = route[Branch[0],k]
                
                tmp = np.array([[ALPijs[i0,jj,s] for jj in self.N[k]] for s in Branch])
                #print('FLAG')
                #print(tmp)
                #print(np.shape(tmp))
                #print(tmp)
                #print(np.max(tmp,0))
                jmin = self.N[k][np.argmin(np.max(tmp,0))]
                
                #print(jmin, np.argmin(np.max(tmp,0)))
                for s in Branch:
                    route[s,k+1] = jmin
                    
                    
        self.debug = ALPijs        
        self.path = full_route[self.indDepot]
        #print(alpha)
        print(route)
        #self.path = routes

#%%
ex = DRTSP(codes, hours, day,utt)
#s0 = ex.SC[0]
#s4 = ex.SC[4]
#s5 = ex.SC[5]
#s10 = ex.SC[10]



#for s in ex.SC:
#    print(ex.solvePrecient(s))

#a0 = ex.solvePrecient(s0)

#route = a0[2]
#arrival_times = ex.arrival_times
#a10 = ex.solvePrecient(s10)
#a4 = ex.solvePrecient(s4)
#a5 = ex.solvePrecient(s5)

#print(Regret(17,route, utt, ex.SC[0], hours, 5))
#print(a0[0])




ex.solve()

Results = np.zeros([ex.nSC+1, ex.nSC])

for s in range(ex.nSC):
    opt = ex.pROUTES[s]
    rec = ex.path[s]
    
    Rrec = Regret(ex.n, rec, utt, ex.SC[s], hours, 5)
    Ropt = Regret(ex.n, opt, utt, ex.SC[s], hours, 5)

    
    
    Results[-1, s] = Rrec/Ropt
    
    
    for s2 in range(ex.nSC):
        other = ex.pROUTES[s2]
        
        Roth = Regret(ex.n, other, utt, ex.SC[s], hours, 5)
        
        Results[s2, s] = (Roth/Ropt)

Results
#print(a4, s4)
#print(a5, s5)
#print(a0)
#print(a10)

#%%

big_mat_utt = np.load('big_mat.npy')
postcodes = np.load('codes.npy')


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


tt = np.array([[big_mat_utt[i, j, :] for i in inds] for j in inds])
tt2 = np.array([[tt[i,j]*np.random.uniform(2,4) if np.random.randint(0,2) else tt[i,j] for j in range(nnodes)] for i in range(nnodes)])
tt3 = np.array([[tt2[i,j]*np.random.uniform(2,4) if np.random.randint(0,2) else tt2[i,j] for j in range(nnodes)] for i in range(nnodes)])






#%%















a#%%
a,b,c = ex.solvePrecient(ex.SC[0])
print(a)
print(b)
print(c)

print(Regret(17, c, utt, ex.SC[0], hours, 5))

#%%
for i in ex.N[0]:
    print(ex.path[i], ex.reg[i])

print(ex.path[ex.indDepot])
print('\n\n')
print(ex.N)




#%%
#example.solveSubpro0blems()
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
inds = np.random.choice(len(postcodes), nnodes)
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

utt = np.load('ex3_dist_mat.npy')

#%%
np.save('ex4_dist_mat.npy', utt)


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


