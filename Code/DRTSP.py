#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:16:00 2017

@author: Jonathan Grant-Peters
"""

import numpy as np
import itertools
import scipy as sp




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
        self.N = [[i for i in self.indCustomers if Hours[i] == j] for j in range(self.K)]
        self.Lambda = np.array([60*i for i in range(self.K)])

            
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

        self.nbranches = np.array([2, 4, 7, 11], dtype = int)
        
        
    
    def solvePrecient(self, s, param = 0):
        #Store regret values here
        regret = np.zeros([self.n])
        routes = [ None for _ in range(self.n)]
        
        #treat the first iteration separately
        k = self.K-1
        j0 = self.indDepot
        for i0 in self.N[k]:
            #Set is the set of customers which are to be visited in between departing
            #from i0 and  arriving at j0,
            Set = np.array([i for i in self.N[k] if i != i0])
            
            #min_regret is to be the smallest regret over all possible subroutes
            min_regret = np.inf;
            
            #loop through all subroutes
            for sigma in itertools.permutations(Set):
                #route is the sequence of customers to be visited in this subproblem
                route = [i0]+[i for i in sigma]+[j0];
                #tt is the reduces travel times matrix for this subproblem
                tt = np.array([[self.utt[ii,jj,s[k]] for jj in route]for ii in route] )
                #arrivals computes the relative arrival time at each customer in this subproblem
                arrivals = Measure(route, tt, self.service_time)
                
                #aj0 is the arrival time at j0, the first customer to be visited in the next
                #time window.
                aj0 = arrivals[-1]
                
                #aim is arrival time at im, the last customer to be visited in 
                #the current time window
                aim = arrivals[-2]
                
                #local regret computes which is bigger, aim, the effect that a delay
                #in arriving at j0 has on the remainder of the journey
                local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + param)
                
                #if we find a smaller regret, store this value as min_regret along
                #with the corresponding subroute.
                if local_regret < min_regret:
                    min_regret = local_regret
                    min_sigma = sigma
            
            #We know can fill in the best route to take to finsh the journey if 
            #we begin the last time window at i0. Similarly, we fill in the 
            #corresponding regret.
            routes[i0] = [i for i in min_sigma]+[self.indDepot]
            regret[i0] = min_regret

        #Now iterate through the main body
        
        #Start where the current time window is the second last one, and work
        #backwards until we get to the first time window.
        for k in range(self.K-2, -1,-1):
            #loop over all customers which we might begin this time window at.
            for i0 in self.N[k]:
                #min_regret is to be the minimum regret given a starting customer
                #i0, over all subroutes sigma, and local destinations j0.
                min_regret = np.inf
                
                #loop over all local destinations, that is customers in the following
                #time window.
                for j0 in self.N[k+1]:
                    #Set is the set of customers which are to be visited in between departing
                    #from i0 and  arriving at j0,
                    Set = np.array([i for i in self.N[k] if i != i0])
                    
                    #loop through all subroutes
                    for sigma in itertools.permutations(Set):
                        
                        #route is the sequence of customers to be visited in this subproblem
                        route = [i0]+[i for i in sigma]+[j0];
                        #tt is the reduces travel times matrix for this subproblem
                        tt = np.array([[self.utt[ii,jj,s[k]]  for jj in route]for ii in route])

                        #arrivals computes the relative arrival time at each customer in this subproblem
                        arrivals = Measure(route, tt, self.service_time)
                        
                        #aj0 is the arrival time at j0, the first customer to be visited in the next
                        #time window.
                        aj0 = arrivals[-1]
                        #aim is arrival time at im, the last customer to be visited in 
                        #the current time window
                        aim = arrivals[-2]
                        
                        #local regret computes which is bigger, aim, the effect that a delay
                        #in arriving at j0 has on the remainder of the journey
                        local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + regret[j0])
                        
                        #if we find a smaller regret, store this value as min_regret along
                        #with the corresponding subroute and local destination
                        if local_regret < min_regret:
                            min_regret = local_regret
                            sigma_j = sigma
                            jstar = j0
                #We know can fill in the best route to take to finsh the journey if 
                #we begin the last time window at i0. Similarly, we fill in the 
                #corresponding regret.
                routes[i0] = [i for i in sigma_j]+[jstar]+routes[jstar]
                regret[i0] = min_regret

        #Now for the final iteration. This is merely the journey from the depot
        #to the first customer
        min_regret = np.inf
        #i0 is fixed this time
        i0=  self.indDepot
        
        #loop over all local destinations
        for j0 in self.N[0]:
            #we leave from the depot at time 0
            aim = 0
            #label aj0, the arrival time at the first node
            aj0 = aim + self.service_time+self.utt[i0,j0,0]
            
            #regret is the journey time to j0, plus the regret for finishing from j0
            local_regret = aj0 +regret[j0]
            
            #if we find a smaller regret, store this value as min_regret along
            #with the corresponding local destination
            if local_regret < min_regret:
                min_regret = regret[j0];
                jstar = j0
        
        #We know can fill in the best route to take to finsh the journey if 
        #we begin the last time window at i0. Similarly, we fill in the 
        #corresponding regret.
        routes[i0] = [i0]+ [jstar]+routes[jstar]
        regret[i0] = min_regret        
        
        #return the regret for the best route, as well as the route itself.
        return regret[i0], routes[self.indDepot]
    
    
    def solve(self):
        #param is the arteficial time window for returning to the depot. The
        #bigger param is, the more important it is to return promptly to the depot
        #at cost of delays to other customers.
        param = 30
        
        #allocate space for, and compute the regret for the precient solutions. 
        #That is the best possible route for each scenario if that scenario is 
        #known in advance.
        self.pR = np.zeros([self.nSC])
        self.pRoutes = np.zeros([self.nSC, self.n+1], dtype=int)
        for s in range(self.nSC):
            self.pR[s],  self.pRoutes[s] = self.solvePrecient(self.SC[s],param)
            
        #Full route is the matrix of arrays which correspond to the best way to finish
        #the journey if staring from i0, in scenario s.
        full_route = np.empty([self.n, self.nSC], dtype = list)
        
        #Allocate space for the following values:
        #   SIGijs is the best route to take from i to j in scenario s.
        #   PHIijs is the cost to travel from i to j in scenario s
        #   ALPijs is the robustness of travelling from i to j in scenario s
        #   PHIis is the cost of finishing the journey from i in scenarios s.
        #   ALPis is the robustness of finishing the journey from i in scenario s.
        SIGijs = np.empty([self.n, self.n, self.nSC], dtype = list)
        PHIijs = np.empty([self.n, self.n, self.nSC])
        ALPijs = np.empty([self.n, self.n, self.nSC])
        PHIis = np.empty([self.n, self.nSC])
        ALPis = np.empty([self.n, self.nSC])
        
        #treat the first iteration separately as before
        
        #begin with the 'current' time window being the last one
        k = self.K-1
        #the local destination is fixed to be the depot itself
        j0 = self.indDepot
        #loop through all branches of the scenario tree at time interval k+1. In 
        #practice, there should be a unique branch for every scenario at this point.
        for Branch in self.tree[k+1]:
            #loop through all customers which might be the first delivery in the 'current' time window.
            for i0 in self.N[k]:
                #Define 'Set' as the set of customers to be visited after i0, but 
                #in the 'current' time window.
                Set = np.array([i for i in self.N[k] if i != i0])
                
                #loop through all scenarios in the branch
                for s in Branch:
                    #sk is the scenario vector corresponding to s at time k.
                    sk = self.SC[s,k]
                    
                    #min_regret is the minimum over all subroutes of robustness
                    min_regret = np.inf;
                    #loop through all subroutes.
                    for sigma in itertools.permutations(Set):
                        #route is the sequence in which the customers are visited given sigma
                        route = [i0]+[i for i in sigma]+[j0];
                        #tt is the reduced distance matrix.
                        tt = np.array([[self.utt[ii,jj,sk] for jj in route] for ii in route])
                        #arrivals is the vector of local arrival times
                        arrivals = Measure(route, tt, self.service_time)
                        
                        #aj0 and aim are the arrival times at the local destination, and
                        #the final customer in the 'current' time window
                        aj0 = arrivals[-1]
                        aim = arrivals[-2]
                        
                        #determine regret
                        local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + param)
                        
                        #search for smallest regret, and corresponding sigma
                        if local_regret < min_regret:
                            min_regret = local_regret
                            min_sigma = sigma
                    
                    #each branch contains a unique scenario, thus this step is easy
                    PHIijs[i0, j0, s] = min_regret
                    ALPijs[i0, j0, s] = PHIijs[i0, j0, s]/self.pR[s]
                    PHIis[i0, s] = min_regret
                    SIGijs[i0, j0, s] = min_sigma
                    
                #Determine the scenario in the branch which is least stable. In this
                #iteration, there is only one scenario in the branch.
                ss = Branch[np.argmax([ALPijs[i0, j0, sss] for sss in Branch])]

                for s in Branch:
                    #there is only one robustness value for a branch.
                    ALPijs[i0, j0,s] = ALPijs[i0,j0,ss]
                    ALPis[i0, s] = ALPijs[i0,j0,  ss]
                    SIGijs[i0, j0, s] = SIGijs[i0, j0, ss]
                    
                    #Fill in the best way to finish the journey.
                    full_route[i0,s] =[l for l in SIGijs[i0, j0, s]] + [j0]
        
        '''
        print('************************************')
        print('update')
        print('************************************')
        for s in range(self.nSC):
            print([[i, PHIijs[i, self.indDepot, s],ALPijs[i, self.indDepot, s], self.pR[s]] for i in self.N[-1]])
        '''
           
        #All middle iterations
        #start with the 'current' time interval being the second last, and go
        #backwards until the first time interval
        for k in range(self.K-2, -1, -1):
            #loop through all branches of the scenario tree during the current
            #time interval
            for Branch in self.tree[k+1]:
                #loop through all customers which could be the first to be delivered
                #to during the 'current' time window
                for i0 in self.N[k]:
                    #loop through all local destinations
                    for j0 in self.N[k+1]:
                        #define 'Set' to be all customers within the 'current'
                        #time interval which must be visited in between i0 and j0
                        Set = np.array([i for i in self.N[k] if i != i0])
                        
                        #loop through all scenarios in the branch
                        for s in Branch:
                            #sk is the scenario corresponding to s and k
                            sk = self.SC[s,k]
                            
                            #min_regret should be the smallest value of regret
                            #over all subroutes
                            min_regret  = np.inf
                            
                            #loop through all subroutes
                            for sigma in itertools.permutations(Set):
                                #route is the sequence in which the customers are vistited
                                route = [i0]+[i for i in sigma]+[j0];
                                #tt is the reduced travel time matrix
                                tt = np.array([[self.utt[ii,jj,sk]  for jj in route]for ii in route])
                                #the local arrival times at customers in the 'current' time window
                                arrivals = Measure(route, tt, self.service_time)
                                
                                #arrival times at the local destination, and the last node
                                #in the 'current' time window
                                aj0 = arrivals[-1]
                                aim = arrivals[-2]
                                
                                #compute regret
                                local_regret = np.maximum(aim , np.maximum(aj0 - 60,0) + PHIis[j0,s])
                                
                                #find smallest regret
                                if local_regret < min_regret:
                                    min_regret = local_regret
                                    min_sigma = sigma
                            #PHIijs is the smallest value of regret given i0, j0 and s
                            PHIijs[i0, j0, s] = min_regret
                            #ALPijs is the ratio of min regret over the regret for the
                            #precient solution
                            ALPijs[i0, j0, s] = PHIijs[i0, j0, s]/self.pR[s] 
                            #SIG is the optimal subroute
                            SIGijs[i0, j0, s] = min_sigma
                        
                            
                            
                        print('\nSanity Check')
                        for s in Branch:
                            print(i0, j0,s, ALPijs[i0, j0, s], PHIijs[i0, j0, s], self.pR[s])
                        
                        #ss is the scenario which gives the worst results
                        ss = Branch[np.argmax([ALPijs[i0, j0, sss] for sss in Branch])]
                                    
                        print(ALPijs[i0, j0, ss])
                        print('over\n\n')
            
                                    
                        for s in Branch:
                            #within a branch, all values of ALP and SIG are constant
                            SIGijs[i0,j0,s] = SIGijs[i0, j0, ss]
                            ALPijs[i0, j0,s] = ALPijs[i0,j0,ss]
                            
                    #determine the cost of finishing the journey from each starting
                    #point for each scenario
                    for s in Branch:
                        vals =np.array([ PHIijs[i0, jj, s] for jj in self.N[k+1]])
                        PHIis[i0, s] = np.min(vals)
                        
                        #find the optimal local destination
                        jstar = self.N[k+1][np.argmin([ALPijs[i0, jj, s] for jj in self.N[k+1]])]
                        
                        #update the 'full route' matrix
                        full_route[i0, s] = [l for l in SIGijs[i0, j0, s]]+[jstar]+full_route[jstar,s]
                        ALPis[i0,s] = ALPijs[i0, jstar, s]
                        
                        print('OPT')
                        print(i0, jstar,s, ALPis[i0,s])
                        print('---------')

        
                        
                        
                        
        #finally the last iteration
        
        #for the final stage, Branch is just the set of all scenarios
        Branch = [i for i in range( self.nSC)]
        #i0 is fixed as the depot
        i0 = self.indDepot
        
        #loop through all potential first arrivals
        for j0 in self.N[0]:
            #loop through all scenarios
            for s in Branch:
                #we depart the depot at time 0
                aim = 0
                aj0 = aim +self.service_time+ self.utt[i0,j0,0]
                
                #regret is the time of our first delivery plus the regret of finishing
                #from there
                PHIijs[i0,j0,s] =  np.maximum(aj0,0)+ PHIis[j0,s]
                ALPijs[i0,j0,s] = PHIijs[i0,j0,s]/self.pR[s]
                
            print('\nSanity Check')
            for s in Branch:
                print(i0, j0,s, ALPijs[i0, j0, s], PHIijs[i0, j0, s], self.pR[s])
            
            #ss is the worst performing scenario
            ss = Branch[np.argmax([ALPijs[i0, j0, sss] for sss in Branch])]
                        
            print(ALPijs[i0, j0, ss])
            print('over\n\n')    
            
            #all values of ALP are constant within a branch
            for s in self.tree[0]:
                #robustness refers to the worst case performance
                ALPijs[i0, j0,s] = ALPijs[i0,j0,ss]
        
        #finally, fill in the best route
        for s in range(self.nSC):                        
            jstar = self.N[0][np.argmin([ALPijs[i0, jj, s] for jj in self.N[0]])]
            full_route[i0,s] = [i0] +[jstar] +full_route[jstar,s]
            print('OPT')
            print(i0, jstar,s, ALPijs[i0,jstar,s])
            print('---------')
      
        self.path = full_route[self.indDepot]
        print(self.path)

        

    
def Measure(route, tt, service_time):
    #measure the time required to travel along route, given travel times tt and service time.
    arrival_times = np.zeros([len(route)]); 
    for i in range((len(route)-1)):
        arrival_times[i+1] = arrival_times[i] + service_time + tt[i,i+1]
    return arrival_times
        
def Regret(n,route, utt, scenario, DW, servicetime):
    #compute the regret for a paricular route given a scenario
    K = np.max(DW)+1;
    regret = 0;
    for k in range(K-1, -1, -1):
        local_regret = 0;
        
        local_inds = [route.index(i) for i in route if DW[i] ==k]
        local_n = len(local_inds)
        local_regret = 0
        local_delay = 0
        
        sk = scenario[k]
        for j0 in range(1, local_n):
            if DW[route[local_inds[j0-1]]]==-1:
                local_regret = local_regret + servicetime + utt[route[local_inds[j0-1]], route[local_inds[j0]], 0]
            else:
                local_regret = local_regret + servicetime + utt[route[local_inds[j0-1]], route[local_inds[j0]], sk]
        local_delay = local_regret+servicetime+utt[route[local_inds[-1]], route[local_inds[-1]+1], sk]-60
        
        regret = np.maximum(local_regret, regret + np.maximum(local_delay, 0) )
    return regret    

