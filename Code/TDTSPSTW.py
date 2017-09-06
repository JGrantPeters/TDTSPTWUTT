#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:22:03 2017

@author: Jonathan Peters
"""


import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import cplex
import scipy.sparse as ss
import sets



#%%

class TDTSPSTW(object):
    def __init__(self, nnodes, DeliveryWindows,travel_times, Time_Windows, service_time):
        #nnodes should be an integer, equal to the number of customers, plus one more for the depot.
        #Timewindows should be a matrix with nnodes rows and 2 columns. For a given row, the two values in the matrix refer to the time after which the delivery may be made, and the time before which the delivery must be made.
        
        if np.size(DeliveryWindows,0)!= nnodes:
            raise ValueError("Invalid input, there should be a time window for each customer, and for the depot itself")
            
        if np.size(travel_times, 0)!= nnodes:
            raise ValueError("Invalid input, the first and 2nd dimensional lengths of the travel time tensor should equal the number of nodes")
        
        #record the number of nodes
        self.n = nnodes
        
        #record the delivery windows for each node
        self.DW = DeliveryWindows;
        
        #record the number of time windows which exist
        self.K = np.size(travel_times,2) - 1       
        
        #Record the times of the day when travel times change. That is, between self.TW[j] and self.TW[j+1] one framework applies. However, after self.TW[j], then the rules changes. Overall, travel times must be piecewise linear though, so things don't change too much.
        self.Theta = [self.DW[0,0]]+[i for i in Time_Windows]+[self.DW[0,1]+60]
        
        #We introduce a dummy variable chi_{i,k} which implies that node i is departed during time slot k. Due to Delivery Windows, certain values of chi are restricted in advance, similar to how certain edges are restricted in advance. We record the list of chi's which might take a value of 1, ignoring the others which are restricted to 0.
        self.departureSlots = self.ImportantDepartureSlots()
        
        #record the number of possible chi's which exist. 
        self.kappa = len(self.slots)        
        
  
        
        
        #identify the edges which might be used, and ignore the others.
        self.edges = self.ImportantEdges(travel_times)
        
        #record the number of possible edges
        self.m = np.size(self.edges, 0)
        
        #a big number
        self.M = 10000
            
        self.service_time = service_time

        self.K2 = np.sum([len(self.outSet[i])*len(self.departureSlots[i]) for i in range(self.n)])
 
        
        
        print(self.departureSlots)
        
        
        
        #one variable for each edges, one service time commencement variable for each node, one waiting time variable for each node, one regret variable for each node, one decision variable for each edge x travel time window indicating during which such window is that node departed from
        self.nvars = 2*self.m+2*self.n + self.kappa
        self.neqcons = 3*self.n
        self.nineqcons = self.m + self.K2 + 3*self.n

        
        #one variable for each meaningful edge, one for the arrival time at each node, and one to measure how early the van arrived at each node.
        
        #record the matrix of refined travel times (tt for short), such that tt[e,k] refers to the time taken to travel along edge 'e' at time change point Theta[k]
        self.tt = np.array([travel_times[self.edges[i][0], self.edges[i][1], :] for i in range(self.m)])
        
        self.obj =[0]*(2*self.m+self.kappa)+ [0]*(self.n)+[1]*self.n
        
        
        #elf.ConstraintLHS()
        
    
    def indexMap(self, i,j):
        return self.ij_to_e[i,j]-1

    def slotMap(self, i,j):
        return self.slotInv[i,j] -1
    
    def ImportantDepartureSlots(self):
        possibilities = [[] for i in range(self.n)]
        

        possibilities[0]=[0]
        
        for i in range(1,self.n):
            arr = [j for j in self.DW[i, 0]-self.Theta ]
            valmin = np.min([j for j in arr if j>0])
            kmin = arr.index(valmin)
            
            arr = [j for j in self.Theta - self.DW[i,1]]
            valmax = np.min([j for j in arr if j >0])
            kmax = arr.index(valmax)
            
            possibilities[i]=[j for j in range(kmin,kmax)]
        
        self.slots = []
        for i in range(self.n):
            for j in possibilities[i]:
                self.slots.append([i,j])
        #self.slots = [[i,j] for j in possibilities[i] for i in range(self.nnodes)]
        
        slotInv = np.zeros([self.n, self.K])
        count = 1;
        for slot in self.slots:
            slotInv[slot[0], slot[1]] = count;
            count+=1
        
        self.slotInv = ss.csc_matrix(slotInv, dtype=int)

        
        return possibilities
        
    def ImportantEdges(self, travel_times):
       #This function takes the complete graph with 'nnodes' nodes, and refines it to the important edges. For example if TW[i,2]<TW[j,1], then the edge from j to i is redundant and can be ignored.
       edges = []
       
       self.inSet = [[] for _ in range(self.n)]
       self.outSet = [[] for _ in range(self.n)]
       
       ij_to_e = np.zeros([self.n, self.n])
       
       
       #Full_Set = sets.Set([i for i in range(self.nnodes)])
       
       Ordering= [[sets.Set([]) for _ in range(3)] for _ in range(self.n)]
       
       
       #This loop figues out the basic ordering hierarchy
       
       for node in range(1,self.n):
           
           for other in range(self.n):
               if(node!=other):
                   if other !=0:
                       tmp = np.min([travel_times[node, other,k] for k in self.departureSlots[node]])
                       tmp2 = np.min([travel_times[other,node,k] for k in self.departureSlots[other]])
                       
                       if self.DW[node, 0]+ tmp> self.DW[other,1]:
                           Ordering[node][0].add(other)
                       elif self.DW[other,0]+tmp2> self.DW[node,1]:
                           Ordering[node][2].add(other)
                       else:
                           Ordering[node][1].add(other)
                   
       #This loop completes the hierarchy procedure by identifying which nodes
       #are 'much' greater in the hierarchy than others.

       for node in range(1,self.n):
           
           for between in list(Ordering[node][2]):
               
               for outside in list(Ordering[between][2]):
                   if outside in Ordering[node][2]:
                       #Ordering[node][4].add(outside)
                       Ordering[node][2].remove(outside);
                       #Ordering[outside][0].add(node);
                       Ordering[outside][0].remove(node)
       
       
       #Finally we need to sort out the unique status of the depot in the hierarchy.
       
       #First find the set of potential last stops.
       start = 1;
       future_target = Ordering[start][2]
       while len(list(future_target)) !=0:
           start = list(Ordering[start][2])[0]
           future_target = Ordering[start][2]

       Ordering[0][0].update(Ordering[start][1])
       Ordering[0][0].add(start)
       for i in list(Ordering[0][0]):
           Ordering[i][2].add(0)
       
       
       
       #Now the set of potential first stops.
       start = 1;
       future_target = Ordering[start][0]
       while len(list(future_target)) !=0:
           #print(list(future_target))
           start = list(Ordering[start][0])[0]
           future_target = Ordering[start][0]
           
       Ordering[0][2].update(Ordering[start][1])
       Ordering[0][2].add(start)
       for i in list(Ordering[0][2]):
           Ordering[i][0].add(0)
       
       
       
       #print('\n\n')
       #for i in Ordering: print(i)
       
       
       count = 0;
       for i in range(self.n):
           for j in list(Ordering[i][1]):
               count+=1
               edges.append([i,j])
               self.inSet[j].append(i)
               self.outSet[i].append(j)
               ij_to_e[i,j] = count
           
           for j in list(Ordering[i][2]):
               count+=1
               edges.append([i,j])
               self.inSet[j].append(i)
               self.outSet[i].append(j)
               ij_to_e[i,j] = count
                   
       #print(count)
       self.ij_to_e = ss.csc_matrix(ij_to_e, dtype=int)
       return edges
   
    
    def ConstraintLHS(self):
        A = np.empty(self.nvars, dtype=cplex.SparsePair)
        
          
        #Loop through the columns of the constraint matrix
        for i in range(self.nvars):
            
            #first deal with the edge variables
            if i < self.m:
                
                inds = [self.edges[i][0], self.n+self.edges[i][1], self.neqcons+i]
                vals = [1,1,self.M]
                A[i] = cplex.SparsePair(ind = inds, val = vals);
            
            #next deal with the travel time variables
            elif i < 2*self.m:
                inds = [self.neqcons+i-self.m]+[self.neqcons+self.m + (i-self.m)*self.K +k for k in range(self.K)]
                vals = [-1]+[1 for _ in range(self.K)]
                
                A[i] = cplex.SparsePair(ind = inds, val = vals)
                
            #next the chi departure slot variables
            elif i <2*self.m +self.kappa:
                #the ik coordinates of this chi variable are:
                base = 2*self.m
                chii = self.slots[i-base][0]
                chik = self.slots[i-base][1]
                
                
                ind1 = [chii+2*self.n]
                val1 = [-1 ]
                
                ee = [self.indexMap(chii, j) for j in self.outSet[chii]]
                print(chii, chik, ee, self.edges[ee[0]])
                
                ind2 = [self.neqcons + self.m + self.K*e + chik for e in ee]
                val2 = [-self.M for _ in ind2]
                                
                ind3 = [chii+self.neqcons+self.m*(1+self.K), chii+self.neqcons+self.m*(1+self.K)+self.n]
                val3 = [-self.Theta[chik+1], -self.Theta[chik]]
                
                A[i] = cplex.SparsePair(ind = ind1+ind2+ind3, val = val1+val2+val3)
                
            elif i <2*self.m + self.kappa + self.n:
                base = 2*self.m + self.kappa
                
                #ii = self.edges[i-base][0]
                #jj = self.edges[i-base][1]
                
                ind1 = [self.neqcons+ self.indexMap(jj, i-base) for jj in self.inSet[i-base]]
                val1 = [1 for _ in ind1]
                ind2 = [self.neqcons+self.indexMap(i-base,jj) for jj in self.outSet[i-base]]
                val2 = [-1 if i-base else 0 for _ in ind2]
                
                ee = [self.indexMap(i-base,jj) for jj in self.outSet[i-base]]
                
                ind3 = [self.neqcons+self.m+ self.K*self.indexMap(i-base,jj) +k for k in range(self.K) for jj in self.outSet[i-base]]
                
                e = i-base
                
                val3 = [-(self.tt[e,k+1] - self.tt[e,k])/(self.Theta[k+1]-self.Theta[k]) if i-base else 0 for k in range(self.K) for _ in self.outSet[i-base]]
                
                
                ind4 = [i-base +self.neqcons+self.m*(self.K+1), i-base +self.neqcons+self.m*(self.K+1)+self.n]
                val4 = [1 for _ in ind4]
                
                A[i] = cplex.SparsePair(ind = ind1+ind2+ind3+ind4, val = val1+val2+val3+val4)
                
            elif i<2*self.m + self.kappa +2*self.n:
                base = 2*self.m + self.kappa + self.n
                A[i] = cplex.SparsePair(ind=[i-base+self.neqcons + self.m*(self.K+1) + 2*self.n], val = [-1])
                
        return A;
    
    def ConstraintRHS(self):
        #the rhs for all equality bounds is 1
        b = np.ones(self.neqcons + self.nineqcons)
        
        for i in range(self.nineqcons):
            if i < self.m:
                #First the 'big M' constraint which relates T_i to T_j           
                b[self.neqcons +i] = (self.M + self.service_time)
            elif i<(self.K+1)*self.m:
                #Next the 'big M' constraint which determine travel time, with respect to time of day. There are K*m of these constraints, one for each edge per time window. We order these constraints such that constraint e*K + k refers to edge: 'e' and time windows 'k'.
                ee = (i-self.m)//self.K
                kk = (i-self.m)% self.K
                
                
                tmp = self.tt[ee,kk] + (self.tt[ee, kk+1]-self.tt[ee,kk])/(self.Theta[kk+1]-self.Theta[kk]) * (self.service_time - self.Theta[kk]) - self.M
                b[self.neqcons +i] = tmp
            
            elif i <(self.K+1)*self.m + 2*self.n:
                #rhs for constraints which determine the chi variable, that is during which time slot the node i is departed from.
                b[self.neqcons+i] = -self.service_time
            elif i <(self.K+1)*self.m + 3*self.n:
                #Finally we determine lateness
                b[self.neqcons+i] = self.DW[i-((self.K+1)*self.m + 2*self.n), 1]
        
        
        con_type = "E"*self.neqcons + "G"*(self.m*(self.K+1)) + "L"*self.n + "G"*self.n + "L" *self.n

        return b, con_type
    
    def formulate(self):
        problem = cplex.Cplex();
        
        problem.objective.set_sense(problem.objective.sense.minimize)
        
        self.names = ["e_"+str(i[0])+","+str(i[1]) for i in self.edges]+ ["t_"+str(i[0])+","+str(i[1]) for i in self.edges] + ["Chi_"+str(i[0])+","+str(i[1]) for i in self.slots] + ["T"+str(i) for i in range(self.n)]  +["r"+str(i) for i in range(self.n)]
        
        
        my_ubs = [1 for _ in range(self.m)] +[self.M for _ in range(self.m)]+[1 for _ in range(self.kappa)] +[self.DW[i,1] for i in range(self.n)] +[self.M for _ in range(self.n)]
        
        my_lbs = [0 for _ in range(2*self.m+self.kappa)]+[self.DW[j,0] for j in range(self.n)]+[0 for _ in range(self.n)]
        
        my_types = [problem.variables.type.binary for _ in range(self.m)]+[problem.variables.type.continuous for _ in range(self.m)]+[problem.variables.type.binary for _ in range(self.kappa)] +[problem.variables.type.continuous for _ in range(2*self.n)]
        
        my_rhs, con_type = self.ConstraintRHS()

        my_con_names = ["c"+str(i) for i in range(self.neqcons + self.nineqcons)]
        problem.linear_constraints.add( rhs = my_rhs, senses = con_type, names = my_con_names)
        
        
        
        #All of the lower bounds take the default value of 0
                
        
        self.LHS = self.ConstraintLHS()
        
        
        #print(len(self.obj), len(self.names), len(my_ubs), len(my_lbs), len(my_types), len(self.LHS))
        
        #print(self.obj)
        problem.variables.add(obj = self.obj, names = self.names, ub = my_ubs, lb = my_lbs, types = my_types, columns =  self.LHS)
        
        return problem
        
     
        
    
    
    




#%%
       
toy = TDTSPSTW(toy_nnodes, toy_TimeWindows, toy_travel_times, TW, servicetime)   
print('\n\n')
print(toy.DW)
print(toy.Theta)
print('\n\n')

print(toy.edges)  

print(toy.ConstraintRHS())
print(toy.ConstraintLHS())
tmp = toy.formulate();
tmp.write("current_test.lp")


#%%

tmp.solve()
tmp.solution.status[tmp.solution.get_status()]

#toy.ConstraintLHS()

    
#%%
servicetime = 5    
toy_nnodes = 5;
ntw = 5;

hour = 60;

toy_customer_bounds = np.random.uniform(low=0,high=8*hour, size=toy_nnodes-1)

toy_TimeWindows = np.empty([toy_nnodes, 2])
toy_TimeWindows[0] = np.array([0, 8.5*hour])
toy_TimeWindows[1:,0] = toy_customer_bounds;
toy_TimeWindows[1:, 1] = toy_customer_bounds+hour;

toy_travel_times = np.array([[[0 if i==j else np.random.uniform(low = 5, high = hour) for k in range(ntw+1)] for i in range(toy_nnodes) ]for j in range(toy_nnodes)] )

TW = np.sort(np.random.uniform(0, 8.5*hour, size=(ntw-1)))


tol_delay = 5
intol_delay = 10
alpha = 5
beta = 100




    


    
#%%

tmp.solve()
print(tmp.solution.get_objective_value())
x = tmp.solution.get_values()
print('\n\n')
for i in x: print i

print('\n\n')

print(toy.travel_times_refined)