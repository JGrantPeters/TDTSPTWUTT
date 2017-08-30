#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:11:41 2017

@author: user
"""


import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import cplex
import scipy.sparse as ss
import sets


toy_nnodes = 20;

hour = 60;

toy_customer_bounds = np.random.uniform(low=0,high=8*hour, size=toy_nnodes-1)

toy_TimeWindows = np.empty([toy_nnodes, 2])
toy_TimeWindows[0] = np.array([0, 8.5*hour])
toy_TimeWindows[1:,0] = toy_customer_bounds;
toy_TimeWindows[1:, 1] = toy_customer_bounds+hour;

toy_travel_times_min = np.array([[0 if i==j else np.random.uniform(low = 5, high = hour) for i in range(toy_nnodes) ]for j in range(toy_nnodes)])

toy_travel_times_max = np.array([[toy_travel_times_min[i,j] + np.random.gamma(shape = hour/4, scale = np.random.uniform(low=0.5, high =2 )) if i != j else 0 for i in range(toy_nnodes)] for j in range(toy_nnodes) ])


#%%

class TSPTW(object):
    def __init__(self, nnodes, TimeWindows, best_case_travel_times, worst_case_travel_times ):
        #nnodes should be an integer, equal to the number of customers, plus one more for the depot.
        #Timewindows should be a matrix with nnodes rows and 2 columns. For a given row, the two values in the matrix refer to the time after which the delivery may be made, and the time before which the delivery must be made.
        
        if np.size(TimeWindows,0)!= nnodes:
            raise ValueError("Invalid input, there should be a time window for each customer, and for the depot itself")
        
        self.nnodes = nnodes;
        self.TW = TimeWindows;
        self.M = self.TW[0,1] - self.TW[0,0]
        
        self.edges = self.ImportantEdges()
        self.nedges = np.size(self.edges, 0)
        
        self.neqcons = 2*self.nnodes
        self.nineqcons = 2*self.nedges
        
        self.nvars = self.nedges+2*self.nnodes
        #one variable for each meaningful edge, one for the arrival time at each node, and one to measure how early the van arrived at each node.
        
        self.travel_times = [best_case_travel_times[self.edges[i][0], self.edges[i][1]] for i in range(self.nedges)]
        
        self.obj =[0]*self.nedges +[0]*self.nnodes+ [1]+[0]*(self.nnodes-1)
        
        print self.obj
        
    
    def ImportantEdges(self):
       #This function takes the complete graph with 'nnodes' nodes, and refines it to the important edges. For example if TW[i,2]<TW[j,1], then the edge from j to i is redundant and can be ignored.
       edges = []
       
       self.inSet = [[] for _ in range(self.nnodes)]
       self.outSet = [[] for _ in range(self.nnodes)]
       
       ij_to_e = np.zeros([self.nnodes, self.nnodes])
       
       
       #Full_Set = sets.Set([i for i in range(self.nnodes)])
       
       Ordering= [[sets.Set([]) for _ in range(3)] for _ in range(self.nnodes)]
       
       
       #This loop figues out the basic ordering hierarchy
       
       for node in range(1,self.nnodes):
           
           for other in range(self.nnodes):
               if(node!=other):
                   if other !=0:
                       if self.TW[node, 0] > self.TW[other,1]:
                           Ordering[node][0].add(other)
                       elif self.TW[other,0] > self.TW[node,1]:
                           Ordering[node][2].add(other)
                       else:
                           Ordering[node][1].add(other)
                   
       #This loop completes the hierarchy procedure by identifying which nodes
       #are 'much' greater in the hierarchy than others.

       for node in range(1,self.nnodes):
           
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
       
       
       
       print('\n\n')
       for i in Ordering: print(i)
       
       
       count = 0;
       for i in range(self.nnodes):
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
                   
       print(count)
       self.ij_to_e = ss.csc_matrix(ij_to_e, dtype=int)
       return edges
   
    def indexMap(self, i,j):
        return self.ij_to_e[i,j]-1
    
    def ConstraintLHS(self):
        A = np.empty(self.nvars, dtype=cplex.SparsePair)
        
        #Loop through the columns of the constraint matrix
        for i in range(self.nvars):
            if i < self.nedges:
                A[i] = cplex.SparsePair(ind=[self.edges[i][0], self.edges[i][1]+self.nnodes, i+self.neqcons, i+self.neqcons+self.nedges], val=[1,1,self.M, self.M])
            if i >= self.nedges and i < self.nedges+self.nnodes:
                
                #a bunch of intermediate calculations
                basicind = []
                basicval = []
                
                tmp = self.inSet[i-self.nedges]
                tmp2 = [self.indexMap(k, i-self.nedges) for k in tmp]
                
                tmp_size = len(tmp2)
                
                compind = [self.neqcons+ k for k in tmp2]+[self.neqcons+self.nedges+k for k in tmp2]
                #compval = [1]*tmp_size +[-1]*tmp_size;
                compval = [1 if i-self.nnodes else 0 for _ in tmp2]+[-1 if i-self.nnodes else 0 for _ in tmp2]
                
                A[i] = cplex.SparsePair(ind = basicind+compind , val = basicval+compval)
                
            if i >= self.nedges+self.nnodes:
                
                basicind = []
                basicval = []
                
                tmp = self.inSet[i-self.nedges-self.nnodes]
                tmp2 = [self.indexMap(j, i-self.nedges-self.nnodes) for j in tmp]
                
                tmp_size = len(tmp2)
                
                comp1ind = [self.neqcons+k for k in tmp2]+[self.neqcons+self.nedges+k for k in tmp2]
                comp1val = [-1]*tmp_size+[1]*tmp_size
            
                tmp = self.outSet[i-self.nedges-self.nnodes]
                tmp2 = [self.indexMap(i-self.nedges-self.nnodes, j) for j in tmp]
                
                tmp_size = len(tmp2)
                
                comp2ind = [self.neqcons+k for k in tmp2]+[self.neqcons+self.nedges+k for k in tmp2]
                comp2val = [1 if i-self.nedges-self.nnodes else 0 for _ in tmp2]+[-1 if i-self.nedges-self.nnodes else 0 for _ in tmp2]
                
                
                print("FLAG")
                print(len(comp2ind))
                print(len(comp2val))
                print(comp2val)
                
                A[i] = cplex.SparsePair(ind = basicind+comp1ind+comp2ind, val = basicval+comp1val+comp2val)
                
        return A;
    
    def ConstraintRHS(self):
        b = np.ones(self.neqcons + self.nineqcons)
        
        relax_scale = 1
        for i in range(self.nineqcons):
            if i < self.nedges:
                b[self.neqcons +i] = (self.M-self.travel_times[i])*relax_scale
            else:
                b[self.neqcons +i] = (self.M+self.travel_times[i-self.nedges])*relax_scale
        con_type = "E"*self.neqcons + "L"*self.nineqcons
        
        return b, con_type
    
    def solve(self):
        problem = cplex.Cplex();
        
        problem.objective.set_sense(problem.objective.sense.minimize)
        
        self.names = ["e"+str(i) for i in range(self.nedges)]+ ["w"+str(i) for i in range(self.nnodes)] + ["T"+str(i) for i in range(self.nnodes)] 
        
        
        my_ubs = [1 for _ in range(self.nedges)] +[self.TW[0,1]-self.TW[0,0] for _ in range(self.nnodes)]+[self.M for i in range(self.nnodes)]
        
        my_lbs = [0 for _ in range(self.nedges+self.nnodes)]+[self.TW[j,0] for j in range(self.nnodes)]
        
        my_types = [problem.variables.type.binary for _ in range(self.nedges)]+[problem.variables.type.continuous for _ in range(2*self.nnodes)]
        
        my_rhs, con_type = self.ConstraintRHS()

        my_con_names = ["c"+str(i) for i in range(self.neqcons + self.nineqcons)]
        problem.linear_constraints.add( rhs = my_rhs, senses = con_type, names = my_con_names)
        
        
        
        #All of the lower bounds take the default value of 0
                
        
        self.LHS = self.ConstraintLHS()
        
        print(self.obj)
        problem.variables.add(obj = self.obj, names = self.names, ub = my_ubs, lb = my_lbs, types = my_types, columns =  self.LHS)
        problem.variables.set_upper_bounds('w0', 0)
        
        return problem
        
        


#%%
       
toy = TSPTW(toy_nnodes, toy_TimeWindows, toy_travel_times_min, toy_travel_times_max)   
print('\n\n')
print(toy.TW)
print('\n\n')

print(toy.edges)  


tmp = toy.solve();
tmp.write("current_test.lp")
    
#%%

tmp.solve()
print(tmp.solution.get_objective_value())
x = tmp.solution.get_values()

for i in x: print i

print('\n\n')

print(toy.travel_times)
