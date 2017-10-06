#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:22:03 2017

@author: Jonathan Peters
"""


import numpy as np
import cplex
import scipy.sparse as ss


#This object solves the Discretely Time Dependent Travelling Salesman Problem with Time Windows
class DTDTSPSTW(object):
    def __init__(self, nnodes, DeliveryWindows,travel_times, Time_Windows, service_time, penalty_params, name = 'discrete_problem.lp'):
        #nnodes should be an integer, equal to the number of customers, plus one more for the depot.
        #Timewindows should be a matrix with nnodes rows and 2 columns. For a given row, the two values in the matrix refer to the time after which the delivery may be made, and the time before which the delivery must be made.
        
        if np.size(DeliveryWindows,0)!= nnodes:
            raise ValueError("Invalid input, there should be a time window for each customer, and for the depot itself")
            
        if np.size(travel_times, 0)!= nnodes:
            raise ValueError("Invalid input, the first and 2nd dimensional lengths of the travel time tensor should equal the number of nodes")
        
        self.name = name
        self.tolerable_lateness = -1.0 * penalty_params[2]/(penalty_params[1] - penalty_params[0])
        self.service_time = service_time

        #record the number of nodes
        self.n = nnodes
        
        #record the delivery windows for each node
        self.DW = DeliveryWindows;
        
        #record the number of time windows which exist
        self.K = np.size(Time_Windows)       
        
        #Record the times of the day when travel times change. That is, between self.TW[j] and self.TW[j+1] one framework applies. However, after self.TW[j], then the rules changes. Overall, travel times must be piecewise linear though, so things don't change too much.
        self.Theta = Time_Windows
        
        #We introduce a dummy variable y_{i,k} which implies that node i is departed during time slot k. Due to Delivery Windows, certain values of y are restricted in advance, similar to how certain edges are restricted in advance. We record the list of y's which might take a value of 1, ignoring the others which are restricted to 0.
        self.departureSlots = self.ImportantDepartureSlots()
        
        #record the number of possible y's which exist. 
        self.kappa = len(self.slots)        
        
  
        
        
        #identify the edges which might be used, and ignore the others.
        self.edges = self.ImportantEdges(travel_times)
        
        #record the number of possible edges
        self.m = np.size(self.edges, 0)
        
        #a big number
        self.M = 100000
            

        self.combos = [[ii,jj, kk] for ii in range(self.n) for jj in self.outSet[ii] for kk in      self.departureSlots[ii]]
        self.K2 = len(self.combos)
        
        
        temp = np.zeros([self.n, self.n, self.K])
        
        count = 1
        for i in self.combos:
            temp[i[0], i[1], i[2] ]= count
            count +=1
        
        self.ijk_to_ind = [[] for _ in range(self.K)]
        for k in range(self.K):
            self.ijk_to_ind[k] = ss.csc_matrix(temp[:, :,k], dtype=int)        
        
        #print(self.departureSlots)
        
        
        
        #one variable for each edges, one service time commencement variable for each node, one waiting time variable for each node, one regret variable for each node, one decision variable for each edge x travel time window indicating during which such window is that node departed from
        self.nvars = 2*self.m+2*self.n + self.kappa-1
        self.neqcons = 3*self.n + self.m
        self.nineqcons = self.m + 3*(self.n-1)

        
        #one variable for each meaningful edge, one for the arrival time at each node, and one to measure how early the van arrived at each node.
        
        #record the matrix of refined travel times (tt for short), such that tt[e,k] refers to the time taken to travel along edge 'e' at time change point Theta[k]
        self.tt = np.array([travel_times[self.edges[i][0], self.edges[i][1], :] for i in range(self.m)])
        
        self.obj =[0]*(2*self.m+self.kappa)+ [1]+[0]*(self.n-1)+[penalty_params[0]]*(self.n-1)
        
        
    
    def indexSlotMap(self, i,j,k):
        return self.ijk_to_ind[k][i,j]-1
    
    def indexMap(self, i,j):
        return int(self.ij_to_e[i,j]-1)

    def slotMap(self, i,j):
        return self.slotInv[i,j] -1
          
    
    def ImportantDepartureSlots(self):
        possibilities = [[] for i in range(self.n)]
        

        possibilities[0]=[0]
        
        for i in range(1,self.n):
            arr = [j for j in self.DW[i, 0]-self.Theta ]
            valmin = np.min([j for j in arr if j>0])
            kmin = arr.index(valmin)
            
            arr = [j for j in self.Theta -self.tolerable_lateness-self.service_time- self.DW[i,1]]
            valmax = np.min([j for j in arr if j >0])
            kmax = arr.index(valmax)
            
            possibilities[i]=[j for j in range(kmin,kmax)]
        
        self.slots = []
        for i in range(self.n):
            for j in possibilities[i]:
                self.slots.append([i,j])
        
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
       
       Ordering= [[set([]) for _ in range(3)] for _ in range(self.n)]
       
       
       #This loop figues out the basic ordering hierarchy
       
       for node in range(1,self.n):
           
           for other in range(self.n):
               if(node!=other):
                   if other !=0:
                       tmp = np.min([travel_times[node, other,k] for k in self.departureSlots[node]])
                       tmp2 = np.min([travel_times[other,node,k] for k in self.departureSlots[other]])
                       
                       if self.DW[node, 0]+ tmp> self.DW[other,1]+self.tolerable_lateness:
                           Ordering[node][0].add(other)
                       elif self.DW[other,0]+tmp2> self.DW[node,1]+self.tolerable_lateness:
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


        for i in range(self.nvars):
            
            #first deal with the edge variables
            if i < self.m:
                
                inds = [self.edges[i][0], self.n+self.edges[i][1], self.neqcons+i]
                vals = [1,1,-self.M]
                A[i] = cplex.SparsePair(ind = inds, val = vals);
            
            #next deal with the travel time variables
            elif i < 2*self.m:
                
                inds = [3*self.n + i-self.m]+[self.neqcons+i-self.m]
                vals = [1]+[-1]
                                
                A[i] = cplex.SparsePair(ind = inds, val = vals)
                
            #next the y departure slot variables
            elif i <2*self.m +self.kappa:
                #the ik coordinates of this y variable are:
                base = 2*self.m
                yi = self.slots[i-base][0]
                yk = self.slots[i-base][1]
                
                ind1 = [yi+2*self.n]
                val1 = [1 ]
                
                ind2 = [3*self.n + self.indexMap(yi, jj) for jj in self.outSet[yi]]
                val2 = [-self.tt[self.indexMap(yi,jj), yk] for jj in self.outSet[yi]]
                
                if yi:
                    ind3 = [yi+self.neqcons+self.m -1, yi+self.neqcons+self.m +self.n-2]
                    val3 = [-self.Theta[yk+1], -self.Theta[yk]]
                else:
                    ind3 = []
                    val3 = []
                
                A[i] = cplex.SparsePair(ind = ind1+ind2+ind3, val = val1+val2+val3)
            
            #now the arrival time dummy variables
            elif i <2*self.m + self.kappa + self.n:
                base = 2*self.m + self.kappa
                ii = i-base
                
                ind1 = [self.neqcons+ self.indexMap(jj, ii) for jj in self.inSet[ii]]
                val1 = [1 for _ in ind1]
                ind2 = [self.neqcons+self.indexMap(ii,jj) for jj in self.outSet[ii]]
                val2 = [-1 if ii else 0 for _ in ind2]

                if ii:
                    ind4 = [ii +self.neqcons+self.m -1 , ii +self.neqcons+self.m+self.n-2]
                    val4 = [1 for _ in ind4]
                    
                    ind5 = [ii+self.neqcons+self.m  + 2*self.n-3]
                    val5 = [1]
                else:
                    ind4 = []
                    val4 = []
                    
                    ind5 = []
                    val5 = []
                
                A[i] = cplex.SparsePair(ind = ind1+ind2+ind4+ind5, val = val1+val2+val4+val5)
                
            elif i<2*self.m + self.kappa +2*self.n-1:
                base = 2*self.m + self.kappa + self.n

                A[i] = cplex.SparsePair(ind=[i-base+self.neqcons + self.m+ 2*(self.n-1)], val = [-1])
        return A;
    
    def ConstraintRHS(self):
        #the rhs for all equality bounds is 1
        b = np.ones(self.neqcons + self.nineqcons)
        b[3*self.n: self.neqcons] = np.zeros(self.m)
        
        for i in range(self.nineqcons):
            if i < self.m:
                #First the 'big M' constraint which relates T_i to T_j           
                b[self.neqcons +i] = (-self.M + self.service_time)
            
            elif i <self.m + 2*(self.n-1):
                #rhs for constraints which determine the y variable, that is during which time slot the node i is departed from.
                b[self.neqcons+i] = -self.service_time
            elif i <self.m + 3*(self.n-1):
                #Finally we determine lateness
                b[self.neqcons+i] = self.DW[i-(self.m + 2*(self.n-1))+1, 1]
        
        
        con_type = "E"*self.neqcons + "G"*(self.m) + "L"*(self.n-1) + "G"*(self.n-1) + "L" *(self.n-1)

        return b, con_type
    
    def formulate(self):
        #create cplex problem object
        problem = cplex.Cplex();
        
        #start the timer...
        self.timestart = problem.get_time()
        
        #declare that we aim to minimize the object
        problem.objective.set_sense(problem.objective.sense.minimize)
        
        #name each variable
        self.names = ["e_"+str(i[0])+","+str(i[1]) for i in self.edges]+ ["t_"+str(i[0])+","+str(i[1]) for i in self.edges] + ["y_"+str(i[0])+","+str(i[1]) for i in self.slots] + ["T"+str(i) for i in range(self.n)]  +["r"+str(i) for i in range(1,self.n)]
        
        #insert upper bounds
        my_ubs = [1 for _ in range(self.m)] +[self.M for _ in range(self.m)]+[1 for _ in range(self.kappa)] +[self.M for i in range(self.n)] +[self.M for _ in range(self.n-1)]
        
        #insert lower bounds
        my_lbs = [0 for _ in range(2*self.m+self.kappa)]+[self.DW[j,0] for j in range(self.n)]+[0 for _ in range(self.n-1)]
        
        #declare variable types
        my_types = [problem.variables.type.binary for _ in range(self.m)]+[problem.variables.type.continuous for _ in range(self.m)]+[problem.variables.type.binary for _ in range(self.kappa)] +[problem.variables.type.continuous for _ in range(2*self.n-1)]
        
        #compute constraint vector
        my_rhs, con_type = self.ConstraintRHS()
        
        #name constraints
        my_con_names = ["c"+str(i) for i in range(self.neqcons + self.nineqcons)]
        
        #add linear constraint details to cplex object                
        problem.linear_constraints.add( rhs = my_rhs, senses = con_type, names = my_con_names)

        #compute constaint matrix
        self.LHS = self.ConstraintLHS()
        
        #finish constructing cplex problem object
        problem.variables.add(obj = self.obj, names = self.names, ub = my_ubs, lb = my_lbs, types = my_types, columns =  self.LHS)

        #return cplex problem object.
        return problem
        
    def solve(self):
        #get cplex problem object
        p = self.formulate();
        #write cplex problem to file
        p.write(self.name)
        #solve cplex problem
        p.solve();
        self.timeend = p.get_time()
        
        #check if cplex was successful
        self.success = (p.solution.status[p.solution.get_status()]=='MIP_optimal')
        #record time taken
        self.time_taken = self.timeend - self.timestart
        
        if self.success:
            sol = p.solution.get_values()
            #record summary details
            self.edge_vals = sol[:self.m]
            self.travel_time_vals = sol[self.m:2*self.m]
            self.time_slot_vals = sol[2*self.m:2*self.m+self.kappa]
            self.arrival_time_vals = sol[2*self.m+self.kappa: 2*self.m+self.kappa+self.n]
            self.lateness_vals = sol[2*self.m+self.kappa +self.n: 2*self.m+self.kappa+2*self.n]
        return p

    def summary(self):
          if self.success:
              self.travelled_edges = [self.edges[i] for i in range(self.m) if self.edge_vals[i]]
              		              
              self.route_info = [[self.edges[i], self.travel_time_vals[i]] for i in range(self.m) if self.edge_vals[i]]
              		              
              tour_route=[0];
              
              while len(tour_route)<self.n+1:
                 tour_route.append(self.travelled_edges[tour_route[-1]][1])
                 
              print('\n\nCustomers are visited in the following sequence')
              print(tour_route)
              
              print('\n\nThis is how late each customer is arrived to at')
              print(self.lateness_vals)
              print('\n\nThe arrival time at each customer')
              print(self.arrival_time_vals)
              print('Timet taken for journey')
              print(self.time_taken)
          else:
              print('No solution exists')
