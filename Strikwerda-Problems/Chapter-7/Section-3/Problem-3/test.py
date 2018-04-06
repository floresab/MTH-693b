# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:34:26 2018

@author: flore
"""
    C1 = b1*lamb/4
    C2 = b2*lamb/4
    steps = int(1.0/(lamb*h)) + 1
    X = [0]*dim
    Y = [0]*dim
    for t in range(1,steps):
        time = t*lamb*h
        
        #loop over y
        for j in range(1,dim-1):
            #Thomas Algorithim
            X[0] = blah #Boundary data
            Y[0] = blah
            #loop over x
            for i in range(1,dim-1):
                dd = NEXT_[i][j] + C2*(NEXT_[i][j+1]-NEXT_[i][j-1])
                denom = 1 + C1*(2-X[i])
                X[i] = C1/denom
                Y[i] = (dd + C1*Y[i-1])/denom
            NEXT_half[-1] = blah #bounary data at x = 1
            for i in range(dim-1,-1,-1):
                NEXT_half[i][j] = X[i]*NEXT_half[i][j] + Y[i]
                
                
        for i in range(1,dim-1):
            ###################
