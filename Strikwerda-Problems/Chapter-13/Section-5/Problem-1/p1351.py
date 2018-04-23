# -*- coding: utf-8 -*-
"""
Author   : Abraham Flores
File     : p1351.py
Language : Python 3.6
Created  : 4/22/2018
Edited   : 4/23/2018

San Digeo State University 
MTH 693b : Computational Partial Differential Equations

Strikwerda 13.5.1 : SOR method and Poisson's equation

Posson's Equation: 
    u_xx + u_yy = -2cos(x)sin(y)
    
    x = [0,1]
    y = [0,1]

    Exact Solution: 
        
        u(x,y) = cos(x)sin(y)

    h = 1/10, 1/20, 1/40
    omega = 2/(1+pi*h)
    
    Boundaries: Exact Solution
    
    Stop iterations at Tolerance of order 7 : (10^(-7))
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Successive Over Relaxation Method: 
    Solves Ax = b 
    an iteravive method with a given tolerance to achieve 
    
    Parameters: 
        grid: grid points (x)
        grid_forcing: Related forcing term in possion's equations (b)
        n : Length and Width of A
        omega : relaxation factor ([0,2])
        tol : tolerance to achieve
    
    Returns: 
        Number of Iterations Ran
"""
def SOR(grid,grid_forcing,n,omega,tol):
#Assume grid is intialized
#repeat until convergence
  iters = 0
  converged = False
  while(not converged):
      change = 0
      #loop over inner grid
      for i in range(1,n-1):
          for j in range(1,n-1):
              
              #SOR METHOD
              sigma = grid[i+1][j] + grid[i-1][j] + grid[i][j+1] +grid[i][j-1]
              point_change = omega*((grid_forcing[i][j]-sigma)/(-4)-grid[i][j])
              grid[i][j] += point_change
              
              #ADD to L2 change Norm
              change += (point_change)**2
      iters+=1       
      #check if convergence is reached
      if (np.sqrt(change) < tol):
          converged = True
          
  return iters
        
def surf_plot(x,y,U,title,fileLoc):
    sns.set(font_scale = 2.0)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    fig,ax = plt.subplots()
    fig.set_size_inches(14.4,9)
    
    X,Y = np.meshgrid(x,y)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # Plot the contour
    plt.pcolor(X, Y, U,vmin=0,vmax=1.0)
    #legend
    clb = plt.colorbar()
    clb.set_label(r'$U(t,X,Y)$', labelpad=40, rotation=270)
    plt.xlabel('X (spatial)')
    plt.ylabel('Y (spatial)')
    plt.title(title)

    plt.savefig(fileLoc+'.png')
    plt.close()
       
if __name__=="__main__":
    grid_spacing = [1/10.0,1/20.0,1/40.0]
    tol = 10**(-7)
    
    for h in grid_spacing:
        x = np.arange(0,1+h,h)
        y = np.arange(0,1+h,h)
        n = len(x)
        
        X,Y = np.meshgrid(x,y)
        
        grid_forcing = -2*np.cos(X)*np.sin(Y)*h**2
        omega = 2/(1+np.pi*h)
        #intialize Grid
        grid = np.zeros((n,n))
        grid[0] = np.cos(x)*np.sin(0)
        grid[-1] = np.cos(x)*np.sin(1)
        
        for i in range(1,n):
            grid[i][0] = np.cos(0)*np.sin(y[i])
            grid[i][-1] = np.cos(1)*np.sin(y[i])

        #RUN SOR
        iters = SOR(grid,grid_forcing,n,omega,tol)
        print(iters)
        
        #Exact Solution
        exact = np.cos(X)*np.sin(Y)
        
        #plot
        surf_plot(x,y,exact,"EXACT h: "+str(h),"Figures/EXACT_h_"+str(h))
        surf_plot(x,y,grid,"SOR h: "+str(h),"Figures/SOR_h_"+str(h))
        surf_plot(x,y,abs(grid-exact),"ERROR h: "+str(h),"Figures/ERROR_h_"+str(h))

"""
Report: 

   The method converges extremly quickly with reasonably small errors. The grid spacing 
   has a signifigant impact on the number of iterations required to reach tolerance. This
   implies that this method will break down extremly quickly for large matrices, thus 
   iterative methods are not well suited for high accuraccy 3D problems. 
"""
    
    
    
    

