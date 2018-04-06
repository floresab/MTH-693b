"""
Author   : Abraham Flores
File     : p737.py
Language : Python 3.6
Created  : 3/23/2018
Edited   : 3/29/2018

San Digeo State University 
MTH 693b : Computational Partial Differential Equations

Strikwerda 7.3.7 : Alternating Direction Implicit Methods

Peaceman-Rachford Alogrithim:
    
    Heat equation:
        u_t = b_1*u_x + b_2*u_y
        
    x in [0,1]
    y in [0,1]
    t in [0,1]
    
    Intial Value: 
        -Exact Solution
        
    Boundaries: 
        -Exact Solution
        
    Exact Solution:
        b1 = 2 
        b2 = 1
        u(t,x,y) = exp(1.68*t)*sin[1.2*(x-y)]*cosh[x+2y]
        
        
    dx = dy = dt = 1/10,1/20,1/40 
        
    *Demonstrate second order Accuracy
        
"""

import os,glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import diags

#Generates intial value function
def exact_foo(t,x,y):
    return np.exp(1.68*t)*np.sin(1.2*(x-y))*np.cosh(x+2*y)

#Contour Plot 
def surf_plot(x,y,U,time,title,fileLoc):
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
    plt.pcolor(X, Y, U , vmin=-5, vmax=5)
    ax.annotate("t = "+ str(round(time,3)),xy=(0,0) ,xytext=(.05,0.05),color="w")
    #legend
    clb = plt.colorbar()
    clb.set_label(r'$U(t,X,Y)$', labelpad=40, rotation=270)
    plt.xlabel('X (spatial)')
    plt.ylabel('Y (spatial)')
    plt.title(title)

    plt.savefig(fileLoc+'.png')
    plt.close()
    
"""
Makes a gif given a name and delay for each image in ms

--Assumes the images are in the figures directory
"""
def makeGif(gifName,delay):
    os.chdir('Figures')
    #Create txt file for gif command
    fileList = glob.glob('*.png') #star grabs everything,
    fileList.sort()
    #writes txt file
    file = open('FileList.txt', 'w')
    for item in fileList:
        file.write("%s\n" % item)
    file.close()

    os.system('convert -delay ' + str(delay) + ' @FileList.txt ' + gifName + '.gif')
    os.system('del FileList.txt')
    os.system('del *.png')
    os.chdir('..')
    
def ExactGIF(h,mu):
    #generate array of intial values at t = 0
    x = np.arange(0,1+h,h)
    y = np.arange(0,1+h,h)
    
    X, Y = np.meshgrid(x,y)
    title = \
  "7.3.7: Peaceman-Rachford: h: " +str(round(h,4)) + ", mu: " +str(mu)    
        
    steps = int(1.0/(mu*h**2)) + 2
    for t in range(steps):
        time = t*mu*h**2
        NEXT_ = exact_foo(time,X,Y)
        #plot 
        str_time = '0'*(5-len(str(t)))+str(t)
        outFile = "Figures\exact" + str_time
        surf_plot(x,y,NEXT_,time,title,outFile)
        
    #makeGif
    makeGif("Exact_Solution",10)
    
def Peaceman_Rachford(h,mu):
    b1 = 2
    b2 = 1
    
    C1 = mu*b1/2
    C2 = mu*b2/2
    #generate array of intial values at t = 0
    x = np.arange(0,1+h,h)
    y = np.arange(0,1+h,h)
    
    X,Y = np.meshgrid(x,y)
    
    #dimension of our matrix
    dim = len(x)
    #intialize v{i,j}
    NEXT_ = exact_foo(0,X,Y)
    
    #Generate arrays to be made into the matrices
    X_mat = C1*np.array([np.ones(dim-1),-2*np.ones(dim),np.ones(dim-1)])       
    Y_mat = C2*np.array([np.ones(dim-1),-2*np.ones(dim),np.ones(dim-1)])
    #idenity
    I = np.identity(dim)
    #Location of each diagonal
    offset = [-1,0,1]

    #Generate Matrices
    X_LEFT  =  I - diags(X_mat,offset).toarray()
    X_RIGHT =  I + diags(X_mat,offset).toarray()
    
    Y_LEFT  =  I - diags(Y_mat,offset).toarray()
    Y_RIGHT =  I + diags(Y_mat,offset).toarray()
    
    #Embed boundary conditions on matrix
    X_LEFT[0] *= 0
    X_LEFT[-1] *= 0
    X_LEFT[0][0] = 1#blah
    X_LEFT[-1][-1] = 1#blah
    
    X_RIGHT[0] *= 0
    X_RIGHT[-1] *= 0
    X_RIGHT[0][0] = 1#blah
    X_RIGHT[-1][-1] = 1#blah
    
    Y_LEFT[0] *= 0
    Y_LEFT[-1] *= 0
    Y_LEFT[0][0] = 1#blah
    Y_LEFT[-1][-1] = 1#blah
    
    Y_RIGHT[0] *= 0
    Y_RIGHT[-1] *= 0
    Y_RIGHT[0][0] = 1#blah
    Y_RIGHT[-1][-1] = 1#blah

    #plot intial foo
    title = \
  "7.3.7: Peaceman-Rachford: h: " +str(round(h,4)) + ", mu: " +str(mu)    
    outFile = "Figures\PR00000"
    surf_plot(x,y,NEXT_,0,title,outFile)
    
    steps = int(1.0/(mu*h**2)) + 2
    for t in range(1,steps):
        time = t*mu*h**2        
        #implement Scheme
        #Generate half steps
        #Half Step Boundary Conditions
        BC_left_zero = exact_foo((t-1)*mu*h**2,0,y)
        BC_right_zero = exact_foo(time,0,y)
        BC_left_one = exact_foo((t-1)*mu*h**2,1,y)
        BC_right_one = exact_foo(time,1,y)
        
        NEXT_HALF = [0]*dim
        
        NEXT_HALF[0] =\
        C2/2*np.roll(BC_left_zero,1)+(0.5-C2)*BC_left_zero +C2/2*np.roll(BC_left_zero,-1)\
        -C2/2*np.roll(BC_right_zero,1)+(0.5+C2)*BC_right_zero -C2/2*np.roll(BC_right_zero,-1)

        NEXT_HALF[-1] = \
        C2/2*np.roll(BC_left_one,1)+(0.5-C2)*BC_left_one +C2/2*np.roll(BC_left_one,-1)\
        -C2/2*np.roll(BC_right_one,1)+(0.5+C2)*BC_right_one -C2/2*np.roll(BC_right_one,-1)
        
        NEXT_T = np.transpose(NEXT_)
        for i in range(1,dim-1):
            NEXT_HALF[i]=\
    np.linalg.tensorsolve(X_LEFT,np.matmul(Y_RIGHT,np.asarray(NEXT_T[i])))

        #Generate full step
        NEXT_ = [0]*dim
        for i in range(dim):
            NEXT_[i]=\
    np.linalg.tensorsolve(Y_LEFT,np.matmul(X_RIGHT,np.asarray(NEXT_HALF[i])))
        
        #Boundary Conditions
        NEXT_[0] = exact_foo(time,0,y)
        NEXT_[-1] = exact_foo(time,1,y)
        for i in range(1,dim-1):
            NEXT_[i][0] = exact_foo(time,x[i],0)
            NEXT_[i][-1] = exact_foo(time,x[i],1)
    
        
        EXACT_ = exact_foo(time,X,Y)
        err = abs(EXACT_-NEXT_)
        #plot 
        str_time = '0'*(5-len(str(t)))+str(t)
        outFile = "Figures\PR" + str_time
        surf_plot(x,y,NEXT_,time,title,outFile)
        
    #makeGif
    makeGif("Peaceman_Rachford_h_"+str(h)+"_mu_"+str(mu),10)
    
if __name__ == "__main__": 
    Peaceman_Rachford(1/40,20)
    #ExactGIF(1/40,20)
