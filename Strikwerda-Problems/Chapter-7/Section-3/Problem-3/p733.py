"""
Author   : Abraham Flores
File     : p733.py
Language : Python 3.6
Created  : 3/23/2018
Edited   : 3/28/2018

San Digeo State University 
MTH 693b : Computational Partial Differential Equations

Strikwerda 7.3.3 : Alternating Direction Implicit Methods

Peaceman-Rachford Alogrithim:
    
    Hyperbolic equation:
        u_t + b_1*u_x + b_2*u_y = 0
        
    x in [-1,1]
    y in [-1,1]
    t in [0,1]
    
    Boundaries: 
        Along sides 
        x = -1 , or y =-1
            Exact solution
        x = 1:
            v{n+1,L,j} = v{n,L-1,j}
        y = 1:
            v{n+1,i,M} = v{n,i,M-1}
    
    Exact Solution:
        u(t,x,y) = u_0(x-b_1*t,y-b_2*t)
        
    with: 
                    |(1-2|x|)(1-2|y|) if |x| <= 0.5 and |y| <= 0.5
        u_0(x,y) =  |
                    |    0             Otherwise
"""

import os,glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import diags

#Generates intial value function
def intial_foo(x,y):
    if abs(x) <= 0.5 and abs(x) <= 0.5:
        return (1 - 2*abs(x))*(1-2*abs(y))
    return 0

#Contour Plot 
def surf_plot(x,y,U,time,title,fileLoc):
    sns.set(font_scale = 2.0)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    fig,ax = plt.subplots()
    fig.set_size_inches(14.4,9)
    
    X,Y = np.meshgrid(x,y)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # Plot the contour
    plt.pcolor(X, Y, U , vmin=-1, vmax=1)
    ax.annotate("t = "+ str(round(time,3)),xy=(-1.0,-1.0) ,xytext=(-.925,-.85))
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
    
def ExactGIF(h,lamb):
    b1 = 1
    b2 = 2
    #generate array of intial values at t = 0
    x = np.arange(-1,1+h,h)
    y = np.arange(-1,1+h,h)

    title = \
  "7.3.3: Peaceman-Rachford: h: " +str(round(h,4)) + ", lambda: " +str(lamb)    
        
    steps = int(1.0/(lamb*h)) + 1
    for t in range(steps):
        NEXT_ = []
        time = t*lamb*h
        for dx in x: 
            temp = []
            for dy in y:
                temp.append(intial_foo(dx-b1*time,dy-b2*time))   
            NEXT_.append(np.asarray(temp))
            
        #plot 
        str_time = '0'*(5-len(str(t)))+str(t)
        outFile = "Figures\exact" + str_time
        surf_plot(x,y,np.asarray(NEXT_),time,title,outFile)
        
    #makeGif
    makeGif("Exact_Solution",10)
    
def Peaceman_Rachford(h,lamb):
    b1 = -1
    b2 = -2
        
    C1 = lamb*b1/4
    C2 = lamb*b2/4
    
    #generate array of intial values at t = 0
    x = np.arange(-1,1+h,h)
    y = np.arange(-1,1+h,h)

    #dimension of our matrix
    dim = len(x)
    NEXT_ = []
    #intialize array v{n,m}
    for dx in x: 
        temp = []
        for dy in y:
            temp.append(intial_foo(dx,dy))   
        NEXT_.append(np.asarray(temp))

    #Generate arrays to be made into the matrices
    X_mat = C1*np.array([-1*np.ones(dim-1),np.ones(dim-1)])       
    Y_mat = C2*np.array([-1*np.ones(dim-1),np.ones(dim-1)])

    #idenity
    I = np.identity(dim)
    #Location of each diagonal
    offset = [-1,1]

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
  "7.3.3: Peaceman-Rachford: h: " +str(round(h,4)) + ", lambda: " +str(lamb)    
    outFile = "Figures\PR00000"
    surf_plot(x,y,NEXT_,0,title,outFile)
    
    steps = int(1.0/(lamb*h)) + 1
    for t in range(1,steps):
        time = t*lamb*h        
        #implement Scheme
        #Generate Temporary half steps
        
        NEXT_half = []
        for next_ in np.transpose(NEXT_):
            NEXT_half.append\
    (np.linalg.tensorsolve(X_LEFT,np.matmul(Y_RIGHT,np.asarray(next_))))
        time_h = time - lamb*h/2
        for i in range(dim):
            #x and  y = -1: Exact
            NEXT_half[0][i] = intial_foo(-1-b1*time_h,y[i]-b2*time_h)
            NEXT_half[i][0] = intial_foo(x[i]-b1*time_h,-1-b2*time_h)
            NEXT_half[-1][i] = intial_foo(1-b1*time_h,y[i]-b2*time_h)
            NEXT_half[i][-1] = intial_foo(x[i]-b1*time_h,1-b2*time_h)
        #Generate full step
        NEXT_ = []
        for next_half in NEXT_half:
            NEXT_.append\
    (np.linalg.tensorsolve(Y_LEFT,np.matmul(X_RIGHT,np.asarray(next_half))))
        
        NEXT_ = np.transpose(NEXT_)

        #Boundary Conditions
        for i in range(dim):
            #x and  y = -1: Exact
            NEXT_[0][i] = intial_foo(-1-b1*time,y[i]-b2*time)
            NEXT_[i][0] = intial_foo(x[i]-b1*time,-1-b2*time)
            NEXT_[-1][i] = intial_foo(1-b1*time,y[i]-b2*time)
            NEXT_[i][-1] = intial_foo(x[i]-b1*time,1-b2*time)

        #plot 
        str_time = '0'*(5-len(str(t)))+str(t)
        outFile = "Figures\PR" + str_time
        surf_plot(x,y,np.asarray(NEXT_),time,title,outFile)
        
    #makeGif
    makeGif("Peaceman_Rachford_h_"+str(h)+"_lambda_"+str(lamb),10)
    
if __name__ == "__main__": 
    Peaceman_Rachford(1/20,.5)
    #ExactGIF(1/40,.5)
