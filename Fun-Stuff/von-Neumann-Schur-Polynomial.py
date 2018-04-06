"""
Author   : Abraham Flores
File     : von-Neumann-Schur-Polynomial.py
Language : Python 3.5
Created  : 2/17/2018
Edited   : 2/17/2018

San Digeo State University 
MTH 693b  : Computational Partial Differential Equations

Algorithim - Lecture 08 Slide 31

Start with P(z) of exact degree d, set Neumann Order = 0 

While (d > 0): 

1.   Construct P*(z)
2.   Define C_d = |P*_d(0)|^2 - |P_d(0)|^2
3.   Construct the NEW polynomial Q(z) = 1/z[P*_d(0)P_d(z) - P_d(0)P*_d(z)]
4

  If      : Q(z) == 0 then Neumann Order += 1 and P_d-1(z) := P'(d)
  Else If : The coefficent of degree d - 1 in Q(z) is 0 then the 
            polynomial is not a von Neumann polynomial - Terminate 
  Else    : P_d-1(z) := Q(z)
  
5.   d = d-1
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt

class vonNeumann: 
    #Coefficents ordered 0->d 
    def __init__(self,d_, coeff_):
        self.d  = d_
        self.coeff = coeff_
        self.star = np.conjugate(coeff_[::-1])
        self.NeumannOrder = 0
        
    def Update(self,d_,coeff_):
        self.d = d_
        self.coeff = coeff_
        self.star = np.conjugate(coeff_[::-1])
        
    def Norm(coeff):
        norm = np.zeros(2*len(coeff))
        
        for i in range(len(coeff)):
            for j in range(len(coeff)):
                norm[i+j] += np.conjuagte(coeff[i])*coeff[j]

        return norm
    
    def Derivative(self):
        if (len(self.coeff)==1):
            self.d = 1
            self.coeff = [0]
            self.star = [0]
            return
            
        for order in range(self.d+1):
            self.coeff[order] *= order
            
        self.coeff = self.coeff[1::]
        self.d -= 1
        self.star = np.conjugate(self.coeff[::-1])
    
    """
    
    """
    def vonNeumannPoly(self):
        d_ = self.d
        constraints = []
        flag = 0
        while (d_>0):
            
            constraints.append(self.star[0]*np.conjugate(self.star[0]) - self.coeff[0]*np.conjugate(self.coeff[0]))
            Q = (self.star[0]*self.coeff - self.coeff[0]*self.star)[1::]
            
            if ((Q == -1*Q).all()):#Neat way to check if all coefficents == 0
                self.Derivative()
                self.NeumannOrder += 1
            
            elif (Q[-1] == 0):#Terminate Algorithim -- Not von Neumann Poly
                return -1,None
                
            else: 
                self.Update(d_-1,Q)
                
            d_ -= 1
            
        return flag,constraints
                

   
if __name__ == '__main__':
    d_ = 2
    alpha = sympy.symbols('alpha')
    beta = sympy.symbols('beta')
    
    c1 = np.array([.5,-2,alpha])
    c2 = np.array([1,-8+2j*beta,7+4j*beta])
    c3 = np.array([1,-3,21-12*alpha-12j*beta,23-12*alpha+12j*beta])
    c4 = np.array([-1,8j/3*beta,-4j/3*beta,8j/3*beta,1])
    
    test = vonNeumann(d_,c1)
    x,y = test.vonNeumannPoly()
    
    for constraint in y:
        print(constraint)
        print("")
    print("-----------------") 
    
    aL = sympy.symbols("aL",real=True)
    T = sympy.symbols("T",real =True)
    
    tim = []
    xline = []
    for i in range(100):
        tim.append(y[1].subs(alpha,(3+2j*(i/100.0))/2))
        xline.append(i)
        
    print(np.array((tim<np.asanyarray([0]*len(tim)))))
    plt.plot(xline,tim)

        
        