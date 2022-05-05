import numpy as np

def RK4_vec(IC,a,b,N,n,f,f1 = None):
    x = np.linspace(a,b,int(N)+1)
    h = x[1] - x[0]
    S = np.zeros((len(x),n))
    S[0,:] = IC
        
    k1 = np.zeros([len(x),n])
    k2,k3,k4,K = k1.copy(),k1.copy(),k1.copy(),k1.copy()
    for i in range(len(x)-1):
        k1[i,:] = f(x[i],S[i,:],f1)
        k2[i,:] = f(x[i]+0.5*h,S[i,:]+k1[i,:]*0.5*h,f1)
        k3[i,:] = f(x[i]+0.5*h,S[i,:]+k2[i,:]*0.5*h,f1)
        k4[i,:] = f(x[i]+ h,S[i,:]+k3[i,:]*h,f1)
        
        K[i,:] = (k1[i,:] + 2*k2[i,:] + 2*k3[i,:] + k4[i,:])/6 
        
        S[i+1,:] = S[i,:] + K[i,:]*h
        
    return x,S

def RK2_vec(IC,a,b,N,n,f,f1=None):
    x = np.linspace(a,b,int(N)+1)
    h = x[1] - x[0]
    S = np.zeros((len(x),n))
    S[0,:] = IC
    
    k1 = np.zeros([len(x),n])
    k2,K = k1.copy(),k1.copy()
    for i in range(len(x)-1):
        k1[i,:] = f(x[i],S[i,:],f1)
        k2[i,:] = f(x[i]+h,S[i,:]+k1[i,:]*h,f1)
        
        K[i,:] = (k1[i,:] + k2[i,:])/2
        
        S[i+1,:] = S[i,:] + K[i,:]*h
        
    return x,S


def Euler_vec(IC,a,b,N,n,f,f1=None):
    
    x = np.linspace(a,b,int(N)+1)
    h = x[1] - x[0]
    S = np.zeros((len(x),n))
    S[0,:] = IC
    
    K = np.zeros([len(x),n])
    for i in range(len(x)-1):
        K[i,:] = f(x[i],S[i,:],f1)
        S[i+1,:] = S[i,:] + K[i,:]*h
        
    return x,S


#question specific


if __name__ == "__main__":
    
   import matplotlib.pyplot as plt 
   def slope(x,S):
       return -S[0] + S[1] - S[1]**3

   def function(x,S,f1):
       return [*S[1:],f1(x,S)]
   
   Ic_e = [-1,0]

   last_4 = RK4_vec(Ic_e, 0, 15, 100, 2, function,slope)
   X_2,last_2= RK2_vec(Ic_e, 0, 15, 100, 2, function,slope)
   X_e,last_e = Euler_vec(Ic_e, 0, 15, 100, 2,function,slope)
   
   print(last_4[1][:,0])
   
   plt.plot(last_4[0],last_4[1],'b--v')
  # plt.plot(X_2,last_2[:,0],'r--o')
   #plt.plot(X_e,last_e[:,0],'1--')
   plt.show()


