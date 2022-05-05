import numpy as np
import matplotlib.pyplot as plt
from MyIVP import *
import pandas as pd
from scipy import stats


def slope(x,S,f1=None):
    temp = np.zeros(len(S))
    temp[0] = S[1] - S[2] + x
    temp[1] = 3*x**2
    temp[2] = S[1] + 1/(np.exp(x))
    
    return temp
 
Anfunc_lis   = [lambda x:-0.05*x**5 + 0.25*x**4 + x + 2 - (1/(np.exp(x))), lambda x: x**3 + 1,lambda x:0.25*x**4 + x  - 1/(np.exp(x))]

Anfunc_lis[0] = np.vectorize(Anfunc_lis[0])
Anfunc_lis[1] = np.vectorize(Anfunc_lis[1])
Anfunc_lis[2] = np.vectorize(Anfunc_lis[2])

def Plotting(X_l,Y_e,Y_2,Y_4,func,key = None,key2 = None):
    N = len(X_l)
    fig,ax = plt.subplots()
    ax.plot(X_l,Y_e,'r--*',label = 'Euler method')
    ax.plot(X_l,Y_2,'b--o',label = 'RK2 method')
    ax.plot(X_l,Y_4,'g--v',label = 'RK4 method')
    ax.plot(X_l,func(X_l))
    ax.set_title(f'Solution for Y{key} for xf = {key2} with N = {N}')
    ax.set_xlabel('Independent variable X')
    ax.set_ylabel(f'Y{key}')
    ax.legend()
    ax.grid()
    #plt.savefig(f'func{key}-x{key2}.png')
    plt.show()
    
def Error(a,b,f_an,key,N_d,method):
    N_list = np.logspace(1,N_d,base = 10,num = int(N_d))
    h = (b-a)/N_list
    E = np.zeros(len(N_list))
    P = []
    for i in range(len(N_list)):
        x,D = method(In_c,a,b,N_list[i],3,slope)
        Y_an = f_an(x)
        E[i] = max(abs(D[:,key] - Y_an))
    return np.log10(N_list),np.log10(h),np.log10(E)
    
Error = np.vectorize(Error)

In_c = [1,1,-1]  
x0 = 0 ; x_e = 1
N1 = 10
x_f = [1,2.5,5,7.5,10]
N_l = [10*i for i in x_f] #  keeping the step size constant

#d part
Y_e,Y_2,Y_4 = [],[],[]
X_e = []
for i in range(len(x_f)):
   x1_e,y_e = Euler_vec(In_c, 0, x_f[i], N_l[i], 3, slope)    
   x1_2,y_2 = RK2_vec(In_c, 0, x_f[i], N_l[i], 3, slope) 
   x1_4,y_4 = RK4_vec(In_c, 0, x_f[i],N_l[i], 3, slope) 
   X_e.append(x1_e) ; Y_e.append(y_e)
   Y_2.append(y_2) ;  Y_4.append(y_4)
   
'''
for i in range(len(x_f)):
    for j in range(len(Anfunc_lis)):
        Plotting(X_e[i],Y_e[i][:,j],Y_2[i][:,j],Y_4[i][:,j],Anfunc_lis[j],j+1,x_f[i])
'''

# e Part
M_e = []
M_Rk2,M_Rk4 = [],[]

for j in range(len(Anfunc_lis)):
    for i in x_f:
        M_er = Error(0,i,Anfunc_lis[j],j,3,Euler_vec)
        M_rk2 = Error(0,i,Anfunc_lis[j],j,3,RK2_vec)
        M_rk4 = Error(0,i,Anfunc_lis[j],j,3,RK4_vec)
        M_e.append(M_er)
        M_Rk2.append(M_rk2)
        M_Rk4.append(M_rk4)

def Plotting_2(D1,key,title):
    type1 = {0 :'N',1 : 'h'}
    p = type1[key]
    fig,ax = plt.subplots(1,3)
    plt.gca().legend(('y0','y1'))
    fig.suptitle(title)
    
    for i in range(len(x_f)):
        ax[0].plot(D1[0][key],D1[i][2],'--*',label = f'for xf = {x_f[i]}')
        ax[1].plot(D1[0][key],D1[i+5][2],'--v')
        ax[2].plot(D1[0][key],D1[i+10][2],'--o')

        for j in range(3):
            ax[j].set_title(f'function {j+1}')
            ax[j].set_xlabel(f'log({p})')
            ax[j].set_ylabel('log(E)')
        fig.legend()
    #plt.savefig(f'{title}-for-{p}.png')    
    plt.show()
'''
Plotting_2(M_e,0,'Euler method')
Plotting_2(M_Rk2,0,'RK2 method')        
Plotting_2(M_Rk4,0,'RK4 method')
Plotting_2(M_e,1,'Euler method')
Plotting_2(M_Rk2,1,'RK2 method')        
Plotting_2(M_Rk4,1,'RK4 method')
'''
def slope_err(D):
    
    slope = np.zeros(len(x_f))
    slope_2,slope_3 = slope.copy(),slope.copy() 
    for i in range(len(x_f)):
    
        slope[i], intercept, r_value, p_value, std_err = stats.linregress(D[i][1],D[i][2])
        slope_2[i], intercept, r_value, p_value, std_err = stats.linregress(D[i+5][1],D[i+5][2])
        slope_3[i], intercept, r_value, p_value, std_err = stats.linregress(D[i+10][1],D[i+10][2])
    return slope,slope_2,slope_3
 
'''
f1,f2,f3 = slope_err(M_Rk2)

data2 = {'final x':x_f,'Function 1':f1,'Function 2':f2,'function 3': f3}
df2 = pd.DataFrame(data = data2)

print(df2)
df2.to_csv('slope_rk2.csv')
'''

def diff_tolerence(In_c,a,b,m,f,N_max,tol,var,method,f1 = None):
    max_n = np.floor(np.log2(N_max))
    n_array = np.logspace(2,max_n,base=2,num = int(max_n)-1)
    H = []
    G = []
    for i in range(len(n_array)):
        x,y = method(In_c,a,b,n_array[i],m,f,f1=None)
        H.append(y)
        G.append(x)    
    for i in range(len(n_array)-1):
        Y = np.zeros((2,int(n_array[i])+1))
        Y[0] = H[i][:,var-1]
        J = H[i+1][:,var-1]
        Y[1] = J[::2]
        den = np.reciprocal(Y[1])
        ty = abs(Y[1] - Y[0])
        err =  max(np.multiply(ty,den))

        if err <= tol:
           return n_array[i+1]
    return

#Tolerence table
'''
M1 = np.zeros(len(x_f)-2)
M2,M3 = M1.copy(),M1.copy()
for i in range(len(x_f)-2):    
    M1[i] = diff_tolerence(In_c,0,x_f[i],3,slope,2**18,0.5*10**(-3),1,Euler_vec)   
    M2[i] = diff_tolerence(In_c,0,x_f[i],3,slope,2**16,0.5*10**(-3),1,RK2_vec)
    M3[i] = diff_tolerence(In_c,0,x_f[i],3,slope,2**16,0.5*10**(-3),1,RK4_vec)
     
d1 = {'final x' : x_f[:3],'Euler method':M1,'RK2 method':M2,'RK4 method':M3}     
df = pd.DataFrame(data = d1)
print(df)        
df.to_csv('tolerence.csv')
'''


#Programming part

def slope_2(x,S,f1 = None):
    temp = np.zeros(len(S))
    temp[0] = S[1]
    temp[1] = np.exp(2*x)*np.sin(x) - 2*S[0] + 2*S[1]
    
    return temp

ic = [-0.4,-0.6]

x,T = RK2_vec(ic, 0, 1, 5, 2, slope_2)

data3 = {'Xi':np.linspace(0,1,6),'Yi':T[:,0]}

df3 = pd.DataFrame(data = data3)

df3.to_csv('last.csv')

x_e,T_e = Euler_vec(ic,0,1,20,2,slope_2)
x_r2,T_r2 = RK2_vec(ic,0,1,20,2,slope_2)
x_r4,T_r4 = RK4_vec(ic,0,1,20,2,slope_2)

plt.plot(x_e,T_e[:,0],label = 'Euler method')
plt.plot(x_r2,T_r2[:,0],label = 'RK2 method')
plt.plot(x_r4,T_r4[:,0],label = 'RK4 method')
plt.title('Numerically calculated y for h = 0.05')
plt.savefig('Last.png')
plt.legend()
plt.show()