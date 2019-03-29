from __future__ import division
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import random

#-----------------------------------------------------------------------------
HUGE=10.**25
TINY=10.**(-25)

def ACO(X):
  #folowing 4 lines are designed to avoid warning messages
  tmpX = np.where(X == 0, -1, X)
  tmpXbig = np.where(np.abs(X) > 1, tmpX, 1/tmpX)
  tmpXbig = np.where(tmpXbig < 0, HUGE, tmpXbig)
  tmpXsmall = 1/tmpXbig
  return ( np.where(np.abs(X) < 1,
                      np.arccos(tmpXsmall),
                      np.arccosh(tmpXbig)
                     ) 
           )

def SurfaceDensity_tilde_NFW(X):
    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    
    denom = np.log(4.) - 1.
    Xminus1 = X-1.
    Xsquaredminus1 = X*X - 1.
    return   ( np.where(abs(Xminus1) < 0.001, 
                        1./3. - 0.4*Xminus1, 
                        (1. 
                         - ACO(1./X) / np.sqrt(abs(Xsquaredminus1))
                         ) 
                        / Xsquaredminus1 
                        )
               / denom 
             )
def SurfaceDensity_opt(X,Y,loga,e,PA,center_x,center_y):
  
   PA=np.pi/2-PA/180*np.pi
   a=(10.)**loga
   b=a*(1-e)
   X0=np.sqrt((((X-center_x)*np.cos(PA)+(Y-center_y)*np.sin(PA))/a)**2+((-(X-center_x)*np.sin(PA)+(Y-center_y)*np.cos(PA))/b)**2)
   return(np.where(X0<TINY,0*X0,SurfaceDensity_tilde_NFW(X0)))


def Prob(Pos_x,Pos_y,plate_radius,X_pos,Y_pos,loga,e,PA,center_x,center_y,sigma_cluster):

   sigma_cluster=(10.)**sigma_cluster
   a=(10.)**loga
   N=len(Pos_x)
   X_pos=X_pos+center_x
   Y_pos=Y_pos+center_y
   H_0=((np.pi*plate_radius**2)/len(X_pos))*np.sum(SurfaceDensity_opt(X_pos,Y_pos,loga,e,PA,center_x,center_y))
   sigma_background=(N-sigma_cluster*H_0)/(np.pi*plate_radius**2)
   return((sigma_cluster*SurfaceDensity_opt(Pos_x,Pos_y,loga,e,PA,center_x,center_y)+sigma_background)/N)



def Likelihood(params,*args):

   loga,e,PA,center_x,center_y,sigma_cluster=params
   Pos_x,Pos_y,plate_radius,X_pos,Y_pos=args
   S=Prob(Pos_x,Pos_y,plate_radius,X_pos,Y_pos,loga,e,PA,center_x,center_y,sigma_cluster)
   #J=np.ma.masked_equal(S,0)
   #S=J.compressed()
   S=S[S>0]
   #/nethome/artis/STAGEM2/Solv/module.py:29: RuntimeWarning: divide by zero encountered in double_scalars
  #- ACO(1./X) / np.sqrt(abs(Xsquaredminus1))
   S=S[S<HUGE]
   S=[value for value in S if not math.isnan(value)]
   #S=S[~np.isnan(S)]
   return(-np.sum(np.log(S)))


def mock_maker(N_amas,r_s,e,PA,centre,bg):
  
    datafile="/nethome/artis/STAGEM2/Acad_NFW100000_c3_skyRA0Dec0_ellip0PA80.dat"
    PA0=PA
    PA=np.pi/2-PA/180*np.pi
    center_x=centre[0]
    center_y=centre[1]

    X=(10**(2.0877+r_s))*np.loadtxt(datafile)[:,0]
    Y=(10**(2.0877+r_s))*np.loadtxt(datafile)[:,1]
    Y=Y*(1-e)

    X0=X*np.cos(PA)-Y*np.sin(PA)
    Y0=X*np.sin(PA)+Y*np.cos(PA)

    del X,Y
    """
    indices=random.sample(range(0,len(X0)),1000)

    x=np.dot(np.diag(X0[indices]),np.ones((len(X0[indices]),len(X0[indices]))))
    y=np.dot(np.diag(Y0[indices]),np.ones((len(Y0[indices]),len(Y0[indices]))))

    x=x-np.transpose(x)
    y=y-np.transpose(y)

    r=np.triu(np.sqrt(x**2+y**2))
    r=r[np.where(r>0)]
    plate_radius=2*np.median(r)
    N_background=int(bg*(np.pi*(plate_radius*60)**2))

    del x,y
    """
    plate_radius=1.8356*(10**r_s)    #Empirique. Il faut y revenir!
    N_background=int(bg*(np.pi*(plate_radius*60)**2))

    
    print('Avec le rayon d echelle choisi, le rayon du plateau doit etre de '+ str(plate_radius)+' degres')    

    X1=X0[np.sqrt(X0**2+Y0**2)<plate_radius]
    Y1=Y0[np.sqrt(X0**2+Y0**2)<plate_radius]

#_______________________________________________________________________________

    N_count0=np.array([1280,640,320,160,80,40,20])
    N_count=np.array([1280,640,320,160,80,40,20])-N_background

    for i in range(0,len(N_count)):
        if N_count[i]>0:
            for j in range(0,N_amas):
    
                 N=len(X1)
                 N_elim=N-N_count[i]
                 elim=random.sample(range(0,N),N_elim)
                 X=np.delete(X1,elim)
                 Y=np.delete(Y1,elim)

                 X_back=np.array([])
                 Y_back=np.array([])

                 while len(X_back) < N_background:
                       q=np.random.rand(1)
                       p=np.random.rand(1)
                       r=np.sqrt(q)*plate_radius
                       theta=2*np.pi*p-np.pi
                       a=r*np.cos(theta)
                       b=r*np.sin(theta)
                       X_back=np.append(X_back,a)
                       Y_back=np.append(Y_back,b)

                 X=np.append(X,X_back)
                 Y=np.append(Y,Y_back)
                 X=X+center_x
                 Y=Y+center_y
                 datafile="/nethome/artis/STAGEM2/Solv/mockcluster/MockNFW"+str(N_count0[i])+"ellip"+str(int(10*e))+"loga"+str(abs(r_s))+"PA"+str(int(PA0))+"center"+str(center_x)+"With_background"+str(bg)+"num"+str(j)+".dat"
                 np.savetxt(datafile,np.c_[X,Y])
        if N_count[i] < 0:
                       print('Attention! Il y a '+str(N_background)+' galaxies dans le background. Vous ne pouvez donc pas choisir un amas a '+str(N_count0[i])+' elements')


    return (plate_radius)

  

def sigmaplusbiais(precision,value,real_value,N_galac):

   all_values=np.array(['time','r_s','e','PA','cent_x','cent_y','bg'])
   N=np.where(all_values==value)
   sigma=np.zeros(len(N_galac))
   for i in range(len(N_galac)):
     sigma[i]=np.sqrt(np.sum(precision[:,N,i]**2)/len(precision[:,N,i]))
   err_sigma=np.zeros(len(N_galac))
   for i in range(len(N_galac)):
     V=np.zeros(30)
     for j in range(0,30):
       boot=np.array(random.sample(range(N_amas),30))
       X=precision[:,N,i][boot]
       V[j]=np.sqrt(np.sum(X**2)/30-(np.sum(X)/30)**2)
     err_sigma[i]=np.sqrt(np.sum(V**2)/30-(np.sum(V)/30)**2)
    
   biais=np.zeros(7)
   for i in range(0,7):
       biais_loga[i]=np.sum(precision[:,N,i])/len(precision[:,N,i])-real_value


   err_biais=np.zeros(7)
   for i in range(0,7):
       V=np.zeros(30)
       for j in range(0,30):       
           boot=np.array(random.sample(range(0,100),30))
           X=precision[:,N,i][boot]
           V[j]=np.sum(precision[:,N,i][boot])/len(precision[:,N,i][boot])-real_value

         
       err_biais[i]=np.sqrt(np.sum(V**2)/30-(np.sum(V)/30)**2)
  

   return(np.array([sigma_loga,err_sigma,biais,err_biais]))
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
def mock_maker_params(N_amas,param_to_test,r_s,e,PA,centre,plate_radius_test,bg,N_gal):
  
    datafile="/nethome/artis/STAGEM2/Acad_NFW100000_c3_skyRA0Dec0_ellip0PA80.dat"

    if param_to_test[0]='r_s':
      PA0=PA
      PA=np.pi/2-PA/180*np.pi
      center_x=centre[0]
      center_y=centre[1]
      all_r_s=np.linspace(param_to_test[1],param_to_test[2],param_to_test[3])
      for i in range(len(all_r_s)):
         X=(10**(2.0877+all_r_s[i]))*np.loadtxt(datafile)[:,0]
         Y=(10**(2.0877+all_r_s[i]))*np.loadtxt(datafile)[:,1]
         Y=Y*(1-e)

         X0=X*np.cos(PA)-Y*np.sin(PA)
         Y0=X*np.sin(PA)+Y*np.cos(PA)
         del X,Y
         plate_radius=1.8356*(10**all_r_s[i])    #Empirique. Il faut y revenir!
         N_background=int(bg*(np.pi*(plate_radius*60)**2))
         X1=X0[np.sqrt(X0**2+Y0**2)<plate_radius]
         Y1=Y0[np.sqrt(X0**2+Y0**2)<plate_radius]
         N_count0=N_gals
         N_count=N_gals-N_background
      

      
    
    PA0=PA
    PA=np.pi/2-PA/180*np.pi
    center_x=centre[0]
    center_y=centre[1]

    X=(10**(2.0877+r_s))*np.loadtxt(datafile)[:,0]
    Y=(10**(2.0877+r_s))*np.loadtxt(datafile)[:,1]
    Y=Y*(1-e)

    X0=X*np.cos(PA)-Y*np.sin(PA)
    Y0=X*np.sin(PA)+Y*np.cos(PA)

    del X,Y
    """
    indices=random.sample(range(0,len(X0)),1000)

    x=np.dot(np.diag(X0[indices]),np.ones((len(X0[indices]),len(X0[indices]))))
    y=np.dot(np.diag(Y0[indices]),np.ones((len(Y0[indices]),len(Y0[indices]))))

    x=x-np.transpose(x)
    y=y-np.transpose(y)

    r=np.triu(np.sqrt(x**2+y**2))
    r=r[np.where(r>0)]
    plate_radius=2*np.median(r)
    N_background=int(bg*(np.pi*(plate_radius*60)**2))

    del x,y
    """
    plate_radius=1.8356*(10**r_s)    #Empirique. Il faut y revenir!
    N_background=int(bg*(np.pi*(plate_radius*60)**2))

    
    print('Avec le rayon d echelle choisi, le rayon du plateau doit etre de '+ str(plate_radius)+' degres')    

    X1=X0[np.sqrt(X0**2+Y0**2)<plate_radius]
    Y1=Y0[np.sqrt(X0**2+Y0**2)<plate_radius]

#_______________________________________________________________________________

    N_count0=np.array([1280,640,320,160,80,40,20])
    N_count=np.array([1280,640,320,160,80,40,20])-N_background

    for i in range(0,len(N_count)):
        if N_count[i]>0:
            for j in range(0,N_amas):
    
                 N=len(X1)
                 N_elim=N-N_count[i]
                 elim=random.sample(range(0,N),N_elim)
                 X=np.delete(X1,elim)
                 Y=np.delete(Y1,elim)

                 X_back=np.array([])
                 Y_back=np.array([])

                 while len(X_back) < N_background:
                       q=np.random.rand(1)
                       p=np.random.rand(1)
                       r=np.sqrt(q)*plate_radius
                       theta=2*np.pi*p-np.pi
                       a=r*np.cos(theta)
                       b=r*np.sin(theta)
                       X_back=np.append(X_back,a)
                       Y_back=np.append(Y_back,b)

                 X=np.append(X,X_back)
                 Y=np.append(Y,Y_back)
                 X=X+center_x
                 Y=Y+center_y
                 datafile="/nethome/artis/STAGEM2/Solv/mockcluster/MockNFW"+str(N_count0[i])+"ellip"+str(int(10*e))+"loga"+str(abs(r_s))+"PA"+str(int(PA0))+"center"+str(center_x)+"With_background"+str(bg)+"num"+str(j)+".dat"
                 np.savetxt(datafile,np.c_[X,Y])
        if N_count[i] < 0:
                       print('Attention! Il y a '+str(N_background)+' galaxies dans le background. Vous ne pouvez donc pas choisir un amas a '+str(N_count0[i])+' elements')


    return (plate_radius)
