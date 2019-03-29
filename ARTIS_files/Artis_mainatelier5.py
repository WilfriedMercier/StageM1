from moduleatelier5 import *
import time
#from scipy.optimize import differential_evolution
from scipy.optimize import minimize

#-----------------------------------------------------------------------------
#MAIN
datafile=["Mock_clusterNFW1000ellip0loga0PA0center0and0No_background.dat","Mock_clusterNFW1000ellip0loga-1.5PA0center0and0No_background.dat","Mock_clusterNFW1000ellip0loga3PA0center0and0No_background.dat","Mock_clusterNFW1000ellip0loga2PA0center0and0No_background.dat","Mock_clusterNFW1000ellip0loga0.01PA0center0and0No_background.dat","Mock_clusterNFW1000ellip0loga-1PA0center0and0No_background.dat","Mock_clusterNFW1000ellip0loga1.7PA0center0and0No_background.dat","Mock_clusterNFW1000ellip0loga-0.8PA0center0and0No_background.dat","/nethome/artis/STAGEM2/Acad_NFW100000_c3_skyRA0Dec0_ellip0PA80.dat","Mock_clusterNFW1000ellip6loga-PA0center0and0No_background.dat","Mock_clusterNFW1000ellip6loga-PA30center0and0No_background.dat","Mock_clusterNFW5000ellip6loga-PA30center0and0No_background.dat","Mock_clusterNFW5000ellip6loga-PA30center0and0With_background.dat","Mock_clusterNFW1000ellip6loga-PA30center0and0With_background.dat","Mock_clusterNFW640ellip6loga-PA30center0and0With_background95.dat","Mock_clusterNFW320ellip6loga-PA30center0and0With_background95.dat","Mock_clusterNFW160ellip6loga-PA30center0and0With_background95.dat","Mock_clusterNFW80ellip6loga-PA30center0and0With_background95.dat","Mock_clusterNFW40ellip6loga-PA30center0and0With_background95.dat","Mock_clusterNFW20ellip6loga-PA30center0and0With_background95.dat","/nethome/artis/STAGEM2/Solv/mockcluster/MockNFW1280ellip5loga2.08PA50center0With_background20num10.dat"]

"""
i=20

Pos_x=np.loadtxt(datafile[i])[:,0]
Pos_y=np.loadtxt(datafile[i])[:,1]


tol=0.0001

bounds=[(-3,0),(0.0001,0.99),(-90,90),(-0.003,0.003),(-0.003,0.003),(4.5,6.5)] 
a=time.time()
#res=differential_evolution(Likelihood1,bounds,args=(Pos_x,Pos_y),tol=tol)
res=differential_evolution(Likelihood1,bounds,args=(Pos_x,Pos_y),tol=tol)
print(res,time.time()-a)


Interest=np.zeros((7,7))
for i in range(13,20):
    Pos_x=np.loadtxt(datafile[i])[:,0]
    Pos_y=np.loadtxt(datafile[i])[:,1]
    tol=0.0001

    bounds=[(-3,0),(0.0001,0.99),(-90,90),(-0.003,0.003),(-0.003,0.003),(4.5,6.5)]
    v_loga=0
    v_e=0
    v_PA=0
    v_center_x=0
    v_center_y=0
    v_sigma_cluster=0
    v_time=0
    for j in range(0,20):
        
        a=time.time()
        res=differential_evolution(Likelihood1,bounds,args=(Pos_x,Pos_y),tol=tol)
        b=time.time()-a
        
        v_time=v_time+b
        v_loga=v_loga+res.x[0]
        v_e=v_e+res.x[1]
        v_PA=v_PA+res.x[2]
        v_center_x=v_center_x+res.x[3]
        v_center_y=v_center_y+res.x[4]
        v_sigma_cluster=v_sigma_cluster+res.x[5]

    Interest[i-13,0]=v_time/20
    Interest[i-13,1]=v_loga/20
    Interest[i-13,2]=v_e/20
    Interest[i-13,3]=v_PA/20
    Interest[i-13,4]=v_center_x/20
    Interest[i-13,5]=v_center_y/20
    Interest[i-13,6]=v_sigma_cluster/20

np.savetxt("precision.dat",Interest)    
"""
"""
N_count=[1280,640,320,160,80,40,20]
logtol=np.array([-2.,-3.,-4.,-5.,-6.])
tole=10**logtol

for k in range(0,len(logtol)):
    tol=tole[k]
    for i in range(0,len(N_count)):
        precision=np.zeros((100,7))
        for j in range(0,100):
            datafile="/nethome/artis/STAGEM2/Solv/mockcluster/MockNFW"+str(N_count[i])+"ellip0loga2.08PA50center0With_background20num"+str(j)+".dat"
            Pos_x=np.loadtxt(datafile)[:,0]
            Pos_y=np.loadtxt(datafile)[:,1]

            bounds=[(-3,0),(0.0001,0.99),(-90,90),(-0.003,0.003),(-0.003,0.003),(4.5,6.5)]
        
            a=time.time()
            res=differential_evolution(Likelihood1,bounds,args=(Pos_x,Pos_y),tol=tol)
            b=time.time()-a
            precision[j,:]=[res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],b]

        np.savetxt("precision"+str(N_count[i])+"tol"+str(int(logtol[k]))+".dat",precision)     
    
        
"""

"""

N_count=[1280,640,320,160,80,40,20]
plate_radius=0.015  #-2.084772
N0=10000
Theta=2*np.pi*np.random.rand(N0)
R=plate_radius*np.sqrt(np.random.rand(N0))
X_pos=R*np.cos(Theta)
Y_pos=R*np.sin(Theta)

for i in range(0,7):
    precision=np.zeros((100,7))
    for j in range(0,100):
        datafile="/nethome/artis/STAGEM2/Solv/mockcluster/MockNFW"+str(N_count[i])+"ellip5loga2.08PA50center0With_background3num"+str(j)+".dat"
        Pos_x=np.loadtxt(datafile)[:,0]
        Pos_y=np.loadtxt(datafile)[:,1]
        tol=0.0001

        H_0=((np.pi*plate_radius**2)/N0)*np.sum(SurfaceDensity_opt(X_pos,Y_pos,-2.08,0.5,50,0,0))
        sigma_c_av=np.log10(N_count[i]/H_0)          
        bounds=[(-3,0),(0.001,0.999),(-90,90),(-0.003,0.003),(-0.003,0.003),(sigma_c_av-0.3,sigma_c_av+0.3)]
        
        a=time.time()
        res=differential_evolution(Likelihood1,bounds,args=(Pos_x,Pos_y),tol=tol)
        b=time.time()-a
        precision[j,:]=[res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],b]

    np.savetxt("precision"+str(N_count[i])+"n1.dat",precision) 

"""

datafile="/nethome/artis/STAGEM2/Solv/mockcluster/MockellipNFW640ellip5loga2.08PA50center0With_background0num10.dat"

Pos_x=np.loadtxt(datafile)[:,0]
Pos_y=np.loadtxt(datafile)[:,1]

Center=guess_center(Pos_x,Pos_y)

bounds=((-3,0),(0.001,0.999),(-90,90),(Center[0]-0.003,Center[0]+0.003),(Center[1]-0.003,Center[1]+0.003),(4,6))

r_max=max(np.sqrt((Center[0]-Pos_x)**2+(Center[1]-Pos_y)**2))
cons=({'type': 'ineq', 'fun':lambda x: (10**x[0])*(1-x[1])-r_max},
      {'type': 'ineq', 'fun':lambda x: d_circle_ellipse(x[3],x[4],Pos_x,Pos_y,x[0],x[1],x[2])})

t0=time.time()
res= minimize(Likelihood2, (-1.5,0.5,45,0,0,5),args=(Pos_x,Pos_y), method='SLSQP', bounds=bounds,constraints=cons)
t=time.time()-t0
print(res,t)
      
        

        

