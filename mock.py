#! /usr/bin/python3
# -*- coding: latin-1 -*-

# Program to generate random clusters

# Author: Gary Mamon

from __future__ import division
import numpy as np
# import sys as sys
# import datetime
# from scipy.optimize import minimize
# from scipy.optimize import minimize_scalar
# from scipy.optimize import differential_evolution
# from scipy.optimize import fmin_tnc
from scipy import interpolate
# from astropy import units as u
# from astropy.coordinates import SkyCoord
import pandas as pd
import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
import matplotlib.patches as pat

def CheckType(X,module,varname):
    """check that type is float, int, or numpy array"""
    
    # author: Gary Mamon
    
    t = type(X)
    if t is not float and t is not np.float64 and t is not int and t is not np.ndarray:
        raise print('ERROR in ', module, ' ', varname, 
                         ' is of type ', t, 
                         ', but it must be a float or integer')
                           

def CheckTypeInt(X,module,varname):
    """check that type is int"""
    
    # author: Gary Mamon
    
    t = type(X)
    if t is not int:
        raise print('ERROR in ', module, ' ', varname,
                         ' is of type ', t, 
                         ', but it must be an integer')

def CheckTypeIntorFloat(X,module,varname):
    """check that type is int or float"""
    
    # author: Gary Mamon
    
    t = type(X)
    if t is not int and t is not float and t is not np.float64:
        raise print('ERROR in ', module, ' ', varname,
                         ' is of type ', t, 
                         ', but it must be an integer or float')

def CheckTypeBool(X,module,varname):
    """check that type is bool"""
    
    # author: Gary Mamon
    
    t = type(X)
    if t is not bool and t is not np.bool_:
        raise print('ERROR in ', module, ' ', varname,
                         ' is of type ', t, 
                         ', but it must be boolean')


def ACO(X):
    """ArcCos for |X| < 1, ArcCosh for |X| >= 1
    arg: X (float, int, or numpy array)"""

    # author: Gary Mamon

    CheckType(X,'ACO','X')
    
    # following 4 lines is to avoid warning messages
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
    """Dimensionless cluster surface density for an NFW profile. 
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2)
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon

    #  t = type(X)

    # check that input is integer or float or numpy array
    CheckType(X,'SurfaceDensity_tilde_NFW','X')
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    minX = np.min(X)
    if np.min(X) <= 0.:
        raise print('ERROR in SurfaceDensity_tilde_NFW: min(X) = ', 
                         minX, ' cannot be <= 0')

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

def SurfaceDensity_tilde_coredNFW(X):
    """Dimensionless cluster surface density for a cored-NFW profile: rho(x) ~ 1/(1+x)^3
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is radius of slope -2 
      (not the natural scale radius for which x=r/a!)
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon

    #  t = type(X)

    # check that input is integer or float or numpy array
    CheckType(X,'SurfaceDensity_tilde_coredNFW','X')
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.min(X) <= 0.:
        raise print('ERROR in SurfaceDensity_tilde_coredNFW: X cannot be <= 0')

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 2.1x10^-6)
    denom = np.log(3.) - 8./9.
    Xsquared = X*X
    Xminushalf = X-0.5
    Xsquaredtimes4minus1 = 4.*Xsquared-1.
    return (np.where(abs(Xminushalf) < 0.001, 
                     0.4 - 24./35.*Xminushalf, 
                     (8.*Xsquared + 1. 
                      - 12.*Xsquared*ACO(0.5/X) / np.sqrt(abs(Xsquaredtimes4minus1))
                      )
                     / Xsquaredtimes4minus1**2
                     ) 
                / denom
            )

def SurfaceDensity_tilde_Uniform(X):
    """Dimensionless cluster surface density for uniform model.
    arg: X = R/R_1 (positive float or array of positive floats), where R_1 is scale radius (radius where uniform model stops)
    returns: Sigma(R_1 X) (float, or array of floats)"""
    return(np.where(
            X<1.,1.,0.
            )
           )

def ProjectedNumber_tilde_NFW(X):
    """Dimensionless cluster projected number for an NFW profile. 
    arg: X = R/r_s (positive float or array of positive floats)
    returns: N_proj(X r_{-2}) / N_proj(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(X,'ProjectedNumber_tilde_NFW','X')

    # stop with error message if input values are < 0 (unphysical)
    if np.min(X) < 0.:
        raise print('ERROR in ProjectedNumber_tilde_NFW: X cannot be <= 0')

    # compute dimensionless projected number
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-7)
    denom = np.log(2.) - 0.5
    # mp = (np.where(abs(X-1.) < 0.001, 
    #                   1. - np.log(2.) + (X-1.)/3., 
    #                   ACO(1./X) / 
    #                    np.sqrt(abs(1.-X*X)) 
    #                    + np.log(0.5*X)
    #                   )
    #          / denom)
    # print (X)
    # print(mp)
    Xtmp0 = np.where(X==0,1,X)
    Xtmp1 = np.where(X==1,0,X)
    return ( np.where(X==0.,
                      0.,
                      np.where(abs(X-1.) < 0.001, 
                               1. - np.log(2.) + (X-1.)/3., 
                               ACO(1./Xtmp0) / 
                               np.sqrt(abs(1.-Xtmp1*Xtmp1)) 
                               + np.log(0.5*Xtmp0)
                      ) / denom
                )
    )

def ProjectedNumber_tilde_coredNFW(X):
    """Dimensionless cluster projected number for a cored NFW profile: rho(x) ~ 1/(1+x)^3
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is radius of slope -2 
      (not the natural scale radius for which x=r/a!)
    returns: N(X r_{-2}) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(X,'ProjectedNumber_tilde_coredNFW','X')

    # stop with error message if input values are < 0 (unphysical)
    if np.min(X) < 0.:
        raise print('ERROR in ProjectedNumber_tilde_coredNFW: X cannot be < 0')

    # compute dimensionless projected number
    #   using series expansion for |X-1/2| < 0.001 (relative accuracy better than 4.11x10^-7)
    denom = np.log(3.) - 8./9.
    Xsquared = X*X
    Xminushalf = X-0.5
    Xsquaredtimes4minus1 = 4.*Xsquared-1.
    
    return (np.where(X==0.,0.,
                     np.where(abs(Xminushalf) < 0.001, 
                     5./6. - np.log(2.) + 0.4*Xminushalf, 
                     (
                (6*Xsquared-1.)*ACO(0.5/X) / np.sqrt(abs(Xsquaredtimes4minus1))
                + np.log(X)*Xsquaredtimes4minus1 - 2.*Xsquared
                )
                     /Xsquaredtimes4minus1
                     )
                     )
            / denom
            )

def Number_tilde_NFW(x):
    """Dimensionless cluster 3D number for an NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(x,'Number_tilde_NFW','x')

    # stop with error message if input values are < 0 (unphysical)
    if np.min(x) < 0.:
        raise print('ERROR in Number_tilde_NFW: x cannot be < 0')

    return ((np.log(x+1)-x/(x+1)) / (np.log(2)-0.5))

def Number_tilde_coredNFW(x):
    """Dimensionless cluster 3D number for a cored NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(x,'Number_tilde_coredNFW','x')

    # stop with error message if input values are < 0 (unphysical)
    if np.min(x) < 0.:
        raise print('ERROR in Number_tilde_coredNFW: x cannot be < 0')

    return ((np.log(2*x+1)-2*x*(3*x+1)/(2*x+1)**2) / (np.log(3)-8./9.))

def Number_tilde_Uniform(x):
    """Dimensionless cluster 3D number for a uniform surface density profile. 
    arg: x = r/R_1 (cutoff radius)
    returns: N_3D(x R_1) / (Sigma/R_1) (float, or array of floats)"""

    # author: Gary Mamon

     # check that input is integer or float or numpy array
    CheckType(x,'Number_tilde_Uniform','x')

    # stop with error message if input values are < 0 (unphysical)
    if np.min(x) < 0.:
        raise print('ERROR in Number_tidle_Uniform: x cannot be < 0')

    return np.where(x >= 1, 0, 1 / (np.pi * np.sqrt(1-x*x)))
   
def ProjectedNumber_tilde_Uniform(X):
    """Dimensionless cluster projected number for a uniform model.
    arg: X = R/R_1 (positive float or array of positive floats), where R_1 is scale radius (radius where uniform model stops)
    returns: N(R_1 X) / N(R_1) (float, or array of floats)"""
    return(np.where(X<1, X*X, 1.))

def SurfaceDensity_tilde(X,model):
    """Dimensionless cluster surface density
    arguments:
        X = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2]  (float or array of floats)"""
    if model == 'NFW':
        return SurfaceDensity_tilde_NFW(X)
    elif model == 'coredNFW':
        return SurfaceDensity_tilde_coredNFW(X)
    elif model == 'uniform':
        return SurfaceDensityUniform(X)
    else:
        raise print('ERROR in SurfaceDensity_tilde: model = ', model, ' is not recognized')

def ProjectedNumber_tilde(X,model):
    """Dimensionless cluster projected number
    arguments:
        X = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
    returns: N_proj(R) / N(r_{-2}) (float or array of floats)"""

    # print("ProjectedNumber_tilde: a = ", a)
    # print("ProjectedNumber_tilde: scale_radius = ", scale_radius)
    if verbosity >= 4:
        print("ProjectedNumber_tilde: ellipticity=",ellipticity)
    if np.any(X < -TINY):
        raise print('ERROR in ProjectedNumber_tilde: X = ', X, ' cannot be negative')
    # elif np.any(X < TINY:
    #     return 0
    # elif X < min_R_over_rminus2:
    #     raise print('ERROR in ProjectedNumber_tilde: X = ', X, ' <= critical value = ', 
    #                 min_R_over_rminus2) 
    elif model == 'NFW':
        return ProjectedNumber_tilde_NFW(X)
    elif model == 'coredNFW':
        return ProjectedNumber_tilde_coredNFW(X)
    elif model == 'uniform':
        return ProjectedNumberUniform(X)
    else:
        raise print('ERROR in ProjectedNumber_tilde: model = ', model, ' is not recognized')

def Number_tilde(x,model):
    """Dimensionless cluster projected number
    arguments:
        x = r/r_scale (dimensionless 3D radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW' or 'Uniform' 
        r_scale = r_{-2} (NFW or coredNFW) or R_cut (Uniform)"""

    if verbosity >= 4:
        print("Number_tilde: ellipticity=",ellipticity)

    options = {
        "NFW" : Number_tilde_NFW,
        "coredNFW" : Number_tilde_coredNFW,
        "uniform" : Number_tilde_Uniform
        }

    options[model](x)
    
def guess_ellip_PA(RA,Dec,RA_cen,Dec_cen):
    dx =  (RA  - RA_cen) * np.cos(np.deg2rad(Dec_cen))
    dy =   Dec - Dec_cen   
    
    # handle crossing of RA=0
    dx = np.where(dx >= 180.,  dx - 360., dx)
    dx = np.where(dx <= -180., dx + 360., dx)

    # # guess ellipticity and PA from 2nd moments
    
    xymean = np.mean(dx*dy)
    x2mean = np.mean(dx*dx)
    y2mean = np.mean(dy*dy)
    tan2theta = 2*xymean/(x2mean-y2mean)
    # theta = 0.5*np.arctan(tan2theta) - np.pi/2.
    theta = 0.5*np.arctan(tan2theta)
    # if xymean  < 0:
    #     theta = -1*theta
    PA_pred = theta / degree
    if PA_pred < 0.:
        PA_pred = PA_pred + 180.
    A2 = (x2mean+y2mean)/2. + np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    B2 = (x2mean+y2mean)/2. - np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    ellipticity_pred = 1 - np.sqrt(B2/A2)
    return ellipticity_pred, PA_pred

def MAIN_MAIN_MAIN():
    # dummy function to better see where MAIN starts below
    return('1')

### MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN
### MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN
### MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN
### MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN
### MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN

if __name__ == '__main__':
    """Main program for PROFCL"""

    # author: Gary Mamon

    # initialization
    degree    = np.pi/180.
    models    = ('NFW', 'coredNFW', 'Uniform')
    HUGE      = 1.e30
    TINY      = 1.e-8
    verbosity = 0

    model = input("Enter model (NFW, coredNFW, Uniform): ")
    radius = float(input("Enter scale radius of model (arcmin): ")) / 60. # in deg
    str_position = input("Enter cluster RA & Dec (deg): ")
    (RA_cen,Dec_cen) = [float(x) for x in str_position.split()]
    str_e_PA = input("Enter cluster ellipticity & PA (deg): ")
    (ellipticity, PA) = [float(x) for x in str_e_PA.split()]
    background = float(input("Enter cluster background surface density (arcmin^(-2)): ")) * 3600. # in deg^-2
    N_model = int(input("Enter number of cluster galaxies: "))
    R_circle = float(input("Enter radius of circle (arcmin): ")) / 60. # in deg
    
    # background galaxies

    q = np.random.random_sample(int(round(np.pi * R_circle**2 * background)))
    R = np.sqrt(q) * R_circle
    q = np.random.random_sample(int(round(np.pi * R_circle**2 * background)))
    theta = 2 * np.pi * q
    Dec_bg = Dec_cen + R * np.cos(theta)
    RA_bg = RA_cen + R * np.sin(theta) / np.cos(RA_cen * degree)
    types_bg = np.array(['bg'] * len(RA_bg))
    print("len(RA_bg)=",len(RA_bg), "len(types_bg)=", len(types_bg))

    # model galaxies

    Xmax = R_circle/(1-ellipticity) / radius
    
    # equal spaced knots in arcsinh X
    Nknots = 100
    asinhX0 = np.linspace(0., np.arcsinh(Xmax), num=Nknots+1)
    X0 = np.sinh(asinhX0)
    ratio = ProjectedNumber_tilde(X0,model) / ProjectedNumber_tilde(Xmax,model)
    # if model == 'NFW':
    #     N0 = ProjectedNumber_tilde_NFW(Xmax)
    # elif model == 'cNFW':
    #     N0 = ProjectedNumber_tilde_coredNFW(Xmax)
    # else:
    #     raise print ('Random_radius: model = ', model, ' not recognized')

    # ratio = ProjectedNumber_tilde_NFW(X0) / N0

    # spline on knots of asinh(equal spaced)
    asinhratio = np.arcsinh(ratio)
    spline = interpolate.splrep(asinhratio,asinhX0,s=0)

    qr = np.random.random_sample(N_model)
    qtheta = np.random.random_sample(N_model)

    asinhX_spline = interpolate.splev(np.arcsinh(qr), spline, der=0, ext=2)
    R = radius * np.sinh(asinhX_spline) # in deg
    theta = 2 * np.pi * qtheta
    u = R * np.cos(theta) # in deg
    v = R * np.sin(theta) # in deg

    # compress
    # v = v * (1-ellipticity)
    v = R * np.sin(theta) * (1.-ellipticity) # in deg

    # restrict to circle
    Rnew = np.sqrt(u*u + v*v) # in deg
    condition = Rnew < R_circle
    u = u[condition]
    v = v[condition]

    # rotate
    print("before dx & dy: type(PA)=",type(PA))
    dx = u * np.sin(PA * degree) + v * np.cos(PA * degree)
    dy = -u * np.cos(PA * degree) + v * np.sin(PA * degree)

    # shift
    Dec_model = Dec_cen + dy
    RA_model = RA_cen - dx / np.cos(Dec_cen * degree)
    types_model = np.array(['model'] * len(RA_model))
    print ("type(types_model) = ", type(types_model))
    print("len(RA_model)=",len(RA_model), "len(types_model)=", len(types_model))    

    # print
    RA = RA_bg
    Dec = Dec_bg
    types = types_bg
    RA = np.append(RA,RA_model)
    Dec = np.append(Dec,Dec_model)
    types = np.append(types,types_model)
    print ("len(RA) = " ,len(RA))
    print ("len(Dec) = " ,len(Dec))
    print ("len(types) = " ,len(types))
    print ("type(RA) = " ,type(RA))
    print ("type(Dec) = " ,type(Dec))
    print ("type(types) = " ,type(types))
    # np.savetxt('mock.dat',(RA,Dec,types),fmt='%8.4f %8.4f %s\n')
    print ("RA=",RA)
    print ("Dec=",Dec)
    print("types=",types)
    tab=np.transpose((RA,Dec,types))
    print("tab=",tab)
    print("type(tab)=",type(tab))
    df = pd.DataFrame(data=tab)
    data_file = 'MockNFW' + str(N_model) + 'ellip' + str(ellipticity) + 'loga' + str(round(np.log10(radius),3)) + 'PA' + str(int(PA)) + 'center' + str(RA_cen) + '_' + str(Dec_cen) + 'With_background' + str(int(background)) + 'num0.dat'
    df.to_csv(data_file,sep=' ')
    # np.savetxt('mock2.dat',tab,fmt='%8.4f %8.4f %10s\n')

    # plot

    msize = np.sqrt(1000/len(RA))
    plt.plot(RA_bg,Dec_bg,'x',markersize=msize)
    plt.plot(RA_model,Dec_model,'ro',markersize=msize)
    plt.axis('scaled')
    # plt.tick_params(axis='x',which='both')
    # plt.grid(ls='dotted')
    plt.xlim(RA_cen+1.2*R_circle,RA_cen-1.2*R_circle)
    plt.ylim(Dec_cen-1.2*R_circle,Dec_cen+1.2*R_circle)
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    # circle for region
    x = np.linspace(-R_circle, R_circle, 500)
    y = np.sqrt((R_circle)**2-x*x)
    plt.plot(RA_cen+x,Dec_cen+y,'g')
    plt.plot(RA_cen+x,Dec_cen-y,'g')
    plt.show()
