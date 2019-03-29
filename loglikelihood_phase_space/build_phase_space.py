#! /usr/local/bin/python3
# -*- coding: latin-1 -*-


# Program to extract structuiral parameters (size, ellipticity, position angle, center, background) of clusters

# Author: Gary Mamon, with help from Christophe Adami, Yuba Amoura, Emanuel Artis and Eliott Mamon

from __future__ import division
import numpy as np
import sys as sys
import getopt as getopt
import datetime
import time
import os
from scipy import interpolate
from astropy import units as u
from astropy.coordinates import SkyCoord

# constants
degree = np.pi   /  180.
arcsec = degree  / 3600.

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
    """ArcCos for X < 1, ArcCosh for X >= 1
    arg: X (float, int, or numpy array)"""

    # author: Gary Mamon

    CheckType(X,'ACO','X')

    # following 4 lines is to avoid warning messages
    tmpX = np.where(X == 0, -1, X)
    tmpXbig = np.where(X > 1, tmpX, 1/tmpX)
    tmpXbig = np.where(tmpXbig < 0, HUGE, tmpXbig)
    tmpXsmall = 1/tmpXbig

    return ( np.where(X < 1,
                      np.arccos(tmpXsmall),
                      np.arccosh(tmpXbig)
                     ) 
           )

def ACOgen(X):
    """ArcCos for |X| < 1, ArcCosh for |X| >= 1
    arg: X (float, int, or numpy array)"""

    # author: Gary Mamon

    CheckType(X,'ACO','X')
    Xabs = np.abs(X)
    
    # following 4 lines is to avoid warning messages
    tmpX = np.where(X == 0, -1, X)
    tmpXbig = np.where(Xabs > 1, tmpX, 1/tmpX)
    tmpXbig = np.where(tmpXbig < 0, HUGE, tmpXbig)
    tmpXsmall = 1/tmpXbig
    return ( np.where(Xabs < 1,
                      np.arccos(tmpXsmall),
                      np.arccosh(tmpXbig)
                     ) 
           )

def SurfaceDensity_tilde_NFW(X):
    """Dimensionless cluster surface density for an NFW profile.
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2)
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon
    # check that input is integer or float or numpy array
    CheckType(X,'SurfaceDensity_tilde_NFW','X')

    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    minX = np.min(X)
    if np.min(X) <= 0.:
        print('ERROR in SurfaceDensity_tilde_NFW: min(X) = ',
                         minX, ' cannot be <= 0')
        raise ValueError

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    denom = np.log(4.) - 1.
    Xminus1 = X-1.
    Xsquaredminus1 = X*X - 1.

    return (np.where(abs(Xminus1) < 0.001,
                     1./3. - 0.4*Xminus1,
                     (1. - ACO(1./X)/np.sqrt(abs(Xsquaredminus1))) / (Xsquaredminus1)
                    )/denom)

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

    Xtmp0 = np.where(X==0,1,X)
    Xtmp1 = np.where(X==1,0,X)
    return (np.where(X==0.,
                     0.,
                     np.where(abs(X-1.) < 0.001,
                              1. - np.log(2.) + (X-1.)/3.,
                              ACO(1./Xtmp0) / np.sqrt(abs(1.-Xtmp1*Xtmp1)) + np.log(0.5*Xtmp0)
                             )/denom))

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

    return ((np.log(x+1)-x/(x+1)) / (np.log(2.)-0.5))

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
    if X < -TINY:
        raise print('ERROR in ProjectedNumber_tilde: X = ', X, ' cannot be negative')
    elif X < TINY:
        return 0
    elif X < min_R_over_rminus2:
        raise print('ERROR in ProjectedNumber_tilde: X = ', X, ' <= critical value = ', 
                    min_R_over_rminus2) 
    elif model == 'NFW':
        if ellipticity == 0:
            return ProjectedNumber_tilde_NFW(X)
        else:
            return ProjectedNumber_tilde_ellip_NFW(X,ellipticity)
    elif model == 'coredNFW':
        if ellipticity == 0:
            return ProjectedNumber_tilde_coredNFW(X)
        else:
            return ProjectedNumber_tilde_ellip_coredNFW(X,ellipticity)
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

    # author: Gary Mamon
    
    if verbosity >= 4:
        print("Number_tilde: ellipticity=",ellipticity)

    options = {
        "NFW" : Number_tilde_NFW,
        "coredNFW" : Number_tilde_coredNFW,
        "uniform" : Number_tilde_Uniform
        }

    return options[model](x)

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

def guess_center(RA, Dec):
    return np.median(RA), np.median(Dec)

def PROFCL_prob_uv(u, v, R_min, R_max, scale_radius, elliipticity, background, model):
    """probability of projected radii for given circular model parameters
    arguments: 
        u: coordinate along major axis (deg)
        v: coordinate along minor axis (deg)
        R_min: minimum allowed sky projected radius (deg)
        R_max: maximum allowed sky projected radius (deg)
        scale_radius:   radius of 3D density slope -2) [same units as R_circ, R_min and R_max]
        ellipticity
        background: uniform background [units of 1/(units of radii)^2, i.e. deg^{-2}]
        model: 'NFW' or 'coredNFW' or 'Uniform'
    returns: p(data|model)"""  

    # authors: Yuba Amoura & Gary Mamon

    if verbosity >= 4:
        print("entering prob_uv: R_min=",R_min)
    bova = 1. - ellipticity

    # uniform model
    if (model == 'Uniform'):
        return(1/(np.pi*10**(2.*log_scale_radius)*bova))

    # non-uniform models
    a   = scale_radius # radius of slope -2 in deg
    dR2 = R_max*R_max - R_min*R_min

    X_min_tmp = R_min/a
    X_max_tmp = R_max/a

    Nproj_tilde_R_min   = ProjectedNumber_tilde(X_min_tmp,model)
    Nproj_tilde_R_max   = ProjectedNumber_tilde(X_max_tmp,model)
    dNproj_tilde        = Nproj_tilde_R_max - Nproj_tilde_R_min

    # predicted number in model in sphere of radius a
    N_tot   = len(u)
    N_bg    = np.pi * background * dR2
    Nofa    = (N_tot - N_bg) / dNproj_tilde

    if verbosity >= 2:
        print("a R_min R_max = ", a, R_min, R_max)
        print("N_tot N_bg background = ", N_tot, N_bg, background)
        print("bova = ", bova)

    if Nofa < 0:
        if background_flag:
            if verbosity >= 2:
                print("Nofa = ", Nofa, " N_tot = ", N_tot, " background=", background, " dR2 = ", dR2, " R_min = ", 
                      R_min, " R_max = ", R_max, " dNproj_tilde =", dNproj_tilde, " N_p(R_max) = ", Nproj_tilde_R_max, " N_p(R_min) = ", Nproj_tilde_R_min)
            numerator = 0
        else:
            print("Nofa = ", Nofa, " N_tot = ", N_tot, " background=", background, " dR2 = ", dR2, " R_min = ", 
                      R_min, " R_max = ", R_max, " dNproj_tilde =", dNproj_tilde, " N_p(R_max) = ", Nproj_tilde_R_max, " N_p(R_min) = ", Nproj_tilde_R_min, "\n")
            raise ValueError

    R_circ      = np.sqrt(u*u + (v/bova)*(v/bova))
    radius = np.sqrt(u*u+v*v)

    numerator   = (radius)*(Nofa / (np.pi*a*a*bova) * SurfaceDensity_tilde(R_circ/a,model) + background)
    #numerator   = Nofa / (np.pi*a*a*bova) * SurfaceDensity_tilde(R_circ/a,model) + background
    denominator = N_tot

    return (numerator / denominator)

def PROFCL_LogLikelihood(params, RA, Dec, prob_membership, R_min, R_max, model):
    """general -log likelihood of cluster given galaxy positions
    arguments:
    RA, Dec: coordinates of galaxies [numpy arrays of floats]
    prob_membership: probability of membership in cluster of galaxies [numpy array of floats, between 0 and 1]
    RA_cen, Dec_cen: coordinates of cluster center [floats]
    log_scale_radius: log of scale radius of the profile (where scale radius is in degrees) [float]
    ellipticity: ellipticity of cluster (1-b/a, so that 0 = circular, and 1 = linear) [float]
    PA: position angle of cluster (degrees from North going East) [float]
    log_background: uniform background of cluster (in deg^{-2}) [float]
    model: density model (NFW or coredNFW or Uniform) [char]
    R_min, R_max:  minimum and maximum allowed projected radii on sky (degrees) [floats]
    returns: -log likelihood [float]
    assumptions: not too close to celestial pole 
    (solution for close to celestial pole [not yet implemented]: 
    1. convert to galactic coordinates, 
    2. fit,
    3. convert back to celestial coordinates for PA)"""

    # authors: Gary Mamon with help from Yuba Amoura, Christophe Adami & Eliott Mamon
    global iPass
    global scale_radius
    global RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background

    if verbosity >= 4:
        print("entering LogLikelihood: R_min = ", R_min)
    iPass = iPass + 1

    # read function arguments (parameters and extra arguments)

    RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background = params
    # RA, Dec, prob_membership, R_min, R_max, model = args

    if np.isnan(RA_cen):
        raise print('ERROR in PROFCL_LogLikelihood: RA_cen is NaN!')
    if np.isnan(Dec_cen):
        raise print('ERROR in PROFCL_LogLikelihood: Dec_cen is NaN!')

    ## checks on types of arguments
    # check that galaxy positions and probabilities are in numpy arrays
    if type(RA) is not np.ndarray:
        raise print('ERROR in PROFCL_lnlikelihood: RA must be numpy array')
    if type(Dec) is not np.ndarray:
        raise print('ERROR in PROFCL_lnlikelihood: Dec must be numpy array')
    if type(prob_membership) is not np.ndarray:
        raise print('ERROR in PROFCL_lnlikelihood: prob_membership must be numpy array')

    # check that cluster center position are floats
    CheckTypeIntorFloat(RA_cen,'PROFCL_LogLIkelihood','RA_cen')
    CheckTypeIntorFloat(Dec_cen,'PROFCL_LogLIkelihood','Dec_cen')

    # check that ellipticity and position angle are floats
    CheckTypeIntorFloat(ellipticity,'PROFCL_LogLIkelihood','ellipticity')
    CheckTypeIntorFloat(PA,'PROFCL_LogLIkelihood','PA')

    # check that min and max projected radii, and loga are floats or ints
    CheckTypeIntorFloat(R_min,'PROFCL_LogLIkelihood','R_min')
    CheckTypeIntorFloat(R_max,'PROFCL_LogLIkelihood','R_max')
    CheckTypeIntorFloat(log_scale_radius,'PROFCL_LogLIkelihood','log_scale_radius')

    # check if out of bounds
    if RA_cen < RA_cen_minallow - TINY:
        if verbosity >= 2:
            print ("RA_cen = ", RA_cen, " < RA_cen_min_allow = ", RA_cen_minallow)
        return(HUGE)
    elif RA_cen > RA_cen_maxallow + TINY:
        if verbosity >= 2:
            print ("RA_cen = ", RA_cen, " > RA_cen_max_allow = ", RA_cen_maxallow)
        return(HUGE)
    elif Dec_cen < Dec_cen_minallow - TINY:
        if verbosity >= 2:
            print ("Dec_cen = ", Dec_cen, " < Dec_cen_min_allow = ", Dec_cen_minallow)
        return(HUGE)
    elif Dec_cen > Dec_cen_maxallow + TINY:
        if verbosity >= 2:
            print ("Dec_cen = ", Dec_cen, " > Dec_cen_max_allow = ", Dec_cen_maxallow)
        return(HUGE)
    elif log_scale_radius < log_scale_radius_minallow - TINY:
        if verbosity >= 2:
            print ("log_scale_radius = ", log_scale_radius, " < log_scale_radius_min_allow = ", log_scale_radius_minallow)
        return(HUGE)
    elif log_scale_radius > log_scale_radius_maxallow + TINY:
        if verbosity >= 2:
            print ("log_scale_radius = ", log_scale_radius, " > log_scale_radius_max_allow = ", log_scale_radius_maxallow)
        return(HUGE)
    elif log_background < log_background_minallow - TINY:
        if verbosity >= 2:
            print ("log_background = ", log_background, " < log_background_min_allow = ", log_background_minallow)
        return(HUGE)
    elif log_background > log_background_maxallow + TINY:
        if verbosity >= 2:
            print ("log_background = ", log_background, " > log_background_max_allow = ", log_background_maxallow)
        return(HUGE)
    elif ellipticity < ellipticity_minallow - TINY:
        if verbosity >= 2:
            print ("ellipticity = ", ellipticity, " < ellipticity_min_allow = ", ellipticity_minallow)
        return(HUGE)
    elif ellipticity > ellipticity_maxallow + TINY:
        if verbosity >= 2:
            print ("ellipticity = ", ellipticity, " > ellipticity_max_allow = ", ellipticity_maxallow)
        return(HUGE)
    elif PA < PA_minallow - TINY:
        if verbosity >= 2:
            print ("PA = ", PA, " < PA_min_allow = ", PA_minallow)
        return(HUGE)
    elif PA > PA_maxallow + TINY:
        if verbosity >= 2:
            print ("PA = ", PA, " > PA_max_allow = ", PA_maxallow)
        return(HUGE)

    ## checks on values of arguments

    # check that RAs are between 0 and 360 degrees
    RA_min = np.min(RA)
    if RA_min < 0.:
        raise print('ERROR in PROFCL_lnlikelihood: min(RA) = ', 
                    RA_min, ' must be >= 0')        
    RA_max = np.max(RA)
    if RA_max > 360.:
        raise print('ERROR in PROFCL_lnlikelihood: max(RA) = ', 
                    RA_max, ' must be <= 360')        
    if RA_cen < 0.:
        raise print('ERROR in PROFCL_lnlikelihood: RA_cen = ', 
                    RA_cen, ' must be >= 0') 
    if RA_cen > 360.:
        raise print('ERROR in PROFCL_lnlikelihood: RA_cen = ', 
                    RA_cen, ' must be <= 360') 
    
    Dec_min = np.min(Dec)
    if Dec_min < -90.:
        raise print('ERROR in PROFCL_lnlikelihood: min(Dec) = ', 
                    Dec_min, ' must be >= -90')        
    Dec_max = np.max(Dec)
    if Dec_max > 90.:
        raise print('ERROR in PROFCL_lnlikelihood: max(Dec) = ', 
                    Dec_max, ' must be <= 90')        
    if Dec_cen < -90.:
        raise print('ERROR in PROFCL_lnlikelihood: Dec_cen = ', 
                    Dec_cen, ' must be >= -90') 
    if Dec_cen > 90.:
        raise print('ERROR in PROFCL_lnlikelihood: Dec_cen = ', 
                    Dec_cen, ' must be <= 90') 
    
    # check that ellipticity is between 0 and 1
    if ellipticity < 0. or ellipticity > 1.:
        print('ellipticity_minallow=',ellipticity_minallow)
        raise print('ERROR in PROFCL_lnlikelihood: ellipticity = ', 
                    ellipticity, ' must be between 0 and 1')     
    
    # check that model is known
    if model != 'NFW' and model != 'coredNFW' and model != 'Uniform':
        raise print('ERROR in PROFCL_lnlikelihood: model = ', 
                    model, ' is not implemented')
    
    # check that R_min > 0 for NFW (to avoid infinite surface densities)
    # or R_min >= 0 for coredNFW
    if R_min <= 0. and model == 'NFW':
        raise print('ERROR in PROFCL_lnlikelihood: R_min must be > 0 for NFW model')
    elif R_min < 0.:
        raise print('ERROR in PROFCL_lnlikelihood: R_min must be >= 0 for coredNFW model')

    # check that R_max > R_min
    if R_max <= R_min:
        raise print('ERROR in PROFCL_lnlikelihood: R_min = ', 
                    Rmin, ' must be < than R_max = ', R_max)

    # check that coordinates are not too close to Celestial Pole
    max_allowed_Dec = 80.
    Dec_abs_max = np.max(np.abs(Dec))
    if Dec_abs_max > max_allowed_Dec:
        raise print('ERROR in PROFCL_lnlikelihood: max(abs(Dec)) = ',
                    Dec_abs_max, ' too close to pole!')

    ## transform from RA,Dec to cartesian and then to projected radii

    # transform coordinate units from degrees to radians
    RA_in_rd      =      RA * degree
    Dec_in_rd     =     Dec * degree
    RA_cen_in_rd  =  RA_cen * degree
    Dec_cen_in_rd = Dec_cen * degree
    PA_in_rd      =      PA * degree

    dx =  - (RA - RA_cen) * np.cos(Dec_cen_in_rd)
    dy =     Dec - Dec_cen

    # handle crossing of RA=0
    dx = np.where(dx >= 180.,  dx - 360., dx)
    dx = np.where(dx <= -180., dx + 360., dx)

    # rotate to axes of cluster (careful: PA measures angle East from North)
    u = - dx * np.sin(PA_in_rd) + dy * np.cos(PA_in_rd)
    v = - dx * np.cos(PA_in_rd) - dy * np.sin(PA_in_rd)
    if verbosity >= 2:
        print("Log_Likelihood: len(u) = ", len(u))

    # equivalent circular projected radii (in degrees)
    R_circ_squared = u*u + v*v/(1.-ellipticity)**2.
    R_circ_squared_min = np.min(R_circ_squared)
    R_circ = np.sqrt(R_circ_squared)

    # linear variables
    scale_radius = 10.**log_scale_radius        # in deg
    background   = 10.**log_background          # in deg^{-2}

    # dimensionless radius
    X_circ = R_circ / scale_radius

    # check that circular radii are within limits of Mathematica fit
    if np.any(X_circ < min_R_over_rminus2) or np.any(X_circ > max_R_over_rminus2):
        print("log_likelihood: off limits for r_s min(X) max(X) X_min X_max= ", scale_radius, min_R_over_rminus2, max_R_over_rminus2, np.min(X_circ), np.max(X_circ))
        return (HUGE)
    elif verbosity >= 4:
        print("OK for r_s = ", scale_radius)

    # sky projected radius
    R = np.sqrt(u*u + v*v)

    ## likelihood calculation
    # restrict projected radii and membership probabilities to those within limits
    condition = np.logical_and(R >= R_min, R <= R_max)
    if verbosity >= 2:
        print("Log_Likelihood: N_tot = ", len(R[condition]))
    if len(R[condition]) == 0:
        print("ERROR in PROFCL_lnlikelihood: ",
              "no galaxies found with projected radius between ",
              R_min, " and ", R_max,
              " around RA = ", RA_cen, " and Dec = ", Dec_cen)
        return(HUGE)
    if R_max/scale_radius > max_R_over_rminus2:
        if verbosity >= 2:
            print("PROFCL_lnlikelihood: R_max a max_Rovera = ", R_max, scale_radius, max_R_over_rminus2)
        return(HUGE)

    prob = PROFCL_prob_uv(u[condition], v[condition], R_min, R_max, scale_radius, ellipticity, background, model)
    if verbosity >= 3:
        print("\n\n PROBABILITY = ", prob, "\n\n")
    if np.any(prob<=0):
        print("one p = 0, EXITING LOG_LIKELIHOOD FUNCTION")
        return(HUGE)
    if verbosity >= 2:
        print("OK")

    lnlikminus = -1*np.sum(prob_membership[condition]*np.log(prob))

    # optional print 
    if verbosity >= 4:
        print('>>> pass = ', iPass,
              '-lnlik = ',       lnlikminus,
              'RA_cen = ',           RA_cen,
              'Dec_cen = ',          Dec_cen,
              'log_scale_radius = ', log_scale_radius,
              'ellipticity = ',      ellipticity,
              'PA = ',               PA,
              'log_background = ',   log_background
              )

        np.savetxt('DEBUG/profcl_debug2.dat' + str(iPass),np.c_[RA[condition],Dec[condition],u[condition],v[condition],R_circ[condition],prob])
        ftest = open('DEBUG/profcl_debug2.dat' + str(iPass),'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:10.3f}\n".format(RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, lnlikminus))
        ftest.close()
        # np.savetxt(sys.stdout,np.c_[RA_filter,Dec_filter,u_filter,v_filter,R_circ_filter,prob])
        ftest = open(debug_file,'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:10.3f}\n".format(RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, lnlikminus))
        ftest.close()

    return lnlikminus

def PROFCL_Fit(RA, Dec, prob_membership, 
               RA_cen, Dec_cen, log_scale_radius, 
               ellipticity, PA, log_background, 
               bounds,
               R_min, R_max, model, 
               background_flag, recenter_flag, ellipticity_flag, 
               function=PROFCL_LogLikelihood,
               method='Nelder-Mead', bound_min=None, bound_max=None):
    '''Maxmimum Likelihood Estimate of 3D scale radius 
       (where slope of 3D density profile is -2) given projected data
    arguments:
        RA, Dec: coordinates of galaxies [numpy arrays of floats]
        prob_membership: probability of membership in cluster of galaxies [numpy array of floats, between 0 and 1]
        RA_cen, Dec_cen: coordinates of cluster center [floats]
        log_scale_radius: log of scale radius of the profile (where scale radius is in degrees) [float]
        ellipticity: ellipticity of cluster (1-b/a, so that 0 = circular, and 1 = linear) [float]
        PA: position angle of cluster (degrees from North going East) [float]
        log_background: uniform background of cluster (in deg^{-2}) [float]
        model: density model (NFW or coredNFW or Uniform) [char]
        R_min:     minimum allowed projected radius (float)
        R_max:     maximum allowed projected radius (float)
        background_flag: flag for background fit (True for including background in fit, False for no background)
        recenter_flag: flag for recentering (True for recentering, False for keeping DETCL center)
        elliptical_flag: flag for elliptical fit (True for elliptical fit, False for circular fit)
        function: name of function to minimize
        method:    minimization method (string)
                        'Nelder-Mead':  Simplex 
                        'Powell':       Powell's method   
                        'BFGS':         Broyden-Fletcher-Goldfarb-Shanno, using gradient
                        'L-BFGS-B':     Broyden-Fletcher-Goldfarb-Shanno, using gradient
                        'SLSQP':        Sequential Least-Squares Programming
                        'Newton-CG':    Newton's conjugate gradient
                        'Brent':        Brent method (only for univariate data)
                        'diff-evol':    Differntial Evolution
        bound_min: minimum bound on log(r_scale)
        bound_max: maximum bound on log(r_scale)'''

    # authors: Gary Mamon & Yuba Amoura, with help from Eliott Mamon

    global iPass
    if verbosity >= 4:
        print("entering Fit: R_min=",R_min)

    ##################################
    #  checks on types of arguments  #
    ##################################

    # check that input positions are in numpy arrays
    if not isinstance(RA,np.ndarray):
        raise print('ERROR in PROFCL_fit: RA must be numpy array')

    if not isinstance(Dec,np.ndarray):
        raise print('ERROR in PROFCL_fit: Dec must be numpy array')

    # check that min and max projected radii are floats or ints
    CheckTypeIntorFloat(R_min,'PROFCL_fit','R_min')
    CheckTypeIntorFloat(R_max,'PROFCL_fit','R_max')

    # check that model is a string
    t = type(model)
    if t is not str:
        raise print('ERROR in PROFCL_fit: model is ', 
                         t, ' ... it must be a str')

    # check that flags are boolean
    CheckTypeBool(background_flag,'PROFCL_fit','background_flag')
    CheckTypeBool(recenter_flag,'PROFCL_fit','recenter_flag')
    CheckTypeBool(ellipticity_flag,'PROFCL_fit','ellipticity_flag')

    # check that method is a string
    t = type(method)
    if t is not str:
        raise print('ERROR in PROFCL_fit: method is ', 
                         t, ' ... it must be a str')

    ## checks on values of arguments

    # check that R_min > 0 (to avoid infinite surface densities)
    if R_min <= 0:
        raise print('ERROR in PROFCL_fit: R_min = ', 
                         R_min, ' must be > 0')

    # check that R_max > R_min
    if R_max < R_min:
        raise print('ERROR in PROFCL_fit: R_max = ', 
                         R_max, ' must be > R_min = ', R_min )

    # check model
    if model != 'NFW' and model != 'coredNFW' and  model != 'Uniform':
        raise print('ERROR in PROFCL_fit: model = ', 
                         model, ' not recognized... must be NFW or coredNFW or Uniform')

    # function of one variable
    if np.isnan(RA_cen):
        raise print('ERROR in PROFCL_fit: RA_cen is NaN!')
    if np.isnan(Dec_cen):
        raise print('ERROR in PROFCL_fit: Dec_cen is NaN!')

    cond = np.logical_and(R_sky >= R_min, R_sky <= R_max)
    if verbosity >= 2:
        print("Fit: N_tot = ", len(R_sky[cond]))

    #############################
    #  End of checking section  #
    #############################

    params = np.array([
                       RA_cen,Dec_cen,log_scale_radius,
                       ellipticity,PA,log_background
                      ]
                     )

    maxfev = 500
    iPass = 0
    if   method == 'Nelder-Mead':
        return minimize              (PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
                        method=method, tol=tolerance, options={'maxfev':maxfev})
    elif method == 'Powell':
        return minimize              (PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'ftol':tolerance, 'maxfev':maxfev})
    elif method == 'CG' or method == 'BFGS':
        return minimize              (PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'gtol':tolerance, 'maxiter':maxfev})
    elif method == 'Newton-CG':
        return minimize              (PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'xtol':tolerance, 'maxiter':maxfev})
    elif method == 'L-BFGS-B':
        return minimize              (PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'ftol':tolerance, 'maxfun':maxfev})
    elif method == 'SLSQP':
        return minimize              (PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
                        method=method, jac=None, bounds=bounds, tol=None, options={'ftol':tolerance, 'maxiter':maxfev})
    elif method == 'TNC':
        return minimize              (PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
                        method=method, jac=None, bounds=bounds, tol=None, options={'xtol':tolerance, 'maxiter':maxfev})
    elif method == 'fmin_tnc':
        return fmin_tnc              (PROFCL_LogLikelihood, params, fprime=None, args=(RA, Dec, prob_membership, R_min, R_max, model),  approx_grad=True, bounds=bounds, epsilon=1.e-8, ftol=tolerance)
    elif method == 'Diff-Evol':
        return differential_evolution(PROFCL_LogLikelihood, bounds, args=(RA, Dec, prob_membership, R_min, R_max, model), atol=tolerance)
    else:
        raise print('ERROR in PROFCL_fit: method = ', method, ' is not yet implemented')

def Median_Separation(RA,Dec):
    """Median separation of all galaxies (in degrees)
    assumes cartesian trigonometry"""

    # author Gary Mamon & Wilfried Mercier

    separation_squared = np.empty([1,0])
    i_gal = np.arange(0,len(RA))
    i_gal2 = np.arange(0,len(RA)-1)
    Dec_median = np.median(Dec)
    cos_Dec_median = np.cos(Dec_median * degree)
    for i in i_gal2:
        # galaxies of higher index
        (RA2,Dec2) = np.array([RA,Dec])
        RA2 = RA2[i_gal > i]
        Dec2 = Dec2[i_gal > i]

        # separations
        separation_x = (RA2-RA[i]) * cos_Dec_median
        separation_y = (Dec2-Dec[i])
        sep_squared = np.array(separation_x**2. + separation_y**2.)
        separation_squared = np.append(separation_squared,sep_squared)

    return np.median(np.sqrt(separation_squared))

def PROFCL_OpenFile(file):
    """Open file into descriptor
       Usage: file_descriptor = PROFCL_OpenFile(filename)"""

    # author: Gary Mamon

    try:
        descriptor = open(file,'r')
        return(descriptor)
    except IOError:
        print('PROFCL_OpenFile: cannot open ', file)
        exit()

    
def PROFCL_ReadClusterGalaxies(cluster_galaxy_file, column_RA, column_pmem, column_z, tflag):
    """Read cluster galaxy data
    arg: file-descriptor"""

    # author: Gary Mamon

    if verbosity >= 1:
        print("reading", cluster_galaxy_file, "...")

    f_gal = PROFCL_OpenFile(cluster_galaxy_file)
    if tflag == "M":
        tab_galaxies = np.genfromtxt(f_gal, skip_header=1, dtype='str')
    else:
        tab_galaxies = np.loadtxt(f_gal)

    f_gal.close()
    num_columns = tab_galaxies.shape[1]

    if verbosity >= 3:
        print("number of columns found is ", num_columns)
    if column_RA > num_columns or column_z > num_columns or column_pmem > num_columns:
        print("PROFCL_ReadClusterGalaxies: data file =",cluster_galaxy_file, " with ", num_columns, " columns")
        raise print("column_RA=",column_RA,"column_pmem=",column_pmem,"column_z=",column_z)
    
    RA = tab_galaxies[:,column_RA-1].astype(float)
    Dec = tab_galaxies[:,column_RA].astype(float)

    if column_pmem > 0:
        prob_membership = tab_galaxies[:,column_pmem-1]
    else:
        prob_membership = 0.*RA + 1.

    if column_z > 0:
        z = tab_galaxies[:,column_z-1]
    else:
        z = 0*RA - 1.

    if verbosity >= 1:
        print("found", len(RA), 'galaxies')

    return (RA, Dec, prob_membership, z)

def PROFCL_ReadClusterCenters(clusters_file,column_RA0):
    """Read cluster centers
    arg: file-descriptor"""

    # author: Gary Mamon

    if verbosity >= 1:
        print("reading", clusters_file, "...")

    f_clusters = PROFCL_OpenFile(clusters_file)
    tab_clusters = np.loadtxt(f_clusters)
    all_ID = tab_clusters[:,0]
    all_RA0 = tab_clusters[:,column_RA0-1]
    all_Dec0 = tab_clusters[:,column_RA0]

    return(all_ID, all_RA0, all_Dec0)

def convertMethod(method):
    """Convert method abbreviations
    arg: abbreviation
    returns: method full name"""

    # author: Gary Mamon

    if method == 'NM' or method == 'nm' or method == 'nelder-mead':
        method2 = 'Nelder-Mead'
    elif method == 'powell':
        method2 = 'Powell'
    elif method == 'cg':
        method2 = 'CG'
    elif method == 'bfgs':
        method2 = 'BFGS'
    elif method == 'newton-cg' or method == 'ncg' or method == 'NCG':
        method2 = 'Newton-CG'
    elif method == 'l-bfgs-b' or method == 'LBB' or method == 'lbb':
        method2 = 'L-BFGS-B'
    elif method == 'slsqp':
        method2 = 'SLSQP'
    elif method == 'tnc':
        method2 = 'TNC'
    elif method == 'fmin_TNC':
        method2 = 'fmin_tnc'
    elif method == 'de' or method == 'diffevol' or method == 'DiffEvol':
        method2 = 'Diff-Evol'
    elif method == 't':
        method2 = 'test'
    else:
        method2 = method
    return(method2)

def convertModel(model):
    """Convert model abbreviations
    arg: abbreviation
    returns: model name"""

    # author: Gary Mamon

    if model == 'n' or model == 'N' or model == 'nfw':
        model2 = 'NFW'
    elif model == 'c' or model == 'C' or model == 'cnfw':
        model2 = 'cNFW'
    elif model == 'u' or model == 'U' or model == 'uniform':
        model2 = 'Uniform'
    else:
        model2 = model
    return(model2)

def ProjectedNumber_tilde_ellip_NFW(X,ellipticity):
    """Dimensionless projected mass for non-circular NFW models
    args:
    X = R_sky/r_{-2}  (positive float or array of positive floats), where r_{-2} is radius of slope -2
    ellipticity = 1-b/a (0 for circular)
    returns: N(X r_{-2}) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon
    # source: ~gam/EUCLID/CLUST/PROCL/denomElliptical.nb

    # N_points = int(1e5)
    # N_knots  = int(1e4)
    N_knots  = 100
    if verbosity >= 5:
        # print ("ProjectedNUmber_tilde_ellip_NFW: verbosity=",verbosity)
        print('using 2D polynomial for ellipticity = ', ellipticity)
    lX = np.log10(X)
    e = ellipticity
    if N_points == 0:
        # analytical approximation
        lNprojtilde = (
            0.2002159160168399 + 0.23482973073606245*e + 
            0.028443816507694702*e**2 + 3.0346488960850246*e**3 - 
            32.92395216847275*e**4 + 155.31214454203342*e**5 - 
            384.7956580823655*e**6 + 524.185430033757*e**7 - 
            372.1186576278279*e**8 + 107.73575331855518*e**9 + 
            1.084957208707404*lX - 0.15550331288872482*e*lX + 
            0.19686182416407058*e**2*lX - 4.369613060146462*e**3*lX + 
            25.786051119038408*e**4*lX - 76.01463442935163*e**5*lX + 
            118.12160576401868*e**6*lX - 93.01512548879035*e**7*lX + 
            29.20821583872627*e**8*lX - 0.350953887814871*lX**2 - 
            0.024303352180603605*e*lX**2 - 0.1287797997529538*e**2*lX**2 + 
            2.173015888479342*e**3*lX**2 - 8.937397688350035*e**4*lX**2 + 
            17.468998705433673*e**5*lX**2 - 16.251979189333717*e**6*lX**2 + 
            5.885069528670919*e**7*lX**2 - 0.012941807861877342*lX**3 + 
            0.03797412877170318*e*lX**3 + 0.10378160335237464*e**2*lX**3 - 
            0.5184579855114362*e**3*lX**3 + 1.2834168703734363*e**4*lX**3 - 
            1.4165466726424798*e**5*lX**3 + 0.5653714436995129*e**6*lX**3 + 
            0.04317093844555722*lX**4 + 0.013619786789711666*e*lX**4 - 
            0.07157446386996426*e**2*lX**4 + 0.12635271935992576*e**3*lX**4 - 
            0.1623323869598711*e**4*lX**4 + 0.06594832410639553*e**5*lX**4 + 
            0.0005189446937787153*lX**5 - 0.012170985301529685*e*lX**5 + 
            0.0078104820069108665*e**2*lX**5 - 
            0.012168623850966566*e**3*lX**5 + 0.01120734375450095*e**4*lX**5 - 
            0.0063164825849164104*lX**6 - 0.0003229562648197668*e*lX**6 + 
            0.004797249087277705*e**2*lX**6 - 
            0.0006839516501486773*e**3*lX**6 + 0.0005190658690337241*lX**7 + 
            0.001323550523203948*e*lX**7 - 0.0009722709478153854*e**2*lX**7 + 
            0.0004615622537881461*lX**8 - 0.0002037464879060379*e*lX**8 - 
            0.00008236148148039739*lX**9)
        Nprojtilde = 10**lNprojtilde
    else:
        # draw N_points points in circle of radius R_region/(b/a)
        if X * 10.**log_scale_radius < R_max:
            Nprojtilde = ProjectedNumber_tilde_NFW(X)
        else:
            print(log_scale_radius, scale_radius, ellipticity)
            (x_rand,y_rand) = Random_xy(X*10.**log_scale_radius/(1-ellipticity),model,N_points,N_knots,ellipticity,PA)
            R_rand_sq = (x_rand-RA_cen_init)**2 + (y_rand-Dec_cen_init)**2
            R_rand_sq_in_circle = R_rand_sq[R_rand_sq < R_max*R_max]
            # R_rand_sq_in_scale_radius = R_rand_sq[R_rand_sq < 10.**(2*log_scale_radius)]
            # Nprojtilde = len(R_rand_sq_in_scale_radius) / len(R_rand_sq_in_circle)
            Nprojtilde = len(R_rand_sq_in_circle) / N_points * ProjectedNumber_tilde_NFW(X)
            if verbosity >=3:
                print("fraction in circle = ", len(R_rand_sq_in_circle) / N_point)
                # print(len(R_rand_sq_in_scale_radius), len(R_rand_sq_in_circle))

    return (Nprojtilde)


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

    #Default input values
    verbosity, model_test_tmp, method, Rmaxovervir, tolerance, min_members, do_ellipticity = 0, "NFW", "test", -1., 0.0001, 1, "e"
    test_flag, median_flag, do_center, do_background, id_cluster_min_max, N_points         = "M", "n", "n", "bg", "0 0", 0
    cluster_richness, cluster_ellipticity, cluster_PA, background, fits_flag               = 1280, 0.4, 10, 3000, False
    cluster_loga, cluster_cen_RA, cluster_cen_Dec, rank_list, rank_list_asstr              = -2.0, 10.0, 20.0, np.asarray([1, 1]), '[1 1]'
    data_galaxies_dir, data_galaxies_prefix, data_clusters_dir, data_clusters_prefix       = "../Mocks_Mamon_test", "none", "/tmp_mnt/broque/poubelle1/tmp/gam/EUCLID/CFC/CFC4/AMICO", "AMICO_CFC4_Phase1_de"
    galaxies_positions_dir, galaxies_positions_prefix                                      = "/nethome/gam/MOCKS/EUCLID_CFC/ClusterChallenge_IV/DURHAM", "Durham_Photreal_300deg2_blind"
    output_file, detection_file, association_file                                          = "test.output", "PROFCL_detection.fits", "PROFCL_association.fits"
    saveinput_file, loadinput_file                                                         = "PROFCL_input.dat", "PROFCL_input.dat"
    auto = "auto"
    iPass = 0

    # initialization
    version = "1.15"
    vdate = "24 April 2018"
    version_date   = version + ' (' + vdate + ')'
    falsetruearr = np.array([False, True])
    falsearr = np.array([False])
    truearr = np.array([True])
    degree    = np.pi/180.
    models    = ('NFW', 'coredNFW', 'Uniform')
    methods_dict = {'brent' : 'brent', 'Nelder-Mead' : 'nm', 'BFGS' : 'bfgs', 'L-BFGS-B' : 'lbb', 'TNC' : 'tnc', 'fmin_tnc' : 'fmin', 'Diff-Evol' : 'de', 'test' : 't'}
    HUGE      = 1.e30
    TINY      = 1.e-8
    RA_crit   =  2 # critical RA for RA coordinate transformation (deg)
    Dec_crit  = 80 # critical Dec for frame transformation (deg)
    R_min_over_maxR = 0.02  # minimum allowed projected radius over maximum observed radius (from DETCL center)
    R_max_over_maxR = 0.8     # maximum allowed projected radius over maximum observed radius (from DETCL center)
    min_R_over_rminus2 = 0.01  # minimal ratio R/r_minus2 where fit for projected mass is performed in advance
    max_R_over_rminus2 = 1000.  # maximal ratio R/r_minus2 where fit for projected mass is performed in advance
    column_RA = 1
    column_RA_cen = 10
    column_pmem = 0
    column_z = 3
    column_z_cen = 2
    debug_file = 'DEBUG/PROFCL.debug'
    fmt  = "{:19} {:5} {:40} {:12} {:.1e} {:d} {:d} {:d} {:5} {:8} {:8.4f} {:8.4f} {:10.3f} {:5.3f} {:3.0f} {:6.2f} {:6.1f} {:15.7G} {:15.7G} {:5d} {:8.4f} {:8.4f} {:10.3f} {:5.3f} {:3.0f} {:6.2f} {:8.3f}"
    fmt_tmp = "{:19} {:5} {:40} {:12} {:.1e} {:d} {:d} {:d} {:5} {:8} {:8.4f}"
    fmt2 = fmt + "\n"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    model_test = convertModel(model_test_tmp)
    # questions for cluster file
    if test_flag == "d" or test_flag == "f":
        cluster_richness    = -1
        cluster_ellipticity = -1
        cluster_PA          = -1
        background          = -1
        Rmaxovervir         = -1
        mock_number         = -1

    elif test_flag in ("a", "A", "M"):
        cluster_ellipticity_10 = str(int(10*float(cluster_ellipticity)))

    elif test_flag == "a":
        if float(background) > 0.:
            galaxy_file = 'Acad_' + model_test + cluster_richness + '_c3_skyRA15Dec10_ellip' + cluster_ellipticity_10 + 'PA' + cluster_PA + '_bg' + background + '_Rmax' + Rmaxovervir + '.dat'
        else:
            galaxy_file = 'Acad_' + model_test + cluster_richness + '_c3_skyRA15Dec10_ellip' + cluster_ellipticity_10 + 'PA' + cluster_PA + '.dat'
        id_cluster_min = 1
        id_cluster_max = 1
        column_z = 0

    elif test_flag in ("A", "M"):
        id_cluster_min, id_cluster_max = map(int, id_cluster_min_max.split())
        column_z = 0

    elif test_flag == "f":
        data_galaxies_prefix = "galaxies_inhalo_clusterlM14"
        data_clusters_dir = "."
        data_clusters_prefix = "lM14_lookup"
        id_cluster_min, id_cluster_max = map(int, id_cluster_min_max.split())

    else:
        raise print("PROF-CL (MAIN): cannot recognize choice = ", test_flag)

    if do_center == 'c':
        recenter_flags = truearr
    elif do_center == 'n':
        recenter_flags = falsearr
    else:
        recenter_flags = falsetruearr

    if do_ellipticity == 'e':
        ellipticity_flags = truearr
    elif do_ellipticity == 'n':
        ellipticity_flags = falsearr
    else:
        ellipticity_flags = falsetruearr

    N_points = int(N_points)
    # integrate_model_in_region = 'm'
    if N_points < 10000:
        if (do_ellipticity == 'e' and do_center == 'n') and N_points != 0:
            raise print("The number should be 0 or >= 10000")
        elif do_center == 'c':
            raise print("N_points should be >= 10000")

    if do_background == 'bg':
        background_flags = truearr
    elif do_background == 'n':
        background_flags = falsearr
    else:
        background_flags = falsetruearr

    # query minimization method:
    found  = 0
    while found == 0:
        t = type(method)
        if t is not str:
            raise print('ERROR in PROFCL_Main: method is of type ', 
                         t, ', but must be str')
        method = convertMethod(method)
        for meth in list(methods_dict.values()) + list(methods_dict.keys()):
            if method == meth:
                found = 1
                break
        if method == 'test':
            found = 1
        if found == 0:
            print(method, ' not recognized')
            raise print("method must be one among: ", list(methods_dict.values()) + list(methods_dict.keys()))

    tolerance = float(tolerance)

    cluster_range = [0]
    for id_cluster in cluster_range:
        print ("\nid_cluster = ", id_cluster)

        # convert cluster ID to cluster galaxy file name
        # TBD

        # galaxy file

        if test_flag == "f":
            galaxy_file0 = data_galaxies_prefix + '_' + str(id_cluster) + '.dat'
            galaxy_file = data_galaxies_dir + '/' + galaxy_file0
        elif test_flag == "d":
            galaxy_file = str(id_cluster) + "/" + galaxy_file0
        elif test_flag == "a":
            if float(background) > 0.:
                galaxy_file0 = 'Acad_' + str(model_test) + str(cluster_richness) + '_c3_skyRA15Dec10_ellip' + str(cluster_ellipticity_10) + 'PA' + str(cluster_PA) + '_bg' + str(background) + '_Rmax' + str(Rmaxovervir) + '.dat'
            else:
                galaxy_file0 = 'Acad_' + str(model_test) + str(cluster_richness) + '_c3_skyRA15Dec10_ellip' + str(cluster_ellipticity_10) + 'PA' + str(cluster_PA) + '.dat'
        elif test_flag == "A":
            galaxy_file0 = 'MockNFW' + str(cluster_richness) + 'ellip' + str(cluster_ellipticity_10) + 'loga2.08PA50center0With_background20num' +  str(id_cluster) + '.dat'
        elif test_flag == "M":
            galaxy_file0 = 'MockNFW' + str(cluster_richness) + 'ellip' + str(cluster_ellipticity) + 'loga' + str(cluster_loga) + 'PA' + str(cluster_PA) + 'center' + str(cluster_cen_RA) + "_" + str(cluster_cen_Dec) + "With_background" + str(background) + 'num' + str(id_cluster) + '.dat'

        if test_flag in ("a", "A", "M"):
            galaxy_file = data_galaxies_dir + '/' + galaxy_file0

        if verbosity >= 1:
            print("galaxy_file = ", galaxy_file)

        # cluster center
        if test_flag == "d" or test_flag == "f":
            RA_cen_init0 = np.asscalar(RA_cen_all[ID==id_cluster])
            Dec_cen_init0 = np.asscalar(Dec_cen_all[ID==id_cluster])
            if verbosity >= 1:
                print("RA_cen_init0=",RA_cen_init0,"Dec_cen_init0=",Dec_cen_init0)

        # read galaxy data (coordinates in degrees)
        if test_flag in ("A", "a", "f"):
            (RA, Dec, prob_membership, z) = PROFCL_ReadClusterGalaxies(galaxy_file, column_RA, column_pmem, column_z, test_flag)
        elif test_flag == "M":
            (RA, Dec, prob_membership, bg_or_model) = PROFCL_ReadClusterGalaxies(galaxy_file, 2, 0, 4, test_flag)
            bg_RA   = RA[bg_or_model == "bg"] #separate background and NFW-like data (useful for debugging)
            bg_Dec  = Dec[bg_or_model == "bg"]
            mod_RA  = RA[bg_or_model == "model"]
            mod_Dec = Dec[bg_or_model == "model"]

        elif test_flag == "d":
            RA              = RA_gal[ID_c == id_cluster]
            Dec             = Dec_gal[ID_c == id_cluster]
            prob_membership = p_c[ID_c == id_cluster]

        N_data = len(RA)
        if verbosity >= 1:
            print("*** cluster ", id_cluster, "RA_cen=", RA_cen_init0, "Dec_cen=",Dec_cen_init0,"N_data =",N_data)

        # skip cluster if too few members
        if N_data < min_members:
            if verbosity >= 1:
                print("skipping cluster ", id_cluster, " which only has ", N_data, " members")
            continue

        # if median separation was activated
        if median_flag == "y" or median_flag == "o":
            #compute computation time
            start = time.clock()
            # median separation (arcmin)
            if verbosity >= 1 and N_data > 100:
                print("computing median separation ...")
            log_median_separation = np.log10(Median_Separation(RA,Dec))
            time_taken = time.clock() - start
            if verbosity >= 1:
                print("log_median_separation (deg)=",log_median_separation)

        #predict center position
        RA_cen_init0, Dec_cen_init0 = guess_center(RA, Dec)
        if verbosity >= 1:
            print("\nPredicted center position before shifting : ", RA_cen_init0, Dec_cen_init0)

        # shift RA and RA_cen by 180 degrees if close to 0
        Dec_cen_init = Dec_cen_init0
        RA_cen_init  = RA_cen_init0
        if RA_cen_init < RA_crit or RA_cen_init > 360.-RA_crit:
            RA_shift_flag = True
            RA_cen_init += 180.
            if RA_cen_init > 360:
                RA_cen_init -= 360.
            RA += 180.
            RA = np.where(RA > 360., RA-360., RA)
            if verbosity >= 1:
                print("close to RA=0: shifting RA by 180 degrees")
        else:
            RA_shift_flag = False

        # convert to galactic if near pole
        if np.abs(Dec_cen_init) > Dec_crit:
            eq2gal_flag = True
            coords_cen = SkyCoord(RA_cen_init,Dec_cen_init,frame='icrs',unit='deg')
            coords_galactic = coords_cen.galactic
            RA_cen_init = coords_galactic.l.deg
            Dec_cen_init= coords_galactic.b.deg

            coords = SkyCoord(RA,Dec,frame='icrs',unit='deg')
            coords_galactic = coords.galactic
            RA = coords_galactic.l.deg
            Dec = coords_galactic.b.deg
            if verbosity >= 3:
                print("shifting position to galactic coordinates")
        else:
            eq2gal_flag = False

        if verbosity >= 1:
            print("Predicted center position after shifting : RA = %", RA_cen_init, Dec_cen_init, "\n")

        # min and max projected angular radii from maximum separation from center (in degrees)
        cosR_sky =  (np.sin(Dec*degree)*np.sin(Dec_cen_init*degree)
                     + np.cos(Dec*degree)*np.cos(Dec_cen_init*degree)*np.cos((RA-RA_cen_init)*degree))
        if np.max(cosR_sky > 1):
            print("max(cos(R_sky)) - 1 = ", np.max(cosR_sky)-1)
        cosR_sky_tmp = np.where(cosR_sky > 1, 1, cosR_sky)
        R_sky = np.where(cosR_sky > 1, 0, np.arccos(cosR_sky_tmp) / degree)
        maxR = np.max(R_sky)
        R_min = R_min_over_maxR * maxR

        if verbosity>=1:
            print("MAIN: R_min set to ",R_min)
        R_max = R_max_over_maxR * maxR
        if test_flag == "a":
            R_max = 0.04932   # 2 r_vir

        # maximum background surface density to prevent negative numerator in galaxy probability
        condition = np.logical_and(R_sky >= R_min, R_sky <= R_max)
        log_background_maxallow_default = np.log10(len(R_sky[condition])/(2*np.pi*(R_max*R_max-R_min*R_min)))    ########### Guess it's not possible to have more background (noise) than data (signal), hence max put to half total surface density of galaxies #########$

        # restrict projected radii and membership probabilities to those within limits
        RA = RA[condition]
        Dec = Dec[condition]
        prob_membership = prob_membership[condition]
        if verbosity >= 1:
            print(len(RA), " galaxies within R_min = ", R_min, " and R_max = ", R_max) 

        if len(RA) == 0:
           print("ERROR in PROFCL_lnlikelihood: ",
              "no galaxies found with projected radius between ",
              R_min, " and ", R_max,
              " around RA = ", RA_cen_init, " and Dec = ", Dec_cen_init)

        # guess ellipticity and PA using 2nd moments of distribution
        ellipticity_pred, PA_pred = guess_ellip_PA(RA,Dec,RA_cen_init,Dec_cen_init)

        # prepare fits for various cases:
        # for recenter_flag in falsetruearr:                 #  fixed or free center
        N_params = 1    # log-scale-radius
        for recenter_flag in recenter_flags:
            if recenter_flag:
                N_params += 2
            # for ellipticity_flag in falsetruearr:          #  circular or elliptical model
            for ellipticity_flag in ellipticity_flags:
                if ellipticity_flag:
                    N_params += 2
                # for background_flag in falsetruearr:       #  without or with background
                for background_flag in background_flags:
                    if verbosity >= 2:
                        print("** MAIN: recenter ellipticity background flags = ", recenter_flag, ellipticity_flag, background_flag)
                    if background_flag:
                        N_params += 1

                    # bounds on parameters according to case
                    for model in models:
                        if model != model_test:
                            continue
                        if verbosity >= 2:
                            print('* MAIN: model = ', model)

                        log_scale_radius_maxallow = np.log10(R_min / min_R_over_rminus2)
                        if R_min == 0:
                            log_scale_radius_minallow = log_scale_radius_maxallow - 2
                        else:
                            log_scale_radius_minallow = np.log10(R_max / max_R_over_rminus2)
                        ### FUDGE FOR TESTING
                            log_scale_radius_minallow = np.log10(R_min)
                            log_scale_radius_maxallow = np.log10(R_max)

                        log_scale_radius = np.log10 ( 1.35 * np.median(R_sky) )
                        # factor 1.35 is predicted median for NFW and coredNFW of concentration 3

                        if not recenter_flag:
                            RA_cen_minallow  = RA_cen_init
                            RA_cen_maxallow  = RA_cen_init
                            Dec_cen_minallow = Dec_cen_init
                            Dec_cen_maxallow = Dec_cen_init
                        else:
                            RA_cen_minallow  = RA_cen_init  - 0.5 * R_max / np.cos(Dec_cen_init * degree)
                            RA_cen_maxallow  = RA_cen_init  + 0.5 * R_max / np.cos(Dec_cen_init * degree)
                            Dec_cen_minallow = Dec_cen_init - 0.5 * R_max
                            Dec_cen_maxallow = Dec_cen_init + 0.5 * R_max
                        RA_cen = RA_cen_init
                        Dec_cen = Dec_cen_init

                        if not ellipticity_flag:
                            ellipticity_minallow = 0.
                            ellipticity_maxallow = 0.
                            PA_minallow = 0.
                            PA_maxallow = 0.
                            ellipticity = 0.
                            PA = 0.
                        else:
                            ellipticity_minallow = 0.
                            ellipticity_maxallow = 0.99
                            PA_minallow = 0.0
                            PA_maxallow = 180.
                            ellipticity = ellipticity_pred
                            PA = PA_pred

                        if not background_flag:
                            log_background_minallow = -99.
                            log_background_maxallow = -99.
                            #log_background_minallow = 0
                            #log_background_maxallow = 0
                        else:
                            # observations (Metcalfe+06) => integrated counts of 80/arcmin^2 down to H=24.5
                            #                            => log(arcmin^2 background) ~= 1.9
                            #                               expect 30/arcmin^2 in Euclid
                            #                               => log ~= 1.5
                            # use consistency in z_phot     => log ~= 0.5
                            # values in deg^{-2}
                            log_background_maxallow = log_background_maxallow_default
                            log_background_minallow = log_background_maxallow - 6
                            # do not set max background here, as it was set above

                        # Testing a method to find a more accurate guess for background (actually poorly determined after testing it except by empirically adjusting it)
                        # Find four corners of cluster and their distance to closest neighbour
                        # compute mean of surface density (1 gal/disk of radius equal to the distance to closest neighbour) weighted by the square inverse of the distance (to avoid large contributions from tails of NFW)
                        leftmost_RA     = np.min(RA)
                        leftmost_Dec    = Dec[RA == leftmost_RA]
                        rightmost_RA    = np.max(RA)
                        rightmost_Dec   = Dec[RA == rightmost_RA]
                        topmost_Dec     = np.max(Dec)
                        topmost_RA      = RA[Dec == topmost_Dec]
                        bottommost_Dec  = np.min(Dec)
                        bottommost_RA   = RA[Dec == bottommost_Dec]

                        test = np.sort(np.sqrt((RA-leftmost_RA)**2 + (Dec-leftmost_Dec)**2)[RA != leftmost_RA])
                        closest_to_leftmost_distance   = np.partition(np.sqrt((RA-leftmost_RA)**2 + (Dec-leftmost_Dec)**2), 11)[10]
                        closest_to_rightmost_distance  = np.partition(np.sqrt((RA-rightmost_RA)**2 + (Dec-rightmost_Dec)**2), 11)[10]
                        closest_to_topmost_distance    = np.partition(np.sqrt((RA-topmost_RA)**2 + (Dec-topmost_Dec)**2), 11)[10]
                        closest_to_bottommost_distance = np.partition(np.sqrt((RA-bottommost_RA)**2 + (Dec-bottommost_Dec)**2), 11)[10]

                        leftmost_surf_density, rightmost_surf_density = 2./(np.pi*closest_to_leftmost_distance**2), 2./(np.pi*closest_to_rightmost_distance**2)
                        topmost_surf_density, bottommost_surf_density = 2./(np.pi*closest_to_topmost_distance**2), 2./(np.pi*closest_to_bottommost_distance**2)

                        log_background  = np.log10(np.average([leftmost_surf_density, rightmost_surf_density, topmost_surf_density, bottommost_surf_density],
                                                              weights = np.array([closest_to_leftmost_distance**2, closest_to_rightmost_distance**2, closest_to_topmost_distance**2, closest_to_bottommost_distance**2]))/1000.)

                        bounds = np.array([(RA_cen_minallow,RA_cen_maxallow),
                                           (Dec_cen_minallow,Dec_cen_maxallow),
                                           (log_scale_radius_minallow,log_scale_radius_maxallow),
                                           (ellipticity_minallow,ellipticity_maxallow),
                                           (PA_minallow,PA_maxallow),
                                           (log_background_minallow,log_background_maxallow)
                                           ])

                        # interactive testing method
                        size = 40
                        log_scale_list   = np.linspace(-3, -0.7, size)
                        e_list           = np.linspace(0, 0.99, size)
                        PA_list          = np.linspace(0, 180, size)
                        log_bg_list      = np.linspace(-5, 5, size)
                        tot = size**4
                        cnt = 0
                        start = time.time()
                        f = open("output_phase_space_RA" + str(round(RA_cen)) + "_Dec" + str(round(Dec_cen)) + "_withradiusinprobuv.dat", "a")
                        f.write("log_scale_list   = np.linspace(-3, -0.7, size) | e_list           = np.linspace(0, 0.99, size) | PA_list          = np.linspace(0, 180, size) | log_bg_list      = np.linspace(-5, 5, size)\n")
                        for i in log_scale_list:
                            for j in e_list:
                                for k in PA_list:
                                    for l in log_bg_list:
                                        cnt+=1

                                        log_scale_radius, ellipticity, PA = i, j, k
                                        log_background = l
                                        log_scale_radius_minallow = log_scale_radius
                                        log_scale_radius_maxallow = log_scale_radius
                                        RA_cen_minallow = RA_cen
                                        RA_cen_maxallow = RA_cen
                                        Dec_cen_minallow = Dec_cen
                                        Dec_cen_maxallow = Dec_cen
                                        ellipticity_maxallow = ellipticity
                                        ellipticity_minallow = ellipticity
                                        PA_minallow = PA
                                        PA_maxallow = PA
                                        log_background_minallow = log_background
                                        log_background_maxallow = log_background

                                        # - ln L
                                        params = RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background
                                        lnlikminus = PROFCL_LogLikelihood(params,RA,Dec,prob_membership,
                                                                        R_min, R_max, model)

                                        if lnlikminus > 0:
                                            lnlikminus = 0
                                        f.write(str(log_scale_radius) + " " + str(ellipticity) + " " + str(PA) + " " + str(log_background) + " " + str(lnlikminus) + "\n")
                                        print(cnt/tot*100, "% effectues\nTemps restant ", (time.time() - start)*(tot-cnt)/60., " min")
                                        start = time.time()

                        f.close()
                        exit()
