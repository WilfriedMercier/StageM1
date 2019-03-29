#! /usr/local/bin/python3
# -*- coding: latin-1 -*-


# Program to extract structuiral parameters (size, ellipticity, position angle, center, background) of clusters

# Author: Gary Mamon, with help from Christophe Adami, Yuba Amoura, Emanuel Artis and Eliott Mamon

from __future__ import division
import numpy as np
import sys as sys
import datetime
import time
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution
from scipy.optimize import fmin_tnc
from scipy import interpolate
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib
#matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt
plt.style.use('classic')

"""PROFCL performs maximum likelihood fits of cluster radial profiles to galaxy positions
    . considering a uniform background
    . re-centering,
    . performing an elliptical fit
it returns the scale radius by a maximum likelihood method, and optionally:
    the background, 
    the new position of the center, 
    and the ellpiticity and position angle
This version loops over clusters, and for each cluster performs 2 x 2 x 2 x 2 = 16 fits
 for the different combination of the 3 flags and 2 models. 9 possible minimization methods are provided.

* Version 1.1 (2 Jan 2017)
        > handles RA near zero
        > handles clusters near equatorial poles
        > formatted output
        > cluster center is now properly re-initialized

* Version 1.1.2 (19 Jan 2017)
        > min |Dec| for frame transformation changed from 10 deg to 80 deg
        > added welcome printout with version number and date
        > print to standard output is now formatted
        > allow to enter minimization method in lower case
        > updated comments

* Version 1.2 (6 June 2017)
        > saves input parameters to file
        > allows default parameters on input (from previous used ones)

* Version 1.3 (6 July 2017)
        > now handles academic mocks from Emmanuel Artis
        > ReadClusterData rewritten for faster read (nupy.loadtxt instead of line by line)

* Version 1.4 (13 July 2017)
        > forces centers of academic mocks to (15,10) (Gary) or (0,0) (Emmanuel Artis)
        > now reads (relative) tolerance of minimization as input

* Version 1.5 (25 Sept 2017)
        > computes median separation
        > saves date and time for each fit
        > saves median separation for each cluster

* Version 1.6 (20 Oct 2017)
        > circular region by Monte-Carlo integration
        > changed sign of x to be that of RA
        > ability to loop over Emmanuel's mocks
        > created PROFCL_OpenFile function

* Version 1.6.1 (29 Jan 2018)
        > split version into version and vdate and version_date
        > save version on output

* Version 1.7 (7 Feb 2018)
        > fixed bug for elliptical region with Monte-Carlo: use global variables set in log_likelihood
        > replaced integrate_model... by N_points (0 for no integration)

* Version 1.8 (13 Feb 2018)
        > now handles FLAGSHIP log M > 14 clusters in (RA,Dec) = (10-20,10-20) square region.
        > reads cluster file for centers

* Version 1.10 (17 Apr 2018)
        > changed tolerance criteria to:
                TNC: absolute on parameters
                SLSQP: absolute on function (lnlik)

* Version 1.11 (24 Apr 2018)
        > fixed bug (found by W. Mercier) in Median_Separation

* Version 1.12 (27 Apr 2018) by W. Mercier
        > Added median_flag for turning on/off median separation
        > Added computation time for each cluster in output file
        > Added RA_cen, Dec_cen, log_scale_radius, ellipticity, PA and log_background initial guesses in output file
        > Added command line argument to set output file name (if not by default it will be PROFCL.output)
        > Added auto flag for automating procedure:
            > set "manual" in input file (last line) for typing parameters as before
            > set otherwise to trigger automated computation

* Version 1.13 (3 May 2018) by W. Mercier & G. Mamon
        > Added full command line arguments
        > Corrected minor bug in RA when returning to cartesian coordinates
        > Added median center for the guess
        > Monte Carlo estimation of Nproj_tilde limited to circle is corrected
            > bfgs still does not work (seems related to bounds)
            > Monte Carlo stille yields predicted values at the end of computation (which takes place)

* Version 1.14 (10 May 2018) by G. Mamon
        > Monte-Carlo method: 
            > reduced N_knots of spline from 10^4 to 100
            > included N_knots for coredNFW model

* Version 1.15 (22 May 2018) by G. Mamon
            > included model NFWtrunc:
                  --> using surface density and number density from Mamon et al. (2010)
            > modified fitting code to handle extra parameter (R_cut)
            > replaced np.min tests by np.any for speed
            > replaced deprecated time.clock() by time.process_time()
            > models is now a numpy.array (to test inclusion with np.any)
            > models_tmp --> model

* Version 1.16 (24 May 2018) by G. Mamon & W. Mercier
            > rescale_flag allows to have fitted parameters with same precision
            > de-blending up to 3 clusters.
            > now test inclusion with if var in ('',''):

Not yet implemented:
    * red vs blue galaxies
    * forcing prob_mem = 1 for all galaxies
    * uncertainties from bootstraps (but the systematic errors should completely dominate the statistical errors)
    * priors on parameters
    ...

** Geometry:
* DET-CL center: RA_cen_init, Dec_cen_init
* new center: RA_cen, Dec_cen
* polar coordinates: PA measured East from North
* cartesian coordinates: 
    centered on DET-CL center
    x increases as RA (towards East, hence backwards in standard cartesian plot)
    y increases with Dec
* elliptical coordinates:
    u: along major axis, points South for PA=0
    v: along minor axis, points West (right) for PA=0
* transformations:
    x = RA_ref + (RA-RA_ref) * \emph{cos(Dec_ref)
}    y = Dec_ref + (Dec-Dec_ref) = Dec
    x = - u sin PA - v cos PA
    y = - u cos PA + v sin PA
    u = - x sin PA - y cos PA
    v = - x cos PA + y sin PA
    
"""


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
    if np.any(X < 0):
        raise print ("ACO argument is negative: ", X)
    
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
    # code is probably deprecated, and replaced by ACO

    CheckType(X,'ACOgen','X')
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

    #  t = type(X)

    # check that input is integer or float or numpy array
    CheckType(X,'SurfaceDensity_tilde_NFW','X')
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise print('ERROR in SurfaceDensity_tilde_NFW: min(X) = ', 
                         np.min(X), ' cannot be <= 0')

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
    if np.any(X) <= 0.:
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

def SurfaceDensity_tilde_NFWtrunc(X,Xm):
    """Dimensionless cluster surface density for a truncated NFW profile. 
    args: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2), Xm = R_truncation/R_{-2}
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon
    # source: Mamon, Biviano & Murante (2010), eq. (B.4)

    #  t = type(X)

    # check that input is integer or float or numpy array
    CheckType(X,'SurfaceDensity_tilde_NFWtrunc','X')
    CheckType(Xm,'SurfaceDensity_tilde_NFWtrunc','Xm')
    
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise print('ERROR in SurfaceDensity_tilde_NFWtrunc: min(X) = ', 
                    np.min(X), ' cannot be <= 0')

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    denom = np.log(4.) - 1.
    Xminus1 = X - 1.
    Xsquaredminus1 = X*X - 1.
    sqrtXmsquaredminus1 = np.sqrt(Xm*Xm - 1.)
    Xmplus1 = Xm + 1.

    return   ( np.select(
        [abs(Xminus1) < 0.001, X > 0 and X < Xm],
        [sqrtXsquaredminus1*(Xm+2)/(3.*Xmplus1**2.) + Xminus1*(2-Xm-4*Xm*Xm-2*Xm**3)/(5*Xmplus1**2.*sqrtXmsquaredminus1),
         ACO((X*X + Xm)/(X*(Xm+1.))) / ((1.-X*X) * np.sqrt(np.abs(X*X-1.))) - np.sqrt(Xm*Xm-X*X)/((1.-X*X)*(Xm+1))],
        default=0) / denom)

def ProjectedNumber_tilde_NFW(X):
    """Dimensionless cluster projected number for an NFW profile. 
    arg: X = R/r_s (positive float or array of positive floats)
    returns: N_proj(X r_{-2}) / N_proj(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(X,'ProjectedNumber_tilde_NFW','X')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(X) < 0.:
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
    if np.any(X) < 0.:
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

def ProjectedNumber_tilde_Uniform(X):
    """Dimensionless cluster projected number for a uniform surface density profile
    arg: X = R/R_cut (positive float or array of positive floats), where R_cut is radius of slope -2 
      (not the natural scale radius for which x=r/a!)
    returns: N(X R_cut) / N(R_cut) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(X,'ProjectedNumber_tilde_Uniform','X')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(X) < 0.:
        raise print('ERROR in ProjectedNumber_tilde_Uniform: X cannot be < 0')

    return(np.where(X<1, X*X, 1.))

def ProjectedNumber_tilde_NFWtrunc(X,Xm):
    """Dimensionless cluster projected number for a truncated NFW profile. 
    args: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2), Xm = R_truncation/R_{-2}
    returns: N(r_{-2} X) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon
    # source: Mamon, Biviano & Murante (2010), eq. (B.4)

    # check that input is integer or float or numpy array
    CheckType(X,'ProjectedNumber_tilde_NFWtrunc','X')
    CheckType(Xm,'ProjectedNumber_tilde_NFWtrunc','Xm')
    
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise print('ERROR in ProjectedNumber_tilde_NFWtrunc: min(X) = ', 
                    np.min(X), ' cannot be <= 0')

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    denom = np.log(2.) - 0.5
    Xminus1 = X - 1.
    Xsquaredminus1 = X*X - 1.
    sqrtXmsquaredminus1 = np.sqrt(Xm*Xm - 1.)
    sqrtXmsqminusXsq = np.sqrt(Xm*Xm - X*X)
    Xmplus1 = Xm + 1.

    return   ( np.select(
        [abs(Xminus1) < 0.001, X > 0 and X < Xm],
        [np.log((Xm+1.)*(Xm-sqrtXmsqminusXsq)) - Xm/(Xm+1) + 2.*np.sqrt((Xm+1.)/(Xm-1.)),
         (sqrtXmsqminusXsq-Xm)/(Xm+1.) + np.log(Xmplus1*(Xm-sqrtXmsqminusXsq)/X) + ACO((X*X+Xm)/(X*(Xm+1.)))
        ],
        default=np.log(Xm+1)-Xm/(Xm+1.)
    ) / denom)

def Number_tilde_NFW(x):
    """Dimensionless cluster 3D number for an NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(x,'Number_tilde_NFW','x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise print('ERROR in Number_tilde_NFW: x cannot be < 0')

    return ((np.log(x+1)-x/(x+1)) / (np.log(2.)-0.5))

def Number_tilde_coredNFW(x):
    """Dimensionless cluster 3D number for a cored NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(x,'Number_tilde_coredNFW','x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
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
    if np.any(x) < 0.:
        raise print('ERROR in Number_tidle_Uniform: x cannot be < 0')

    return np.where(x >= 1, 0, 1 / (np.pi * np.sqrt(1-x*x)))
   
def Random_radius(Xmax,model,Npoints,Nknots):
    """Random R/r_{-2} from Monte Carlo for circular NFW model
    args: R_max/R_{-2} model (NFW|cNFW) number-of-random-points number-of-knots"""

    # author: Gary Mamon

    CheckType(Xmax,'Random_radius','Xmax')
    CheckTypeInt(Npoints,'Random_radius','Npoints')

    # random numbers
    q = np.random.random_sample(Npoints)

    # equal spaced knots in arcsinh X
    asinhX0 = np.linspace(0., np.arcsinh(Xmax), num=Nknots+1)

    X0 = np.sinh(asinhX0)
    if model == 'NFW':
        N0 = ProjectedNumber_tilde_NFW(Xmax)
    elif model == 'cNFW':
        N0 = ProjectedNumber_tilde_coredNFW(Xmax)
    elif model == 'NFWtrunc':
        N0 = ProjectedNumber_tilde_NFWtrunc(Xmax)
    else:
        raise print ('Random_radius: model = ', model, ' not recognized')
    
    ratio = ProjectedNumber_tilde_NFW(X0) / N0

    # spline on knots of asinh(equal spaced)
    asinhratio = np.arcsinh(ratio)
    t = time.time()
    spline = interpolate.splrep(asinhratio,asinhX0,s=0)
    if verbosity >= 2:
        print("compute spline: time = ", time.time()-t)
    t = time.time()        
    asinhq = np.arcsinh(q)
    if verbosity >= 2:
        print("asinh(q): time = ", time.time()-t)
    t = time.time()        
    asinhX_spline = interpolate.splev(asinhq, spline, der=0, ext=2)
    if verbosity >= 2:
        print("evaluate spline: time = ", time.time()-t)
    return (np.sinh(asinhX_spline))

def Random_xy(Rmax,model,Npoints,Nknots,ellipticity,PA):
    """Random x & y (in deg) from Monte Carlo for circular model (NFW or cNFW)
    args: R_max model (NFW|cNFW) number-of-random-points number-of-knots ellipticity PA"""
    # print("Random_xy: PA=", PA)
    R_random = scale_radius * Random_radius(Rmax/scale_radius,model,Npoints,Nknots)
    PA_random = 2 * np.pi * np.random.random_sample(Npoints) # in rd
    # u_random = R_random * np.cos(PA_random)
    # v_random = R_random * np.sin(PA_random) / (1-ellipticity)
    # v_random = R_random * np.cos(PA_random) / (1-ellipticity) #AAAAAAAAAAAAAAAAAAAAAAA
    theta_random = 2 * np.pi * np.random.random_sample(Npoints) # in rd
    u_random = R_random * np.cos(theta_random)
    v_random = R_random * np.sin(theta_random) * (1-ellipticity)
    x0_random = RA_cen_init + (RA_cen - RA_cen_init) * np.cos(Dec_cen_init * degree) 
    y0_random = Dec_cen
    # x_random = x0_random + u_random*np.sin(PA_random) + v_random*np.cos(PA_random)
    # y_random = y0_random + u_random*np.cos(PA_random) - v_random*np.sin(PA_random)
    x_random = x0_random - u_random*np.sin(PA*degree) - v_random*np.cos(PA*degree)
    y_random = y0_random + u_random*np.cos(PA*degree) - v_random*np.sin(PA*degree)

    if verbosity >= 3:
        print("IS TRUE : ", x0_random == RA_cen_init, y0_random == Dec_cen_init)
    # plt.plot(PA_random, R_random, ".")
    # plt.axes().set_aspect('equal','datalim')
    # plt.plot(u_random, v_random, ".")
    # plt.plot(x_random-x0_random, y_random-y0_random, ".")
    # plt.grid()
    # plt.xscale('log')
    # plt.show()

    return(x_random, y_random)
    
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
            (x_rand,y_rand) = Random_xy(X*10.**log_scale_radius/(1-ellipticity),'NFW',N_points,N_knots,ellipticity,PA)
            R_rand_sq = (x_rand-RA_cen_init)**2 + (y_rand-Dec_cen_init)**2
            R_rand_sq_in_circle = R_rand_sq[R_rand_sq < R_max*R_max]
            # R_rand_sq_in_scale_radius = R_rand_sq[R_rand_sq < 10.**(2*log_scale_radius)]
            # Nprojtilde = len(R_rand_sq_in_scale_radius) / len(R_rand_sq_in_circle)
            Nprojtilde = len(R_rand_sq_in_circle) / N_points * ProjectedNumber_tilde_NFW(X)
            if verbosity >=3:
                print("fraction in circle = ", len(R_rand_sq_in_circle) / N_point)
                # print(len(R_rand_sq_in_scale_radius), len(R_rand_sq_in_circle))

    return (Nprojtilde)

def ProjectedNumber_tilde_ellip_coredNFW(X,ellipticity):
    """Dimensionless projected mass for non-circular cored NFW models
    args:
    X = R/r_{-2}  (positive float or array of positive floats), where r_{-2} is radius of slope -2
    ellipticity = 1-b/a (0 for circular)
    returns: N(X r_{-2}) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon
    # source: ~gam/EUCLID/CLUST/PROCL/denomElliptical.nb

    N_knots  = 100
    if verbosity >= 3:
        print('using 2D polynomial')
    lX = np.log10(X)
    e = ellipticity
    if N_points == 0:
        lNprojtilde = (
            0.21076779174081403 - 0.1673534076933856*e - 
            0.9471677808222536*e**2 + 11.648473045614114*e**3 - 
            91.92475409478227*e**4 + 422.8544124895236*e**5 - 
            1206.605470683992*e**6 + 2152.6556515394586*e**7 - 
            2336.252720403306*e**8 + 1409.651246367505*e**9 - 
            362.82577003936643*e**10 + 1.1400775160218775*lX - 
            0.24603956803791907*e*lX + 0.4746353855804624*e**2*lX - 
            5.213784368168905*e**3*lX + 28.349333190289443*e**4*lX - 
            95.44143806235569*e**5*lX + 196.77037041806182*e**6*lX - 
            242.5768688683326*e**7*lX + 164.00212699954048*e**8*lX - 
            46.77921433973666*e**9*lX - 0.47280190984201714*lX**2 + 
            0.030724988640708772*e*lX**2 + 0.14209201391142387*e**2*lX**2 - 
            0.755436616271162*e**3*lX**2 + 3.306367265271173*e**4*lX**2 - 
            7.25557673533242*e**5*lX**2 + 9.429315278575027*e**6*lX**2 - 
            6.660238987320651*e**7*lX**2 + 2.04545992649397*e**8*lX**2 + 
            0.03394971337975079*lX**3 + 0.09887824821508472*e*lX**3 - 
            0.18041596878156793*e**2*lX**3 + 0.6289610806099004*e**3*lX**3 - 
            1.4556318193802276*e**4*lX**3 + 2.1239832585391083*e**5*lX**3 - 
            1.8325143147948293*e**6*lX**3 + 0.6369289158521704*e**7*lX**3 + 
            0.07315774564006589*lX**4 - 0.037041022300377306*e*lX**4 + 
            0.0029908382801743685*e**2*lX**4 - 0.03572991462536126*e**3*lX**4 - 
            0.05039173454869054*e**4*lX**4 + 0.06826024306255776*e**5*lX**4 - 
            0.028441143677024536*e**6*lX**4 - 0.019219238751868855*lX**5 - 
            0.02361318179363677*e*lX**5 + 0.0405966969727285*e**2*lX**5 - 
            0.052053157027219105*e**3*lX**5 + 0.05969376194544227*e**4*lX**5 - 
            0.01240643979930337*e**5*lX**5 - 0.01026942895674158*lX**6 + 
            0.01301415707276946*e*lX**6 - 0.007109228236235994*e**2*lX**6 + 
            0.014751475808259498*e**3*lX**6 - 0.008400229615749667*e**4*lX**6 + 
            0.004545329673990146*lX**7 + 0.0011480281966753895*e*lX**7 - 
            0.002874103492006819*e**2*lX**7 - 0.0009871609971554144*e**3*lX**7 + 
            0.0003921813852493623*lX**8 - 0.0014751021188585689*e*lX**8 + 
            0.0006830554871586946*e**2*lX**8 - 0.0004114331203583239*lX**9 + 
            0.00020132121960998451*e*lX**9 + 0.00005094309326516718*lX**10
        )
    else:
        if X * 10.**log_scale_radius < R_max:
            Nprojtilde = ProjectedNumber_tilde_cNFW(X)
        else:
            (x_rand,y_rand) = Random_xy(X*10.**log_scale_radius/(1-ellipticity),'cNFW',N_points,N_knots,ellipticity,PA)
            R_rand_sq = (x_rand-RA_cen_init)**2 + (y_rand-Dec_cen_init)**2
            R_rand_sq_in_circle = R_rand_sq[R_rand_sq < R_max*R_max]
            # R_rand_sq_in_scale_radius = R_rand_sq[R_rand_sq < 10.**(2*log_scale_radius)]
            # Nprojtilde = len(R_rand_sq_in_scale_radius) / len(R_rand_sq_in_circle)
            Nprojtilde = len(R_rand_sq_in_circle) / N_points * ProjectedNumber_tilde_cNFW(X)
            if verbosity >=3:
                print("fraction in circle = ", len(R_rand_sq_in_circle) / N_point)
                # print(len(R_rand_sq_in_scale_radius), len(R_rand_sq_in_circle))

    return (Nprojtilde)

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
    elif model == 'NFWtrunc':
        return SurfaceDensity_tilde_NFWtrunc(X)
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
    elif model == 'NFWtrunc':
        if ellipticity == 0:
            return ProjectedNumber_tilde_NFWtrunc(X)
        else:
            return ProjectedNumber_tilde_ellip_NFWtrunc(X,ellipticity)
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

def PROFCL_prob_uv(u, v, R_min, R_max, scale_radius, elliipticity, background, R_cut, model):
    """probability density of projected radii for given circular model parameters
    arguments: 
        u:              coordinate along major axis (deg)
        v:              coordinate along minor axis (deg)
        R_min:          minimum allowed sky projected radius (deg)
        R_max:          maximum allowed sky projected radius (deg)
        scale_radius:   radius of 3D density slope -2) [same units as R_circ, R_min and R_max]
        ellipticity
        background:     uniform background [units of 1/(units of radii)^2, i.e. deg^{-2}]
        model:          'NFW' or 'coredNFW' or 'Uniform' or 'NFWtrunc'
    returns: p(data|model)"""  

    # authors: Yuba Amoura & Gary Mamon

    # to speed things up, no checks are performed here: they are performed in calling function likelihood_Rproj_NFW
    # hence following lines are commented

    # # check that input projected radii is in numpy array
    # if type(R_circ) is not numpy.ndarray:
    #     raise print('ERROR in PROFCL_prob_Rproj_NFW: R_circ must be numpy array')

    # # check that min and max projected radii, and loga are floats or ints
    # t = type(R_min)
    # if (t is not float and t is not int):
    #     raise print('ERROR in PROFCL_prob_Rproj_NFW: R_min must be a float or int')
    # t = type(R_max)
    # if (t is not float and t is not int):
    #     raise print('ERROR in PROFCL_prob_Rproj_NFW: R_max must be a float or int')
    # t = type(loga)
    # if (t is not float and t is not int):
    #     raise print('ERROR in PROFCL_prob_Rproj_NFW: loga must be a float or int')

    # # check that R_min > 0 (to avoid infinite surface densities)
    # if R_min <= 0:
    #     raise print('ERROR in PROFCL_prob_Rproj_NFW: R_min must be > 0')

    # # check that data is within limits
    # if (np.min(R_circ) < R_min or np.max(R_circ) > R_max):
    #     raise print('ERROR in PROFCL_prob_Rproj_NFW: data must lie within limits')

    # if R_circ < min_R_over_rminus2 * scale_radius:

    if verbosity >= 4:
        print("entering prob_uv: R_min=",R_min)
    bova = 1. - ellipticity

    # uniform model
    if (model == 'Uniform'):
        return(1/(np.pi*10**(2.*log_scale_radius)*bova))

    # non-uniform models
    a = scale_radius # radius of slope -2 in deg
    dR2 = R_max*R_max - R_min*R_min

    X_min_tmp = R_min/a
    X_max_tmp = R_max/a
    # Nproj_R_min = ProjectedNumber_tilde(R_min/a,model)
    if model == 'NFWtrunc':
        Nproj_tilde_R_min = ProjectedNumber_tilde_NFWtrunc(X_min_tmp,R_cut/a)
        Nproj_tilde_R_max = ProjectedNumber_tilde_NFWtrunc(X_max_tmp,R_cut/a)
    else:
        Nproj_tilde_R_min = ProjectedNumber_tilde(X_min_tmp,model)
        Nproj_tilde_R_max = ProjectedNumber_tilde(X_max_tmp,model)
    #print("Nproj_R_min = ", Nproj_tilde_R_min)
    # Nproj_R_max = ProjectedNumber_tilde(R_max/a,model)

    #print("Nproj_R_max = ", Nproj_tilde_R_max)
    dNproj_tilde = Nproj_tilde_R_max - Nproj_tilde_R_min     #AAAAAAAAAAAAAAAAAAAAAAAAAAAA
    # predicted number in model in sphere of radius a 
    N_tot = len(u)
    N_bg = np.pi * background * dR2
    Nofa = (N_tot - N_bg) / dNproj_tilde
    if verbosity >= 2:
        print("a R_min R_max = ", a, R_min, R_max)
        print("N_tot N_bg background = ", N_tot, N_bg, background) 
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

    # numerator = (2. * R_circ * 
    #                (
    #                     Nofa * SurfaceDensity_tilde(R_circ/a,model)/(a*a)
    #                     + np.pi*background
    #                )
    #             )
    # numerator = (Nofa/(np.pi*a*a) * SurfaceDensity_tilde(R_circ/a,model)
    #                     + background)
    # numerator = (2. * R_circ * Nofa/(a*a) * SurfaceDensity_tilde(R_circ/a,model)
    #                     + 2. * np.pi * R_circ * background)
    if verbosity >= 2:
        print("bova=",bova)
    R_circ = np.sqrt(u*u + (v/bova)*(v/bova))
    # numerator = 2*np.pi * R_circ * (Nofa / (a*a) / (np.pi * bova) * SurfaceDensity_tilde(R_circ/a,model) + background)
    if model == 'NFWtrunc':
        numerator = Nofa / (a*a) / (np.pi * bova) * SurfaceDensity_tilde_NFWtrunc(R_circ/a,R_cut/a) + background
    else:
        numerator = Nofa / (a*a) / (np.pi * bova) * SurfaceDensity_tilde(R_circ/a,model) + background
        # numerator = Nofa / (a*a) / (np.pi) * SurfaceDensity_tilde(R_circ/a,model) + background
    denominator = N_tot
                # (Nofa * 
                #      (
                #         ProjectedNumber_tilde(R_max/a,model)
                #         - ProjectedNumber_tilde(R_min/a,model)
                #      )
                #      + np.pi*background*(R_max*R_max-R_min*R_min)
                #   )
    # print("model = ", model, " Rmin = ", R_min, " R_max = ", R_max, " a = ", a, " denom = ", denominator, "dNproj_tilde = ", dNproj_tilde)
    # print("R = ", R_circ,  "X = ", R_circ/a, " num = ", numerator, "sd = ", SurfaceDensity_tilde(R_circ/a,model))
    # print("p = ", numerator/denominator)
    return (numerator / denominator)
    # return (numerator / (0*numerator + denominator))

def PROFCL_PenaltyFunction(NormalizedDistance):
    """Penalty function for nearby subclusters
    NormalziedDistance: dimensionless distance"""
    return(np.where(
        NormalizedDistance<1,1/NormalizedDistance**2-1,
        0
    )

# def PROFCL_LogLikelihood(params, *args):

def PROFCL_LogLikelihood(params, data, R_min, R_max, model):
    """general -log likelihood of cluster given galaxy positions
    arguments:
    params:             parameters to fit
    RA, Dec:            coordinates of galaxies (data, in deg) 
    prob_membership:    probabilities of membership of galaxies (data)
    R_min, R_max:       minimum and maximum allowed projected radii on sky (degrees) [floats]
    model:              density model (NFW or coredNFW or Uniform) [char]

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
        print("entering LogLikelihood: R_min=",R_min)
    iPass = iPass + 1

    # read data
    RA, Dec, prob_membership = data
    
    # check that galaxy positions and probabilities are in numpy arrays
    if type(RA) is not np.ndarray:
        raise print('ERROR in PROFCL_lnlikelihood: RA must be numpy array')
    if type(Dec) is not np.ndarray:
        raise print('ERROR in PROFCL_lnlikelihood: Dec must be numpy array')
    if type(prob_membership) is not np.ndarray:
        raise print('ERROR in PROFCL_lnlikelihood: prob_membership must be numpy array')

    # read function arguments (parameters and extra arguments)
    log_background, R_cut = params[0:1]
    
    # loop over number of possible sub-clusters
    # RA_cen, Dec_cen, log_scale_radius, ellipticity, PA = np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    RA_cen, Dec_cen, log_scale_radius, ellipticity, PA = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for n in range(n_subclusters):
        RA_cen           = np.append(RA_cen,params[n+2]))
        Dec_cen          = np.append(Dec_cen,params[n+3])
        log_scale_radius = np.append(log_scale_radius,params[n+4)])
        ellipticity      = np.append(ellipticity,params[n+5)])
        PA               = np.append(PA,params[n+6)])
        
    if rescale_flag == 'y':
        RA_cen          = R_max * np.cos(Dec_cen_init*degree) * RA_cen
        Dec_cen         = R_max * Dec_cen
        ellipticity     = 5. * ellipticity
        PA              = 500. * PA
        log_background  = 10. * log_bakcground
        R_cut           = 10.**(5.*R_cut)

    ## checks on types of arguments
    
    if np.isnan(RA_cen):
        raise print('ERROR in PROFCL_LogLikelihood: RA_cen is NaN!')
    if np.isnan(Dec_cen):
        raise print('ERROR in PROFCL_LogLikelihood: Dec_cen is NaN!')

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
    
    # check that Decs are between -90 and 90 degrees
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
    
    # # check that model is known
    # if model != 'NFW' and model != 'coredNFW' and model != 'Uniform':
    #     raise print('ERROR in PROFCL_lnlikelihood: model = ', 
    #                 model, ' is not implemented')

    # check that R_min > 0 for NFW (to avoid infinite surface densities)
    # or R_min >= 0 for coredNFW
    if R_min <= 0. and (model == 'NFW' or model == 'NFWtrunc'):
        raise print('ERROR in PROFCL_lnlikelihood: R_min must be > 0 for ', model, ' model')
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
    
    # x     = np.cos(Dec_in_rd)     * np.cos(RA_in_rd)
    # y     = np.cos(Dec_in_rd)     * np.sin(RA_in_rd)
    # x_cen = np.cos(Dec_cen_in_rd) * np.cos(RA_cen_in_rd)
    # y_cen = np.cos(Dec_cen_in_rd) * np.sin(RA_cen_in_rd)
    # transform coordinates to relative to central ones
    # dx = x - x_cen
    # dy = y - y_cen

    # relative positions (in degrees, careful that x points to decreasing RA)
    # dx =  (RA - RA_cen) * np.cos(Dec_cen_in_rd)
    # dy =  Dec - Dec_cen   
    dx =  - (RA - RA_cen) * np.cos(Dec_cen_in_rd)
    dy =     Dec - Dec_cen   
    
    # handle crossing of RA=0
    dx = np.where(dx >= 180.,  dx - 360., dx)
    dx = np.where(dx <= -180., dx + 360., dx)

    # # # guess ellipticity and PA from 2nd moments
    
    # xymean = np.mean(dx*dy)
    # x2mean = np.mean(dx*dx)
    # y2mean = np.mean(dy*dy)
    # tan2theta = 2*xymean/(x2mean-y2mean)
    # theta = 0.5*np.arctan(tan2theta) - np.pi/2.
    # # if xymean  < 0:
    # #     theta = -1*theta
    # PA_predict = theta / degree
    # if PA_predict < -90.:
    #     PA_predict = PA_predict + 180.
    # A2 = (x2mean+y2mean)/2. + np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    # B2 = (x2mean+y2mean)/2. - np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    # ellipticity_pred = 1 - np.sqrt(B2/A2)
    # print('predicted values: ellipticity = ', ellipticity_pred, ' PA = ', PA_predict,
    #       ' xymean=', xymean, ' tan2theta=', tan2theta)

    # exit()

    # rotate to axes of cluster (careful: PA measures angle East from North)
    # u = - dx * np.sin(PA_in_rd) - dy * np.cos(PA_in_rd)
    # v = - dx * np.cos(PA_in_rd) + dy * np.sin(PA_in_rd)
    u = - dx * np.sin(PA_in_rd) + dy * np.cos(PA_in_rd)
    v = - dx * np.cos(PA_in_rd) - dy * np.sin(PA_in_rd)
    if verbosity >= 2:
        print("Log_Likelihood: len(u) = ", len(u))
    # np.savetxt('PROFCL.uv',np.array([u,v]).T,header='u v')
    # exit()

    # equivalent circular projected radii (in degrees)
    R_circ_squared = u*u + v*v/(1.-ellipticity)**2.
    R_circ_squared_min = np.min(R_circ_squared)

    # Gerry Williger suggets to omit following test, since it cannot happen!
    if R_circ_squared_min < 0.:
        # if min(R^2) < -1*(1 arcsec)^2, set to 0, else stop
        if R_circ_squared_min > -1/3600.**2.:
            if verbosity >= 2:
                print("min(R^2) slightly < 0!")
            R_circ_squared = np.where(R_circ_squared < 0.,
                                      0.,
                                      R_circ_squared
                                      )
        else:
            raise print('ERROR in PROFCL_lnlikelihood: min(R^2) = ', 
                        R_circ_squared_min/3600.**2, 
                        ' arcsec^2 ... cannot be < 0')
    R_circ = np.sqrt(R_circ_squared)

    # linear variables
    scale_radius = 10.**log_scale_radius        # in deg
    background   = 10.**log_background          # in deg^{-2}

    truncation_radius = R_cut
    
    # dimensionless radius
    X_circ = R_circ / scale_radius

    # check that circular radii are within limits of Mathematica fit
    if np.any(X_circ < min_R_over_rminus2) or np.any(X_circ > max_R_over_rminus2):
        if verbosity >= 2:
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

    # print("u = ", u[condition])
    # print("v = ", v[condition])
    prob = PROFCL_prob_uv(u[condition],v[condition],R_min,R_max,scale_radius,ellipticity,background,truncation_radius,model)
    if verbosity >= 3:
        print("\n\n PROBABILITY = ", prob, "\n\n")
    if np.any(prob<=0):
        if verbosity >= 2:
            print("one p = 0, EXITING LOG_LIKELIHOOD FUNCTION")
        return(HUGE)

    if verbosity >= 2:
        print("OK")

    # print("prob=",prob)
    lnlikminus = -1*np.sum(prob_membership[condition]*np.log(prob))

    # add penalty for nearby subclusters

    distances = np.array([])
    for n in range(n_subclusters-1):
        nn = range(n+1,n_subclusters)
        pos1 = ap.coordinates.SkyCoord(RA_cen[n],Dec_cen[n],unit='deg')
        pos = ap.coordinates.SkyCoord(RA_cen[nn],Dec_cen[nn],unit='deg')
        distance_tmp = pos1.separation(pos)
        distances = np.append(distances,distance_tmp)
    distances = np.asarray(distances)

    if len(distances) > 0:
        penalty = PenaltyFunction(distances)
    else:
        penalty = 0.

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
        # print('RA_filter = ',RA_filter)
        # print('Dec_filter = ',Dec_filter)
        # print('u_filter = ',u_filter)
        # print('v_filter = ',v_filter)
        # print('R_circ_filter = ',R_circ_filter)
        # print('prob = ',prob)
        # ftest = open('profcl_debug2.dat','a')
        # ftest.write("{0:8.4f} {1:8.4f} {2:13.4e} {3:13.4e} {4:12.4e} {5:12.4e}\n".format(RA_filter,Dec_filter,u_filter,v_filter,R_circ_filter,prob))
        # ftest.close()
        np.savetxt('DEBUG/profcl_debug2.dat' + str(iPass),np.c_[RA[condition],Dec[condition],u[condition],v[condition],R_circ[condition],prob])
        ftest = open('DEBUG/profcl_debug2.dat' + str(iPass),'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:4.1f} {7:10.3f}\n".format(RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, 60.*R_cut, lnlikminus))
        ftest.close()
        # np.savetxt(sys.stdout,np.c_[RA_filter,Dec_filter,u_filter,v_filter,R_circ_filter,prob])
        ftest = open(debug_file,'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:4.1f} {7:10.3f}\n".format(RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, 60.*R_cut, lnlikminus))
        ftest.close()

    # return -ln likelihood
    return (lnlikminus + penalty)

def PROFCL_Fit(data, params,
               bounds,
               R_min, R_max, model, 
               background_flag, recenter_flag, ellipticity_flag, 
               function=PROFCL_LogLikelihood,
               method='Nelder-Mead', bound_min=None, bound_max=None):
    '''Maxmimum Likelihood Estimate of 3D scale radius 
       (where slope of 3D density profile is -2) given projected data
    arguments:
        RA, Dec:                coordinates of galaxies [numpy arrays of floats]
        prob_membership:        probability of membership in cluster of galaxies [numpy array of floats, between 0 and 1]
        RA_cen, Dec_cen:        coordinates of cluster center [floats]
        log_scale_radius:       log of scale radius of the profile (where scale radius is in degrees) [float]
        ellipticity:            ellipticity of cluster (1-b/a, so that 0 = circular, and 1 = linear) [float]
        PA:                     position angle of cluster (degrees from North going East) [float]
        log_background:         uniform background of cluster (in deg^{-2}) [float]
        R_cut:                  truncation radius (NFWtrunc model) [float]
        model:                  density model (NFW or coredNFW or Uniform) [char]
        R_min:                  minimum allowed projected radius (float)
        R_max:                  maximum allowed projected radius (float)
        background_flag:        flag for background fit (True for including background in fit, False for no background)
        recenter_flag:          flag for recentering (True for recentering, False for keeping DETCL center)
        elliptical_flag:        flag for elliptical fit (True for elliptical fit, False for circular fit)
        function:               name of function to minimize
        method:                 minimization method (string)
                        'Nelder-Mead':  Simplex 
                        'Powell':       Powell's method   
                        'BFGS':         Broyden-Fletcher-Goldfarb-Shanno, using gradient
                        'L-BFGS-B':     Broyden-Fletcher-Goldfarb-Shanno, using gradient
                        'SLSQP':        Sequential Least-Squares Programming
                        'Newton-CG':    Newton's conjugate gradient
                        'Brent':        Brent method (only for univariate data)
                        'diff-evol':    Differntial Evolution
        bound_min:              minimum bound on log(r_scale)
        bound_max:              maximum bound on log(r_scale)'''

    # authors: Gary Mamon & Yuba Amoura, with help from Eliott Mamon

    global iPass
    if verbosity >= 4:
        print("entering Fit: R_min=",R_min)

    ## checks on types of arguments

    # # check that input positions are in numpy arrays
    # if not isinstance(RA,np.ndarray):
    #     raise print('ERROR in PROFCL_Fit: RA must be numpy array')

    # if not isinstance(Dec,np.ndarray):
    #     raise print('ERROR in PROFCL_Fit: Dec must be numpy array')

    # check that min and max projected radii are floats or ints
    CheckTypeIntorFloat(R_min,'PROFCL_Fit','R_min')
    CheckTypeIntorFloat(R_max,'PROFCL_Fit','R_max')

    # check that model is a string
    t = type(model)
    if t is not str:
        raise print('ERROR in PROFCL_Fit: model is ', 
                         t, ' ... it must be a str')
                             
    # check that flags are boolean
    CheckTypeBool(background_flag,'PROFCL_Fit','background_flag')
    CheckTypeBool(recenter_flag,'PROFCL_Fit','recenter_flag')
    CheckTypeBool(ellipticity_flag,'PROFCL_Fit','ellipticity_flag')
    
    # check that method is a string
    t = type(method)
    if t is not str:
        raise print('ERROR in PROFCL_Fit: method is ', 
                         t, ' ... it must be a str')
                             
    ## checks on values of arguments
    
    # check that R_min > 0 (to avoid infinite surface densities)
    if R_min <= 0:
        raise print('ERROR in PROFCL_Fit: R_min = ', 
                         R_min, ' must be > 0')
                             
    # check that R_max > R_min
    if R_max < R_min:
        raise print('ERROR in PROFCL_Fit: R_max = ', 
                         R_max, ' must be > R_min = ', R_min )
                             
    # check model
    if model != 'NFW' and model != 'coredNFW' and  model != 'Uniform':
        raise print('ERROR in PROFCL_Fit: model = ', 
                         model, ' not recognized... must be NFW or coredNFW or Uniform')
    
    # function of one variable
    if np.isnan(RA_cen):
        raise print('ERROR in PROFCL_Fit: RA_cen is NaN!')
    if np.isnan(Dec_cen):
        raise print('ERROR in PROFCL_Fit: Dec_cen is NaN!')

    cond = np.logical_and(R_sky >= R_min, R_sky <= R_max)
    if verbosity >= 2:
        print("Fit: N_tot = ", len(R_sky[cond]))
    # params = np.array([
    #                    RA_cen,Dec_cen,log_scale_radius,
    #                    ellipticity,PA,log_background,R_cut
    #                   ]
    #                  )

    # minimization
    # if method == 'brent' or method == 'Brent':
    #     if recenter_flag:
    #         raise print('ERROR in PROFCL_Fit: brent minimization method cannot be used for fits with re-centering')
    #     if ellipticity_flag:
    #         raise print('ERROR in PROFCL_Fit: brent minimization method cannot be used for elliptical fits')
        
    #     if bound_min is None or bound_max is None:
    #         return minimize_scalar(f, method='brent', tol=0.001)
    #     else:
    #         return minimize_scalar(f, bounds=(bound_min,bound_max), 
    #                                    method='bounded', tol=0.001)
    # else:
    maxfev = 500
    iPass = 0
    if   method == 'Nelder-Mead':
        # return minimize(PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
        #                 method=method, tol=tolerance, bounds=bounds, options={'fatol':tol, 'maxfev':maxfev})
        return minimize              (PROFCL_LogLikelihood, params, args=(data, R_min, R_max, model), 
                        method=method, tol=tolerance, options={'maxfev':maxfev})
    elif method == 'Powell':
        return minimize              (PROFCL_LogLikelihood, params, args=(data, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'ftol':tolerance, 'maxfev':maxfev})
    elif method == 'CG' or method == 'BFGS':
        return minimize              (PROFCL_LogLikelihood, params, args=(data, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'gtol':tolerance, 'maxiter':maxfev})
    elif method == 'Newton-CG':
        return minimize              (PROFCL_LogLikelihood, params, args=(data, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'xtol':tolerance, 'maxiter':maxfev})
    elif method == 'L-BFGS-B':
        return minimize              (PROFCL_LogLikelihood, params, args=(data, R_min, R_max, model), 
                        method=method, tol=tolerance, bounds=bounds, options={'ftol':tolerance, 'maxfun':maxfev})
    elif method == 'SLSQP':
        return minimize              (PROFCL_LogLikelihood, params, args=(data, R_min, R_max, model), 
                        method=method, jac=None, bounds=bounds, tol=None, options={'ftol':tolerance, 'maxiter':maxfev})
    elif method == 'TNC':
        return minimize              (PROFCL_LogLikelihood, params, args=(data, R_min, R_max, model), 
                        method=method, jac=None, bounds=bounds, tol=None, options={'xtol':tolerance, 'maxiter':maxfev})
    elif method == 'fmin_tnc':
        return fmin_tnc              (PROFCL_LogLikelihood, params, fprime=None, args=(data, R_min, R_max, model),  approx_grad=True, bounds=bounds, epsilon=1.e-8, ftol=tolerance)
    elif method == 'Diff-Evol':
        return differential_evolution(PROFCL_LogLikelihood, bounds, args=(data, R_min, R_max, model), atol=tolerance)
    else:
        raise print('ERROR in PROFCL_Fit: method = ', method, ' is not yet implemented')

# def PROFCL_ReadClusters(cluster_file)
#     """Read cluster global data"""

#     # author: Gary Mamon

#     # open file
#     try:
#         f = open(cluster_file,'r')
#     except IOError:
#         print('PROF_ReadClusterData: cannot open ', cluster_file)
#         exit()

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

    
def PROFCL_ReadClusterGalaxies(cluster_galaxy_file,column_RA,column_pmem,column_z):
    """Read cluster galaxy data
    arg: file-descriptor"""

    # author: Gary Mamon

    if verbosity >= 1:
        print("reading", cluster_galaxy_file, "...")

    f_gal = PROFCL_OpenFile(cluster_galaxy_file)
    tab_galaxies = np.loadtxt(f_gal)
    f_gal.close()

    num_columns = tab_galaxies.shape[1]
    if verbosity >= 3:
        print("number of columns found is ", num_columns)
    if column_RA > num_columns or column_z > num_columns or column_pmem > num_columns:
        print("PROFCL_ReadClusterGalaxies: data file =",cluster_galaxy_file, " with ", num_columns, " columns")
        raise print("column_RA=",column_RA,"column_pmem=",column_pmem,"column_z=",column_z)
    
    RA = tab_galaxies[:,column_RA-1]
    Dec = tab_galaxies[:,column_RA]
    
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

    return (RA, Dec, prob_membership)

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

    if method in ('NM','nm','nelder-mead'):
        method2 = 'Nelder-Mead'
    elif method in ('powell','pow','Pow'):
        method2 = 'Powell'
    elif method == 'cg':
        method2 = 'CG'
    elif method == 'bfgs':
        method2 = 'BFGS'
    elif method in ('newton-cg','ncg','NCG'):
        method2 = 'Newton-CG'
    elif method in ('l-bfgs-b','LBB','lbb'):
        method2 = 'L-BFGS-B'
    elif method == 'slsqp':
        method2 = 'SLSQP'
    elif method in ('tnc','t'):
        method2 = 'TNC'
    elif method == 'fmin_TNC':
        method2 = 'fmin_tnc'
    elif method in ('de','diffevol','DiffEvol'):
        method2 = 'Diff-Evol'
    elif method == 'T':
        method2 = 'test'
    else:
        method2 = method
    return(method2)

def convertModel(model):
    """Convert model abbreviations
    arg: abbreviation
    returns: model name"""

    # author: Gary Mamon

    if model in ('n','N','nfw','Nfw'):
        model2 = 'NFW'
    elif model in ('c','C','cnfw','coredNFW'):
        model2 = 'cNFW'
    elif model in ('u','U','uniform'):
        model2 = 'Uniform'
    elif model in ('nt','NFWt'):
        model2 = 'NFWtrunc'
    else:
        model2 = model
    return(model2)


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

    ###################################################################################################################
    # Typical command line:
    # python PROFCL_v.1.12.py -man -verb 0 -model NFW -meth tnc -tol 0.0001 -ellip 0 -median -center -minmax 0 99 -mock A 1280 0.5 50 20
    #
    # Mandatory arguments if manual mode: -man -mock x x x x x
    # -man and -auto are the only parameters which have a fixed position (first argument)
    # Unspecified arguments will have a value by default           
    # No arguments are required for automatic
    # If neither -man nor -auto are specified, the program will ask to enter values (and show thoses stored in input file)
    ####################################################################################################################

    # constants

    degree = np.pi   /  180.
    arcmin = degree / 60.
    arcsec = degree  / 3600.

    nb_arguments = len(sys.argv)
    #testing if enough arguments
    if nb_arguments >= 2:
        #checking if first argument known
        if sys.argv[1] == "-man":
            #default values
            verbosity, model_tmp, method, Rmaxovervir, tolerance, min_members, data_galaxies_dir, do_ellipticity = 0, "NFW", "tnc", -1., 0.0001, 1, "galaxies", "n"
            test_flag, median_flag, do_center, do_background, id_cluster_min_max                                      = "A", "n", "n", "n", "0 1"
            output_file = "test.output"

            auto     = "options"
            offset   = 0 #an offset in order to not take into account parameters following options such as -meth when looping
            list_arg = sys.argv[2:]
            size     = len(list_arg)
            for i in range(size):
                if i+offset >= size: #if the end is reached exit loop
                    break

                arg = list_arg[i+offset]
                if arg == "-verb": #verbosity must be followed by a number between 0 and 4
                    offset  += 1
                    verbosity = int(list_arg[i+offset])
                    continue
                if arg == "-model": #model must be followed by a string
                    offset += 1
                    model_tmp = list_arg[i+offset]
                    continue
                if arg == "-meth": #method must be followed by a string
                    offset += 1
                    method = list_arg[i+offset]
                    continue
                if arg == "-RmaxoverVir": #ratio of radii must be followed by a number
                    offset += 1
                    Rmaxovervir = float(list_arg[i+offset])
                    continue
                if arg == "-tol": #tolerance must be followed by a number
                    offset += 1
                    tolerance = float(list_arg[i+offset])
                    continue
                if arg == "-minmemb": #minimum member must be followed by a number
                    offset += 1
                    min_members = int(list_arg[i+offset])
                    continue
                if arg == "-datadir": #directory of galaxy data must be followed by a string
                    offset += 1
                    data_galaxies_dir = list_arg[i+offset]
                    continue
                if arg == "-ell": #ellipticity must be followed by a number
                    offset += 1
                    N_points = int(list_arg[i+offset])
                    do_ellipticity = "e"
                    continue
                if arg == "-output": #output file option must be followed by a string
                    offset += 1
                    output_file = list_arg[i+offset]
                    continue

                if arg == "-median": #median separation flag must either be followed by "only" for computing median separation alone or by nothing
                    if i+offset+1 < size and list_arg[i+offset+1] == "o":
                        offset += 1
                        median_flag = "o"
                    else:
                        median_flag = "y"
                    continue

                if arg == "-minmax": #minimum and maximum ID clusters must be followed by two numbers (like this for instance : 0 99)
                    id_cluster_min_max = list_arg[i+offset+1] + " " + list_arg[i+offset+2]
                    offset += 2
                    continue

                if arg == "-center": #centering option must be followed by nothing (other than another option)
                    do_center = "c"
                    continue
                if arg == "-back": #background option
                    do_background = "bg"
                    continue

                if arg == "-data": #data choice
                    offset += 1
                    test_flag = list_arg[i+offset]
                    if test_flag == "A": #Emmanuel Academic must be followed in the same order by cluster richness (number), ellipticity (number), PA (number), background (number)
                        cluster_richness    = int(list_arg[i+offset+1])
                        cluster_ellipticity = float(list_arg[i+offset+2])
                        cluster_PA          = float(list_arg[i+offset+3])
                        background          = int(list_arg[i+offset+4])
                        offset += 4
                    continue

                else:
                    print("Unknow parameter ", list_arg[i+offset])
                    exit(-1)
            #print(verbosity, test_flag, model_tmp, method, Rmaxovervir, tolerance, min_members, data_galaxies_dir, do_ellipticity, median_flag, id_cluster_min_max, do_center, do_background, output_file, "\n", cluster_richness, cluster_ellipticity, cluster_PA, background)
            #exit(0)

        elif sys.argv[1] == "-auto":
            auto = "auto"
        else:
            print("Unknown first argument ", sys.argv[1], ". Please enter either -man followed by all parameters, -auto or nothing")
            exit(-2)
    #if not enough arguments use manual mode (program asks to enter values manually and stores them in input file for later purposes)
    else:
        auto = "manual"

    # initialization
    version = "1.16"
    vdate = "24 May 2018"
    version_date   = version + ' (' + vdate + ')'
    falsetruearr = np.array([False, True])
    falsearr = np.array([False])
    truearr = np.array([True])
    degree    = np.pi/180.
    models    = np.array(['NFW', 'coredNFW', 'Uniform', 'NFWtrunc'])
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
    fmt  = "{:19} {:5} {:40} {:12} {:.1e} {:d} {:d} {:d} {:5} {:8} {:8.4f} {:8.4f} {:10.3f} {:5.3f} {:3.0f} {:6.2f} {:6.3G} {:6.1f} {:15.7G} {:15.7G} {:5d} {:8.4f} {:8.4f} {:10.3f} {:5.3f} {:3.0f} {:6.2f} {:8.3f}"
    fmt_tmp = "{:19} {:5} {:40} {:12} {:.1e} {:d} {:d} {:d} {:5} {:8} {:8.4f}"
    fmt2 = fmt + "\n"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # ftest = open(debug_file,'w')
    # ftest.close()

    print ("Welcome to PROFCL: version", version_date)

    # read default answers
    if auto == "manual" or auto == "auto":
        f_test_found = 1
        try:
            f_test = open("PROFCL_input.dat","r")
        except IOError:
            f_test_found = 0
            print("PROFCL (Main): cannot open PROCL_input.dat")
        else:
            with f_test:
                params_list = f_test.read().splitlines()
                [verbosity,test_flag,model_tmp,cluster_richness,cluster_ellipticity,cluster_PA,background,Rmaxoverrvir,do_center,do_ellipticity,N_points,do_background,method,tolerance,id_cluster_min_max,min_members,data_galaxies_dir, median_flag, output_file] = params_list

    # verbosity
    if auto == "manual":
        if 'verbosity' in locals():
            verbosity = input("Enter verbosity (0 -> minimal, 1 -> some, 2 -> more, 3 -> all) [%s]: "%verbosity) or verbosity
            # verbosity = int(input("Enter verbosity (0 -> minimal, 1 -> some, 2 -> more, 3 -> all) [%s]: "%verbosity + chr(8)*4))
        else:
            verbosity = input("Enter verbosity (0 -> minimal, 1 -> some, 2 -> more, 3 -> all): ")

        if 'test_flag' in locals():
            test_flag = input("Enter d for data, a for academic (Gary), A for academic (Emmanuel), f for Flagship, t to test [%s]: "%test_flag) or test_flag
        else:
            test_flag = input("Enter d for data, a for academic (Gary), A for academic (Emmanuel), f for Flagship, t to test: ")
    verbosity = int(verbosity)

    # questions for cluster file
    if test_flag in ('d','f'):
        cluster_richness = -1
        cluster_ellipticity = -1
        cluster_PA = -1
        background = -1
        Rmaxoverrvir = -1
        mock_number = -1

    elif test_flag in ('a','A'):
        if auto == "manual":
            if 'cluster_richness' in locals():
                cluster_richness = input("Enter cluster richness to test [%s]: "%cluster_richness) or cluster_richness
            else:
                cluster_richness = input("Enter cluster richness to test: ")
            if 'cluster_ellipticity' in locals():
                cluster_ellipticity = input("Enter cluster ellipticity to test [%s]: "%cluster_ellipticity) or cluster_ellipticity
            else:
                cluster_ellipticity = input("Enter cluster ellipticity to test: ")
        cluster_ellipticity_10 = str(int(10*float(cluster_ellipticity)))

    if test_flag in ('d','f'):
        if auto == "manual":
            if 'model_tmp' in locals():
                model_tmp = input("Enter cluster model: (n for NFW, c for coredNFW, u for Uniform) [%s]: "%model_tmp) or model_tmp
            else:
                model_tmp = input("Enter cluster model: (n for NFW, c for coredNFW, u for Uniform): ")
    elif test_flag == 'A':        # Academic mock from Emmanuel Artis
        model_tmp = 'NFW'
        cluster_PA = 50
        background = 20
        Rmaxoverrvir = -1
    elif test_flag == 'a':      # Academic mock from Gary Mamon
        if auto == "manual":
            if 'model_tmp' in locals():
                model_tmp = input("Enter cluster model: (n for NFW, c for coredNFW, u for Uniform) [%s]: "%model_tmp) or model_tmp
            else:
                model_tmp = input("Enter cluster model: (n for NFW, c for coredNFW, u for Uniform): ")

            if 'cluster_PA' in locals():
                cluster_PA = input("Enter cluster PA to test [%s]: "%cluster_PA) or cluster_PA
            else:
                cluster_PA = input("Enter cluster PA to test: ")
            if 'background' in locals():
                background = input("Enter background density (arcmin^-2) [%s]: "%background) or background
            else:
                background = input("Enter background density (arcmin^-2): ")
            if 'Rmaxoverrvir' in locals():
                Rmaxoverrvir = input("Enter R_max/r_vir [%s]: "%Rmaxoverrvir) or Rmaxoverrvir
            else:
                Rmaxoverrvir = input("Enter R_max/r_vir: ")
            if f_test_found > 0:
                f_test.close()
    elif test_flag = 't':       # file to test
        if 'data_galaxies_dir' in locals():
            data_galaxies_dir = input("Enter galaxy data directory [%s]: "%data_galaxies_dir) or data_galaxies_dir
        else:
            data_galaxies_dir = input("Enter galaxy data directory: ")
        if 'data_galaxies_prefix' in locals():
            data_galaxies_prefix = input("Enter galaxy data file prefix [%s]: "%data_galaxies_prefix) or data_galaxies_prefix
        else:
            data_galaxies_prefix = input("Enter galaxy data file prefix: ")

                
    model = convertModel(model_tmp)
    # mock file name prefix and minimum number of member galaxies

    min_members = 1
    if test_flag == 'd':
        if auto == "manual":
            if 'data_galaxies_dir' in locals():
                data_galaxies_dir = input("Enter galaxy data directory [%s]: "%data_galaxies_dir) or data_galaxies_dir
            else:
                data_galaxies_dir = input("Enter galaxy data directory: ")
            if 'data_galaxies_prefix' in locals():
                data_galaxies_prefix = input("Enter galaxy data file prefix [%s]: "%data_galaxies_prefix) or data_galaxies_prefix
            else:
                data_galaxies_prefix = input("Enter galaxy data file prefix: ")
            if 'data_clusters_dir' in locals():
                data_clusters_dir = input("Enter cluster data directory [%s]: "%data_clusters_dir) or data_clusters_dir
            else:
                data_clusters_dir = input("Enter cluster data directory: ")
            if 'data_clusters_prefix' in locals():
                data_clusters_prefix = input("Enter cluster data file prefix [%s]: "%data_clusters_prefix) or data_clusters_prefix
            else:
                data_clusters_prefix = input("Enter cluster data file prefix: ")
            if 'id_cluster_min_max' in locals():
                id_cluster_min_max = input("Enter minimum and maximum cluster IDs [%s]: "%id_cluster_min_max) or id_cluster_min_max
            else:
                id_cluster_min_max = input("Enter minimum and maximum cluster IDs: ")
            if 'min_members' in locals():
                min_members = int(input("Enter minimum number of cluster members [%s]: "%min_members) or min_members)
            else:
                min_members = int(input("Enter minimum number of cluster members: "))
        id_cluster_min, id_cluster_max = map(int, id_cluster_min_max.split())

    elif test_flag == 'a':
        if float(background) > 0.:
            galaxy_file = 'Acad_' + model + cluster_richness + '_c3_skyRA15Dec10_ellip' + cluster_ellipticity_10 + 'PA' + cluster_PA + '_bg' + background + '_Rmax' + Rmaxoverrvir + '.dat'
        else:
            galaxy_file = 'Acad_' + model + cluster_richness + '_c3_skyRA15Dec10_ellip' + cluster_ellipticity_10 + 'PA' + cluster_PA + '.dat'
        id_cluster_min = 1
        id_cluster_max = 1
        column_z = 0

    elif test_flag == 'A':
        # if 'mock_number' in locals():
        #     mock_number = input("Enter random realization (0-99, -1 for all) [%s]: "%mock_number) or mock_number
        # else:
        #     mock_number = input("Enter random realization (0-99, -1 for all): ")
        # if mock_number == '-1':
        #     id_cluster_min = 0
        #     id_cluster_max = 99
        # else:
        #     id_cluster_min = int(mock_number)
        #     id_cluster_max = int(mock_number)
        if auto == "manual":
            if 'id_cluster_min_max' in locals():
                id_cluster_min_max = input("Enter minimum and maximum cluster IDs [%s]: "%id_cluster_min_max) or id_cluster_min_max
            else:
                id_cluster_min_max = input("Enter minimum and maximum cluster IDs: ")

            #Directory containing input data
            if 'data_galaxies_dir' in locals():
                data_galaxies_dir = input("Enter galaxy data directory [%s]: "%data_galaxies_dir) or data_galaxies_dir
            else:
                data_galaxies_dir = input("Enter galaxy data directory: ")
        id_cluster_min, id_cluster_max = map(int, id_cluster_min_max.split())
        column_z = 0

    elif test_flag == 'f':
        if auto == "manual":
            if 'data_galaxies_dir' in locals():
                data_galaxies_dir = input("Enter galaxy data directory [%s]: "%data_galaxies_dir) or data_galaxies_dir
            else:
                data_galaxies_dir = input("Enter galaxy data directory: ")
            if 'id_cluster_min_max' in locals():
                id_cluster_min_max = input("Enter minimum and maximum cluster IDs [%s]: "%id_cluster_min_max) or id_cluster_min_max
            else:
                id_cluster_min_max = input("Enter minimum and maximum cluster IDs: ")
            if 'min_members' in locals():
                min_members = int(input("Enter minimum number of cluster members [%s]: "%min_members) or min_members)
            else:
                min_members = int(input("Enter minimum number of cluster members: "))
        data_galaxies_prefix = "galaxies_inhalo_clusterlM14"
        data_clusters_dir = "."
        data_clusters_prefix = "lM14_lookup"
        id_cluster_min, id_cluster_max = map(int, id_cluster_min_max.split())

    else:
        raise print("PROF-CL (MAIN): cannot recognize choice = ", test_flag)

    # min & max projected (angular) radii

    # if 'R_min_allow' in locals():
    #     R_min_kpc_allow = int(input("Enter minimum allowed projected physical radius (kpc) [%s]: "%R_min_kpc_allow) or R_min_kpc_allow)
    # else:
    #     R_min_kpc_allow = int(input("Enter minimum allowed projected physical radius (kpc): "))
    # if 'R_max_allow' in locals():
    #     R_max_kpc_allow = int(input("Enter maximum allowed projected physical radius (kpc) [%s]: "%R_max_kpc_allow) or R_max_kpc_allow)
    # else:
    #     R_max_kpc_allow = int(input("Enter maximum allowed projected physical radius (kpc): "))
    
    # query strategies:

    # 1) re-centering
    if auto == "manual":
        ok = 0
        while ok == 0:
            if 'do_center' in locals():
                do_center = input("Enter c for centering, n for none, b for both [%s]: "%do_center) or do_center
            else:
                do_center = input("Enter c for centering, n for none, b for both: ")
            # GAM: need to add BCG one day
            if do_center in ('c','n','b'):
                ok = 1
    if do_center == 'c':
        recenter_flags = truearr
    elif do_center == 'n':
        recenter_flags = falsearr
    else:
        recenter_flags = falsetruearr

    # 2) ellipticity
    if auto == "manual":
        ok = 0
        while ok == 0:
            if 'do_ellipticity' in locals():
                do_ellipticity = input("Enter e for ellipticity/PA, n for none, b for both [%s]: "%do_ellipticity) or do_ellipticity
            else:
                do_ellipticity = input("Enter e for ellipticity/PA, n for none, b for both: ")
            if do_ellipticity in ('e','n','b'):
                ok = 1
    if do_ellipticity == 'e':
        ellipticity_flags = truearr
    elif do_ellipticity == 'n':
        ellipticity_flags = falsearr
    else:
        ellipticity_flags = falsetruearr

    # integration of model within region: analytical (a), Monte-Carlo (m), or none (x)?
    if auto == "manual":
        if do_ellipticity == 'e' and do_center == 'n':
            if 'N_points' in locals():
                N_points = input("Enter 0 for approximate analytical or N_points >= 10000 for Monte-Carlo [%s]: "%N_points) or N_points
            else:
                N_points = input("Enter 0 for approximate analytical or N_points >= 10000 for Monte-Carlo: ")
        elif do_center == 'c':
            if 'N_points' in locals():
                N_points = input("Enter N_points for Monte-Carlo [%s]: "%N_points) or N_points
            else:
                N_points = input("Enter N_points for Monte-Carlo: ")
        else:
            N_points = 0
            # integrate_model_in_region = 'x'

    N_points = int(N_points)
    # integrate_model_in_region = 'm'
    if N_points < 10000:
        if (do_ellipticity == 'e' and do_center == 'n') and N_points != 0:
            raise print("The number should be 0 or >= 10000")
        elif do_center == 'c':
            raise print("N_points should be >= 10000")

    # 3) background
    if auto == "manual":
        ok = 0
        while ok == 0:
            if 'do_background' in locals():
                do_background = input("Enter bg for background, n for none, b for both [%s]: "%do_background) or do_background
            else:
                do_background = input("Enter bg for background, n for none, b for both: ")
            if do_background in ('bg','n','b'):
                ok = 1
    if do_background == 'bg':
        background_flags = truearr
    elif do_background == 'n':
        background_flags = falsearr
    else:
        background_flags = falsetruearr

    # query minimization method:
    found  = 0
    while found == 0:
        if auto == "manual":
            print("Available minimization methods are ", list(methods_dict.keys()))
            print("... can be abbreviated as ", list(methods_dict.values()))
            if 'method' in locals():
                method = input("Enter minimization method (lower case OK, 'test' for test) [%s]: "%method) or method
            else:
                method = input("Enter minimization method (lower case OK, 'test' for test): ")
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

    # re-scale parameters?

    ok = 0
    while ok == 0:
        if 'do_rescale_params' in locals():
            do_rescale_params = input("Rescale parameters for similar precisions? (y|n) [%s]: "%do_rescale_params) or do_rescale_params
        else:
            do_rescale_params = input("Rescale parameters for similar precisions? (y|n): ")
        if do_rescale_params in ('y','n'):
            ok = 1

    # sub-clusters

    if 'n_subclusters' in locals():
        n_subclusters = int(input("Enter maximum number of subclusters: [%s]: "%n_subclusters) or n_subclusters)
    else:
        n_subclusters = int(input("Enter maximum number of subclusters: "))

    # query tolerance for minimization
    if auto == "manual":
        if 'tolerance' in locals():
            tolerance = input("Enter relative tolerance of minimization method [%s]: " %tolerance) or tolerance
        else:
            tolerance = input("Enter relative tolerance of minimization method: ")
    tolerance = float(tolerance)

    #Check if user wants to perform median separation method
    if auto == "manual":
        if 'median_flag' in locals():
            median_flag = input("Do you want to perform median separation method (y for yes, n for no, o for median separation only) [%s]: " %median_flag) or median_flag
        else:
            median_flag = input("Do you want to perform median separation method")

    # save option choices to file for use as defaults in next run
    my_list = [verbosity,test_flag,model,cluster_richness,cluster_ellipticity,cluster_PA,background,Rmaxoverrvir,do_center,do_ellipticity,N_points,do_background,method,tolerance,id_cluster_min_max,min_members, data_galaxies_dir, median_flag, output_file]
    # print("my_list = ", my_list)
    f_test = open('PROFCL_input.dat','w')
    for item in my_list:
        if verbosity >= 3:
            print("item = ", item)
        f_test.write("%s\n" % item)
    f_test.close()

    # cluster centers
    if test_flag == 'd' or test_flag == 'f':
        clusters_file = data_clusters_dir + '/' + data_clusters_prefix + '.dat'
        (ID_all, RA_cen_all, Dec_cen_all) = PROFCL_ReadClusterCenters(clusters_file,column_RA_cen)
        idx_clusters = np.arange(0, len(ID_all))
    elif test_flag == 'a':
        (RA_cen_init0, Dec_cen_init0) = (15, 10)
    elif test_flag == 'A':
        (RA_cen_init0, Dec_cen_init0) = (0, 0)

    # open output file in append mode
    f_out = open(output_file, 'a')

    # loop over clusters

    if test_flag == 'f':
        print("type ID_all =",type(ID_all))
        print("ID_all=",ID_all)
    
    for id_cluster in range(id_cluster_min,id_cluster_max+1):
        print ("\nid_cluster = ", id_cluster)

        # convert cluster ID to cluster galaxy file name
        # TBD

        # galaxy file

        if test_flag == 'd' or test_flag == 'f':
            galaxy_file0 = data_galaxies_prefix + '_' + str(id_cluster) + '.dat'
            galaxy_file = data_galaxies_dir + '/' + galaxy_file0
        elif test_flag == 'a':
            if float(background) > 0.:
                galaxy_file0 = 'Acad_' + str(model) + str(cluster_richness) + '_c3_skyRA15Dec10_ellip' + str(cluster_ellipticity_10) + 'PA' + str(cluster_PA) + '_bg' + str(background) + '_Rmax' + str(Rmaxoverrvir) + '.dat'
            else:
                galaxy_file0 = 'Acad_' + str(model) + str(cluster_richness) + '_c3_skyRA15Dec10_ellip' + str(cluster_ellipticity_10) + 'PA' + str(cluster_PA) + '.dat'
        elif test_flag == 'A':
            galaxy_file0 = 'MockNFW' + str(cluster_richness) + 'ellip' + str(cluster_ellipticity_10) + 'loga2.08PA50center0With_background20num' +  str(id_cluster) + '.dat'

        if test_flag == 'a' or test_flag == 'A':
            galaxy_file = data_galaxies_dir + '/' + galaxy_file0

        if verbosity >= 1:
            print("galaxy_file = ", galaxy_file)
            
        # cluster center
        if test_flag == 'd' or test_flag == 'f':
            RA_cen_init0 = np.asscalar(RA_cen_all[ID_all==id_cluster])
            Dec_cen_init0 = np.asscalar(Dec_cen_all[ID_all==id_cluster])
            if verbosity >= 1:
                print("RA_cen_init0=",RA_cen_init0,"Dec_cen_init0=",Dec_cen_init0)
            
        # read galaxy data (coordinates in degrees)
        (RA, Dec, prob_membership) = PROFCL_ReadClusterGalaxies(galaxy_file,column_RA,column_pmem,column_z)
        N_data = len(RA)
        if verbosity >= 1:
            print("*** cluster ", id_cluster, "RA_cen=", RA_cen_init0, "Dec_cen=",Dec_cen_init0,"N_data =",N_data)

        # skip cluster if too few members
        if N_data < min_members:
            if verbosity >= 1:
                print("skipping cluster ", id_cluster, " which only has ", N_data, " members")
            continue

        # if median separation was activated
        if median_flag == 'y' or median_flag == 'o':
            #compute computation time
            start = time.process_time()
            # median separation (arcmin)
            if verbosity >= 1 and N_data > 100:
                print("computing median separation ...")
            log_median_separation = np.log10(Median_Separation(RA,Dec))
            time_taken = time.process_time() - start
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
            RA += 180
        if RA_cen_init > 360:
            RA_cen_init -= 360.
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
            print("MAIN: min(cosR_sky) = ", np.min(cosR_sky))
            print("MAIN: maxR = ", maxR)
            print("MAIN: R_min set to ",R_min)
        R_max = R_max_over_maxR * maxR
        if test_flag == 'a':
            R_max = 0.04932   # 2 r_vir

        # maximum background surface density to prevent negative numerator in galaxy probability
        condition = np.logical_and(R_sky >= R_min, R_sky <= R_max)
        log_background_maxallow_default = np.log10(len(R_sky[condition])/(np.pi*(R_max*R_max-R_min*R_min)))

        # reduce by factor to avoid errors from cartesian geometry
        log_background_maxallow_default = np.log10(0.75) + log_background_maxallow_default
                        
        print("MAIN: R_min (deg) = ", R_min, "\n      R_max (deg) = ", R_max, "\n      N_tot = ", len(R_sky[condition]))

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

        # # test ellipticity and PA
        # log_scale_radius = -3.
        # ellipticity = 0.
        # PA = 0.
        # log_background = -99.
        # params =  RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background
        # model = models[0]
        # RA_cen_minallow = RA_cen - 1
        # RA_cen_maxallow = RA_cen + 1
        # Dec_cen_minallow = Dec_cen - 1
        # Dec_cen_maxallow = Dec_cen + 1
        # log_scale_radius_minallow = log_scale_radius - 1
        # log_scale_radius_maxallow = log_scale_radius + 1
        # log_background_minallow = log_background - 1
        # log_background_maxallow = log_background + 1
        # ellipticity_minallow = ellipticity - 1
        # ellipticity_maxallow = ellipticity + 1
        # PA_minallow = PA - 1
        # PA_maxallow = PA + 1
        # PROFCL_LogLikelihood(params, RA, Dec, prob_membership, R_min, R_max, model)

        # guess ellipticity and PA using 2nd moments of distribution
        
        ellipticity_pred, PA_pred = guess_ellip_PA(RA,Dec,RA_cen_init,Dec_cen_init)
        print("MAIN: ellipticity_pred = ",ellipticity_pred,"\n      PA_pred = ", PA_pred)

        # bounds on parameters according to case
        
        if not np.any(models == model):
            print("model ", model, " not recognized")
            exit()
                        
        # prepare fits for various cases:
        # for recenter_flag in falsetruearr:                 #  fixed or free center
        N_params = 1    # log-scale-radius
        for recenter_flag in recenter_flags:
            if recenter_flag:
                N_params = N_params + 2
            # for ellipticity_flag in falsetruearr:          #  circular or elliptical model
            for ellipticity_flag in ellipticity_flags:
                if ellipticity_flag:
                    N_params = N_params + 2
                # for background_flag in falsetruearr:       #  without or with background
                for background_flag in background_flags:
                    if verbosity >= 2:
                        print("** MAIN: recenter ellipticity background flags = ", 
                              recenter_flag, ellipticity_flag, background_flag)
                    if background_flag:
                        N_params = N_params + 1

                    if verbosity >= 2:
                        print("* MAIN: model = ", model)

                    log_scale_radius_maxallow = np.log10(R_min / min_R_over_rminus2)
                    if R_min == 0:
                        log_scale_radius_minallow = log_scale_radius_maxallow - 2
                    else:
                        log_scale_radius_minallow = np.log10(R_max / max_R_over_rminus2)
                        ### FUDGE FOR TESTING
                        log_scale_radius_minallow = np.log10(R_min)
                        log_scale_radius_maxallow = np.log10(R_max)
                        # else:
                        #     log_scale_radius_minallow = np.log10(R_min)
                        # guess
                    log_scale_radius = np.log10 ( 1.35 * np.median(R_sky) )
                    # factor 1.35 is predicted median for NFW and coredNFW of concentration 3

                    # data: start with initial value of central position
                    # mocks: start with median position
                    # if test_flag != 'd':
                    #     RA_cen = np.median(RA)
                    #     Dec_cen = np.median(Dec)
                    # else:
                    #     RA_cen  = RA_cen_init
                    #     Dec_cen = Dec_cen_init
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
                        ellipticity_maxallow = 0.8
                        PA_minallow = 0.0
                        PA_maxallow = 180.
                        ellipticity = ellipticity_pred
                        PA = PA_pred

                    if not background_flag:
                        log_background_minallow = -99.
                        log_background_maxallow = -99.
                    else:
                        # observations (Metcalfe+06) => integrated counts of 80/arcmin^2 down to H=24.5
                        #                            => log(arcmin^2 background) ~= 1.9
                        #                               expect 30/arcmin^2 in Euclid
                        #                               => log ~= 1.5
                        # use consistency in z_phot     => log ~= 0.5
                        # values in deg^{-2}
                        log_background_maxallow = log_background_maxallow_default
                        # log_background_minallow = 0.5 - 2.5 + 2.*np.log10(60.)
                        # if log_background_minallow > log_background_maxallow:
                        #     log_background_minallow = log_background_maxallow - 1
                        log_background_minallow = log_background_maxallow - 6
                        # log_background_maxallow = 0.5 + 2.5 + 2.*np.log10(60.)
                        # do not set max background here, as it was set above

                        # TEMPORARY FOR TESTS!
                        # RA_cen = RA_cen - 0.3*(RA_cen_maxallow-RA_cen_minallow)
                        # Dec_cen = Dec_cen + 0.4*(Dec_cen_maxallow-Dec_cen_minallow)
                        
                    log_background = (log_background_minallow
                                          + log_background_maxallow) / 2.

                    # add extra parameter for NFWtrunc
                        
                    if model == 'NFWtrunc':
                        R_cut_minallow = R_max / 5
                        R_cut_maxallow = 2 * R_max
                        R_cut          = 0.7 * R_max
                        N_params       = N_params + 1
                    else:
                        R_cut_minallow = 2 * R_max
                        R_cut_maxallow = R_cut_minallow
                        R_cut = R_cut_minallow

                    if verbosity >= 1:
                        print("RA_cen_minallow = ", RA_cen_minallow, "RA_cen_maxallow = ", RA_cen_maxallow, " RA_guess = ", RA_cen)
                        print("Dec_cen_minallow = ", Dec_cen_minallow, "Dec_cen_maxallow = ", Dec_cen_maxallow, " Dec_guess = ", Dec_cen)
                        print("log_scale_radius_minallow = ", log_scale_radius_minallow, "log_scale_radius_maxallow = ", log_scale_radius_maxallow, " log_scale_radius_guess = ", log_scale_radius)
                        print("log_background_minallow = ", log_background_minallow, "log_background_maxallow = ", log_background_maxallow, " log_background_guess = ", log_background)
                        print("ellipticity_minallow = ", ellipticity_minallow, "ellipticity_maxallow = ", ellipticity_maxallow, " ellipticity_guess = ", ellipticity)
                        print("PA_minallow = ", PA_minallow, "PA_maxallow = ", PA_maxallow, " PA_guess = ", PA)
                        print("R_cut_minallow = ", R_cut_minallow, "R_cut_maxallow = ", R_cut_maxallow, " R_cut_guess = ", R_cut)

                    bounds = np.array([(RA_cen_minallow,RA_cen_maxallow),
                                       (Dec_cen_minallow,Dec_cen_maxallow),
                                       (log_scale_radius_minallow,log_scale_radius_maxallow),
                                       (ellipticity_minallow,ellipticity_maxallow),
                                       (PA_minallow,PA_maxallow),
                                       (log_background_minallow,log_background_maxallow),
                                       (R_cut_minallow,R_cut_maxallow)
                    ])
                        
                    # interactive testing method
                    if method == 'test':
                        cont = 'y'
                        iPass = 0
                        while cont == 'y':
                            # enter desired parameters
                            print("enter value")
                            # log_scale_radius in deg
                            log_scale_radius = float(input("Enter log_scale_radius (arcmin): ")) - np.log10(60.)
                            RA_cen = float(input("Enter RA_cen: "))
                            Dec_cen = float(input("Enter Dec_cen: "))
                            ellipticity = float(input("Enter ellipticity: "))
                            PA = float(input("Enter PA: "))
                            background = float(input("Enter background (arcmin^{-2}): "))
                            R_cut = float(input("Enter truncation radius (arcmin): "))
                            log_background = np.log10 (background * 3600.)        # in deg^{-2}
                            # set min and max allowed values to desired values
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
                            R_cut_minallow = R_cut
                            R_cut_maxallow = R_cut
                            if rescale_flag == 'y':
                                params = [RA_cen/R_max/np.cos(Dec_cen_init), Dec_cen/R_max, log_scale_radius, ellipticity/5., PA/500., log_background/10., np.log10(R_cut)/5.]
                            else:
                                params = [RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, R_cut]
                            if model == 'NFWtrunc':
                                R_cut = float(input("Enter truncation radius (arcmin): "))
                                R_cut_minallow = R_cut
                                R_cut_maxallow = R_cut
                                # params = params + [R_cut]

                            # - ln L

                            lnlikminus = PROFCL_LogLikelihood(params,RA,Dec,prob_membership,
                                                                      R_min, R_max, model)
                            print("-lnL = ", lnlikminus)
                            cont = input("Continue (y|n)? ")
                        exit()
                        # end of test block

                    # loop over max number of sub-clusters
                    for n_sub in range(n_subclusters):

                        #time taken till there by the program
                        start_time = time.process_time()

                        #make sure previous predicted values are not overwritten
                        log_scale_radius_pred = log_scale_radius
                        ellipticity_pred      = ellipticity
                        RA_cen_pred           = RA_cen
                        Dec_cen_pred          = Dec_cen
                        PA_pred               = PA
                        log_background_pred   = log_background

                        if model == 'NFWtrunc':
                            R_cut_pred        = RA_cut

                        if median_flag != 'o':
                            # perform maximum likelihood fit

                            # predict central position

                            # see mail of 24 May

                            
                            if rescale_flag == 'y':
                                # convert physical parameters to fitted parameters, so that desired absolute precisions are equal among all parameters
                                params = [log_background/10., np.log10(R_cut)/5.], RA_cen/R_max/np.cos(Dec_cen_init*degree), Dec_cen/R_max, log_scale_radius, ellipticity/5., PA/500.]
                            else:
                                params = [log_background, R_cut, RA_cen, Dec_cen, log_scale_radius, ellipticity, PA]
                            fit_results = PROFCL_Fit(data, params, bounds,
                                                     R_min, R_max, model, 
                                                     background_flag, recenter_flag, ellipticity_flag, 
                                                     PROFCL_LogLikelihood, method)

                            #get time taken for the computation
                            time_taken = time.process_time() - start_time
                            print("Computation time = ", time_taken)

                            params_bestfit = fit_results.x

                            print("echo : ", fit_results.nfev, fit_results.nit, fit_results.success)

                            if verbosity >=1:
                                print ("params_bestfit = ",params_bestfit)

                            [RA_cen_bestfit, Dec_cen_bestfit, log_scale_radius_bestfit, ellipticity_bestfit, PA_bestfit, log_background_bestfit, R_cut_bestfit] = params_bestfit
                            # convert fitted parameters to physical values
                            if rescale_flag == 'y':
                                RA_cen_bestfit, Dec_cen_bestfit = R_max * np.cos(Dec_cen_init*degree) * RA_cen_bestfit, R_max * Dec_cen_bestfit
                                ellipticity_bestfit = 5. * ellipticity_bestfit
                                PA_bestfit = 500. * PA_bestfit
                                log_background_bestfit = 10. * log_bakcground_bestfit
                                R_cut_bestfit = 10.**(5.*R_cut_bestfit)

                            # convert central position back to equatorial coordinates if galactic coordinates had been used
                            if eq2gal_flag:
                                coords_galactic = SkyCoord(RA_cen,Dec_cen,frame='galactic',unit='deg')
                                coords_equatorial = coords_galactic.icrs
                                RA_cen_pred  = coords_equatorial.ra
                                Dec_cen_pred = coords_equatorial.dec
                                print("test   ", RA_cen_pred, Dec_cen_pred)

                            # shift back RA_cen if initial RA_cen was close to 0 and make sure that RA_cen is not negative
                            if RA_shift_flag:
                                RA_cen_bestfit -= 180.
                                RA_cen_pred    -= 180.
                                if verbosity >= 2:
                                    print("shifting back RA, now at ", RA_cen_bestfit)
                            if RA_cen_bestfit < 0:
                                RA_cen_bestfit += 360.
                            if RA_cen_pred < 0:
                                RA_cen_pred += 360.
                            print("test2   ", RA_cen_pred, Dec_cen_pred)

                            # Bayesian evidence:
                            # Note that N_data is the initial value, 
                            # which gets cut by R_min,R_max, 
                            # but this matters little, since BIC ~ ln(N_data) + cst)
                            BIC = 2.*fit_results.fun + N_params * np.log(N_data)

                            if verbosity >= 4:
                                print("date_time=",date_time)
                                print("galaxy_file0=",galaxy_file0)
                                print("method=",method)
                                print("tolerance=",tolerance)
                                print("recenter_flag=",recenter_flag)
                                print("ellipticity_flag=",ellipticity_flag)
                                print("background_flag=",background_flag)
                                print("N_points=",N_points)
                                print("model=",model)
                                for i in range(5+1):
                                    print ("i params_bestfit = ", i, params_bestfit[i])
                                    print ("-ln L_max = ", fit_results.fun, "type = ", type(fit_results.fun))
                                    #lnL_MLE_minus = fit_results.fun.astype(float)
                                    #print("now type = ", type(lnL_MLE_minus))
                                    print ("BIC = ", BIC, "type = ", type(BIC))
                                    print ("N_eval = ", fit_results.nfev, "type = ", type(fit_results.nfev))
                                    print("bounds = ", bounds)

                            # cluster 3D normalization N(r_scale) = N(r_{-2}): only makes sense for circular fit without re-centering

                            Ntot_maxR = len(RA)
                            # N_r_minus_bestfit = (Ntot_maxR - np.pi * maxR**2 * 10**log_background_bestfit) / Number_tilde(maxR/10**log_scale_radius_bestfit,model)
                            num = Ntot_maxR - np.pi * maxR**2 * 10**log_background_bestfit
                            denom = Number_tilde(maxR/10**log_scale_radius_bestfit,model)
                            print("MAIN: num = ",num, "\n      denom = ", denom, "\n      maxR = ", maxR,"\n      arg = ", maxR/10**log_scale_radius_bestfit)
                            print("      log_scale_radius_bestfit = ",log_scale_radius_bestfit)
                            N_r_minus_bestfit = num/denom

                            # print results to screen
                            print (" date/time          cluID/file              method      tol c e b int model    RA0       Dec0    log_r_{-2} ellip  PA log_bg   -lnL       BIC      Npasses     RA_pred       Dec_pred        log_r_pred      ellip_pred       PA_pred       log_bg_pred       Computation_time(s)")

                            sN_points = '%.g' % N_points
                            print (fmt.format(
                                date_time, version, galaxy_file, method, tolerance,
                                recenter_flag, ellipticity_flag, background_flag, sN_points, model, 
                                RA_cen_bestfit, Dec_cen_bestfit, log_scale_radius_bestfit, ellipticity_bestfit, PA_bestfit, log_background_bestfit, R_cut_bestfit, N_r_minus_bestfit,
                                fit_results.fun, BIC, fit_results.nfev,
                                RA_cen_pred, Dec_cen_pred, log_scale_radius_pred, ellipticity_pred, PA_pred, log_background_pred, time_taken)
                            )
                            # print results to file
                            f_out.write (fmt2.format(
                                date_time, version, galaxy_file, method, tolerance, recenter_flag, ellipticity_flag, 
                                background_flag, sN_points, model, 
                                RA_cen_bestfit, Dec_cen_bestfit, log_scale_radius_bestfit, ellipticity_bestfit, PA_bestfit, log_background_bestfit, R_cut_bestfit, N_r_minus_bestfit,
                                fit_results.fun, BIC, fit_results.nfev,
                                RA_cen_pred, Dec_cen_pred, log_scale_radius_pred, ellipticity_pred, PA_pred, log_background_pred, time_taken)
                            )

        # print also median separation using same format
        if median_flag == 'y' or median_flag == 'o':
            log_background_best_fit = -99
            print        (fmt.format(date_time, version, galaxy_file, "median-sep", 0, 0, 0, 0, 'x', "none", RA_cen_init0, Dec_cen_init0, log_median_separation, 0, 0, log_background_best_fit, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, time_taken))
            f_out.write (fmt2.format(date_time, version, galaxy_file, "median-sep", 0, 0, 0, 0, 'x', "none", RA_cen_init0, Dec_cen_init0, log_median_separation, 0, 0, log_background_best_fit, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, time_taken))

    # end of cluster loop

    f_out.close()

    ## EXTRAS for testing

    ## transform from RA,Dec to cartesian and then to projected radii
    
    # transform coordinate units from degrees to radians
    RA_in_rd      =      RA * degree
    Dec_in_rd     =     Dec * degree
    RA_cen_in_rd  =  RA_cen * degree
    Dec_cen_in_rd = Dec_cen * degree
    PA_in_rd      =      PA * degree
    
    # x     = np.cos(Dec_in_rd)     * np.cos(RA_in_rd)
    # y     = np.cos(Dec_in_rd)     * np.sin(RA_in_rd)
    # x_cen = np.cos(Dec_cen_in_rd) * np.cos(RA_cen_in_rd)
    # y_cen = np.cos(Dec_cen_in_rd) * np.sin(RA_cen_in_rd)
    # transform coordinates to relative to central ones
    # dx = x - x_cen
    # dy = y - y_cen

    # relative positions (in degrees, careful that x points to decreasing RA)
    dx =  - (RA  - RA_cen) * np.cos(Dec_cen_in_rd)
    dy =     Dec - Dec_cen   
    
    # handle crossing of RA=0
    dx = np.where(dx >= 180.,  dx - 360., dx)
    dx = np.where(dx <= -180., dx + 360., dx)

    # # # guess ellipticity and PA from 2nd moments
    
    # xymean = np.mean(dx*dy)
    # x2mean = np.mean(dx*dx)
    # y2mean = np.mean(dy*dy)
    # tan2theta = 2*xymean/(x2mean-y2mean)
    # theta = 0.5*np.arctan(tan2theta) - np.pi/2.
    # # if xymean  < 0:
    # #     theta = -1*theta
    # PA_predict = theta / degree
    # if PA_predict < -90.:
    #     PA_predict = PA_predict + 180.
    # A2 = (x2mean+y2mean)/2. + np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    # B2 = (x2mean+y2mean)/2. - np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    # ellipticity_pred = 1 - np.sqrt(B2/A2)
    # print('predicted values: ellipticity = ', ellipticity_pred, ' PA = ', PA_predict,
    #       ' xymean=', xymean, ' tan2theta=', tan2theta)

    # exit()

    # rotate to axes of cluster (careful: PA measures angle East from North)
    u = - dx * np.sin(PA_in_rd) + dy * np.cos(PA_in_rd)
    v = - dx * np.cos(PA_in_rd) - dy * np.sin(PA_in_rd)
    # np.savetxt('PROFCL.uv',np.array([u,v]).T,header='u v')
    # exit()

    # equivalent circular projected radii (in degrees)
    R_circ_squared = u*u + v*v/(1.-ellipticity)**2.
    R_circ_squared_min = np.min(R_circ_squared)
    if R_circ_squared_min < 0.:
        # if min(R^2) < -1*(1 arcsec)^2, set to 0, else stop
        if R_circ_squared_min > -1/3600.**2.:
            if verbosity >= 2:
                print("min(R^2) slightly < 0!")
            R_circ_squared = np.where(R_circ_squared < 0.,
                                      0.,
                                      R_circ_squared
                                      )
        else:
            raise print('ERROR in PROFCL_lnlikelihood: min(R^2) = ', 
                        R_circ_squared_min/3600.**2, 
                        ' arcsec^2 ... cannot be < 0')

    # print ("last printouts...")
    # R_circ = np.sqrt(R_circ_squared)
    # # print("dx = ", dx)
    # # print("u = ", u)
    # print("R = ", R_circ)
    # print("median(R) = ", np.median(R_circ))
    # a_Python = 10**log_scale_radius_bestfit
    # a_SM = 0.01207
    # print("a_Python=", a_Python, "a_SM=",a_SM)
    # p_Python = PROFCL_prob_Rproj(R_circ, R_min, R_max, a_Python, model)
    # p_SM = PROFCL_prob_Rproj(R_circ, R_min, R_max, a_SM, model)
    # print("-ln p(a_Python) =", -np.log(p_Python))
    # print("-ln p(a_SM) =", -np.log(p_SM))
    # print(" why are -ln p not arrays?")

    # end of loops on cases and model
