#! /usr/local/bin/python3
# -*- coding: latin-1 -*-


# Program to extract structuiral parameters (size, ellipticity, position angle, center, background) of clusters

# Author: Gary Mamon, with help from Christophe Adami, Yuba Amoura, Emanuel Artis and Eliott Mamon

from __future__ import division
import numpy as np
import sys as sys
import datetime
import time
import getopt as getopt
from scipy.optimize import minimize, differential_evolution
from scipy import interpolate, integrate
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import matplotlib
matplotlib.use('GTK3Cairo')
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
            > incldued N_knots for coredNFW model

* Version 1.14.1 (31 May 2018) by G. Mamon
        > major re-write
        > fixed several bugs for elliptical and/or non-centered fits 
        > fixed bug on final N(r_{-2}) normalization
        > added penalty function for unconstrained minimization methods

* Version 2.0 (4 June 1989) by G. Mamon & W. Mercier
  (merge of versions 1.14.1 of G. Mamon and 1.17 of W. Mercier)
 
Not yet implemented:
    * red vs blue galaxies
    * forcing prob_mem = 1 for all galaxies
    * uncertainties from bootstraps (but the systematic errors should completely dominate the statistical errors)
    * split of cluster into sub-clusters (when cluster is obvisouly a merger of several)
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

def AngularSeparation(RA1,Dec1,RA2,Dec2,choice='cart'):
    """angular separation given (RA,Dec) pairs [input and output in degrees]
    args: RA_1, Dec_1, RA_2, Dec_2, choice
    choice can be 'astropy', 'trig', or 'cart'"""

    # author: Gary Mamon

    # tests indicate that:
    # 1) trig is identical to astropy, but 50x faster!
    # 2) cart is within 10^-4 for separations of 6 arcmin, but 2x faster than trig
    
    if choice == 'cart':
        DeltaRA = (RA2-RA1) * np.cos(0.5*(Dec1+Dec2)*degree)
        DeltaDec = Dec2-Dec1
        separation = np.sqrt(DeltaRA*DeltaRA + DeltaDec*DeltaDec)
    elif choice == 'trig':
        cosSeparation = np.sin(Dec1*degree)*np.sin(Dec2*degree) \
                        + np.cos(Dec1*degree)*np.cos(Dec2*degree)*np.cos((RA2-RA1)*degree)
        separation = ACO(cosSeparation) / degree
    elif choice == 'astropy':
        c1 = SkyCoord(RA1,Dec1,unit='deg')
        c2 = SkyCoord(RA2,Dec2,unit='deg')
        separation = c1.separation(c2).deg
    else:
        raise print("AngularSeparation: cannot recognize choice = ", choice)

    return(separation)

def RADec_from_dxdy(dx,dy):
    """celestial coordinates given cartesian coordinates relative to new center (all in deg)"""

    # author: Gary Mamon
    # cosDec_cen_init, RA_cen, Dec_cen arrive as global variables
    
    RA  = cosDec_cen_init * RA_cen - dx
    Dec = Dec_cen                  + dy

    # handle crossing of RA=0
    RA = np.where(RA >= 360.,  RA - 360., RA)
    
    return(RA,Dec)

def xy_from_dxdy(dx,dy):
    """cartesian coordinates relative to center of circular region (x points to -RA, y to +Dec) given cartesian coordinates relative to new center"""

    # author: Gary Mamon
    # Delta_x_cen and Delta_y_cen arrive as global variables
    
    x = dx + Delta_x_cen
    y = dy + Delta_y_cen
    return (x,y)

def dxdy_from_xy(x,y):
    """cartesian coordinates relative to new center (x points to -RA, y to +Dec) given cartesian coordinates relative to center fo circular region"""
    # author: Gary Mamon
    # Delta_x_cen and Delta_y_cen arrive as global variables
    
    dx = x + Delta_x_cen
    dy = y - Delta_y_cen
    return(dx,dy)

def dxdy_from_uv(u,v):
    """cartesian coordinates relative to center (x points to -RA, y to +Dec) given elliptical coordinates"""
    # author: Gary Mamon
    # cosPA and sinPA arrive as global variables
    
    dx = -u * sinPA - v * cosPA
    dy =  u * cosPA - v * sinPA
    return(dx,dy)

def dxdy_from_RADec(RA,Dec):
    """cartesian coordinates relative to center (x points to -RA, y to +Dec) given celestial coordinates"""
    # author: Gary Mamon
    # RA_cen, Dec_cen, cosDec_cen_init arrive as global variables

    dx =  -1.* (RA  - RA_cen) * cosDec_cen_init
    dy =        Dec - Dec_cen   
    
    # handle crossing of RA=0
    dx = np.where(dx >= 180.,  dx - 360., dx)
    dx = np.where(dx <= -180., dx + 360., dx)

    return(dx,dy)

def dxdy_from_RADec2(RA,Dec,RA_cen,Dec_cen):
    """cartesian coordinates relative to center (x points to -RA, y to +Dec) given celestial coordinates:
       same as dxdy_from_RADec but with fewer globals arriving"""
    # author: Gary Mamon
    # only cosDec_cen_init arrive as global variables

    dx =  -1.* (RA  - RA_cen) * cosDec_cen_init
    dy =        Dec - Dec_cen   
    
    # handle crossing of RA=0
    dx = np.where(dx >= 180.,  dx - 360., dx)
    dx = np.where(dx <= -180., dx + 360., dx)

    return(dx,dy)

def uv_from_dxdy(dx,dy):
    """elliptical coordinates u (along major axis) and v (along minor axis), given cartesian coordinates relative to ellipse (all in deg)"""
    # author: Gary Mamon
    # sinPA and cosPA arrives as a global variables

    # rotate to axes of cluster (careful: PA measures angle East from North)
    u = - dx * sinPA + dy * cosPA
    v = - dx * cosPA - dy * sinPA
    return(u,v)

def uv_from_xy(x,y):
    """elliptical coordinates u (along major axis) and v (along minor axis), given cartesian coordinates relative to circle (all in deg)"""
    # author: Gary Mamon

    # cartesian coordinates around new center
    dx,dy = dxdy_from_xy(x,y)

    # rotate to axes of cluster
    u,v = uv_from_dxdy(dx,dy)

    return(u,v)
    
def uv_from_RADec(RA,Dec):
    """elliptical coordinates u (along major axis) and v (along minor axis), given celestial coordinates and PA (all in deg)"""
    # author: Gary Mamon
    # RA_cen, Dec_cen and PA_in_rd arrive as global variables

    # cartesian coordinates around new center
    dx,dy = dxdy_from_RADec(RA,Dec)
    
    # rotate to axes of cluster
    u,v = uv_from_dxdy(dx,dy)

    return(u,v)

def RADec_from_uv(u,v):
    """celestial coordinates given elliptical coordinates u (along major axis) and v (along minor axis)"""
    # author: Gary Mamon

    # cartesian coordinates around new center    
    dx,dy = dxdy_from_uv(u,v)
    RA,Dec = RADec_from_dxdy(dx,dy)
    
    return(RA,Dec)

def R_ellip_from_dxdy(dx,dy):
    """elliptical equivalent radius, given celestial coordinates and PA (all in deg) and ellipticity"""
    # author: Gary Mamon
    # ellipticity arrives as a global variable

    # rotate to axes of cluster
    u,v = uv_from_dxdy(dx,dy)
    v_decompressed = v/(1.-ellipticity)
    return np.sqrt(u*u + v_decompressed*v_decompressed)

def R_ellip_from_RADec(RA,Dec):
    """elliptical equivalent radius, given celestial coordinates and PA (all in deg) and ellipticity"""
    # author: Gary Mamon
    # ellipticity arrives as a global variable
    
    # rotate to axes of cluster
    u,v = uv_from_RADec(RA,Dec)
    v_decompressed = v/(1.-ellipticity)
    return np.sqrt(u*u + v_decompressed*v_decompressed)

def R_ellip_from_xy(x,y):
    """elliptical equivalent radius, given cartesian coordinates around circular region (all in deg)"""
    # author: Gary Mamon
    # ellipticity arrives as a global variable

    # rotate to axes of cluster
    u,v = uv_from_xy(x,y)
    v_decompressed = v/(1.-ellipticity)
    return np.sqrt(u*u + v_decompressed*v_decompressed)

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
    CheckType(X,"SurfaceDensity_tilde_coredNFW",'X')
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise print("ERROR in SurfaceDensity_tilde_coredNFW: min(X) = ", np.min(X), "cannot be <= 0")

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
    if np.any(X) < 0.:
        raise print("ERROR in ProjectedNumber_tilde_NFW: min(X) = ", np.min(X), "cannot be <= 0")

    # compute dimensionless projected number
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-7)
    denom = np.log(2.) - 0.5
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
        raise print("ERROR in ProjectedNumber_tilde_coredNFW: min(X) = ", np.min(X), "cannot be < 0")

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

def ProjectedNumber_tilde_Uniform(X):
    """Dimensionless cluster projected number for a uniform surface density profile
    arg: X = R/R_cut (positive float or array of positive floats), where R_cut is radius of slope -2 
      (not the natural scale radius for which x=r/a!)
    returns: N(X R_cut) / N(R_cut) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(X,"ProjectedNumber_tilde_Uniform","X")

    # stop with error message if input values are < 0 (unphysical)
    if np.any(X) < 0.:
        raise print("ERROR in ProjectedNumber_tilde_Uniform: X cannot be < 0")

    return(np.where(X<1, X*X, 1.))

def Number_tilde_NFW(x):
    """Dimensionless cluster 3D number for an NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(x,"Number_tilde_NFW","x")

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise print("ERROR in Number_tilde_NFW: min(x) = ", np.min(x), "cannot be < 0")

    return ((np.log(x+1)-x/(x+1)) / (np.log(2.)-0.5))

def Number_tilde_coredNFW(x):
    """Dimensionless cluster 3D number for a cored NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    CheckType(x,"Number_tilde_coredNFW",'x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise print("ERROR in Number_tilde_coredNFW: min(x) = ", np.min(x), " cannot be < 0")

    return ((np.log(2*x+1)-2*x*(3*x+1)/(2*x+1)**2) / (np.log(3)-8./9.))

def Number_tilde_NFWtrunc(x):
    """Dimensionless cluster 3D number for an NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # cutoff radius / scale radius
    X_cut = 10.**(log_R_cut-log_scale_radius)
    
    # check that input is integer or float or numpy array
    CheckType(x,"Number_tilde_NFWtrunc",'x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise print("ERROR in Number_tilde_NFWtrunc: min(x) = ", np.min(x), "cannot be < 0")

    xtmp = np.where(x < X_cut, x, X_cut)
    return ((np.log(xtmp+1)-xtmp/(xtmp+1)) / (np.log(2.)-0.5))

def Number_tilde_Uniform(x):
    """Dimensionless cluster 3D number for a uniform surface density profile. 
    arg: x = r/R_1 (cutoff radius)
    returns: N_3D(x R_1) / (Sigma/R_1) (float, or array of floats)"""

    # author: Gary Mamon

     # check that input is integer or float or numpy array
    CheckType(x,"Number_tilde_Uniform",'x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise print("ERROR in Number_tidle_Uniform: min(x) = ", np.min(x), "cannot be < 0")

    return np.where(x >= 1, 0, 1 / (np.pi * np.sqrt(1-x*x)))
   
def Random_radius(Xmax,model,Npoints,Nknots):
    """Random R/r_{-2} from Monte Carlo for circular NFW model
    args: R_max/R_{-2} model (NFW|coredNFW) number-of-random-points number-of-knots"""

    # author: Gary Mamon

    CheckType(Xmax,"Random_radius","Xmax")
    CheckTypeInt(Npoints,"Random_radius","Npoints")

    # random numbers
    q = np.random.random_sample(Npoints)

    # equal spaced knots in arcsinh X
    asinhX0 = np.linspace(0., np.arcsinh(Xmax), num=Nknots+1)

    X0 = np.sinh(asinhX0)
    if model == "NFW":
        N0 = ProjectedNumber_tilde_NFW(Xmax)
    elif model == "coredNFW":
        N0 = ProjectedNumber_tilde_coredNFW(Xmax)
    elif model == 'NFWtrunc':
        N0 = ProjectedNumber_tilde_NFWtrunc(Xmax)
    else:
        raise print ("Random_radius: model = ", model, " not recognized")
    
    ratio = ProjectedNumber_tilde_NFW(X0) / N0

    # spline on knots of asinh(equal spaced)
    asinhratio = np.arcsinh(ratio)
    # t = time.process_time()
    spline = interpolate.splrep(asinhratio,asinhX0,s=0)
    if verbosity >= 2:
        print("compute spline: time = ", time.process_time()-t)
    # t = time.process_time()        
    asinhq = np.arcsinh(q)
    if verbosity >= 2:
        print("asinh(q): time = ", time.process_time()-t)
    # t = time.process_time()        
    asinhX_spline = interpolate.splev(asinhq, spline, der=0, ext=2)
    if verbosity >= 2:
        print("evaluate spline: time = ", time.process_time()-t)
    return (np.sinh(asinhX_spline))

def Random_xy(Rmax,model,Npoints,Nknots,ellipticity,PA):
    """Random x & y (in deg) from Monte Carlo for circular model (NFW or coredNFW)
    args: R_max model (NFW|coredNFW) number-of-random-points number-of-knots ellipticity PA"""
    R_random = scale_radius * Random_radius(Rmax/scale_radius,model,Npoints,Nknots)
    PA_random = 2 * np.pi * np.random.random_sample(Npoints) # in rd
    theta_random = 2 * np.pi * np.random.random_sample(Npoints) # in rd
    u_random = R_random * np.cos(theta_random)
    v_random = R_random * np.sin(theta_random) * (1-ellipticity)
    x0_random = RA_cen_init + (RA_cen - RA_cen_init) * np.cos(Dec_cen_init * degree) 
    y0_random = Dec_cen
    x_random = x0_random - u_random*np.sin(PA*degree) - v_random*np.cos(PA*degree)
    y_random = y0_random + u_random*np.cos(PA*degree) - v_random*np.sin(PA*degree)

    if verbosity >= 3:
        print("IS TRUE : ", x0_random == RA_cen_init, y0_random == Dec_cen_init)
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
        print("using 2D polynomial for ellipticity = ", ellipticity)
    e = ellipticity
    if N_points == 0 and DeltaCenter < TINY_SHIFT_POS and X < min_R_over_rminus2:
        # print("ProjectedNumber_tilde_ellip_NFW: series expansion")
        Nprojtilde = X*X/ln16minus2 / (1.-e) * (-1. - 2.*np.log(0.25*X*(2.-e)/(1.-e)))
    elif N_points == 0 and DeltaCenter < TINY_SHIFT_POS:
        # analytical approximation, only for centered ellipses
        lX = np.log10(X)
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
            0.00008236148148039739*lX**9
        )
        Nprojtilde = 10. ** lNprojtilde
    elif N_points < 0 and DeltaCenter < TINY_SHIFT_POS:
        # evaluate double integral
        print("ProjectedNumber_tilde_ellip_NFW: quadrature")
        f = lambda V, U: SurfaceDensity_tilde_NFW(np.sqrt(U*U+V*V/(1-ellipticity)**2))
        Nprojtilde = integrate.dblquad(f,0,X,lambda U: 0, lambda U: np.sqrt(X*X-U*U), epsabs=0., epsrel=0.001)
        Nprojtilde = 4/(np.pi*(1.-ellipticity)) * Nprojtilde[0]
    else:
        # Monte Carlo integration
        print("ProjectedNumber_tilde_ellip_NFW: Monte Carlo")
        if DeltaCenter_over_a < 1.e-6:
            X_ellip = Random_radius(X/(1.-ellipticity), "NFW", N_points, 100)
        else:
            X_ellip = Random_radius((X+DeltaCenter_over_a)/(1.-ellipticity), "NFW", N_points, 100)
        phi = 2. * np.pi * np.random.random_sample(N_points)
        U = X_ellip * np.cos(phi)
        V = (1.-ellipticity) * X_ellip * np.sin(phi)
        if DeltaCenter_over_a < 1.e-6:
            X_sky_MC = np.sqrt(U*U + V*V)
        else:
            dX = - U*np.sin(PA_in_degrees) - V*np.cos(PA_in_degrees)
            dY =   U*np.cos(PA_in_degrees) - V*np.cos(PA_in_degrees)
            X_MC = -(RA_cen - RA_cen_init)/np.cos(Dec_cen_init * degree) + dX
            Y_MC = Dec_cen - Dec_cen_init + dY
            X_sky_MC = np.sqrt(X_MC*X_MC + Y_MC*Y_MC)

        X_in_circle = X_sky_MC[X_sky_MC < X]
        frac = len(X_in_circle) / N_points
        Nprojtilde = frac * ProjectedNumber_tilde_NFW(X/(1.-ellipticity))

    # print ("ProjectedNumber_tilde_ellip_NFW: Xmax e Nprojtilde = ", X, ellipticity, Nprojtilde)
    return (Nprojtilde)

def ProjectedNumber_tilde_ellip_coredNFW(X,ellipticity):
    """Dimensionless projected mass for non-circular cored NFW models
    args:
    X = R/r_{-2}  (positive float or array of positive floats), where r_{-2} is radius of slope -2
    ellipticity = 1-b/a (0 for circular)
    returns: N(X r_{-2}) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon
    # source: ~gam/EUCLID/CLUST/PROCL/denomElliptical.nb

    if np.abs(ellipticity) < TINY:
        return(ProjectedNumber_tilde_coredNFW(X))
    
    N_knots  = 100
    if verbosity >= 3:
        print("using 2D polynomial")
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
        Nprojtilde = 10. ** lNprojtilde
        
    return (Nprojtilde)

def ProjectedNumber_tilde_Uniform(X):
    """Dimensionless cluster projected number for a uniform model.
    arg: X = R/R_1 (positive float or array of positive floats), where R_1 is scale radius (radius where uniform model stops)
    returns: N(R_1 X) / N(R_1) (float, or array of floats)"""

    # author: Gary Mamon

    return(np.where(X<1, X*X, 1.))

def ProjectedNumber_tilde_ellip(R_over_a, model, e, DeltaRA, DeltaDec):
    """Dimensionless cluster projected number of elliptical and/or shifted models relative to circular region
    arguments:
        R_over_a = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
        e: ellipticity (default=0.) [dimensionless]
        DeltaRA, DeltaDec: shift of position of model relative to circular region [deg]
    returns: N_proj(R) / N(r_{-2}) (float or array of floats)"""

    # author: Gary Mamon

    # short-named variables for clarity
    a = scale_radius
    Z = R_over_a
    
    if np.abs(e) < TINY and np.abs(DeltaCenter) < TINY_SHIFT_POS:
        # centered circular
        # print("ProjectedNumber_tilde_ellip: circular ...")
        return (ProjectedNumbertilde(R_over_a,model))
    
    elif np.abs(e) < TINY and model == 'uniform':
        # shifted Uniform
        # print("ProjectedNumber_tilde_ellip: uniform shifted ...")
        shift_x, shift_y = dxdy_from_RADec(RA_cen_init+DeltaRA,Dec_cen_init+DeltaDec)
        d = np.sqrt(shift_x**shift_x + shift_y*shift_y)
        # area is intersection of circles
        # from Wolfram MathWorld http://mathworld.wolfram.com/Circle-CircleIntersection.html
        Rtmp = a * R_over_a
        area =   Rtmp*Rtmp * np.arccos((d*d + Rtmp*Rtmp - a*a) / (2*d*Rtmp)) \
               + a*a * np.arccos((d*d + a*a - Rtmp*Rtmp) / (2*d*a)) \
               - 0.5*np.sqrt((-d+a+Rtmp) * (d+a-Rtmp) * (d-a+Rtmp) * (d+a+Rtmp))
        # FOLLOWING IS PROBABLY INCORRECT!!!
        return(background * area)

    elif DeltaCenter < TINY_SHIFT_POS and N_points == 0 and model in ('NFW','coredNFW'):
        # centered elliptical with polynomial approximation
        # print("ProjectedNumber_tilde_ellip: polynomial ...")
        if model == 'NFW':
            return(ProjectedNumber_tilde_ellip_NFW(R_over_a,e))
        elif model == 'coredNFW':
            return(ProjectedNumber_tilde_ellip_coredNFW(R_over_a,e))

    elif N_points < 0:
        # double integral by quadrature
        # print("ProjectedNumber_tilde_ellip: quadrature ...")
        tol_quadrature = 10.**N_points
        if DeltaCenter < TINY_SHIFT_POS:
            f = lambda V, U: SurfaceDensity_tilde(np.sqrt(U*U+V*V/(1-e)**2),model)
            Nprojtilde = integrate.dblquad(f,0,Z,lambda U: 0, lambda U: np.sqrt(Z*Z-U*U), epsabs=0., epsrel=tol_quadrature)     
            Nprojtilde = 4./(np.pi*(1.-e)) * Nprojtilde[0]
        else:
            # CHECK FOLLOWING LINE!
            f = lambda Y, X: SurfaceDensity_tilde(R_ellip_from_xy(a*X,a*Y)/a,model)
            Nprojtilde = integrate.dblquad(f,-Z,Z,lambda X: -np.sqrt(Z*Z-X*X), lambda X: np.sqrt(Z*Z-X*X), epsabs=0., epsrel=tol_quadrature)
            Nprojtilde = 1./(np.pi*(1.-e)) * Nprojtilde[0]

    elif N_points >= 1000:
        # Monte Carlo
        # print("ProjectedNumber_tilde_ellip: Monte Carlo ... N_points = ", N_points)
        N_knots = 100
        Z_ellip_MC = Random_radius(R_over_a/(1.-e), model, N_points, N_knots)
        phi = 2. * np.pi * np.random.random_sample(N_points)
        U = Z_ellip_MC * np.cos(phi)
        V = (1.-e) * Z_ellip_MC * np.sin(phi)
        if np.abs(DeltaCenter) < TINY_SHIFT_POS:
            Z_MC = np.sqrt(U*U + V*V)
        else:
            # add shift of center
            dX,dY = dxdy_from_uv(U,V)
            X = -1. * DeltaRA/np.cos(Dec_cen_init * degree) + dX
            Y =       DeltaDec                              + dY
            Z_MC = np.sqrt(X*X + Y*Y)
        Z_in_circle = Z_MC[Z_MC < R_over_a]
        frac = len(Z_in_circle) / N_points
        # circular N_proj_tilde times fraction of points inside oversized circle
        Nprojtilde = frac * ProjectedNumber_tilde(R_over_a/(1.-e),model)
    else:
        raise print("ProjectedNumber_tilde_ellip: N_points = ", N_points, \
                    "DeltaCenter = ", DeltaCenter)
    return(Nprojtilde)

def SurfaceDensity_tilde(X,model):
    """Dimensionless cluster surface density
    arguments:
        X = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2]  (float or array of floats)"""

    # author: Gary Mamon

    if model == "NFW":
        return SurfaceDensity_tilde_NFW(X)
    elif model == "coredNFW":
        return SurfaceDensity_tilde_coredNFW(X)
    elif model == 'NFWtrunc':
        return SurfaceDensity_tilde_NFWtrunc(X)
    elif model == "uniform":
        return SurfaceDensity_tilde_Uniform(X)
    else:
        raise print("ERROR in SurfaceDensity_tilde: model = ", model, " is not recognized")

def ProjectedNumber_tilde(X,model,e=0.,DeltaCenter=0.):
    """Dimensionless cluster projected number
    arguments:
        X = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
        e: ellipticity (default=0.) [dimensionless]
        DeltaRA, DeltaDec: shift of position of model relative to circular region [deg]
    returns: N_proj(R) / N(r_{-2}) (float or array of floats)"""

    # author: Gary Mamon

    if verbosity >= 4:
        print("ProjectedNumber_tilde: ellipticity=",ellipticity)
    if X < -TINY:
        raise print("ERROR in ProjectedNumber_tilde: X = ", X, " cannot be negative")
    elif X < TINY:
        return 0
    elif X < min_R_over_rminus2-TINY:
        raise print("ERROR in ProjectedNumber_tilde: X = ", X, " <= critical value = ", 
                    min_R_over_rminus2)

    if np.abs(e) < TINY and DeltaCenter < TINY_SHIFT_POS:
        if model == "NFW":
            return ProjectedNumber_tilde_NFW(X)
        elif model == "coredNFW":
            return ProjectedNumber_tilde_coredNFW(X)
        elif model == 'NFWtrunc':
            return ProjectedNumber_tilde_NFWtrunc(X)
        elif model == "uniform":
            return ProjectedNumber_tilde_Uniform(X)
        else:
            raise print("ProjectedNumber_tilde: cannot recognize model ", model)
    else:
        return ProjectedNumber_tilde_ellip(X,model,e,DeltaRA,DeltaDec)

def Number_tilde(x,model):
    """Dimensionless cluster 3D number
    arguments:
        x = r/r_scale (dimensionless 3D radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW' or 'Uniform' 
        r_scale = r_{-2} (NFW or coredNFW) or R_cut (Uniform)"""

    # author: Gary Mamon
    
    if verbosity >= 4:
        print("Number_tilde: ellipticity=",ellipticity)

    if model == "NFW":
        return Number_tilde_NFW(x)
    elif model == "coredNFW":
        return Number_tilde_coredNFW(x)
    elif model == "NFWtrunc":
        return Number_tilde_NFWtrunc(x)
    elif model == "uniform":
        return Number_tilde_Uniform(x)
    else:
        raise print("ERROR in Number_tilde: model = ", model, " is not recognized")

def PenaltyFunction(x, boundMin, boundMax):
    """normalized penalty Fuunction applied to likelihood when one goes beyond bound"""

    # exp10 arrives as global 

    if boundMax <= boundMin:
        pf = 0.

    else:
        xtmp = (x-boundMin) / (boundMax-boundMin)
        xtmp2 = 2. * np.abs(xtmp-0.5)
        # pf = np.where(xtmp2 > 1, np.exp(10.*xtmp2)-exp10, 0)
        pf = np.where(xtmp2 > 1, 1000. * np.sqrt(x-1.), 0)
        if verbosity >= 3:
            print("PenaltyFunction: x boundMin boundMax = ", x, boundMin, boundMax,\
              " penalty = ", pf)

    return(pf)

def guess_ellip_PA(RA,Dec):
    """ guess ellipticity and PA of cluster according to its 2nd moments """

    # author: Gary Mamon
    
    dx,dy = dxdy_from_RADec2(RA,Dec,RA_cen_init,Dec_cen_init)

    # guess ellipticity and PA from 2nd moments

    xymean = np.mean(dx*dy)
    x2mean = np.mean(dx*dx)
    y2mean = np.mean(dy*dy)
    tan2theta = 2*xymean/(x2mean-y2mean)
    theta = 0.5*np.arctan(tan2theta)
    PA_pred = theta / degree
    if PA_pred < 0.:
        PA_pred = PA_pred + 180.
    A2 = (x2mean+y2mean)/2. + np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    B2 = (x2mean+y2mean)/2. - np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    ellipticity_pred = 1 - np.sqrt(B2/A2)
    return ellipticity_pred, PA_pred

def guess_center(RA, Dec):
    return np.median(RA), np.median(Dec)

def PROFCL_prob_galaxy():
    """probability of projected radii for given galaxy position, model parameters and background
    arguments:
        RA, Dec: celestial coordinates [deg]
        scale_radius: radius of 3D density slope -2) [deg]
        ellipticity [dimensionless]
        background: uniform background [deg^{-2}]
        RA_cen, Dec_cen: (possibly new) center of model [deg]
        model: 'NFW' or 'coredNFW' or 'Uniform'
    returns: p(data|model)"""  

    # author: Gary Mamon

    a = scale_radius    # for clarity
    e = ellipticity     # for clarity
    
    DeltaNproj_tilde = ProjectedNumber_tilde(R_max/a, model, e, DeltaCenter) \
                        - ProjectedNumber_tilde(R_min/a, model, e, DeltaCenter)
    if background < TINY:
        numerator = SurfaceDensity_tilde(R_ellip_over_a, model)
        denominator = a*a * (1.-e) * DeltaNproj_tilde
    else:
        Nofa = (N_tot - np.pi * (R_max*R_max - R_min*R_min) * background) / DeltaNproj_tilde
        numerator = Nofa/(np.pi * a*a * (1.-e)) * SurfaceDensity_tilde(R_ellip_over_a, model) + background
        denominator = N_tot

    return(numerator / denominator)
def PROFCL_Rescale_Params(params_in_fit):
    RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, log_R_cut = params_in_fit

    # re-sccale
    RA_cen          *= R_max * np.cos(Dec_cen*degree)
    Dec_cen         *= R_max
    ellipticity     *= 5.
    PA              *= 500.
    log_background  += 1.   # 10 x background 
    log_R_cut       += 0.7  #  5 x R_cut
    return(RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, log_R_cut)
    
def PROFCL_LogLikelihood(params):
    """general -log likelihood of cluster given galaxy positions
    arguments:
    RA_cen, Dec_cen: coordinates of cluster center [floats]
    log_scale_radius: log of scale radius of the profile (where scale radius is in degrees) [float]
    ellipticity: ellipticity of cluster (1-b/a, so that 0 = circular, and 1 = linear) [float]
    PA: position angle of cluster (degrees from North going East) [float]
    log_background: uniform background of cluster (in deg^{-2}) [float]
    returns: -log likelihood [float]
    assumptions: not too close to celestial pole 
    (solution for close to celestial pole [not yet implemented]: 
    1. convert to galactic coordinates, 
    2. fit,
    3. convert back to celestial coordinates for PA)"""
    # authors: Gary Mamon with help from Yuba Amoura, Christophe Adami & Eliott Mamon
                                    
    global iPass
    global Delta_x_cen, Delta_y_cen
    global DeltaCenter, DeltaCenter_over_a
    global DeltaRA, DeltaDec
    global RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, background, log_R_cut
    global scale_radius
    global PA_in_rd, cosPA, sinPA
    global R_ellip_over_a, DeltaRA_over_a, DeltaDec_over_a
    global in_annulus
    global N_points, N_points_flag

    if verbosity >= 4:
        print("entering LogLikelihood: R_min=",R_min)
    iPass = iPass + 1
                                    
    # read function arguments (parameters and extra arguments)

    if rescale_flag == 'y':
        RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, log_R_cut = PROFCL_Rescale_Params(params)
    else:
        RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, log_R_cut = params
    
    # RA, Dec, prob_membership, R_min, R_max, model = args

    # if np.isnan(RA_cen):
    #     raise print("ERROR in PROFCL_LogLikelihood: RA_cen is NaN!")
    # if np.isnan(Dec_cen):
    #     raise print("ERROR in PROFCL_LogLikelihood: Dec_cen is NaN!")

    # ## checks on types of arguments
    
    # # check that galaxy positions and probabilities are in numpy arrays
    # if type(RA) is not np.ndarray:
    #     raise print("ERROR in PROFCL_lnlikelihood: RA must be numpy array")
    # if type(Dec) is not np.ndarray:
    #     raise print("ERROR in PROFCL_lnlikelihood: Dec must be numpy array")
    # if type(prob_membership) is not np.ndarray:
    #     raise print("ERROR in PROFCL_lnlikelihood: prob_membership must be numpy array")

    # # check that cluster center position are floats
    # CheckTypeIntorFloat(RA_cen,"PROFCL_LogLIkelihood","RA_cen")
    # CheckTypeIntorFloat(Dec_cen,"PROFCL_LogLIkelihood","Dec_cen")
    
    # # check that ellipticity and position angle are floats
    # CheckTypeIntorFloat(ellipticity,"PROFCL_LogLIkelihood","ellipticity")
    # CheckTypeIntorFloat(PA,"PROFCL_LogLIkelihood","PA")

    # # check that min and max projected radii, and loga are floats or ints
    # CheckTypeIntorFloat(R_min,"PROFCL_LogLIkelihood","R_min")
    # CheckTypeIntorFloat(R_max,"PROFCL_LogLIkelihood","R_max")
    # CheckTypeIntorFloat(log_scale_radius,"PROFCL_LogLIkelihood","log_scale_radius")
   
    # # check if out of bounds 
    # if RA_cen < RA_cen_minallow - TINY:
    #     if verbosity >= 2:
    #         print ("RA_cen = ", RA_cen, " < RA_cen_min_allow = ", RA_cen_minallow)
    #     return(HUGE)
    # elif RA_cen > RA_cen_maxallow + TINY:
    #     if verbosity >= 2:
    #         print ("RA_cen = ", RA_cen, " > RA_cen_max_allow = ", RA_cen_maxallow)
    #     return(HUGE)
    # elif Dec_cen < Dec_cen_minallow - TINY:
    #     if verbosity >= 2:
    #         print ("Dec_cen = ", Dec_cen, " < Dec_cen_min_allow = ", Dec_cen_minallow)
    #     return(HUGE)
    # elif Dec_cen > Dec_cen_maxallow + TINY:
    #     if verbosity >= 2:
    #         print ("Dec_cen = ", Dec_cen, " > Dec_cen_max_allow = ", Dec_cen_maxallow)
    #     return(HUGE)
    # elif log_scale_radius < log_scale_radius_minallow - TINY:
    #     if verbosity >= 2:
    #         print ("log_scale_radius = ", log_scale_radius, " < log_scale_radius_min_allow = ", log_scale_radius_minallow)
    #     return(HUGE)
    # elif log_scale_radius > log_scale_radius_maxallow + TINY:
    #     if verbosity >= 2:
    #         print ("log_scale_radius = ", log_scale_radius, " > log_scale_radius_max_allow = ", log_scale_radius_maxallow)
    #     return(HUGE)
    # elif log_background < log_background_minallow - TINY:
    #     if verbosity >= 2:
    #         print ("log_background = ", log_background, " < log_background_min_allow = ", log_background_minallow)
    #     return(HUGE)
    # elif log_background > log_background_maxallow + TINY:
    #     if verbosity >= 2:
    #         print ("log_background = ", log_background, " > log_background_max_allow = ", log_background_maxallow)
    #     return(HUGE)
    # elif ellipticity < ellipticity_minallow - TINY:
    #     if verbosity >= 2:
    #         print ("ellipticity = ", ellipticity, " < ellipticity_min_allow = ", ellipticity_minallow)
    #     return(HUGE)
    # elif ellipticity > ellipticity_maxallow + TINY:
    #     if verbosity >= 2:
    #         print ("ellipticity = ", ellipticity, " > ellipticity_max_allow = ", ellipticity_maxallow)
    #     return(HUGE)
    # elif PA < PA_minallow - TINY:
    #     if verbosity >= 2:
    #         print ("PA = ", PA, " < PA_min_allow = ", PA_minallow)
    #     return(HUGE)
    # elif PA > PA_maxallow + TINY:
    #     if verbosity >= 2:
    #         print ("PA = ", PA, " > PA_max_allow = ", PA_maxallow)
    #     return(HUGE)

    # ## checks on values of arguments
    
    # # check that RAs are between 0 and 360 degrees
    # RA_min = np.min(RA)
    # if RA_min < 0.:
    #     raise print("ERROR in PROFCL_lnlikelihood: min(RA) = ", 
    #                 RA_min, " must be >= 0")        
    # RA_max = np.max(RA)
    # if RA_max > 360.:
    #     raise print("ERROR in PROFCL_lnlikelihood: max(RA) = ", 
    #                 RA_max, " must be <= 360")        
    # if RA_cen < 0.:
    #     raise print("ERROR in PROFCL_lnlikelihood: RA_cen = ", 
    #                 RA_cen, " must be >= 0") 
    # if RA_cen > 360.:
    #     raise print("ERROR in PROFCL_lnlikelihood: RA_cen = ", 
    #                 RA_cen, " must be <= 360") 
    
    # # check that Decs are between -90 and 90 degrees
    # Dec_min = np.min(Dec)
    # if Dec_min < -90.:
    #     raise print("ERROR in PROFCL_lnlikelihood: min(Dec) = ", 
    #                 Dec_min, " must be >= -90")        
    # Dec_max = np.max(Dec)
    # if Dec_max > 90.:
    #     raise print("ERROR in PROFCL_lnlikelihood: max(Dec) = ", 
    #                 Dec_max, " must be <= 90")        
    # if Dec_cen < -90.:
    #     raise print("ERROR in PROFCL_lnlikelihood: Dec_cen = ", 
    #                 Dec_cen, " must be >= -90") 
    # if Dec_cen > 90.:
    #     raise print("ERROR in PROFCL_lnlikelihood: Dec_cen = ", 
    #                 Dec_cen, " must be <= 90") 
    
    # # check that ellipticity is between 0 and 1
    # if ellipticity < 0. or ellipticity > 1.:
    #     print("ellipticity_minallow=",ellipticity_minallow)
    #     raise print("ERROR in PROFCL_lnlikelihood: ellipticity = ", 
    #                 ellipticity, " must be between 0 and 1")     
    
    # # check that model is known
    # if model != "NFW" and model != "coredNFW" and model != "Uniform":
    #     raise print("ERROR in PROFCL_lnlikelihood: model = ", 
    #                 model, " is not implemented")
    
    # # check that R_min > 0 for NFW (to avoid infinite surface densities)
    # # or R_min >= 0 for coredNFW
    # if R_min <= 0. and model == "NFW":
    #     raise print("ERROR in PROFCL_lnlikelihood: R_min must be > 0 for NFW model")
    # elif R_min < 0.:
    #     raise print("ERROR in PROFCL_lnlikelihood: R_min must be >= 0 for coredNFW model")

    # # check that R_max > R_min
    # if R_max <= R_min:
    #     raise print("ERROR in PROFCL_lnlikelihood: R_min = ", 
    #                 Rmin, " must be < than R_max = ", R_max)

    # check that coordinates are not too close to Celestial Pole
    max_allowed_Dec = 80.
    Dec_abs_max = np.max(np.abs(Dec_gal))
    if Dec_abs_max > max_allowed_Dec:
        raise print("ERROR in PROFCL_lnlikelihood: max(abs(Dec)) = ",
                    Dec_abs_max, " too close to pole!")

    ## transform from RA,Dec to cartesian and then to projected radii
    
    # transform coordinate units from degrees to radians
    RA_cen_in_rd  =  RA_cen * degree
    Dec_cen_in_rd = Dec_cen * degree
    PA_in_rd      =      PA * degree
    cosPA         = np.cos(PA_in_rd)
    sinPA         = np.sin(PA_in_rd)

    a = 10. ** log_scale_radius
    DeltaRA = RA_cen - RA_cen_init
    DeltaDec = Dec_cen - Dec_cen_init
    Delta_x_cen,Delta_y_cen = dxdy_from_RADec2(RA_cen,Dec_cen,RA_cen_init,Dec_cen_init)
    # Delta_x_cen = - (RA_cen-RA_cen_init) / np.cos(Dec_cen_init * degree)
    # Delta_y_cen =   Dec_cen-Dec_cen_init
    
    DeltaCenter = np.sqrt(Delta_x_cen*Delta_x_cen + Delta_y_cen*Delta_y_cen)
    # some minimizers will change the center when this is meant to be fixed
    # for N_points = 0, we change N_points to 1000 to compute a penalty
    # flag, and then turn back N_points to 0
    if DeltaCenter > TINY_SHIFT_POS and N_points == 0:
        N_points = 1000
        N_points_flag = True
        
    DeltaCenter_over_a = DeltaCenter / a
    u,v = uv_from_RADec(RA_gal,Dec_gal)
    if verbosity >= 2:
        print("Log_Likelihood: len(u) = ", len(u))

    # linear variables
    scale_radius = 10.**log_scale_radius        # in deg
    background   = 10.**log_background          # in deg^{-2}

    # elliptical radii (in degrees)
    R_ellip = R_ellip_from_RADec(RA_gal,Dec_gal)
    R_ellip_over_a = R_ellip / scale_radius
    DeltaRA_over_a = (RA_cen - RA_cen_init) / scale_radius
    DeltaDec_over_a = (Dec_cen - Dec_cen_init) / scale_radius
    
    # check that elliptical radii are within limits of Mathematica fit
    
    if N_points == 0 and (np.any(R_ellip_over_a < min_R_over_rminus2) or np.any(R_ellip_over_a > max_R_over_rminus2)):
        if verbosity >= 2:
            print("log_likelihood: off limits for r_s min(X) max(X) X_min X_max= ", scale_radius, min_R_over_rminus2, max_R_over_rminus2, np.min(R_ellip_over_a), np.max(R_ellip_over_a))
        return (HUGE)
    elif verbosity >= 4:
        print("OK for r_s = ", scale_radius)

    ## likelihood calculation
    
    if verbosity >= 2:
        print("Log_Likelihood: N_tot = ", len(R_sky))
    if len(R_sky) == 0:
        print("ERROR in PROFCL_lnlikelihood: ",
              "no galaxies found with projected radius between ",
              R_min, " and ", R_max,
              " around RA = ", RA_cen, " and Dec = ", Dec_cen)
        return(HUGE)
    if R_max/scale_radius > max_R_over_rminus2:
        if verbosity >= 2:
            print("PROFCL_lnlikelihood: R_max a max_Rovera = ", R_max, scale_radius, max_R_over_rminus2)
        return(HUGE)

    # print("u = ", u[in_annulus])
    # print("v = ", v[in_annulus])

    # prob = PROFCL_prob_uv(u[in_annulus],v[in_annulus],R_min,R_max,scale_radius,ellipticity,background,model)
    prob = PROFCL_prob_galaxy()
    
    if verbosity >= 3:
        print("\n\n PROBABILITY = ", prob, "\n\n")
    if np.any(prob<=0):
        if verbosity >= 2:
            print("one p = 0, EXITING LOG_LIKELIHOOD FUNCTION")
        return(HUGE)

    if verbosity >= 2:
        print("OK")

    # print("prob=",prob)
    lnlikminus = -1*np.sum(prob_membership*np.log(prob))

    # penalization

    sumPenalization = 0.
    for i in range(len(params)):
        penalization = PenaltyFunction(params[i],bounds[i,0],bounds[i,1])
        sumPenalization += penalization

    if N_points_flag:
        N_points = 0
        N_points_flag = False
        
    # optional print 
    if verbosity >= 2:
        print(">>> pass = ", iPass,
              "-lnlik = ",       lnlikminus,
              "penalization = ",  sumPenalization,
              "RA_cen = ",           RA_cen,
              "Dec_cen = ",          Dec_cen,
              "log_scale_radius = ", log_scale_radius,
              "ellipticity = ",      ellipticity,
              "PA = ",               PA,
              "log_background = ",   log_background
              )
        np.savetxt("DEBUG/profcl_debug2.dat" + str(iPass),np.c_[RA_gal,Dec_gal,u,v,R_ellip,prob])
        ftest = open("DEBUG/profcl_debug2.dat" + str(iPass),'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:10.3f}\n".format(RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, lnlikminus))
        ftest.close()
        ftest = open(debug_file,'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:10.3f}\n".format(RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, lnlikminus))
        ftest.close()

    # return -ln likelihood + penalization
    return (lnlikminus + sumPenalization)

def PROFCL_Fit(RA, Dec, prob_membership, 
               RA_cen, Dec_cen, log_scale_radius, 
               ellipticity, PA, log_background, 
               bounds,
               R_min, R_max, model, 
               background_flag, recenter_flag, ellipticity_flag, 
               function=PROFCL_LogLikelihood,
               method="Nelder-Mead", bound_min=None, bound_max=None):
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

    ## checks on types of arguments

    # check that input positions are in numpy arrays
    if not isinstance(RA,np.ndarray):
        raise print("ERROR in PROFCL_fit: RA must be numpy array")

    if not isinstance(Dec,np.ndarray):
        raise print("ERROR in PROFCL_fit: Dec must be numpy array")

    # check that min and max projected radii are floats or ints
    CheckTypeIntorFloat(R_min,"PROFCL_fit","R_min")
    CheckTypeIntorFloat(R_max,"PROFCL_fit","R_max")

    # check that model is a string
    t = type(model)
    if t is not str:
        raise print("ERROR in PROFCL_fit: model is ", 
                         t, " ... it must be a str")
                             
    # check that flags are boolean
    CheckTypeBool(background_flag,"PROFCL_fit","background_flag")
    CheckTypeBool(recenter_flag,"PROFCL_fit","recenter_flag")
    CheckTypeBool(ellipticity_flag,"PROFCL_fit","ellipticity_flag")
    
    # check that method is a string
    t = type(method)
    if t is not str:
        raise print("ERROR in PROFCL_fit: method is ", 
                         t, " ... it must be a str")
                             
    ## checks on values of arguments
    
    # check that R_min > 0 (to avoid infinite surface densities)
    if R_min <= 0:
        raise print("ERROR in PROFCL_fit: R_min = ", 
                         R_min, " must be > 0")
                             
    # check that R_max > R_min
    if R_max < R_min:
        raise print("ERROR in PROFCL_fit: R_max = ", 
                         R_max, " must be > R_min = ", R_min )
                             
    # check model
    if model != "NFW" and model != "coredNFW" and  model != "Uniform":
        raise print("ERROR in PROFCL_fit: model = ", 
                         model, " not recognized... must be NFW or coredNFW or Uniform")
    
    # function of one variable
    if np.isnan(RA_cen):
        raise print("ERROR in PROFCL_fit: RA_cen is NaN!")
    if np.isnan(Dec_cen):
        raise print("ERROR in PROFCL_fit: Dec_cen is NaN!")

    cond = np.logical_and(R_sky >= R_min, R_sky <= R_max)
    if verbosity >= 2:
        print("Fit: N_tot = ", len(R_sky[cond]))
    params = np.array([
                       RA_cen,Dec_cen,log_scale_radius,
                       ellipticity,PA,log_background
                      ]
                     )

    # minimization
    # if method == 'brent' or method == 'Brent':
    #     if recenter_flag:
    #         raise print('ERROR in PROFCL_fit: brent minimization method cannot be used for fits with re-centering')
    #     if ellipticity_flag:
    #         raise print('ERROR in PROFCL_fit: brent minimization method cannot be used for elliptical fits')
        
    #     if bound_min is None or bound_max is None:
    #         return minimize_scalar(f, method='brent', tol=0.001)
    #     else:
    #         return minimize_scalar(f, bounds=(bound_min,bound_max), 
    #                                    method='bounded', tol=0.001)
    # else:
    maxfev = 500
    iPass = 0
    if   method == "Nelder-Mead":
        # return minimize(PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
        #                 method=method, tol=tolerance, bounds=bounds, options={'fatol':tol, 'maxfev':maxfev})
        return minimize              (PROFCL_LogLikelihood, params, args=(), 
                                      method=method, tol=tolerance, options={"maxfev":maxfev})
    # elif method == "Powell":
    #     return minimize              (PROFCL_LogLikelihood, params, args=(), 
    #                     method=method, tol=tolerance, bounds=bounds, options={"ftol":tolerance, "maxfev":maxfev})
    # elif method == "CG" or method == "BFGS":
    #     return minimize              (PROFCL_LogLikelihood, params, args=(), 
    #                     method=method, tol=tolerance, bounds=bounds, options={"gtol":tolerance, "maxiter":maxfev})
    # elif method == "Newton-CG":
    #     return minimize              (PROFCL_LogLikelihood, params, args=(), 
    #                     method=method, tol=tolerance, bounds=bounds, options={"xtol":tolerance, "maxiter":maxfev})
    elif method == "L-BFGS-B":
        return minimize              (PROFCL_LogLikelihood, params, args=(), 
                        method=method, tol=tolerance, bounds=bounds, options={"ftol":tolerance, "maxfun":maxfev})
    elif method == "SLSQP":
        return minimize              (PROFCL_LogLikelihood, params, args=(), 
                        method=method, jac=None, bounds=bounds, tol=None, options={"ftol":tolerance, "maxiter":maxfev})
    elif method == "TNC":
        return minimize              (PROFCL_LogLikelihood, params, args=(), 
                        method=method, jac=None, bounds=bounds, tol=None, options={"xtol":tolerance, "maxiter":maxfev})
    # elif method == "fmin_TNC":
    #     return fmin_tnc              (PROFCL_LogLikelihood, params, fprime=None, args=(),  approx_grad=True, bounds=bounds, epsilon=1.e-8, ftol=tolerance)
    elif method == "Diff-Evol":
        return differential_evolution(PROFCL_LogLikelihood, bounds, args=(), atol=tolerance)
    else:
        raise print("ERROR in PROFCL_fit: method = ", method, " is not yet implemented")

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

    separation_squared = np.array([])
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
        raise IOError("PROFCL_OpenFile: cannot open ", file)

def PROFCL_ReadClusterGalaxies_oldversion(cluster_galaxy_file,column_RA,column_pmem,column_z):
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
        # set p = 1 for all galaxies
        prob_membership = 0.*RA + 1.

    if column_z > 0:
        z = tab_galaxies[:,column_z-1]
    else:
        z = 0*RA - 1.
        
    if verbosity >= 1:
        print("found", len(RA), "galaxies")

    return (RA, Dec, prob_membership)

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

    RA  = tab_galaxies[:,column_RA-1].astype(float)
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

    if method in ("NM", "nm", "nelder-mead"):
        method2 = "Nelder-Mead"
    # elif method == "powell":
    #     method2 = "Powell"
    # elif method == "cg":
    #     method2 = "CG"
    # elif method == "bfgs":
    #     method2 = "BFGS"
    # elif method in ("newton-cg", "ncg", "NCG"):
    #     method2 = "Newton-CG"
    elif method in ("l-bfgs-b", "LBB", "lbb", "l"):
        method2 = "L-BFGS-B"
    elif method == "slsqp":
        method2 = "SLSQP"
    elif method == "tnc":
        method2 = "TNC"
    # elif method == "fmin_tnc":
    #     method2 = "fmin_TNC"
    elif method in ("de", "DE", "diffevol", "DiffEvol"):
        method2 = "Diff-Evol"
    elif method == "t":
        method2 = "test"
    else:
        method2 = method
    return(method2)

def convertModel(model):
    """Convert model abbreviations
    arg: abbreviation
    returns: model name"""

    # author: Gary Mamon

    if model in ('n', 'N', 'nfw', 'NFW'):
        model2 = 'NFW'
    elif model in ('c', 'C', 'cnfw', 'cNFW'):
        model2 = 'cNFW'
    elif model in ('u', 'U', 'uniform', 'UNIFORM', 'Uniform'):
        model2 = 'Uniform'
    elif model in ('nt', 'NFWt', 'truncNFW', 'tNFW', 'NFWtrunc'):
        model2 = 'NFWtrunc'
    else:
        return False
    return(model2)

def print_help():
    print("NAME\n    PROFCL")
    print("DESCRIPTION\n    Find best cluster parameters by BLE. No option is mandatory. Input file must be located in the same directory in order for the program to load default values.")
    print("    Every option in input file can be changed by typing the desired option and value from the list below.")
    print("SYNOPSIS\n    python3.5 PROFCL_v1.13 [-man options] [-auto] []")
    print("")
    print("OPTIONS   ")
    print("    -A")
    print("             Automatic mode option. If auto option is auto, PROFCL will not ask the user to enter manually each option. If auto option is manual, PROFCL will ask manual entry. Default value is manual.\n")
    print("    -v, --verbosity=NUMBER")
    print("             Verbosity (debug text). Must be followed by an integer between 0 and 5 (information given increases with number).")
    print("    -m, --model=WORD")
    print("             Model option must either be followed by NFW, cNFW or uniform.")
    print("    -M, --method=WORD")
    print("             Method option can be followed by tnc, bfgs, nm, de, lbb or powell.")
    print("    -r, --rmaxovervir=NUMBER")
    print("             Ratio of maximum over minimum allowed radii. Must be followed by a float.")
    print("    -n, --npoints=NUMBER")
    print("             Monte-Carlo option. Must be followed by the number of points (0 for analytical approximation).")
    print("    -t, --tolerance=NUMBER")
    print("             Absolute tolerance. Must be followed by a float.\n")
    print("    -d, --mockdir=WORD")
    print("             Mock data directory option. Must be followed by the directory location. ")
    print("    -o, --output=WORD")
    print("             Output file option. Must be followed by the name of the file.\n")
    print("    -D, --datatype=OPTION")
    print("             Data type to use. OPTION must always be a string between quotation mark. See DATA TYPE section for more information.")
    print("    -a, --adaptativescale")
    print("             Scale option. Adapt precision on each argument for better convergence.")
    print("    -e")
    print("             Ellipticity option.")
    print("    -c")
    print("             Re-centering option.")
    print("    -b")
    print("             Background option")
    print("    -x")
    print("             Extended data option. Only useful for debugging. Meant to disappear in later versions\n.")
    print("    --median=WORD")
    print("             Median option. Must either be y for trigering median separation, n for no, or o for computing only median separation.")
    print("    --minmemb=NUMBER")
    print("             Minimum member of galaxies in a cluster.")
    print("    --min=NUMBER")
    print("             Minimum ID for cluster to test. This option does not work with AMICO data. Instead directly type list of clusters with -D option.")
    print("    --max=NUMBER")
    print("            Maximum ID for cluster to test.")
    print("    --dfile=WORD, --detectionfile=WORD")
    print("             File containg clusters data from AMICO. Only works with AMICO data.")
    print("    --ddir=WORD, --detectiondir=WORD")
    print("             Directory of AMICO detection file.")
    print("    --afile=WORD, --associationfile=WORD")
    print("             File containing galaxies data from AMICO. Only works with AMICO data.")
    print("    --adir=WORD, --associationdir=WORD")
    print("             Directory of AMICO association file.")
    print("    --gfile=WORD, --galaxyfile=WORD")
    print("             File containing full EUCLID galaxies data. Only works with AMICO data.")
    print("    --gdir=WORD, --galaxydir=WORD")
    print("             Directory of EUCLID 200deg full galaxies data file.\n")
    print("    --ndfile=WORD, --newdetectionfile=WORD")
    print("             New detection file built with given selected clusters.")
    print("    --nafile=WORD, --newassociationfile=WORD")
    print("             New association file built with given selected galaxies.\n")
    print("    --loadinput=WORD")
    print("             Input file's name from which options are loaded. Default value is PROFCL_[version of PROFCL]_input.dat.")
    print("    --saveinput=WORD")
    print("             Input file's name within wich options are saved. Default value is PROFCL_[version of PROFCL]_input.dat.\n")
    print("    -s, --subclusters=NUMBER")
    print("             Number of subclusters to test for deblending. NOT YET IMPLEMENTED.")
    print("DATA TYPE\n    Information concerning OPTION for -D, --datatype argument. Default data are Mamon Mocks:\n")
    print("    Artis Mocks")
    print("             OPTION must be a string of the type 'A N1 N2 N3 N4' with N1 cluster richness, N2 cluster ellipticity, N3 cluster PA and N4 background (galaxies/arcmin^2). Default values are 160, 0.5, 50, 1.")
    print("    Mamon Mocks")
    print("             OPTION must be a string of the type 'M N1 N2 N3 N4 N5 N6 N7' with N1 cluster richness, N2 cluster ellipticity, N3 cluster PA, N4 background (galaxies/arcmin^2), ")
    print("             N5 logarithm of scale radius, N6 cluster center RA and N7 cluster center Dec. Default values are 160, 0.5, 50, 1, -2.08, 0, 0.")
    print("    AMICO data")
    print("             OPTION must be a string of the type 'd [N1-N2][N3-N4]', each list corresponding to a range of desired clusters indices. An unlimited amount of lists can be provided. ")
    print("             If no list is given PROFCL will skip detection and association files generation and will directly import data from detection and association files from previous build.")

def Ask_input_values():
    """ Ask values for variables if auto=="manual" was selected"""
    #authors  G. Mamon & W. Mercier

    global mock_dir, galaxy_dir, detection_dir, association_dir, \
           galaxy_file, detection_file, association_file, \
           new_detection_file, new_association_file, output_file, \
           verbosity, model, method, \
           do_ellipticity, do_center, do_background, \
           N_points, tolerance, test_flag, rank_list_asstr, background, \
           cluster_richness, cluster_ellipticity, cluster_ellipticity10, cluster_PA, cluster_loga, cluster_cen_RA, cluster_cen_Dec, \
           Rmaxovervir, min_members, rank_list, id_cluster_min, id_cluster_max, \
           rescale_flag, median_flag, recenter_flags, ellipticity_flags, fits_flag, background_flags, \
           data_galaxies_prefix, data_clusters_dir, data_clusters_prefix

    # verbosity
    ok = False
    while not ok:
        if auto == "manual":
            if verbosity != "":
                verbosity = input("Enter verbosity (0 -> minimal, 1 -> some, 2 -> more, 3 -> all) [%s]: "%verbosity) or verbosity
            else:
                verbosity = input("Enter verbosity (0 -> minimal, 1 -> some, 2 -> more, 3 -> all): ")
        try:
            verbosity = int(verbosity)
            if verbosity >= 0 and verbosity < 4:
                ok = True
            else:
                print("Verbosity must be between 0 and 3 included.")
        except:
            print("Error:", verbosity, "is not a number.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", verbosity, "for verbosity given in input file.")

    # test_flag = data type to use
    ok = False
    while not ok:
        if auto == "manual":
            if test_flag != "":
                test_flag = input("Enter d for data, a for academic (Gary), A for academic (Emmanuel), M for academic (Mamon), f for Flagship [%s]: "%test_flag) or test_flag
            else:
                test_flag = input("Enter d for data, a for academic (Gary), A for academic (Emmanuel), M for academic (Mamon), f for Flagship: ")
        if test_flag not in ('d', 'a', 'A', 'M', 'f'):
            print("Data type must be one among d, a, A, M, f. Given value", test_flag, "is not correct.")
        else:
            ok = True

        if test_flag not in ('d', 'a', 'A', 'M', 'f') and auto != "manual":
            raise ValueError("Wrong value", test_flag, "for data type given in input file.")

    # questions for cluster file
    if test_flag in ("f"):
        cluster_richness    = -1
        cluster_ellipticity = -1
        cluster_PA          = -1
        background          = -1
        Rmaxovervir         = -1

    # read richness, ellipticity, PA and background for Mocks only
    elif test_flag in ("a", "A", "M"):
        ok = False
        while not ok:
            if auto == "manual":
                if cluster_richness != "":
                    cluster_richness = input("Enter cluster richness to test [%s]: "%cluster_richness) or cluster_richness
                else:
                    cluster_richness = input("Enter cluster richness to test: ")
            try:
                cluster_richness = int(cluster_richness)
                if cluster_richness > 0:
                    ok = True
                else:
                    print("Cluster richness must an integer strictly greater than 0.")
            except:
                print("Error:", cluster_richness, "is not a number.")

            if not ok and auto != "manual":
                raise ValueError("Wrong value", cluster_richness, "for richness given in input file.")

        ok = False
        while not ok:
            if auto == "manual":
                if cluster_ellipticity != "":
                    cluster_ellipticity = input("Enter cluster ellipticity to test [%s]: "%cluster_ellipticity) or cluster_ellipticity
                else:
                    cluster_ellipticity = input("Enter cluster ellipticity to test: ")
            try:
                cluster_ellipticity     = float(cluster_ellipticity)
                if cluster_ellipticity >= 0 and cluster_ellipticity < 1:
                    ok = True
                else:
                    print("Cluster ellipticity must be a float between 0 and 1.")
                cluster_ellipticity_10  = 10*cluster_ellipticity
            except:
                print("Error:", cluster_ellipticity, "is not a number.")

            if not ok and auto != "manual":
                raise ValueError("Wrong value", cluster_ellipticity, "for ellipticity given in input file.")

        ok = False
        while not ok:
            if auto == "manual":
                if cluster_PA != "":
                    cluster_PA = input("Enter cluster PA to test (in deg) [%s]: "%cluster_PA) or cluster_PA
                else:
                    cluster_PA = input("Enter cluster PA to test (in deg): ")
            try:
                cluster_PA = int(cluster_PA)
                if cluster_PA >= 0 and cluster_PA < 180:
                    ok = True
                else:
                    print("Cluster PA must be an integer between 0 and 180 degrees.")
            except:
                print("Error:", cluster_PA, "is not a number.")

            if not ok and auto != "manual":
                raise ValueError("Wrong value", cluster_PA, "for position angle given in input file.")

        ok = False
        while not ok:
            if auto == "manual":
                if background != "":
                    background = input("Enter background surface density (arcmin^-2) [%s]: "%background) or background
                else:
                    background = input("Enter background surface density (arcmin^-2): ")
            try:
                background = float(background)
                if background >= 0:
                    ok = True
                else:
                    print("Background surface density must be greater than or equal to 0.")
            except:
                print("Error:", background, "is not a number.")

            if not ok and auto != "manual":
                raise ValueError("Wrong value", background, "for background surface density given in input file.")

    # Minimum allowed number of galaxies per cluster
    ok = False
    while not ok:
        if auto == "manual":
            if min_members != "":
                min_members = input("Enter minimum number of cluster members [%s]: "%min_members) or min_members
            else:
                min_members = input("Enter minimum number of cluster members: ")
        try:
            min_members = int(min_members)
            if min_members >= 1:
                ok = True
            else:
                print("Minimum number of clusters must be greater than  or equal to 1")
        except:
            print("Error:", min_members, "is not a number.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", minmembers, "for minmum number of galaxies per cluster given in input file.")

    # Ratio of R_max over R_vir
    ok = False
    while not ok:
        if auto == "manual":
            if Rmaxovervir != "":
                Rmaxovervir = input("Enter R_max/r_vir [%s]: "%Rmaxovervir) or Rmaxovervir
            else:
                Rmaxovervir = input("Enter R_max/r_vir: ")
        try:
            Rmaxovervir = float(Rmaxovervir)
            if Rmaxovervir > 0:
                ok = True
            else:
                print("R_max/R_vir must be strictly greater than 0.")
        except:
            print("Error:", Rmaxovervir, "is not a number.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", Rmaxovervir, "for R_max/R_vir given in input file.")

    # Read model
    ok = False
    while not ok:
        if auto == "manual":
            if model != "":
                model = input("Enter cluster model: (n for NFW, c for coredNFW, u for Uniform, t for truncatedNFW) [%s]: "%model) or model
            else:
                model = input("Enter cluster model: (n for NFW, c for coredNFW, u for Uniform, t for truncatedNFW): ")
        model    = convertModel(model)
        if model == False:
            model = ""
            if auto == "manual":
                print("Unkown model")
            else:
                raise ValueError("Wrong value", model, "for model given in input file.")
        else:
            ok = True

    # AMICO's data filenames and directorieso
    if test_flag == "d":
        if auto == "manual":
            if association_dir != "":
                association_dir = input("Enter AMICO's association file directory [%s]: "%association_dir) or association_dir
            else:
                association_dir = input("Enter AMICO's association file directory: ")
            if association_file != "":
                association_file = input("Enter AMICO's association file name [%s]: "%association_file) or association_file
            else:
                association_file = input("Enter AMICO's association file name: ")
            if detection_dir != "":
                detection_dir = input("Enter AMICO's detection file directory [%s]: "%detection_dir) or detection_dir
            else:
                detection_dir = input("Enter AMICO's detection file directory: ")
            if detection_file != "":
                detection_file = input("Enter AMICO's detection file name [%s]: "%detection_file) or detection_file
            else:
                detection_file = input("Enter AMICO's detection file name: ")
            if galaxy_dir != "":
                galaxy_dir = input("Enter AMICO's galaxy (200deg) file directory [%s]: "%galaxy_dir) or galaxy_dir
            else:
                galaxy_dir = input("Enter AMICO's galaxy (200deg) file directory: ")
            if galaxy_file != "":
                galaxy_file = input("Enter AMICO's galaxy (200deg) file name [%s]: "%galaxy_file) or galaxy_file
            else:
                galaxy_file = input("Enter AMICO's galaxy (200deg) file name: ")
            if detection_file != "":
                new_detection_file = input("Enter AMICO's new detection file name (containing selected clusters) [%s]: "%new_detection_file) or new_detection_file
            else:
                new_detection_file = input("Enter AMICO's new detection file name (containing selected clusters): ")
            if new_association_file != "":
                new_association_file = input("Enter AMICO's new association file name (containing selected galaxies) [%s]: "%new_association_file) or new_association_file
            else:
                new_association_file = input("Enter AMICO's new association file name (containing selected galaxies): ")

        ok = False
        while not ok:
            if auto == "manual":
                if rank_list_asstr != False:
                    rank_list_asstr = input("Enter lists of cluster IDs [%s]: "%rank_list_asstr) or rank_list_asstr
                else:
                    rank_list_asstr = input("Enter lists of cluster IDs: ")
            if rank_list_asstr == "":
                fits_flag = False
                ok = True
                continue
            try:
                arg = rank_list_asstr.split("]")[:-1] # split a first time the string in an array
                arg = [i.split("-") for i in arg]     # split a second time each element so that we can easily get the limits for the np.arange
            except:
                print("Error: format is not correct. Use -h option for more information.")

            rank_list = np.asarray([])
            for i in range(0, len(arg)):
                try:
                    lim_inf = int(arg[i][0][1:])
                    lim_sup = int(arg[i][1])+1
                    try:
                        rank_list = np.append(rank_list, np.arange(np.min([lim_inf, lim_sup]), np.max([lim_inf, lim_sup]))) # int type for the limits of the range but np.arange builds a numpy array of floats
                        rank_list = rank_list.astype(int) # thus needs to convert array to integers afterwards
                    except:
                        print("Error: issue when converting string to array for rank_list")
                except:
                    print("Error: at least one value is not a number")
            if rank_list != [] and not np.any(rank_list < 1):
                ok = True
            else:
                print("Error: at least one value in cluster rank list is lower than minimum cluster ID")

            if not ok and auto != "manual":
                raise ValueError("Wrong value", rank_list, "for cluster rank list given in input file.")

    if test_flag in ("A", "M", "f"):
        ok = False
        while not ok:
            if auto == "manual":
                if id_cluster_min != "" and id_cluster_max != "":
                    id_cluster_min_max = str(id_cluster_min) + " " + str(id_cluster_max)
                    id_cluster_min_max = input("Enter minimum and maximum cluster IDs [%s]: "%id_cluster_min_max) or id_cluster_min_max
                else:
                    id_cluster_min_max = input("Enter minimum and maximum cluster IDs: ")
            try:
                id_cluster_min, id_cluster_max = map(int, id_cluster_min_max.split())
                if id_cluster_min >= 0 and id_cluster_max >= 0:
                    ok = True
                else:
                    print("Error: either minmum or maximum or both cluster IDs are negative")
            except:
                print("Error: could not map given string to 2 integers")

            if not ok and auto != "manual":
               raise ValueError("Wrong value(s)", id_cluster_min, id_cluster_max, "for minimum and/or maximum cluster ID(s) given in input file.")

        if auto == "manual":
            if mock_dir != "":
                mock_dir = input("Enter mocks data directory [%s]: "%mock_dir) or mock_dir
            else:
                mock_dir = input("Enter mocks data directory: ")

    if test_flag == "a":
        if float(background) > 0.:
            mocki_file = 'Acad_' + model_test + cluster_richness + '_c3_skyRA15Dec10_ellip' + cluster_ellipticity_10 + 'PA' + cluster_PA + '_bg' + background + '_Rmax' + Rmaxovervir + '.dat'
        else:
            mock_file = 'Acad_' + model_test + cluster_richness + '_c3_skyRA15Dec10_ellip' + cluster_ellipticity_10 + 'PA' + cluster_PA + '.dat'
        id_cluster_min = 1
        id_cluster_max = 1
        column_z = 0

    if test_flag == "f":
        data_galaxies_prefix = "galaxies_inhalo_clusterlM14"
        data_clusters_dir = "."
        data_clusters_prefix = "lM14_lookup"

    if test_flag in ("n", "N"):
        if auto == "manual":
            if mock_file != "":
                mock_file = input("Enter mock file's name [%s]:"%mock_file) or mock_file
            else:
                mock_file = input("Enter mock file's name :")

    # Re-centering parameter ?
    ok = False
    while not ok:
        if auto == "manual":
            if do_center != "":
                do_center = input("Enter c for centering, n for none, b for both [%s]: "%do_center) or do_center
            else:
                do_center = input("Enter c for centering, n for none, b for both: ")
        if do_center in ('c', 'n', 'b'):
            ok = True
        else:
            print("Given value", do_center, " for do_center is not correct.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", do_center, "for do_center variable given in input file.")

    if do_center == 'c':
        recenter_flags = truearr
    elif do_center == 'n':
        recenter_flags = falsearr
    else:
        recenter_flags = falsetruearr

    # Ellipticity parameter ?
    ok = False
    while not ok:
        if auto == "manual":
            if do_ellipticity != "":
                do_ellipticity = input("Enter e for ellipticity/PA, n for none, b for both [%s]: "%do_ellipticity) or do_ellipticity
            else:
                do_ellipticity = input("Enter e for ellipticity/PA, n for none, b for both: ")
        if do_ellipticity in ('e', 'n', 'b'):
            ok = True
        else:
            print("Given value", do_ellipticity, "for do_ellipticity is not correct.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", do_ellipticity, "for do_ellipticity variable given in input file.")

    if do_ellipticity == 'e':
        ellipticity_flags = truearr
    elif do_ellipticity == 'n':
        ellipticity_flags = falsearr
    else:
        ellipticity_flags = falsetruearr

    # Number of points for Monte-Carlo or quadrature ?
    ok = 0
    while not ok:
        if auto == "manual":
            if do_ellipticity == 'e' and do_center == 'n':
                if "N_points" != "":
                    N_points = input("Enter 0 for approximate analytical, N_points >= 10000 for Monte-Carlo or N_points < 0 for quadrature [%s]: "%N_points) or N_points
                else:
                    N_points = input("Enter 0 for approximate analytical, N_points >= 10000 for Monte-Carlo or N_points < 0 for quadrature: ")
            elif do_center == 'c':
                if "N_points" != "":
                    N_points = input("Enter N_points for Monte-Carlo [%s]: "%N_points) or N_points
                else:
                    N_points = input("Enter N_points for Monte-Carlo: ")
            else:
                N_points = 0
        try:
            N_points = int(N_points)
            ok = True
            if do_ellipticity == "e" and do_center == "n" and N_points < 10000 and N_points != 0:
                ok = False
                print("Number of points should be greater than or equal to 10000  (for Monte-Carlo) or 0 (for analytical approximation) with ellipticity option but no centering.")
            elif do_center == "c" and N_points >= 0 and N_points < 10000:
                ok = False
                print("Number of points should be greater than or equal to 10000 (for Monte-Carlo) or strictly less than 0 (for quadrature) with centering option.")
        except:
            print("Error:", N_points, "is not a number.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", N_points, "for number of Monte-Carlo points given in input file.")

    # Background parameter ?
    ok = False
    while not ok:
        if auto == "manual":
            if do_background != "":
                do_background = input("Enter bg for background, n for none, b for both [%s]: "%do_background) or do_background
            else:
                do_background = input("Enter bg for background, n for none, b for both: ")
        if do_background in ('bg', 'n', 'b'):
            ok = True
        else:
            print("Given value", do_background, "of do_background is not correct.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", do_background, "for do_background variable given in input file.")

    if do_background in ('bg'):
        background_flags = truearr
    elif do_background == 'n':
        background_flags = falsearr
    else:
        background_flags = falsetruearr

    # Query minimization method:
    found  = False
    while not found:
        if auto == "manual":
            print("Available minimization methods are ", list(methods_dict.keys()))
            print("... can be abbreviated as ", list(methods_dict.values()))
            if 'method' != "":
                method = input("Enter minimization method (lower case OK, 'test' for test) [%s]: "%method) or method
            else:
                method = input("Enter minimization method (lower case OK, 'test' for test): ")
        method = str(method)
        t = type(method)
        if t is not str:
            raise TypeError('ERROR in PROFCL_Main: method is of type ',
                         t, ', but must be str')

        method = convertMethod(method)
        for meth in list(methods_dict.values()) + list(methods_dict.keys()):
            if method == meth:
                found = True
                break

        if method == 'test':
            found = True
        if not found:
            raise KeyError(method, ' not recognized.', "Method must be one among: ", list(methods_dict.values()) + list(methods_dict.keys()))

    # Query tolerance for minimization
    ok = False
    while not ok:
        if auto == "manual":
            if tolerance != "":
                tolerance = input("Enter relative tolerance of minimization method [%s]: " %tolerance) or tolerance
            else:
                tolerance = input("Enter relative tolerance of minimization method: ")
        try:
            tolerance = float(tolerance)
            if tolerance > 0:
                ok = True
            else:
                print("Tolerance parameter should be a non-zero positive number.")
        except:
            print("Error:", tolerance, "is not a number.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", tolerance, "for tolerance given in input file.")

    # Check if user wants to perform median separation method
    ok = False
    while not ok:
        if auto == "manual":
            if median_flag != "":
                median_flag = input("Do you want to perform median separation method (y for yes, n for no, o for median separation only) [%s]: " %median_flag) or median_flag
            else:
                median_flag = input("Do you want to perform median separation method (y for yes, n for no, o for median separation only):")
        if median_flag in ("y", "n", "o"):
            ok = True
        else:
            print("Median_flag must either be y for yes, n for no or o for only (skips log likelihood computation).")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", median_flag, "for median flag given in input file.")

    # Check if user wants to scale precision on parameters
    ok = False
    while not ok:
        if auto == "manual":
            if rescale_flag != "":
                rescale_flag = input("Do you want to have scaled precision on parameters (y for yes, n for no) [%s]: " %rescale_flag) or rescale_flag
            else:
                rescale_flag = input("Do you want to have scaled precision on parameters (y for yes, n for no):")
        if rescale_flag in ("y", "n"):
            ok = True
        else:
            print("Rescale_flag must either be y for yes or n for no.")

        if not ok and auto != "manual":
            raise ValueError("Wrong value", rescale_flag, "for rescaling flag given in input file.")
    print(auto)

def Save_input_in_file():
    """ Save input values after reading input file and changing values given as parameters in shell"""
    #author W. Mercier

    try:
        f = open(saveinput_file,'w')
    except IOError:
        raise IOError("cannot open file", saveinput_file)

    dict    = {"verbosity":verbosity, "mock_dir":mock_dir, "galaxy_dir":galaxy_dir, "detection_dir":detection_dir, "association_dir":association_dir, 
               "galaxy_file":galaxy_file, "detection_file":detection_file, "association_file":association_file, "new_detection_file":new_detection_file,
               "new_association_file":new_association_file, "output_file":output_file, "model":model, "method":method, "do_ellipticity":do_ellipticity,
               "do_center":do_center, "do_background":do_background, "N_points":N_points, "tolerance":tolerance, "median_flag":median_flag,
               "test_flag":test_flag, "id_cluster_min":id_cluster_min, "id_cluster_max":id_cluster_max, "cluster_richness":cluster_richness,
               "cluster_loga":cluster_loga, "cluster_ellipticity":cluster_ellipticity, "cluster_PA":cluster_PA, "cluster_cen_RA":cluster_cen_RA,
               "cluster_cen_Dec":cluster_cen_Dec, "background":background, "Rmaxovervir":Rmaxovervir, "rank_list_asstr":rank_list_asstr, 
               "min_members":min_members, "rescale_flag":rescale_flag, "mock_file":mock_file}

    for i, j in dict.items():
        f.write( i + "\t" + str(j) + "\n")

def read_input(file):
    """Read input file of default parameters"""
    # authors Gary Mamon & Wilfried Mercier

    global mock_dir, galaxy_dir, detection_dir, association_dir, \
           galaxy_file, detection_file, association_file, \
           new_detection_file, new_association_file, output_file, mock_file, \
           verbosity, model, method, \
           do_ellipticity, do_center, do_background, \
           N_points, tolerance, test_flag, rank_list_asstr, background, \
           id_cluster_min, id_cluster_max, cluster_richness, cluster_ellipticity, cluster_PA, cluster_loga, cluster_cen_RA, cluster_cen_Dec, \
           Rmaxovervir, min_members, \
           rescale_flag, median_flag, recenter_flags, ellipticity_flags, background_flags,\
           data_galaxies_prefix, data_clusters_dir, data_clusters_prefix

    try:
        f = open(file,'r')
        print(file)
    except:
        raise IOError("Cannot open file", file)

    tab = np.genfromtxt(file, dtype=str)
    for i in range(len(tab)):
        var = tab[i][0]
        val = tab[i][1]
        if var == 'verbosity':
            verbosity = int(val)
        elif var == 'mock_dir':
            mock_dir = val
        elif var == 'galaxy_dir':
            galaxy_dir = val
        elif var == 'detection_dir':
            detection_dir = val
        elif var == 'association_dir':
            association_dir = val
        elif var == 'galaxy_file':
            galaxy_file = val
        elif var == 'detection_file':
            detection_file = val
        elif var == 'association_file':
            association_file = val
        elif var == 'new_detection_file':
            new_detection_file = val
        elif var == 'new_association_file':
            new_association_file = val
        elif var == 'output_file':
            output_file = val
        elif var == 'mock_file':
            mock_file = val
        elif var == 'model':
            model = val
        elif var == 'method':
            method = val
        elif var == 'do_ellipticity':
            do_ellipticity = val
        elif var == 'do_center':
            do_center = val
        elif var == 'do_background':
            do_background = val
        elif var == 'N_points':
            N_points = int(val)
        elif var == 'tolerance':
            tolerance = float(val)
        elif var == 'median_flag':
            median_flag = val
        elif var == 'test_flag':
            test_flag = val
        elif var == 'id_cluster_min':
            id_cluster_min = int(val)
        elif var == 'id_cluster_max':
            id_cluster_max = int(val)
        elif var == 'cluster_richness':
            cluster_richness = int(val)
        elif var == 'cluster_loga':
            cluster_loga = float(val)
        elif var == 'cluster_ellipticity':
            cluster_ellipticity = float(val)
        elif var == 'cluster_PA':
            cluster_PA = int(val)
        elif var == 'cluster_cen_RA':
            cluster_cen_RA = float(val)
        elif var == 'cluster_cen_Dec':
            cluster_cen_Dec = float(val)
        elif var == 'background':
            background = float(val)
        elif var == 'Rmaxovervir':
            Rmaxovervir = float(val)
        elif var == 'rank_list_asstr':
            rank_list_asstr = val
        elif var == 'min_members':
            min_members = int(val)
        elif var == 'rescale_flag':
            rescale_flag = val

def MAIN_MAIN_MAIN():
    # dummy function to better see where MAIN starts below
    return('1')

### MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN --- --- MAIN

if __name__ == '__main__':
    """Main program for PROFCL"""

    # author: Gary Mamon & W. Mercier
    # constants
    degree              = np.pi/180.
    exp1                = np.exp(1.)
    exp10               = np.exp(10.)
    HUGE                = 1.e30
    TINY                = 1.e-8
    TINY_SHIFT_POS      = degree / 3600.     # one arcsec

    # initialization
    falsetruearr        = np.array([False, True])
    falsearr            = np.array([False])
    truearr             = np.array([True])
    RA_crit             = 2. # critical RA for RA coordinate transformation (deg)
    Dec_crit            = 80 # critical Dec for frame transformation (deg)
    R_min_over_maxR     = 0.02  # minimum allowed projected radius over maximum observed radius (from DETCL center)
    R_max_over_maxR     = 0.8     # maximum allowed projected radius over maximum observed radius (from DETCL center)
    min_R_over_rminus2  = 0.01  # minimal ratio R/r_minus2 where fit for projected mass is performed in advance
    max_R_over_rminus2  = 1000.  # maximal ratio R/r_minus2 where fit for projected mass is performed in advance
    column_RA           = 1
    column_RA_cen       = 10
    column_pmem         = 0
    column_z            = 3
    column_z_cen        = 2
    N_points_flag       = False
    models              = ("NFW", "coredNFW", "Uniform")
    methods_dict        = {"brent" : "brent", "Nelder-Mead" : "nm", "BFGS" : "bfgs", "L-BFGS-B" : "lbb", "TNC" : "tnc", "Diff-Evol" : "de", "test" : "t"}
    fmt                 = "{:19} {:7} {:40} {:11} {:.1e} {:d} {:d} {:d} {:5} {:8} {:8.4f} {:8.4f} {:7.3f} {:5.3f} {:3.0f} {:6.2f} {:6.1f} {:9.2f} {:9.2f} {:5d} {:8.4f} {:8.4f} {:7.3f} {:5.3f} {:3.0f} {:6.2f} {:8.3f}"
    fmt2                = fmt + "\n"
    date_time           = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    version             = "1.17"
    vdate               = "28 Mai 2018"
    version_date        = version + ' (' + vdate + ')'
    loadinput_file      = "PROFCL_" + version + "_input.dat"
    saveinput_file      = loadinput_file

    # Initialize to empty string (or False) all variables so that they can become global variables
    # rank_list_asstr is set as False because it can actually be an empty string (in order to avoid later bugs)
    # fits_flag is True by default meaning if will build new association and detection file from given selected clusters
    mock_dir, galaxy_dir, detection_dir, association_dir                    = "", "", "", ""
    galaxy_file, detection_file, association_file                           = "", "", ""
    new_detection_file, new_association_file, output_file                   = "", "", ""
    verbosity, model, method                                                = "", "", ""
    do_ellipticity, do_center, do_background                                = "", "", ""
    N_points, tolerance, test_flag, rank_list_asstr, background             = "", "", "", False, ""
    id_cluster_min, id_cluster_max, rank_list                               = "", "", ""
    cluster_richness, cluster_ellipticity, cluster_ellipticity10            = "", "", ""
    cluster_PA, cluster_loga, cluster_cen_RA, cluster_cen_Dec               = "", "", "", ""
    Rmaxovervir, min_members                                                = "", ""
    rescale_flag, median_flag, recenter_flags, ellipticity_flags, fits_flag = "", "", "", "", True
    background_flags                                                        = ""
    data_galaxies_prefix, data_clusters_dir, data_clusters_prefix           = "", "", ""
    mock_file                                                               = ""

    # MAIN local variables
    auto                = "manual"
    extended_data_flag  = False

    # author: Gary Mamon
    optlist, args = getopt.getopt(sys.argv[1:],
                                  "ehcAbaxv:m:M:r:t:d:o:n:s:l:D:s:",
                                  ['verb=', 'ver=', 'verbosity=', 'model=', 'mod=', 'meth=', 'method=', 'met=',
                                   'datatype=', 'ofile=', 'minmax=', 'sub=', 'load=', 
                                   'RmaxoverVir=', 'Rmaxovervir=', 'rmaxoverVir=', 'rmaxovervir=', 
                                   'tol=', 'tolerance=', 'minmemb=', 'Minmemb=', 'mm=', 'dtype=', 'outputfile=', 
                                   'Min=', 'min=', 'MIN=', 'Max=', 'max=', 'MAX=', 
                                   'minmem=', 'Minmem=', 'mockdir=', 'mdir=', 'mockdirectory=', 
                                   'Npoints=', 'npoints=', 'output=', 'median=', 'med=', 
                                   'tole=', 'loadinput=', 'subclusters=', 
                                   'galaxyfile=', 'gfile=', 'galaxydir=', 'gdir=', 
                                   'detectiondir=', 'ddir=', 'associationdir=', 'adir=', 
                                   'detectionfile=', 'dfile=', 'associationfile=', 'afile=', 
                                   'newdetectionfile=', 'ndfile=', 'newassociationfile=', 'nafile=',
                                   'saveinput=', 'help'
                                  ])

    # default parameters
    read_input(loadinput_file)

    # read arguments
    size = len(sys.argv[1:])
    if np.any(np.asarray(optlist) == "-A"):
        auto = "auto"

    # Ask user to enter values if manual mode and convert/check type of some values
    Ask_input_values()

    if size >=1:
        if sys.argv[1] in ("-h", "-help"):
            print_help()
            exit(0)
        else:
            for opt, arg in optlist:
                if opt == "-A":
                    auto = "auto"
                elif opt in ("-v", "--ver", "--verb", "--verbosity"):
                    verbosity = int(arg)
                elif opt in ("-m", "--mod", "--model"):
                    model = arg
                elif opt in ("-M", "--met", "--meth", "--method"):
                    method = arg
                elif opt in ("-r", "--RmaxoverVir", "--Rmaxovervir", "--rmaxoverVir", "--rmaxovervir"):
                    Rmaxovervir = float(arg)
                elif opt in ("-t", "--tol", "-tole", "-tolerance"):
                    tolerance = float(arg)
                elif opt in ("-d", "--mockdir", "--mdir", "--mockdirectory"):
                    data_galaxies_dir = arg
                elif opt in ("-n", "--Npoints", "--npoints"):
                    N_points = int(arg)
                elif opt in ("-o", "--output", "--ouputfile", "--ofile"):
                    output_file = arg
                elif opt in ("-e"):
                    do_ellipticity = "e"
                elif opt in ("-c"):
                    do_center = "c"
                elif opt in ("-b"):
                    do_background = "bg"
                elif opt in ("-a"):
                    rescale_flag = "y"
                elif opt in ("-x"):
                    extended_data_flag = True
                elif opt in ("--saveinput"):
                    saveinput_file = arg
                elif opt in ("--loadinput"):
                    loadinput_file = arg
                elif opt in ("--median", "--med"):
                    median_flag = arg
                elif opt in ("--minmemb", "--Minmemb", "--mm", "--minmem", "--Minmem"):
                    min_members = int(arg)
                elif opt in ("--min", "--Min", "--MIN"):
                    id_cluster_min = arg
                elif opt in ("--max", "--Max", "--MAX"):
                    id_cluster_max = arg
                elif opt in ("-s", "--subclusters", "--sub"):
                    saveinput_file = arg
                elif opt in ("--detectiondir", "--ddir"):
                    detection_dir = arg
                elif opt in ("--associationdir", "--adir"):
                    association_dir = arg
                elif opt in ("--galaxydir", "--gdir"):
                    galaxy_file = arg
                elif opt in ("--detectionfile", "--dfile"):
                    detection_file = arg
                elif opt in ("--associationfile", "--afile"):
                    association_file = arg
                elif opt in ("--galaxyfile", "--gfile"):
                    galaxy_file = arg
                elif opt in ("--newdetectionfile", "--ndfile"):
                    new_detection_file = arg
                elif opt in ("--newassociationfile", "--nafile"):
                    new_association_file = arg
                elif opt in ("-D", "--datatype", "--dtype"):
                    arg = arg.split()
                    test_flag = arg[0].upper()
                    if test_flag in ("M", "m"):
                        if len(arg) == 8:
                            cluster_richness    = int(arg[1])
                            cluster_ellipticity = float(arg[2])
                            cluster_PA          = int(arg[3])
                            background          = int(arg[4])
                            cluster_loga        = float(arg[5])
                            cluster_cen_RA      = float(arg[6])
                            cluster_cen_Dec     = float(arg[7])
                    elif test_flag in ("a", "A"):
                        cluster_loga        = float(arg[1])
                        cluster_richness    = int(arg[2])
                        cluster_ellipticity = float(arg[3])
                        cluster_PA          = int(arg[4])
                        background          = int(arg[5])
                    elif test_flag in ("d", "D"):
                        test_flag = "d"
                        rank_list_asstr = arg[1]
                        rank_list       = np.array([])
                        if len(arg) < 2:
                            fits_flag = False
                        else:
                            arg = rank_list_asstr.split("]")[:-1] # split a first time the string in an array
                            arg = [i.split("-") for i in arg]     # split a second time each element so that we can easily get the limits for the np.arange
                            for i in range(0, len(arg)):
                                rank_list = np.append(rank_list, np.arange(int(arg[i][0][1:]), int(arg[i][1])+1)) # int type for the limits of the range but np.arange builds a numpy array of floats
                            rank_list = rank_list.astype(int) # thus need to convert array to integers afterwards
                    elif test_flag in ("n", "N"):
                        mock_file = arg
                else:
                    raise ValueError("Unknown option", opt, ". Use option -h for more information")

    # Save in input file new parameters
    Save_input_in_file()

    #########################################################
    ### dealing with fits files if AMICO data was choosen ###
    #########################################################
    if test_flag == "d" and fits_flag == True:
        # read AMICO data file
        print("Opening cluster data...")
        with fits.open(data_clusters_dir + "/" + data_clusters_prefix + '.fits', memmap=True) as hdu:
            data                                                                = hdu[1].data
            header                                                              = hdu[1].header
            ID, RA_cen_all, Dec_cen_all, z, z_err, SNR, radius, richness, rank  = data['ID'], data['RA'], data['Dec'], \
                                                                                  data['z'], data['z_err'], data['SNR'], \
                                                                                  data['radius'], data['richness'], data['rank']

        # selecting desired data from rank_list
        ID, RA_cen_all, Dec_cen_all, z = np.take(ID, rank_list-1), np.take(RA_cen_all, rank_list-1), \
                                         np.take(Dec_cen_all, rank_list-1), np.take(z, rank_list-1),
        z_err, SNR, radius, richness   = np.take(z_err, rank_list-1), np.take(SNR, rank_list-1), \
                                         np.take(radius, rank_list-1), np.take(richness, rank_list-1)

        # saving new data in new detection.fits file
        print("Saving selected cluster data...")
        hdul = fits.BinTableHDU.from_columns([fits.Column(name="ID", array=ID, format='J'),
                                              fits.Column(name="RA", array=RA_cen_all, format='E'),
                                              fits.Column(name="Dec", array=Dec_cen_all, format='E'),
                                              fits.Column(name="z", array=z, format='E'),
                                              fits.Column(name="z_err", array=z_err, format='E'),
                                              fits.Column(name="SNR", array=SNR, format='E'),
                                              fits.Column(name="radius", array=radius, format='E'),
                                              fits.Column(name="richness", array=richness, format='E') ], header=header)

        # file of clusters reducted to clusters under study
        fits_name = detection_file
        try:
            hdul.writeto(fits_name)
        except:
            print(fits_name, " file already exists. Could not overwrite it. Please delete it before building a new one.")

        # opening galaxies data .fits file
        print("Opening galaxies data...")
        with fits.open(data_galaxies_dir + "/" + data_galaxies_prefix + ".fits", memmap=True) as hdu:
            data             = hdu[1].data
            header           = hdu[1].header
            ID_g, ID_c, p_c  = data['ID_g'], data['ID_c'], data['p_c']

        # keeping galaxies from desired clusters
        print("Selecting appropriate galaxies...")
        mask    = np.isin(ID_c, rank_list)
        ID_g    = ID_g[mask]
        ID_c    = ID_c[mask]
        p_c     = p_c[mask]

        # getting galaxies RA and Dec values from big file with galaxy positions
        with fits.open(galaxies_positions_dir + "/" + galaxies_positions_prefix + ".fits", memmap=True) as hdu:
            print("Reading galaxies positions...")
            ID_full     = hdu[1].data['ID']
            mask        = np.isin(ID_full, ID_g)
            ID_select   = ID_full[mask]
            RA_gal      = hdu[1].data['ra'][mask]
            Dec_gal     = hdu[1].data['dec'][mask]

            #building an array of indices which links the structure of ID_select/RA and Dec with the structure of ID_g (which contains more values due to galaxies belonging to many clusters)
            unique, indices = np.unique(ID_g, return_inverse=True)
            RA_gal  = RA_gal[indices]
            Dec_gal = Dec_gal[indices]

        # saving new galaxies data in new association .fits file
        print("Saving selected galaxies data...")
        hdul = fits.BinTableHDU.from_columns([fits.Column(name="ID_g", array=ID_g, format='J'),
                                              fits.Column(name="ID_c", array=ID_c, format='J'),
                                              fits.Column(name="p_c", array=p_c, format='E'),
                                              fits.Column(name="RA", array=RA_gal, format='E'),
                                              fits.Column(name="Dec", array=Dec_gal, format='E')], header=header)

        fits_name = association_file
        try:
            hdul.writeto(fits_name)
        except:
            print(fits_name, " file already exists. Could not overwrite it. Please delete it before building a new one.")

    # if no .fits file was built load data from detection and association files
    if fits_flag == False:
        with fits.open(detection_file) as hdu:
            data = hdu[1].data
            ID, RA_cen_all, Dec_cen_all, z, z_err, SNR, radius, richness = data['ID'], data['RA'], data['Dec'], data['z'], data['z_err'], data['SNR'], data['radius'], data['richness']
        with fits.open(association_file) as hdu:
            data = hdu[1].data
            ID_g, ID_c, p_c, RA_gal, Dec_gal = data['ID_g'], data['ID_c'], data['p_c'], data['RA'], data['Dec']

    # extract parameters from data file name for Mocks where full name was given
    if test_flag in ('n', 'N'):
        file_str = mock_file.split('_')
        model_test_tmp = file_str[1]
        ellipstring = file_str[2]
        ellip = 0.1*float(ellipstring[1:])
        PA = float(file_str[3][2:])
        a = float(file_str[4][1:]) / 60.
        log_scale_radius = np.log10(a)
        bg = float(file_str[6][3:]) * 3600.
        log_background = np.log10(bg)
        RA_cen_init0 = float(file_str[7][2:])
        Dec_cen_init0 = float(file_str[8][3:-4])
        id_cluster_min_max = "1 1"

        model_test = convertModel(model_test_tmp)
        ellipticity_flags = falsetruearr

    # cluster centers
    if test_flag == 'f':
        clusters_file = data_clusters_dir + '/' + data_clusters_prefix + ".dat"
        (ID_all, RA_cen_all, Dec_cen_all) = PROFCL_ReadClusterCenters(clusters_file,column_RA_cen)
        idx_clusters = np.arange(0, len(ID_all))
    elif test_flag == "a":
        (RA_cen_init0, Dec_cen_init0) = (15, 10)
    elif test_flag in ("A", "n"):
        (RA_cen_init0, Dec_cen_init0) = (0, 0)

    # open output file in append mode
    f_out = open(output_file, 'a')

    if extended_data_flag:
        extended_data_file  = open("data.debug", "w")
        extended_data_file.write("cluster_id method tol recenter_flag ell_flag bg_flag sN_points model RA_cen_pred RA_cen_bestfit Dec_cen_pred Dec_cen_bestfit lsr_pred lsr_bestfit ell_pred ell_bestfit PA_pred PA_bestfit log10(bg)_pred log10(bg)_bestfit -ln(likelihood) time_taken\n")

    if test_flag == "d":
        cluster_range   = np.unique(ID_c)
        galaxy_file0    = "PROFCL_association.fits/PROFCL_detection.fits"
    else:
        cluster_range = range(id_cluster_min,id_cluster_max+1)
    # loop over clusters
    for id_cluster in range(id_cluster_min,id_cluster_max+1):
        print ("\nid_cluster = ", id_cluster)

        # galaxy file
        if test_flag == "f":
            mock_file0 = data_galaxies_prefix + '_' + str(id_cluster) + '.dat'
            mock_file = data_galaxies_dir + '/' + mock_file0
        elif test_flag == "a":
            if float(background) > 0.:
                mock_file0 = 'Acad_' + str(model_test) + str(cluster_richness) + '_c3_skyRA15Dec10_ellip' + str(cluster_ellipticity_10) + 'PA' + str(cluster_PA) + '_bg' + str(background) + '_Rmax' + str(Rmaxovervir) + '.dat'
            else:
                mock_file0 = 'Acad_' + str(model_test) + str(cluster_richness) + '_c3_skyRA15Dec10_ellip' + str(cluster_ellipticity_10) + 'PA' + str(cluster_PA) + '.dat'
        elif test_flag == "A":
            mock_file0 = 'MockNFW' + str(cluster_richness) + 'ellip' + str(cluster_ellipticity_10) + 'loga2.08PA50center0With_background20num' +  str(id_cluster) + '.dat'
        elif test_flag == "M":
            mock_file0 = 'MockNFW' + str(cluster_richness) + 'ellip' + str(cluster_ellipticity) + 'loga' + str(cluster_loga) + 'PA' + str(cluster_PA) + 'center' + str(cluster_cen_RA) + "_" + str(cluster_cen_Dec) + "With_background" + str(int(background)) + 'num' + str(id_cluster) + '.dat'

        if test_flag in ("a", "A", "M"):
            mock_file = mock_dir + '/' + mock_file0

        # cluster center
        if test_flag in ('d', 'f'):
            RA_cen_init0 = np.asscalar(RA_cen_all[ID_all==id_cluster])
            Dec_cen_init0 = np.asscalar(Dec_cen_all[ID_all==id_cluster])
            if verbosity >= 1:
                print("RA_cen_init0=",RA_cen_init0,"Dec_cen_init0=",Dec_cen_init0)

        # read galaxy data (coordinates in degrees)
        if test_flag in ("A", "a", "f"):
            (RA, Dec, prob_membership, z) = PROFCL_ReadClusterGalaxies(mock_file, column_RA, column_pmem, column_z, test_flag)
        elif test_flag == "M":
            (RA, Dec, prob_membership, bg_or_model) = PROFCL_ReadClusterGalaxies(mock_file, 2, 0, 4, test_flag)
            bg_RA   = RA[bg_or_model == "bg"] #separate background and NFW-like data (useful for debugging)
            bg_Dec  = Dec[bg_or_model == "bg"]
            mod_RA  = RA[bg_or_model == "model"]
            mod_Dec = Dec[bg_or_model == "model"]

        #select galaxies coordinates and probabilities of membership in current cluster
        elif test_flag == "d":
            RA              = RA_gal[ID_c == id_cluster]
            Dec             = Dec_gal[ID_c == id_cluster]
            prob_membership = p_c[ID_c == id_cluster]
            z_clust         = z[ID == id_cluster][0]
            z_err_clust     = z_err[ID == id_cluster][0]
            SNR_clust       = SNR[ID == id_cluster][0]

        N_data = len(RA)
        if verbosity >= 1:
            print("*** cluster ", id_cluster, "RA_cen=", RA_cen_init0, "Dec_cen=",Dec_cen_init0,"N_data =",N_data)

        # skip cluster if too few members
        if N_data < min_members:
            if verbosity >= 1:
                print("skipping cluster ", id_cluster, " which only has ", N_data, " members")
            continue

        # if median separation was activated
        if median_flag in ('y', 'o'):
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
        RA_cen_init0, Dec_cen_init0 = guess_center(RA_gal, Dec_gal)
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
            RA_gal += 180.
            RA_gal = np.where(RA_gal > 360., RA_gal-360., RA_gal)
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
            RA_gal = coords_galactic.l.deg
            Dec_gal = coords_galactic.b.deg
            if verbosity >= 3:
                print("shifting position to galactic coordinates")
        else:
            eq2gal_flag = False

        if verbosity >= 1:
            print("Predicted center position after shifting: ", RA_cen_init, Dec_cen_init, "\n")


        cosDec_cen_init = np.cos(Dec_cen_init * degree)

        # min and max projected angular radii from maximum separation from center (in degrees)
        R_sky   = np.sqrt((RA-RA_cen_init)**2 + (Dec-Dec_cen_init)**2)
        maxR    = np.max(R_sky)
        R_min   = R_min_over_maxR * maxR

        # print("R_sky=", R_sky[:10])
        # print("CPU= ", t2-t1)
        # c1 = SkyCoord(RA_cen_init,Dec_cen_init,unit="deg")
        # c2 = SkyCoord(RA_gal,Dec_gal,unit="deg")
        # R_sky2 = c1.separation(c2).deg
        # print("R_sky2=", R_sky2[:10])
        # t3 = time.process_time()
        # print("astropy: CPU=", t3-t2)
        # exit()
        # AstroPy separations are same but 50x slower to compute!
        maxR = np.max(R_sky)
        R_min = R_min_over_maxR * maxR

        if verbosity>=1:
            print("MAIN: R_min set to ",R_min)

        R_max = R_max_over_maxR * maxR
        if test_flag == 'a':
            R_max = 0.04932   # 2 r_vir

        # maximum background surface density to prevent negative numerator in galaxy probability
        condition = np.logical_and(R_sky >= R_min, R_sky <= R_max)

        # skip cluster if no galaxy lies within limits
        if not np.any(condition):
            continue

        # restrict projected radii and membership probabilities to those within limits
        RA_gal          = RA_gal[condition]
        Dec_gal         = Dec_gal[condition]
        prob_membership = prob_membership[condition]
        R_sky           = R_sky[condition]
        if verbosity >= 1:
            print(len(RA_gal), " galaxies within R_min = ", R_min, " and R_max = ", R_max)

        N_tot = len(RA_gal)
        if N_tot == 0:
            print("ERROR in PROFCL_lnlikelihood: ",
              "no galaxies found with projected radius between ",
              R_min, " and ", R_max,
              " around RA = ", RA_cen_init, " and Dec = ", Dec_cen_init)

        # max background: assume all galaxies in background!
        log_background_maxallow_default = np.log10( N_tot/(np.pi*(R_max*R_max-R_min*R_min)) )
        # reduce by factor 25% to avoid errors from cartesian geometry
        log_background_maxallow_default = np.log10(0.75) + log_background_maxallow_default

        # guess ellipticity and PA using 2nd moments of distribution
        
        ellipticity_pred, PA_pred = guess_ellip_PA(RA_gal,Dec_gal)
        if verbosity >= 2:
            print("MAIN: ellipticity_pred = ",ellipticity_pred,"\n      PA_pred = ", PA_pred)

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

                    # bounds on parameters according to case
                    for model in models:
                        if model != model_test:
                            continue
                        if verbosity >= 3:
                            print('* MAIN: model = ', model)

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
                            log_background_minallow = log_background_maxallow - 6

                        # add extra parameter for NFWtrunc
                        if model == 'NFWtrunc':
                            R_cut_minallow = R_max / 4.
                            R_cut_maxallow = 2. * R_max
                            R_cut          = 0.7 * R_max
                            N_params       += 1
                        else:
                            # irrelevant, since only considered for NFWtrunc model
                            R_cut_minallow = 2. * R_max
                            R_cut_maxallow = R_cut_minallow
                            R_cut = R_cut_minallow

                        log_background = (log_background_minallow
                                          +         log_background_maxallow
                                          ) / 2.

                        if verbosity >= 1:
                            print("RA_cen_minallow = ", RA_cen_minallow, "RA_cen_maxallow = ", RA_cen_maxallow, " RA_guess = ", RA_cen)
                            print("Dec_cen_minallow = ", Dec_cen_minallow, "Dec_cen_maxallow = ", Dec_cen_maxallow, " Dec_guess = ", Dec_cen)
                            print("log_scale_radius_minallow = ", log_scale_radius_minallow, "log_scale_radius_maxallow = ", log_scale_radius_maxallow, " log_scale_radius_guess = ", log_scale_radius)
                            print("log_background_minallow = ", log_background_minallow, "log_background_maxallow = ", log_background_maxallow, " log_background_guess = ", log_background)
                            print("ellipticity_minallow = ", ellipticity_minallow, "ellipticity_maxallow = ", ellipticity_maxallow, " ellipticity_guess = ", ellipticity)
                            print("PA_minallow = ", PA_minallow, "PA_maxallow = ", PA_maxallow, " PA_guess = ", PA)

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

                                # - ln L
                                params = RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background
                                lnlikminus = PROFCL_LogLikelihood(params,RA,Dec,prob_membership,
                                                                R_min, R_max, model)
                                print("-lnL = ", lnlikminus)
                                cont = input("Continue (y|n)? ")
                            exit()

                        #time taken till there by the program
                        start_time = time.process_time()

                        #make sure previous predicted values are not overwritten
                        log_scale_radius_pred = log_scale_radius
                        ellipticity_pred      = ellipticity
                        RA_cen_pred           = RA_cen
                        Dec_cen_pred          = Dec_cen
                        PA_pred               = PA
                        log_background_pred   = log_background

                        if median_flag != 'o':
                            # set bounds (without or with re-scaling)

                            if rescale_flag == "y":
                                params  = [RA_cen/(R_max*np.cos(Dec_cen*degree)), Dec_cen/R_max, log_scale_radius, ellipticity/5., PA/500., log_background/10., np.log10(R_cut)/5.]
                                bounds = np.array([(RA_cen_minallow/(R_max*np.cos(Dec_cen*degree)), RA_cen_maxallow/(R_max*np.cos(Dec_cen*degree))),
                                                   (Dec_cen_minallow/R_max, Dec_cen_maxallow/R_max),
                                                   (log_scale_radius_minallow, log_scale_radius_maxallow),
                                                   (ellipticity_minallow/5., ellipticity_maxallow/5.),
                                                   (PA_minallow/500., PA_maxallow/500.),
                                                   (log_background_minallow-1., log_background_maxallow-1.),
                                                   (np.log10(R_cut_minallow/5.), np.log10(R_cut_maxallow/5.))
                                ])
                            else:
                                params  = [RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_background, R_cut]
                                bounds = np.array([(RA_cen_minallow, RA_cen_maxallow),
                                                   (Dec_cen_minallow, Dec_cen_maxallow),
                                                   (log_scale_radius_minallow, log_scale_radius_maxallow),
                                                   (ellipticity_minallow, ellipticity_maxallow),
                                                   (PA_minallow, PA_maxallow),
                                                   (log_background_minallow, log_background_maxallow),
                                                   (np.log10(R_cut_minallow), np.log10(R_cut_maxallow))
                                ])

                            # perform maximum likelihood fit
                            fit_results = PROFCL_Fit(RA_gal, Dec_gal, prob_membership, 
                                                     RA_cen,Dec_cen,log_scale_radius,
                                                     ellipticity,PA,log_background, 
                                                     bounds,
                                                     R_min, R_max, model, 
                                                     background_flag, recenter_flag, ellipticity_flag, 
                                                     PROFCL_LogLikelihood, method)

                            #get time taken for the computation
                            time_taken = time.process_time() - start_time
                            if verbosity >= 2:
                                print("Computation time = ", time_taken)

                            params_bestfit = fit_results.x

                            if verbosity >= 2:
                                print("N_fev N_it success = ", fit_results.nfev, fit_results.nit, fit_results.success)
                                print ("params_bestfit = ", params_bestfit)

                            [RA_cen_bestfit, Dec_cen_bestfit, log_scale_radius_bestfit, ellipticity_bestfit, PA_bestfit, log_background_bestfit] = params_bestfit
                        if rescale_flag == "y":
                            Dec_cen_bestfit         *= R_max
                            RA_cen_bestfit          *= (R_max * np.cos(Dec_cen_bestfit*degree))
                            ellipticity_bestfit     *= 5.
                            PA_bestfit              *= 500.
                            log_background_bestfit  += 1.
                            R_cut_bestfit           += 0.7


                            # convert central position back to equatorial coordinates if galactic coordinates had been used
                            if eq2gal_flag:
                                coords_galactic = SkyCoord(RA_cen,Dec_cen,frame='galactic',unit='deg')
                                coords_equatorial = coords_galactic.icrs
                                RA_cen_pred  = coords_equatorial.ra
                                Dec_cen_pred = coords_equatorial.dec
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

                            # Bayesian evidence:
                            BIC = 2.*fit_results.fun + N_params * np.log(N_tot)

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
                                    print ("BIC = ", BIC, "type = ", type(BIC))
                                    print ("N_eval = ", fit_results.nfev, "type = ", type(fit_results.nfev))
                                    print("bounds = ", bounds)

                            # cluster 3D normalization N(r_scale_radius) = N(r_{-2}) = N(a)

                            num = N_tot - np.pi * (R_max*R_max-R_Min*R_min) * 10.**log_background_bestfit
                            a_bestfit = 10. ** log_scale_radius_bestfit
                            DeltaCenter_bestfit = AngularSeparation(RA_cen_bestfit,Dec_cen_bestfit,RA_cen_init,Dec_cen_init)
                            denom = \
                                    ProjectedNumber_tilde(R_max/a_bestfit,model,ellipticity_bestfit,DeltaCenter_bestfit) \
                                    - \
                                    ProjectedNumber_tilde(R_min/a_bestfit,model,ellipticity_bestfit,DeltaCenter_bestfit)
                            Nofa_bestfit = num/denom

                            # print results to screen
                            print (" date/time          cluID/file              method      tol c e b int model    RA0       Dec0    log_r_{-2} ellip  PA log_bg   -lnL       BIC      Npasses     RA_pred       Dec_pred        log_r_pred      ellip_pred       PA_pred       log_bg_pred       Computation_time(s)")

                            sN_points = '%.g' % N_points
                            print (fmt.format(
                                date_time, version, galaxy_file, method, tolerance,
                                recenter_flag, ellipticity_flag, background_flag, sN_points, model, 
                                RA_cen_bestfit, Dec_cen_bestfit, log_scale_radius_bestfit, ellipticity_bestfit, PA_bestfit, log_background_bestfit, Nofa_bestfit,
                                fit_results.fun, BIC, fit_results.nfev, RA_cen_pred, Dec_cen_pred, log_scale_radius_pred, ellipticity_pred, PA_pred, log_background_pred, time_taken)
                            )
                            #print("2323 : ", log_scale_radius, log_scale_radius_bestfit, ellipticity, ellipticity_bestfit)
                            # print results to file
                            f_out.write (fmt2.format(
                                date_time, version, galaxy_file, method, tolerance, recenter_flag, ellipticity_flag, 
                                background_flag, sN_points, model, 
                                RA_cen_bestfit, Dec_cen_bestfit, log_scale_radius_bestfit, ellipticity_bestfit, PA_bestfit, log_background_bestfit, Nofa_bestfit,
                                fit_results.fun, BIC, fit_results.nfev, RA_cen_pred, Dec_cen_pred, log_scale_radius_pred, ellipticity_pred, PA_pred, log_background_pred, time_taken)
                            )

        # print also median separation using same format
        if median_flag in ('y', 'o'):
            log_background_best_fit = -99
            print        (fmt.format(date_time, version, galaxy_file, "median-sep", 0, 0, 0, 0, 'x', "none", RA_cen_init0, Dec_cen_init0, log_median_separation, 0, 0, log_background_best_fit, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, time_taken))
            f_out.write (fmt2.format(date_time, version, galaxy_file, "median-sep", 0, 0, 0, 0, 'x', "none", RA_cen_init0, Dec_cen_init0, log_median_separation, 0, 0, log_background_best_fit, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, time_taken))

    # end of cluster loop

    f_out.close()

    ## EXTRAS for testing

    ## transform from RA,Dec to cartesian and then to projected radii
    u,v = uv_from_RADec(RA_gal,Dec_gal)

    # equivalent circular projected radii (in degrees)
    R_ellip_squared = u*u + v*v/(1.-ellipticity)**2.
    if np.any(R_ellip_squared) < 0.:
        R_ellip_squared_min = np.min(R_ellip_squared)
        # if min(R^2) < -1*(1 arcsec)^2, set to 0, else stop
        if R_ellip_squared_min > -1/3600.**2.:
            if verbosity >= 2:
                print("min(R^2) slightly < 0!")
            R_ellip_squared = np.where(R_ellip_squared < 0.,
                                      0.,
                                      R_ellip_squared
                                      )
        else:
            raise print('ERROR in PROFCL_lnlikelihood: min(R^2) = ', 
                        R_ellip_squared_min/3600.**2, 
                        ' arcsec^2 ... cannot be < 0')

