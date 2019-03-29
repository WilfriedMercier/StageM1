# from __future__ import division # ensures floating divisions
import sys
import os as os
from os.path import expanduser
home = expanduser('~')
import numpy as np
import math as m
import pandas as pd
import astropy as ap    # fits is in astropy!
import matplotlib.pyplot as plt
from inspect import getsource
from collections import OrderedDict
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
print ("\nWelcome to Python 3, Gary!\n")

# import pyfits
from astropy.io import fits

def seq(min,max,step=1):
    """Sequence (range) of numbers including end-point  
    bug: for max off step"""
    # author: Gary Mamon
    if (max-min) % step == 0:   # add step if range is integer number of steps
        return np.arange(min,max+step,step)
    else:
        return np.arange(min,max,step)

def getvarname(var):
    """Variable name string"""
    # author: Eliott Mamon

    # copy globals dictionary
    vars_dict = globals().copy()

    # loop over variables in dictinary and return variable string related to variable name
    for key in vars_dict:
        if vars_dict[key] is var:
            if key[0] is not '_':
                return key
            lastFoundUnderscoreName = key
    return lastFoundUnderscoreName

def dataframe_from_arrays(*args, **kwargs):
    """Merge vectors into dataframe with automatic headers
        needs getvarname, pandas"""
    # author: Eliott Mamon
    df_args = pd.DataFrame( OrderedDict(tuple( (getvarname(v),v) for v in args )) )
    df_kwargs = pd.DataFrame(kwargs)
    return pd.concat([df_args, df_kwargs], axis=1)

def getsrc(func):
    print(getsource(func))

def arabic_to_roman(number):
    """Convert arabic positive integer to roman numeral"""
    # authors: taken from Web, http://stackoverflow.com/questions/33486183/convert-from-numbers-to-roman-notation by http://stackoverflow.com/users/8747/robi 
    #   adapted by Gary Mamon (inserted "int" for those who use floating division by default)
    conv = [[1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'],
            [ 100, 'C'], [ 90, 'XC'], [ 50, 'L'], [ 40, 'XL'],
            [  10, 'X'], [  9, 'IX'], [  5, 'V'], [  4, 'IV'],
            [   1, 'I']]
    result = ''
    for denom, roman_digit in conv:
        result += roman_digit*int(number/denom)
        number %= denom
    return result

def plot2(xvec,yvec,xlog=0,ylog=0,grid=1,style='classic'):
    """scatter plot with automatic limits and axis labels"""
    # authors: Eliott & Gary Mamon 

    import matplotlib.style
    # square box
    plt.figure(figsize=(6,6))
    # log axes if necessary
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    # point size
    s = 100/(len(xvec)/10)
    smin=0
    smax=360
    if s > smax:
        s=smax
    if s < smin:
        s=smin
    # add minor ticks
    plt.minorticks_on()
    # plot limits
    if xlog:
        xmin,xmax = np.min(xvec[xvec>0]),np.max(xvec)
    else:
        xmin,xmax = np.min(xvec),np.max(xvec)
    if ylog:
        ymin,ymax = np.min(yvec[yvec>0]),np.max(yvec)
    else:
        ymin,ymax = np.min(yvec),np.max(yvec)
    # go a little beyond edges for plot limits
    dx,dy = xmax-xmin,ymax-ymin
    plt.xlim(xmin-0.05*dx,xmax+0.05*dx)
    plt.ylim(ymin-0.05*dy,ymax+0.05*dy)
    # add grid if desired
    if grid:
        plt.grid(which='both')
    # TeX labels
    with plt.rc_context(rc={'text.usetex': True}):
        plt.scatter(xvec, yvec, s=s)
        # print(rc_context)
        # axis labels
        plt.xlabel(add_brackets(getvarname(xvec)))
        plt.ylabel(add_brackets(getvarname(yvec)))
        # choose plot style from argument
        plt.style.use(style)
        # show full plot
        plt.show()

def add_brackets(string):
    """convert string to handle subscripts"""
    # authors: Eliott & Gary Mamon
    i = string.find('_')
    # string starts with '_': remove '_'!
    if i == 0:
        return string[1:]
    # otherwise: surround what is after '_' by curly brackets (or full string if '_' not found)
    else:
        return string[:i+1]+'{'+string[i+1:]+'}'

def flatten(L):
    """flatten list using recursion and duck typing"""
    # source: http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    # author: http://stackoverflow.com/users/1355221/dansalmo 
    # usage: array = np.array(list(flatten(array)))
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
