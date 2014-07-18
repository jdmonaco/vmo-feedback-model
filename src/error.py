# encoding: utf-8
"""
error.py -- Solutions to the error integral across time (Equations 7 and 8)

Created by Joe Monaco on 2010-09-16.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pylab as plt
from scipy.special import erf
from numpy import pi


# Constants

PEAK = 1.0 # default feedback peak
WIDTH = pi/12 # default feedback width
EPSILON = 0.05 # default residual error
SPEED = 13.325 # cm/s
RADIUS = 35.0 # cm
A0 = 0.5 # phase error at t=0


# Phase error with time:
# Equation 7 in the paper, where eta is the phase error variable:
def alpha(t, a0=A0, A=PEAK, sigma=WIDTH, s=SPEED, r=RADIUS):
    return a0*np.exp(-A*r*sigma*np.sqrt(pi/2)*erf(s*t/(np.sqrt(2)*r*sigma))/s)

# Peak (gamma) with cue size (track angle s.d. in radians)
# Equation 8 in the paper, where A is the peak gain variable:
def gamma(sigma=WIDTH, epsilon=EPSILON, s=SPEED, r=RADIUS):
    return np.log(1/epsilon)*s/(np.sqrt(2*pi)*r*sigma)
    

#
# Convenience functions for plotting solutions:
#

N = 128 # number of points to draw for line plots

def plot_alpha_vs_peak(peaks=[0.1,0.25,0.5,1,2,5], tlim=(-5,5), ax=None, norm=True,
    legend=True, **kwargs):
    """Plot alpha (phase error) across time for different peak feedback gains
    
    Additional keyword arguments are passed to error.alpha().
    """
    peaks = np.asarray(peaks)
    if ax is None:
        f = plt.figure()
        ax = plt.axes()
    else:
        f = ax.get_figure()
    t = np.linspace(tlim[0], tlim[1], N)
    
    # Plot time curves
    for peak in peaks:
        err = alpha(t, A=peak, **kwargs)
        if norm:
            err /= err[0]
        plt.plot(t, err, '-', label='A = %.2f'%peak)
    
    # Set axis attributes
    plt.title(r'$\alpha(t)$')
    plt.xlabel('t (s)')
    plt.ylabel('error')
    plt.xlim(tlim)
    if legend:
        plt.legend()
    return f
    
def plot_alpha_vs_width(widths=[pi/24,pi/12,pi/9,pi/6,pi/4,pi/2], tlim=(-5,5), ax=None, 
    norm=True, legend=True, **kwargs):
    """Plot alpha (phase error) across time for different cue sizes (widths)
    
    Additional keyword arguments are passed to error.alpha().
    """
    widths = np.asarray(widths)
    if ax is None:
        f = plt.figure()
        ax = plt.axes()
    else:
        f = ax.get_figure()
    t = np.linspace(tlim[0], tlim[1], N)
    
    # Plot time curves
    for width in widths:
        err = alpha(t, sigma=width, **kwargs)
        if norm:
            err /= err[0]
        plt.plot(t, err, '-', label=r'$\sigma = %.3f$'%width)
    
    # Set axis attributes
    plt.title(r'$\alpha(t)$')
    plt.xlabel('t (s)')
    plt.ylabel('error')
    plt.xlim(tlim)
    if legend:
        plt.legend()
    return f
    
def plot_peak_isoerror(epsilon=[0.5, 0.2, 0.1, 0.05, 0.01, 0.001], wlim=(pi/32, pi/4),
    fmt='-', ax=None, legend=True, degrees=False, **kwargs):
    """Plot iso-error (iso-epsilon) curves of feedback peak across cue size 
    
    Additional keyword arguments are passed to error.gamma().
    """
    epsilon = np.asarray(epsilon)
    if ax is None:
        f = plt.figure()
        ax = plt.axes()
    else:
        f = ax.get_figure()
    sigma = np.linspace(wlim[0], wlim[1], N)
    
    # Plot width curves
    conv = degrees and 180/pi or 1.0
    for eps in epsilon:
        plt.plot(conv*sigma, gamma(sigma=sigma, epsilon=eps, **kwargs), fmt, 
            label=r'$\epsilon=%.3f$'%eps)
    
    # Set axis attributes
    plt.title(r'$A/\sigma,\epsilon$')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('A')
    plt.xlim(xmin=conv*wlim[0], xmax=conv*wlim[1])
    if legend:
        plt.legend()
    return f
    
def plot_peak_isowidth(width=[pi/32, pi/16, pi/8, pi/4, pi/2], elim=(0.001, 0.9),
    fmt='-', ax=None, legend=True, **kwargs):
    """Plot iso-width (iso-sigma) curves of feedback peak across phase error
    
    Additional keyword arguments are passed to error.gamma().
    """
    width = np.asarray(width)
    if ax is None:
        f = plt.figure()
        ax = plt.axes()
    else:
        f = ax.get_figure()
    eps = np.linspace(elim[0], elim[1], N)
    
    # Plot width curves
    for sigma in width:
        plt.plot(eps, gamma(sigma=sigma, epsilon=eps, **kwargs), fmt, 
            label=r'$\sigma=%.3f$'%sigma)
    
    # Set axis attributes
    plt.title(r'$A/\epsilon,\sigma$')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('A')
    plt.xlim(elim)
    if legend:
        plt.legend()
    return f
