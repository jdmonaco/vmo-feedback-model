# encoding: utf-8
"""
compare.py -- Functions for comparing double-rotation session responses

Exoported namespace: mismatch_response_tally, mismatch_rotation, 
    cluster_mismatch_rotation, population_spatial_correlation, 
    correlation_matrix, correlation_diagonals, common_units

Created by Joe Monaco on 2010-02-11.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
from scipy.stats import pearsonr

# Package imports 
from .trajectory import CircleTrackData
from .session import VMOSession
from .tools.radians import get_angle_array, circle_diff
from .tools.filters import circular_blur


# Functions to compare responses between two sessions

def mismatch_response_tally(STD, MIS, mismatch, angle_tol=0.5, min_corr=0.4,
    **kwargs):
    """Categorize and tally spatial response changes in mismatch session
    
    Both angle tolerance and minimum correlation criteria must be met for a 
    response change to classify as coherent.
    
    Required arguments:
    STD, MIS -- VMOSession objects for STD and MIS session pair
    mismatch -- total mismatch angle for the cue rotation (in degrees)
    
    Keyword arguments:
    angle_tol -- proportional tolerance for matching a cue rotation
    min_corr -- minimum maximal correlation value across rotation
    
    Returns dictionary of tallies: local, distal, ambiguous, on, off.
    """
    if not hasattr(STD, '_get_active_units') or \
        type(MIS) is not type(STD):
        raise ValueError, 'invalid or non-matching session data inputs'
        
    # Count coherent vs non-coherent response changes
    tally = dict(local=0, distal=0, amb=0)
    rotations = mismatch_rotation(STD, MIS, degrees=False, **kwargs)
    mismatch *= (np.pi/180) / 2 # cue rotation in radians
    for i,rotcorr in enumerate(rotations.T):
        rot, corr = rotcorr
        if corr < min_corr:
            tally['amb'] += 1
        else:
            angle_dev = min(
                abs(circle_diff(rot, mismatch)), 
                abs(circle_diff(rot, -mismatch)))
            if angle_dev <= angle_tol * mismatch:
                if rot < np.pi:
                    tally['local'] += 1
                else:
                    tally['distal'] += 1
            else:
                tally['amb'] += 1
    
    # Count remapping (on/off) response changes
    STDclusts = set(STD._get_active_units())
    MISclusts = set(MIS._get_active_units())
    tally['off'] = len(STDclusts.difference(MISclusts))
    tally['on'] = len(MISclusts.difference(STDclusts))
    
    return tally

def mismatch_rotation(STD, MIS, degrees=True, **kwargs):
    """Computes rotation angle and peak correlation for active clusters between
    two sessions
    
    Returns two-row per-cluster array: angle, correlation.
    """
    Rstd, Rmis = comparison_matrices(STD, MIS, **kwargs)
    units, bins = Rstd.shape
    return np.array([cluster_mismatch_rotation(Rstd[c], Rmis[c], degrees=degrees)
        for c in xrange(units)]).T
        
def cluster_mismatch_rotation(Rstd, Rmis, degrees=True):
    """Find the rotation angle for a single cluster ratemap
    
    Ratemap inputs must be one-dimensional arrays representing the whole track.
    
    Keyword arguments:
    degrees -- whether to specify angle in degrees or radians
    
    Returns (angle, correlation) tuple.
    """
    bins = Rstd.shape[0]
    angle = get_angle_array(bins, degrees=degrees)
    corr = np.empty(bins, 'd')
    for offset in xrange(bins):
        MISrot = np.concatenate((Rmis[offset:], Rmis[:offset]))
        corr[offset] = pearsonr(Rstd, MISrot)[0]
    return angle[np.argmax(corr)], corr.max()

def population_spatial_correlation(STD, MIS, **kwargs):
    """Return whole population spatial correlation between two matrices
    """
    A, B = comparison_matrices(STD, MIS, **kwargs)
    return pearsonr(A.flatten(), B.flatten())[0]


# Functions to compute and operate on population correlation matrices

def correlation_matrix(SD, cross=None, **kwargs):
    """Compute a spatial correlation matrix of population-rate vectors
    
    Returns (bins, bins) correlation matrix.
    """
    # Validate arguments and compute population response matrices
    R, R_ = comparison_matrices(SD, cross, **kwargs)
    
    # Compute the correlation matrix
    N_units, bins = R.shape
    C = np.empty((bins, bins), 'd')
    for i in xrange(bins):
        R_i = R[:,i] / np.sqrt(np.dot(R[:,i], R[:,i]))
        for j in xrange(bins):
            R_j = R_[:,j] / np.sqrt(np.dot(R_[:,j], R_[:,j]))
            C[i,j] = np.dot(R_i, R_j)
            
    # Fix any NaN's resulting from silent population responses (rare!)
    C[np.isnan(C)] = 0.0
    return C

def correlation_diagonals(C, use_median=True, centered=False, blur=None):
    """Return the angle bins and diagonals of a correlation matrix
    
    Keyword arguments:
    use_median -- whether to use the median diagonal correlation to collapse 
        the diagonals; if use_median=False, the average is used
    centered -- whether to center the diagonals on [-180, 180]
    blur -- if not None, specifies width in degrees of gaussian blur to be
        applied to diagonal array
    """
    bins = C.shape[0]
    if C.shape != (bins, bins):
        raise ValueError, 'correlation matrix must be square'
    f = use_median and np.median or np.mean
    D = np.empty(bins+1, 'd')
    d = np.empty(bins, 'd')
    offset = 0
    if centered:
        offset = int(bins/2)
        
    # Loop through and collapse correlation diagonals
    for b0 in xrange(bins):
        for b1 in xrange(bins):
            d[b1] = C[b1, np.fmod(offset+b0+b1, bins)]
        D[b0] = f(d)
    if blur is not None:
        D[:bins] = circular_blur(D[:bins], blur)
        
    # Wrap the last point around to the beginning
    D[-1] = D[0]
    last = centered and 180 or 360
    a = np.r_[get_angle_array(bins, degrees=True, zero_center=centered), last]
    return np.array([a, D])


# Functions on lists of VMOSession objects (e.g., full five-session double-
# rotation experiments)

def comparison_matrices(SD, cross, **kwargs):
    """Validate multiple types of arguments for use as comparanda

    SD and cross must be the same type of object (unless cross is None for an
    autocomparison): VMOSession instances, or previously computed population 
    matrices.

    For VMOSession objects, a clusters list is automatically created to be 
    passed in for the get_population_matrix call. This may be overriden by 
    passing in your own clusters list as a keyword argument. Additional keyword 
    arguments are passed to get_population_matrix.

    Returns two valid (units, bins) population matrix references.
    """
    if type(SD) is np.ndarray:
        if SD.ndim != 2:
            raise ValueError, 'expecting 2-dim population matrix'
        R = R_ = SD
        if type(cross) is np.ndarray:
            if cross.shape == SD.shape:
                R_ = cross
            else:
                raise ValueError, 'non-matching population matrices'
    elif hasattr(SD, '_get_active_units'):
        kwargs['norm'] = False
        if type(cross) is type(SD):
            if 'clusters' not in kwargs:
                kwargs['clusters'] = common_units(SD, cross)
        elif cross is not None:
            raise ValueError, 'non-matching session object types'
        R = R_ = SD.get_population_matrix(**kwargs)
        if cross is not None:
            R_ = cross.get_population_matrix(**kwargs)
            if R_.shape != R.shape:
                raise ValueError, 'population matrix size mismatch'
    return R, R_

def common_units(*SD_list):
    """Get a list of the active units that are common to a set of sessions
    """
    # Allow a single python list to be passed in
    if len(SD_list) == 1 and type(SD_list[0]) is list:
        SD_list = SD_list[0]
    
    # Get a list of sets of clusters for each data object
    clust_list = []
    for SD in SD_list:
        clust_list.append(set(SD._get_active_units()))

    # Find the common clusters
    common = clust_list[0]
    for i in xrange(1, len(SD_list)):
        common = common.intersection(clust_list[i])
        
    return list(common)
