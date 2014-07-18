#encoding: utf-8
"""
stats.py -- Toolbox functions for computing statistics 

Exported namespace: smooth_pdf

Written by Joe Monaco
Center for Theoretical Neuroscience
Copyright (c) 2007-2008 Columbia Unversity. All Rights Reserved.  

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""


def smooth_pdf(a, sd=None):
    """Get a smoothed pdf of an array of data for visualization
    
    Keyword arguments:
    sd -- S.D. of the gaussian kernel used to perform the smoothing (default is
        1/20 of the data range)
    
    Return 2-row (x, pdf(x)) smoothed probability density estimate.
    """
    from numpy import array, arange, cumsum, trapz, histogram, diff, r_, c_
    from scipy.signal import gaussian, convolve
    if sd is None:
        sd = 0.05 * a.ptp()
    data = a.copy().flatten() # get 1D copy of array data
    nbins = len(data) > 1000 and len(data) or 1000 # num bins >~ O(len(data))
    f, l = histogram(data, bins=nbins, normed=True) # fine pdf
    sd_bins = sd * (float(nbins) / a.ptp()) # convert sd to bin units
    kern_size = int(10*sd_bins) # sufficient convolution kernel size
    g = gaussian(kern_size, sd_bins) # generate smoothing kernel
    c = cumsum(f, dtype='d') # raw fine-grained cdf
    cext = r_[array((0,)*(2*kern_size), 'd'), c, 
        array((c[-1],)*(2*kern_size), 'd')] # wrap data to kill boundary effect
    cs = convolve(cext, g, mode='same') # smooth the extended cdf
    ps = diff(cs) # differentiate smooth cdf to get smooth pdf
    dl = l[1]-l[0] # get bin delta
    l = r_[arange(l[0]-kern_size*dl, l[0], dl), l, 
        arange(l[-1]+dl, l[-1]+kern_size*dl, dl)] # pad index to match bounds
    ps = ps[kern_size:kern_size+len(l)] # crop pdf to same length as index
    ps /= trapz(ps, x=l) # normalize pdf integral to unity
    return c_[l, ps].T # return 2-row concatenation of x and pdf(x)
