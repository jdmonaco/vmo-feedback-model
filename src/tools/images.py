#encoding: utf-8
"""
images.py -- Toolbox functions for creating and handling image output

Exported namespace: image_blast, array_to_rgba, array_to_image

Written by Joe Monaco
Center for Theoretical Neuroscience
Copyright (c) 2007-2008 Columbia Unversity. All Rights Reserved.  

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

import os
import numpy as np


def image_blast(M, savedir, stem='image', fmt='%s_%03d', rev=False, **kwargs):
    """Save a rank-3 stacked intensity matrix *M* to a set of individual PNG 
    image files in the directory *savedir*.
    
    If *savedir* does not exist it will be created. Set **stem** to specify 
    the filename suffix.
    
    Keyword arguments:
    stem -- file name stem to be used for output images
    fmt -- a unique_path fmt specification (need an %s followed by a %d)
    rev -- indicate use of a reversed fmt specification (%d followed by a %s)
    
    Extra keyword arguments will get passed through to array_to_rgba. See its 
    doc string for details.
    """
    assert M.ndim == 3, 'requires rank-3 array of intensity values'
    d = os.path.realpath(str(savedir))
    if not os.path.exists(d):
        os.makedirs(d)
    stem = os.path.join(d, stem)
    N = M.shape[0]
    first, middle, last = "", "", ""
    for i,m in enumerate(M):
        image_fn = unique_path(stem, fmt=fmt, ext="png", reverse_fmt=rev)
        if i == 0:
            first = image_fn
        elif i == N-1:
            last = image_fn
        array_to_image(m, image_fn, **kwargs)
    if N == 2:
        middle += '\n'
    elif N > 2:
        middle += '\n\t...\n'
    print first, middle, last
    return    

def array_to_rgba(mat, cmap=None, norm=True, cmin=0, cmax=1):
    """Intensity matrix (float64) -> RGBA colormapped matrix (uint8)
    
    Keyword arguments:
    cmap -- a matplotlib.cm colormap object
    norm -- whether the color range is normalized to values in M
    
    If *norm* is set to False:
    cmin -- minimum clipping bound of the color range (default 0)
    cmax -- maximum clipping bound of the color range (default 1)
    """
    if cmap is None:
        from matplotlib import cm
        cmap = cm.hot
    M = mat.copy()
    data_min, data_max = M.min(), M.max()
    if norm:
        cmin, cmax = data_min, data_max
    else:
        if cmin > data_min:
            M[M < cmin] = cmin # clip lower bound
        if cmax < data_max:
            M[M > cmax] = cmax # clip uppder bound
    return cmap((M-cmin)/float(cmax-cmin), bytes=True)

def array_to_image(M, filename, **kwargs):
    """Save matrix, autoscaled, to image file (use PIL fmts)
    
    Keyword arguments are passed to array_to_rgba.
    """
    import sys
    if M.ndim != 2:
        raise ValueError, 'requires rank-2 matrix'
    if sys.platform == "win32":
        import Image
    else:
        from PIL import Image
    img = Image.fromarray(array_to_rgba(M, **kwargs), 'RGBA')
    img.save(filename)
    return

def tiling_dims(N):
    """Square-ish (rows, columns) for tiling N things
    """
    d = np.ceil(np.sqrt(N))
    return int(np.ceil(N / d)), int(d)
