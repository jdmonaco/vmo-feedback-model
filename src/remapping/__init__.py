# encoding: utf-8
"""
Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

CL_LABELS = [   'local', 'distal', 'amb', 'on', 'off'   ]
CL_COLORS = [   (0.1, 1.0, 0.7),    # local -> light green
                (0.9, 0.7, 0.3),    # distal -> light orange
                (0.5, 0.7, 1.0),    # amb -> light blue
                (0.8, 1.0, 0.4),    # on -> yellow
                (0.5, 0.5, 0.5)     # off -> gray
            ]

from .simulate import VMOExperiment
from .mismatch import MismatchAnalysis
from .trends import MismatchTrends
