# encoding: utf-8
"""
Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

import os.path as _path

CENTER_FILE_NAME = "center.xy"
TRACKING_FILE_NAME = "circle_track.csv"

TRAJ_DIR = _path.abspath(_path.split(__file__)[0])
TRACKING_FILE = _path.join(TRAJ_DIR, TRACKING_FILE_NAME)
CENTER_FILE = _path.join(TRAJ_DIR, CENTER_FILE_NAME)

if _path.exists(TRACKING_FILE) and _path.exists(CENTER_FILE):
    print 'Found circle-track data at:\n\t%s\n\t%s'%(TRACKING_FILE, CENTER_FILE)
else:
    raise IOError, \
        'Failed to find trajectory data (\'%s\', \'%s\')!'%(TRACKING_FILE_NAME, CENTER_FILE_NAME)

from circle_track import CircleTrackData, CM_PER_PX
