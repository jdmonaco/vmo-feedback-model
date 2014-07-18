#encoding: utf-8
"""
circle_track.py -- Circle-track trajectory data loader and container
    
Exported namespace: TrajectoryData, CircleTrackData

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
from numpy import pi, diff, r_
from scipy.interpolate import interp1d

# Package imports
from . import TRACKING_FILE, CENTER_FILE
from ..tools.bash import CPrint
from ..tools.filters import quick_boxcar
from ..tools.radians import xy_to_rad_vec, radian

# Traits imports
from enthought.traits.api import HasTraits, Float, Long, Int, Array, Instance

# Tracking conversion constants
CM_PER_PX = 0.25
TWO_PI = 2*pi


class TrajectoryData(HasTraits):

    """
    Array data for the position tracking of a single session
    """
    
    t = Array
    x = Array
    y = Array
    x0 = Float
    y0 = Float
    angle = Array
    
    def __init__(self, pos_data, **traits):
        """Required first argument is the 4-column data array from a Pos.p file
        """
        HasTraits.__init__(self, **traits)
        if not (pos_data[:,-1] == -99).all():
            valid = (pos_data[:,-1] != -99).nonzero()[0]
            pos_data = pos_data[valid]
        self.t, self.x, self.y, self.angle = pos_data.T
    
    def _x0_default(self):
        return (self.x.max() - self.x.min()) / 2

    def _y0_default(self):
        return (self.y.max() - self.y.min()) / 2

    
class CircleTrackData(HasTraits):
    
    """
    Load trajectory data and compute laps and other metadata
    
    Convenience methods:
    elapsed_time_from_timestamp -- convert timestamp to seconds from start
    get_position_functions -- x and y position as a function of elapsed time
    get_running_speed_function -- linear running speed function
    get_velocity_functions -- component velocity functions
    """
    
    out = Instance(CPrint)
    
    # Data attributes
    rat = Int(72)
    day = Int(1)
    session = Int(4)
    sess_start = Long(9028711691L)
    sess_end = Long(9353991209L)
    duration = Float
    trajectory = Instance(TrajectoryData)
        
    # Lap traits
    N_laps = Int
    laps = Array(desc='fence-post lap partition by time-stamp')
    
    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.out = CPrint(prefix=self.__class__.__name__, color='purple')
        self._set_paths()
        self._load_data()
        self._compute_laps()
    
    # Data loading and initialization methods

    def _set_paths(self):
        """Set full paths to filenames and directories containing session data.
        """
        self.pos_file = TRACKING_FILE
        self.center_file = CENTER_FILE
    
    def _load_data(self):
        """Load trajectory data from position tracking and center files
        """
        # Load arena center coordinates
        center_fd = file(self.center_file, 'r')
        center = [float(v) for v in center_fd.readline().rstrip().split(',')]
        center_fd.close()
        
        # Trajectory duration
        self.duration = self.elapsed_time_from_timestamp(self.sess_end)
    
        # Load position tracking data
        pos_data = np.loadtxt(self.pos_file, comments='%', delimiter=',')
        start_ix = (pos_data[:,0] >= self.sess_start).nonzero()[0][0]
        end_ix = (pos_data[:,0] > self.sess_end).nonzero()[0][0]
        pos_data = pos_data[start_ix:end_ix] # crop single session
        pos_data[:,2] = 2*center[1] - pos_data[:,2] # invert y-axis 
        self.trajectory = TrajectoryData(pos_data, x0=center[0], y0=center[1])
        self.out('Loaded trajectory data!')
        
    def _compute_laps(self):
        """Determine the lap partition of the trajectory
        """
        traj = self.trajectory
        
        # Initialize logic-state and loop variables
        alpha = xy_to_rad_vec(traj.x-traj.x0, traj.y-traj.y0)
        alpha_prev = alpha0 = alpha[0]
        alpha_halfwave = float(radian(alpha0 - pi))
        alpha_hw_flag = False
        cur_lap = 0
        laps_list = [self.sess_start]
        for i in xrange(traj.t.size):
            # First filter out any 0/2pi jumps
            if alpha_prev - alpha[i] > pi:
                continue
            # Binary flag logic for determining when the first lap has passed:
            #   Clockwise -> less-than radian comparisons 
            if not alpha_hw_flag:
                if alpha_prev > alpha_halfwave and alpha[i] <= alpha_halfwave:
                    alpha_hw_flag = True
            elif alpha_prev > alpha0 and alpha[i] <= alpha0:
                cur_lap += 1
                alpha_hw_flag = False
                laps_list.append(long(traj.t[i]))
                self.out('Lap #%d completed at t=%.2fs'%(cur_lap, 
                    self.elapsed_time_from_timestamp(traj.t[i])))
            alpha_prev = alpha[i]
        laps_list.append(self.sess_end)
        self.laps = np.array(laps_list)
        self.N_laps = cur_lap
    
    # Convenience functions for working with session data

    def elapsed_time_from_timestamp(self, timestamp):
        """Return elapsed time since start of recording (s) based on timestamp
        """
        return 10**-6 * (timestamp - self.sess_start)

    def get_position_functions(self):
        """Get position functions x(t) and y(t) for this trajectory
    
        Returns a tuple of callable interp1d objects that can be called with a 
        time t in seconds from the beginning of the recording session. 
        """
        # Get trajectory data
        t, x, y = self._get_filtered_tracking_data()
    
        # Add t=0 data point
        if t[0] != 0.0:
            t = r_[0, t]
            x = r_[x[0], x]
            y = r_[y[0], y]
    
        return \
            interp1d(t, x, 
                copy=False, bounds_error=False, fill_value=x[-1]), \
            interp1d(t, y, 
                copy=False, bounds_error=False, fill_value=y[-1])

    def get_running_speed_function(self):
        """Get running speed function r(t)
    
        Returns a callable interp1d object that can be called with a time t in 
        seconds from the beginning of the recording session. 
        """
        # Get trajectory data
        t, x, y = self._get_filtered_tracking_data()
    
        return \
            interp1d(t[1:], np.sqrt((diff(x)/diff(t))**2 + (diff(y)/diff(t))**2), 
                copy=False, bounds_error=False, fill_value=0.0)

    def get_velocity_functions(self):
        """Get velocity components dx(t) and dy(t) for this trajectory
    
        Returns a tuple of callable interp1d objects that can be called with a 
        time t in seconds from the beginning of the recording session. 
        """
        # Get trajectory data
        t, x, y = self._get_filtered_tracking_data()
    
        return \
            interp1d(t[1:], diff(x)/diff(t), 
                copy=False, bounds_error=False, fill_value=0.0), \
            interp1d(t[1:], diff(y)/diff(t), 
                copy=False, bounds_error=False, fill_value=0.0)
    
    # Helper functions for tracking data
    
    def _get_filtered_tracking_data(self):
        """Helper function that returns boxcar-filtered and calibrated position
        tracking data.
    
        Returns (t, x, y) tuple.
        """
        # Unpack trajectory data
        t, x, y = self.trajectory.t, self.trajectory.x, self.trajectory.y
    
        # Recalibrate time-stamps into seconds-from-beginning, x and y to cm
        t = self.elapsed_time_from_timestamp(t)
        x = x * CM_PER_PX
        y = y * CM_PER_PX
    
        # Boxcar filter the position tracking data
        x = quick_boxcar(x, M=4, centered=True)
        y = quick_boxcar(y, M=4, centered=True)
    
        return t, x, y
