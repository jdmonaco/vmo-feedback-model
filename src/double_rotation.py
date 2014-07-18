# encoding: utf-8
"""
double_rotation.py -- Multi-trial subclass of VMOModel that automatically 
    performs a double rotation experiment.

Exported namespace: VMODoubleRotation

Created by Joe Monaco on 2011-02-24. 

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
from numpy import pi

# Package imports
from .vmo import VMOModel
from .tools.radians import radian

# Traits imports
from enthought.traits.api import Trait, Array


class VMODoubleRotation(VMOModel):
    
    """
    VMOModel subclass that performs double-rotation experiments
    
    Constructor arguments:
    mismatch -- List or array of mismatch angles in radians to iterate through
        (defaults to [0, pi/4, pi/2, 3pi/4, pi])
        
    Example:
    >>> model = VMOModel()
    >>> model.advance_all()
    ...
    >>> sessions = VMOSession.get_session_list(model)
    >>> placemaps = CirclePlaceMap.get_placemap_list(sessions)
    """
    
    # Double rotation traits
    mismatch = Trait(list, Array, user=True)

    def get_alpha_local(self):
        """Subclass override; provided to compute local cue rotations
        """
        # Subtracting mismatch angle -> CCW cue rotation
        return float(radian(self.alpha - self.mismatch[self.trial-1]/2))
    
    def get_alpha_distal(self):
        """Subclass override; provided to compute distal cue rotations
        """
        # Adding mismatch angle -> clockwise cue rotation
        return float(radian(self.alpha + self.mismatch[self.trial-1]/2))
    
    def _mismatch_default(self):
        return [0, pi/4, pi/2, 3*pi/4, pi]
    
    def _num_trials_default(self):
        return self.mismatch.size
