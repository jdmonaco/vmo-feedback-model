# encoding: utf-8
"""
placemap.py -- Compute spatial map information from circle-track simulation

Created by Joe Monaco on 2010-02-04.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
from enthought.traits.api import HasTraits, Trait, Instance, Array, Float, Int

# Package imports
from .vmo import VMOModel
from .session import VMOSession
from .tools import circstat
from .tools.array_container import TraitedArrayContainer
from .tools.radians import get_angle_array

# Place field criteria
MAX_RATE_THRESHOLD = 0.2 # e.g., 20% of peak rate
MIN_SIZE_DEGREES = 10


class CirclePlaceMap(TraitedArrayContainer):
    
    """
    Spatial map representation and statistics for circle-track session data
    
    Model simulation results in the form of a completed VMOModel (or subclass) 
    object or VMOSession object can be used to initialize a CirclePlaceMap. 
    """
    
    # Place map statistics
    COM = Array
    COM_fields = Array
    fields = Array
    peak_rate = Float
    maxima = Array
    num_active = Int
    sparsity = Float
    num_fields = Array
    field_size = Array
    
    # Track coverage
    coverage_maps = Array
    track_coverage_map = Array
    track_coverage = Float
    track_repr_map = Array
    track_repr = Float
    
    # Population matrix information
    units = Int
    bins = Int
    Map = Array
    angle = Array
    data = Trait(None, Instance(VMOSession), Instance(VMOModel))
    
    def __init__(self, data, **kwargs):
        HasTraits.__init__(self)
        self.data = data
        if issubclass(self.data.__class__, VMOModel):
            if self.data.num_trials == 1:
                self.data = VMOSession(self.data)
            else:
                self.data.out('Assuming you want trial 1 out of %d!'%
                    self.data.num_trials, error=True)
                self.data = VMOSession(self.data, trial=1)
        
        # Get the population response map
        self.Map = self.data.get_population_matrix(**kwargs)
        self.units, self.bins = self.Map.shape
        self.angle = get_angle_array(self.bins)
        
        # Compute place field sizes, COMs, distribution, etc
        self._find_maxima()
        self._store_fields()
        self._compute_field_info()
        self._compute_coverage()
        
        # Once data is reduced, release reference to original data source
        self.data = None
        
    def _find_maxima(self):
        """Find and store the map peak rate and per-unit maxima locations
        """
        self.maxima = \
            np.array([[self.angle[np.argmax(R)], np.max(R)] for R in self.Map])
        self.peak_rate = self.maxima[:,1].max()
        
    def _store_fields(self):
        """Find and store boolean field arrays for all place fields
        """
        MIN_BIN_SIZE = int(np.ceil(MIN_SIZE_DEGREES / (360.0/self.bins)))
        for unit in xrange(self.units):
            fields = []
            infield = False
            R = self.Map[unit]
            thresh = 0.2 * R.max()
            for i in xrange(self.bins):
                if R.max() == 0:
                    continue
                if R[i] >= thresh:
                    if infield:
                        field[i] = True
                    else:
                        field = np.zeros(self.bins, '?')
                        infield = field[i] = True
                elif infield:
                    infield = False
                    if field.sum() >= MIN_BIN_SIZE:
                        fields.append(field)
            if infield:
                if len(fields) and R[0] >= thresh:
                    fields[0] = np.logical_or(fields[0], field)
                elif field.sum() >= MIN_BIN_SIZE:
                    fields.append(field)
            if len(fields):
                self.fields[unit] = np.array(fields)
    
    def _compute_field_info(self):
        """Compute per-unit and per-field COMs, sizes, etc
        """
        pf_COM = []
        pf_size = []
        for unit, R in enumerate(self.Map):
            if self.fields[unit] is None:
                continue
            self.COM[unit] = circstat.mean(self.angle, w=R)
            self.num_fields[unit] = len(self.fields[unit])
            for field in self.fields[unit]:
                pf_COM.append(circstat.mean(self.angle[field], w=R[field]))
                pf_size.append(field.sum()*(360.0/self.bins))
        self.COM_fields = np.array(pf_COM)
        self.field_size = np.array(pf_size)
        
    def _compute_coverage(self):
        """Compute trajectory coverage, activity and representation
        """
        # Compute individual unit coverage stats
        self.num_active = 0
        for unit, fields in enumerate(self.fields):
            if fields is None:
                continue
            self.coverage_maps[unit] = (fields.sum(axis=0) != 0)
            self.num_active += 1
        self.sparsity = 1 - (self.num_active / float(self.units))
            
        # Compute overall track coverage by fields
        self.track_repr_map = self.coverage_maps.sum(axis=0)
        self.track_repr = self.track_repr_map.mean()
        self.track_coverage_map = (self.track_repr_map != 0)
        self.track_coverage = self.track_coverage_map.sum() / float(self.bins)
        
    @classmethod
    def get_placemap_list(cls, data, **kwargs):
        """Get list of place map objects from a list of session data objects
        
        Keyword arguments are passed into data.get_population_matrix(...).
        """
        maps = []
        if issubclass(type(data), VMOModel):
            if data.num_trials == 1:
                maps = cls(data, **kwargs)
            else:
                sessions = VMOSession.get_session_list(data)
                maps = [cls(trial, **kwargs) for trial in sessions]
        elif type(data) is list:
            maps = [cls(trial, **kwargs) for trial in data]
        elif type(data) is VMOSession:
            maps = cls(data, **kwargs)
        else:
            raise ValueError, 'bad data argument, should be results data'
        return maps
        
    def _COM_default(self):
        return np.zeros(self.units, 'd')
    
    def _num_fields_default(self):
        return np.zeros(self.units, 'h')
    
    def _fields_default(self):
        return np.array([None]*self.units, dtype=object)
    
    def _coverage_maps_default(self):
        return np.zeros((self.units, self.bins), '?')
