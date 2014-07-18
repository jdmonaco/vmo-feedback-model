# encoding: utf-8
"""
session.py -- Post-hoc container for computing and storing VMOModel results

Exported namespace: VMOSession

Copyright (c) 2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
from glob import glob
from numpy import pi
from scipy.signal import hilbert

# Package imports
from .vmo import VMOModel
from .double_rotation import VMODoubleRotation
from .tools.radians import radian, get_angle_histogram
from .tools.bash import CPrint
from .tools.path import unique_path
from .tools.filters import halfwave, circular_blur
from .tools.array_container import TraitedArrayContainer

# Traits imports
from enthought.traits.api import Trait, Instance, Array, Float, Int, false


class VMOSession(TraitedArrayContainer):
    
    """
    A container for a completed VMOModel simulation object that automatically 
    computes input signal envelopes, median thresholds, a population matrix 
    and place cell spatial information for each trial.
    """
    
    out = Instance(CPrint)
    
    params = Trait(dict)
    center = Array
    dt = Float
    trial = Int(1)
    num_trials = Int
    num_units = Int
    angle = Array
    alpha = Array
    t = Array
    x = Array
    y = Array
    laps = Array
    N_laps = Int
    E = Array(desc='units x track response matrix')
    E_laps = Array(desc='units x track x laps response matrix')
    thresh = Float
    R = Array
    R_laps = Array
    I_rate = Array(desc='spatial information')
    sortix = Array(desc='active unit index sorted by max responses')
    active_units = Array(desc='active unit index')
    is_mismatch = false(desc='whether this is a mismatch session')
    mismatch = Int

    # Firing rate smoothing parameters
    default_blur_width = Float(4.3)
    bins = Int(360)

    # Saving time-series data
    I_cache = Array(desc='saved input time-series')
    E_cache = Array(desc='saved envelope time-series')
    save_I = false(desc='save input time-series')
    save_E = false(desc='save envelope time-series')
    
    # Activity threshold for counting a unit as active
    min_spike_count = Trait(0.05, Float, desc='min. fraction pop. max')
    
    def __init__(self, model, **kwargs):
        super(VMOSession, self).__init__(**kwargs)
        try:
            if not model.done:
                raise ValueError, "model simulation must be completed"
        except AttributeError:
            raise ValueError, "argument must be a VMOModel object"
        
        # Get basic data about this model simulation/session
        self.trait_set( params=model.parameter_dict(), 
                        num_trials=model.num_trials, 
                        num_units=model.N_outputs,
                        dt=model.dt, 
                        center=model.center)
        if self.num_trials == 1:
            pm = model.post_mortem()
        else:
            pm = model.post_mortem().get_trial_data(self.trial)
        if hasattr(model, 'mismatch'):
            self.is_mismatch = True
            self.mismatch = int((180/pi)*model.mismatch[self.trial-1])
        self.trait_set(alpha=pm.alpha, x=pm.x, y=pm.y, t=pm.t)
        
        # Compute envelopes, thresholds, and population responses
        self.out('Computing responses for trial %d of %d...'%(self.trial, 
            self.num_trials))
        self._compute_envelopes(pm.I, model.track)
        self._set_threshold()
        self.compute_circle_responses()
        self.out('Done!')
    
    def _compute_envelope_timeseries(self, I_t):
        """Compute raw time-series signal envelopes of oscillatory drive from 
        synaptic drive matrix (timesteps x units).
        """
        # Compute signal envelope via Hilbert transform
        E = amplitude_envelope(I_t)
        
        # Cache the time-series before track binning if specified
        if self.save_I:
            self.out('Warning: saving input cache')
            self.I_cache = I_t
        if self.save_E:
            self.out('Warning: saving envelopes cache')
            self.E_cache = E
        return E
    
    def _compute_envelopes(self, I_theta, track):
        """Compute the amplitude envelopes for each of the output units
        
        Required arguments:
        I_theta -- synaptic drive time-series matrix for all outputs
        track -- CircleTrackData object containing trajectory data
        
        Session and per-lap envelope matrices are computed.
        """
        # Compute envelope time-series
        E = self._compute_envelope_timeseries(I_theta.T)
        
        # Reduce envelope data to binned track angle histogram
        t, alpha = self.t, self.alpha
        angle = np.linspace(0, 2*pi, self.bins+1)
        self.angle = angle[:-1]
        
        # Get completed laps
        lap_times = track.elapsed_time_from_timestamp(track.laps)
        lap_ix = (lap_times<=self.t[-1]).nonzero()[0].max()
        self.laps = lap_times[:lap_ix]
        self.N_laps = lap_ix - 1 # only including *complete* laps, last lap is always incomplete
        
        # Compute track responses: session- and lap-averages
        self.E = np.zeros((self.num_units, self.bins), 'd')
        self.E_laps = \
            np.zeros((self.num_units, self.bins, self.N_laps), 'd')
        for b in xrange(self.bins):
            ix = np.logical_and(
                alpha >= angle[b], alpha < angle[b+1]).nonzero()[0]
            if len(ix):
                self.E[:,b] = E[:,ix].mean(axis=1)
            for lap in xrange(self.N_laps):
                ix = reduce(np.logical_and, 
                    [alpha >= angle[b], alpha < angle[b+1], 
                    t >= self.laps[lap], t < self.laps[lap+1]]).nonzero()[0]
                if len(ix):
                    self.E_laps[:,b,lap] = E[:,ix].mean(axis=1)
    
    def _set_threshold(self):
        """Compute median peak inputs as an activity threshold
        """
        self.thresh = np.median(self.E.max(axis=1))
    
    def compute_circle_responses(self):
        """Top-level function to recompute the population matrix, information
        rates and active place units.
        """
        self._compute_population_matrix()
        self._compute_spatial_information()
        self._set_active_units()

    def _compute_population_matrix(self):
        """Compute radial place field ratemaps for each output unit
        """
        self.R = halfwave(self.E - self.thresh)
        self.R_laps = halfwave(self.E_laps - self.thresh)
    
    def _compute_spatial_information(self):
        """Compute overall spatial information for each output unit
        
        Calculates bits/spike as (Skaggs et al 1993):
        I(R|X) = (1/F) * Sum_i[p(x_i)*f(x_i)*log_2(f(x_i)/F)]
        """
        self.I_rate = np.empty(self.num_units, 'd')
        occ = get_angle_histogram(
            self.x-self.center[0], self.y-self.center[1], self.bins)
        occ *= self.dt # convert occupancy to seconds
        p = occ/occ.sum()
        for i in xrange(self.num_units):
            f = self.R[i]
            F = halfwave(self.E[i]-self.thresh).mean()
            I = p*f*np.log2(f/F)/F
            I[np.isnan(I)] = 0.0 # handle zero-rate bins
            self.I_rate[i] = I.sum()
    
    def _set_active_units(self):
        """Apply minimal firing rate threshold to determine which active units 
        are active.
        """
        self.active_units = (
            self.R.max(axis=1) >= self.min_spike_count*self.R.max()
            ).nonzero()[0]
        self.sortix = self.active_units[
            np.argsort(np.argmax(self.R[self.active_units], axis=1))]
                        
    def get_spatial_information(self, unit=None):
        """Get overall spatial information for the population or a single unit
        """
        return np.squeeze(self.I_rate[unit])
        
    def get_population_matrix(self, bins=None, norm=False, clusters=None,
        smoothing=True, blur_width=None, inplace=False):
        """Retrieve the population response matrix for this session simulation
        
        Keyword arguments:
        bins -- recompute responses for a different number of bins (deprecated)
        norm -- whether to integral normalize each unit's response
        clusters -- optional index array for row-sorting the response matrix;
            if not specified, a peak-location sort of the place-active subset 
            of the population is used by default
        smoothing -- whether to do circular gaussian blur on ratemaps
        blur_width -- width of gaussian window to use for smoothing; a value of 
            None defaults to default_blur_width
        
        Returns (units, bins) matrix of population spatial responses.
        """
        self.compute_circle_responses()
        if clusters is None:
            clusts = self._get_active_units()
        elif type(clusters) in (np.ndarray, list):
            clusts = np.asarray(clusters)
        if inplace:
            R = self.R[clusts] 
        else:
            R = self.R[clusts].copy()
        if smoothing:
            if blur_width is None:
                blur_width = self.default_blur_width
            for Runit in R:
                Runit[:] = circular_blur(Runit, blur_width)
        if norm:
            Rsum = np.trapz(R, axis=1).reshape(R.shape[0], 1)
            Rsum[Rsum==0.0] = 1
            R /= Rsum
        return R
    
    def get_population_lap_matrix(self, clusters=None, smoothing=True, 
        blur_width=None, inplace=False, **kwargs):
        """Construct concatentation of per-lap population response matrices
    
        Keyword arguments:
        clusters -- optional index array for row-sorting the response matrix;
            if not specified, a peak-location sort of the place-active subset 
            of the population is used by default
        smoothing -- whether to do circular gaussian blur on ratemaps
        blur_width -- width of gaussian window to use for smoothing; a value of 
            None defaults to default_blur_width
    
        Returns (N_clusts, bins, N_laps) response matrix.
        """
        self.compute_circle_responses()
        if clusters is None:
            clusts = self._get_active_units()
        elif type(clusters) in (np.ndarray, list):
            clusts = np.asarray(clusters)
        if inplace:
            R = self.R_laps[clusts] 
        else:
            R = self.R_laps[clusts].copy()
        if smoothing:
            if blur_width is None:
                blur_width = self.default_blur_width
            for Runit in R:
                for Rlap in Runit.T:
                    Rlap[:] = circular_blur(Rlap, blur_width)
        return R
            
    def recover_cues(self):
        """Simulate a dummy model with identical cue configuration as was used
        to create this session data. A post-mortem object is returned that can
        be plotted using (e.g.) rat.oi_funcs.plot_external_cues.
        """
        pdict = dict(
            N_theta = 1,
            N_outputs = 1,
            monitoring = False,
            N_cues_local = self.params['N_cues_local'],
            N_cues_distal = self.params['N_cues_distal'],
            local_cue_std = self.params['local_cue_std'],
            distal_cue_std = self.params['distal_cue_std'],
            refresh_fixed_points = False
            )
        if self.is_mismatch:
            pdict.update(mismatch=[(np.pi/180)*self.mismatch])
            klass = VMODoubleRotation
        else:
            klass = VMOModel
        model = klass(**pdict)
        model.advance()
        return model.post_mortem()

    @classmethod
    def get_session_list(cls, model, **kwargs):
        """Convenience method to get a list of VMOSession objects for the
        trials in a model object.
        """
        res = []
        if model.num_trials == 1:
            res = VMOSession(model, **kwargs)
        else:
            res = []
            for trial in xrange(1, model.num_trials+1):
                res.append(VMOSession(model, trial=trial, **kwargs))
        return res
    
    @classmethod
    def save_session_list(cls, session_list, save_dir):
        """Save all sessions in an experiment to the specified directory
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for session in session_list:
            if session.is_mismatch:
                if session.mismatch == 0:
                    fn = 'STD.tar.gz'
                else:
                    fn = 'MIS_%03d.tar.gz'%session.mismatch
                session.tofile(os.path.join(save_dir, fn))
            else:
                fn = unique_path(os.path.join(save_dir, 'session_'), 
                    ext='tar.gz')
                session.tofile(fn)
                
    @classmethod
    def load_session_list(cls, load_dir):
        """Load all sessions from files found in the specified load directory
        """
        files = glob(os.path.join(load_dir, '*.tar.gz'))
        files.sort()
        return [cls.fromfile(fn) for fn in files]

    def _get_active_units(self):
        """Get the list of active place units
        """
        return self.sortix
    
    def _out_default(self):
        return CPrint(prefix=self.__class__.__name__)
        