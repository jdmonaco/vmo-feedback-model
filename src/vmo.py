# encoding: utf-8
"""
vmo.py -- An oscillatory interference model for place-cell activity that 
    provides phase-code feedback from interactions with external cues.

Exported namespace: VMOModel

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np
from numpy import pi
from scipy.interpolate import interp1d

# Package imports
from .core.model import AbstractModel
from .trajectory import CircleTrackData, CM_PER_PX
from .error import gamma, EPSILON
from .tools.radians import radian, xy_to_rad, circle_diff, circle_diff_vec

# Traits imports
from enthought.traits.api import (Trait, Float, CFloat, Int, Array, 
    Tuple, Instance, Callable, Range, true, false)

# Constants 
TWO_PI = 2*pi


class VMOModel(AbstractModel):
    
    """
    Firing-rate model implementation of an oscillatory interference model
    """
    
    label = "Oscillator Model"
    
    # Oscillator traits
    N_theta = Int(1000, user=True)
    theta = Array(desc='oscillator phase')
    theta_t0 = Array(desc='initial phase array for each trial')
    r_theta = Array(desc='oscillator firing rate')
    phi = Array(desc='velocity direction preference')
    VM_matrix = Array
    omega = Float(7, user=True)
    scale = Array(desc='inverse gain of phase modulation')
    lambda_range = Tuple((100, 200), user=True)
    init_random = true(user=True)
    
    # Cue-dependent feedback traits
    dtheta_t = Array(desc='current relative phase distribution')
    dtheta_local = Array(desc='phase reference for local cues')
    dtheta_distal = Array(desc='phase reference for distal cues')
    cue_t_local = Array(desc='local cue crossing times')
    cue_t_distal = Array(desc='distal cue crossing times')
    refresh_fixed_points = true(user=True)
    first_lap = true
    _alpha0 = Float(-1)
    _alpha_prev = Float
    _alpha_halfwave = Float
    _alpha_hw_flag = false
    
    # External cue traits
    epsilon = Range(low=0.0, high=1.0, value=EPSILON, user=True)
    N_cues = Int(3, user=True, desc='default number of cues')
    N_cues_local = Int(user=True)
    N_cues_distal = Int(user=True)    
    cue_spacing_local = Float
    cue_spacing_distal = Float
    cue_std = Float(pi/20, user=True, desc='default cue size')
    local_cue_std = Float(user=True)
    distal_cue_std = Float(user=True)
    cue_offset = Float(user=True)
    local_offset = Float(user=True)
    distal_offset = Float(user=True)
    active_local_cue = Int(track=True)
    active_distal_cue = Int(track=True)
    local_cue_intensity = CFloat
    distal_cue_intensity = CFloat
    C_local = Float(track=True)
    C_distal = Float(track=True)
    F_local = Callable(desc='cue intensity function')
    F_distal = Callable(desc='cue intensity function')
    gamma_local = Float(user=True)
    gamma_distal = Float(user=True)
    
    # Trajectory traits
    track = Instance(CircleTrackData)
    x_t = Callable
    y_t = Callable
    dx_t = Callable
    dy_t = Callable
    center = Array
    
    # Model traits
    N_outputs = Int(500, user=True)
    vel = Array(track=True)
    W = Array
    C_W = Float(0.05, user=True)
    I = Array(track=True)
    x = CFloat(track=True)
    y = CFloat(track=True)
    alpha = Float(track=True)
    alpha_local = Float
    alpha_distal = Float
    dt = 0.010
    
    def __init__(self, **traits):
        AbstractModel.__init__(self, **traits)
        self.track = CircleTrackData()
        self.x_t, self.y_t = self.track.get_position_functions()
        self.dx_t, self.dy_t = self.track.get_velocity_functions()
        self.T = self.track.duration
        traj = self.track.trajectory
        self.center = np.array([traj.x0, traj.y0]) * CM_PER_PX
    
    # Model subclass override methods and phase equation
    
    def trial_setup(self):
        # Initialize first-lap state to store phase fixed points
        if self.refresh_fixed_points:
            self.first_lap = True
            self._alpha0 = -1
            self._alpha_hw_flag = False
            self._alpha_prev = TWO_PI
            self.refresh_fixed_points = False # only first trial by default
        else:
            self.first_lap = False
        
        # Set cue offsets
        self.local_offset = self.cue_offset
        self.distal_offset = self.cue_offset + self.cue_spacing_distal / 2
        
        # Refresh cue intensity functions
        self.F_local = self._F_local_default()
        self.F_distal = self._F_distal_default()
        
        # Initialize phase array and new integrator
        self.theta[:] = self.theta_t0[self.trial-1]
        
    def run_timestep(self):
        # Get tracking and velocity data
        self.trait_set(x=self.x_t(self.t), y=self.y_t(self.t))
        self.vel[:] = self.dx_t(self.t), self.dy_t(self.t)
        
        # Get track position (alpha) and currently active cues
        self.trait_set( alpha=self.get_alpha(),
                        alpha_local=self.get_alpha_local(),
                        alpha_distal=self.get_alpha_distal(),
                        active_local_cue=self.get_active_local_cue(),
                        active_distal_cue=self.get_active_distal_cue()  )
        
        # Compute reference phase difference vector for current timestep
        self.dtheta_t[:] = circle_diff_vec(self.theta, TWO_PI*self.omega*self.t)
        
        # Store phase vector on the first lap while turning off feedback
        if self.first_lap:
            self.store_delta_theta()
            self.local_cue_intensity = self.distal_cue_intensity = 0.0
        else: 
            # Compute the active cue intensity values
            self.local_cue_intensity = \
                self.F_local(
                    circle_diff(self.active_local_cue*self.cue_spacing_local+ 
                        self.local_offset, self.alpha_local))
            self.distal_cue_intensity = \
                self.F_distal(
                    circle_diff(self.active_distal_cue*self.cue_spacing_distal+
                        self.distal_offset, self.alpha_distal))
            
        # Calculate external input gains
        self.set_cue_coefficients()
        
        # Evolve the theta phase equation
        self.theta += (self.dtheta_dt(self.theta) * self.dt)[0]
        
        # Calculate theta firing rates and output synaptic drive
        self.r_theta[:] = np.cos(self.theta)
        self.I[:] = np.dot(self.W, self.r_theta)
        
    def dtheta_dt(self, theta):
        """Implementation of the governing theta-phase equation
        """
        return TWO_PI * (
            # Carrier wave:
            self.omega + 
            # Path integration via velocity modulation:
            np.dot(self.vel, self.VM_matrix) + 
            # Balanced cue-dependent feedback:
            self.C_local * 
                circle_diff_vec(self.dtheta_local[self.active_local_cue], 
                    self.dtheta_t) + 
            self.C_distal * 
                circle_diff_vec(self.dtheta_distal[self.active_distal_cue], 
                    self.dtheta_t)
            )
    
    # Helper functions
    
    def set_cue_coefficients(self):
        """Set the local and distal cue coefficients with gain modulation
        """
        self.C_local = self.gamma_local * self.local_cue_intensity
        self.C_distal = self.gamma_distal * self.distal_cue_intensity

    def store_delta_theta(self):
        """Store relative phase vectors for each external cue around the track
        
        This per-timestep method keeps track of the first clock-wise lap of the
        circle-track trajectory, setting *first_lap* to False once the lap is
        completed. At that point, the modulation traits *local_cue_intensity* 
        and *distal_cue_intensity* are activated, effectively switching on the
        phase feedback mechanism.
        
        During the first lap, initial clock-wise cue crossings trigger the 
        storage of the current reference phase-difference vector in association 
        with the corresponding external cue.
        """
        # Set initial alpha value for trajectory on first timestep
        if self._alpha0 == -1:
            self._alpha0 = self.alpha
            self._alpha_halfwave = float(radian(self._alpha0 - pi))
        else:
            # Binary flag logic for determining when the first lap has passed:
            #   Clockwise -> less-than radian comparisons 
            if not self._alpha_hw_flag:
                if self._alpha_prev > self._alpha_halfwave and \
                    self.alpha <= self._alpha_halfwave:
                    self._alpha_hw_flag = True
            elif self._alpha_prev > self._alpha0 and \
                self.alpha <= self._alpha0:
                self.first_lap = False
                self.out('First lap completed at t=%.2fs'%self.t)
        
            # Store phase reference vectors for local cues
            for i in xrange(self.N_cues_local):
                cue_alpha = \
                    np.fmod(self.local_offset + i*self.cue_spacing_local, 
                        TWO_PI)
                if self._alpha_prev > cue_alpha and \
                    (self.alpha <= cue_alpha or self.alpha-self._alpha_prev > pi):                
                    if self.dtheta_local[i,0] != -99:
                        break
                    self.dtheta_local[i] = \
                        circle_diff_vec(self.theta, TWO_PI*self.omega*self.t)
                    self.cue_t_local[i] = self.t
                    self.out('Stored local cue #%d at t=%.2fs'%(i+1, self.t))
                
            # Store phase reference vectors for distal cues
            for i in xrange(self.N_cues_distal):
                cue_alpha = \
                    np.fmod(self.distal_offset + i*self.cue_spacing_distal,
                        TWO_PI)
                if self._alpha_prev > cue_alpha and \
                    (self.alpha <= cue_alpha or self.alpha-self._alpha_prev > pi):                
                    if self.dtheta_distal[i,0] != -99:
                        break
                    self.dtheta_distal[i] = \
                        circle_diff_vec(self.theta, TWO_PI*self.omega*self.t)
                    self.cue_t_distal[i] = self.t
                    self.out('Stored distal cue #%d at t=%.2fs'%(i+1, self.t))                
        
        # Store current track alpha for next comparison
        self._alpha_prev = self.alpha
        
    def get_alpha(self):
        """Compute and return angle of current location w.r.t. center point
        """
        return xy_to_rad(self.x-self.center[0], self.y-self.center[1])
    
    def get_alpha_local(self):
        """Subclass override; provided to compute local cue rotations
        """
        return self.alpha
    
    def get_alpha_distal(self):
        """Subclass override; provided to compute distal cue rotations
        """
        return self.alpha
    
    def get_active_local_cue(self):
        """Return index for the nearest current local cue
        """
        return \
            int(np.fmod(
                np.round((self.alpha_local - self.local_offset) / 
                    self.cue_spacing_local), 
            self.N_cues_local))
    
    def get_active_distal_cue(self):
        """Return index for the nearest current distal cue
        """
        return \
            int(np.fmod(
                np.round((self.alpha_distal - self.distal_offset) / 
                    self.cue_spacing_distal), 
            self.N_cues_distal))
                    
    # Traits default methods
    
    def _W_default(self):
        fan_in = int(self.C_W*self.N_theta)
        W = np.zeros((self.N_outputs, self.N_theta), 'd')
        W[0,:fan_in] = 1
        for i in xrange(self.N_outputs):
            W[i] = np.random.permutation(W[0])
        return W

    def _theta_default(self):
        return np.zeros(self.N_theta)
    
    def _theta_t0_default(self):
        if self.init_random:
            return TWO_PI*np.random.rand(self.num_trials, self.N_theta)
        else:
            return np.zeros((self.num_trials, self.N_theta)) + \
                TWO_PI*np.random.rand(self.N_theta)
    
    def _r_theta_default(self):
        return np.zeros(self.N_theta)
    
    def _vel_default(self):
        return np.zeros((1, 2), 'd')

    def _phi_default(self):
        return TWO_PI*np.random.rand(self.N_theta)
    
    def _scale_default(self):
        min_lambda, max_lambda = min(self.lambda_range), max(self.lambda_range)
        return min_lambda + \
            (max_lambda-min_lambda) * np.random.rand(self.N_theta)
    
    def _VM_matrix_default(self):
        return np.array([np.cos(self.phi), np.sin(self.phi)]) / self.scale
    
    def _N_cues_local_default(self):
        return self.N_cues
    
    def _N_cues_distal_default(self):
        return self.N_cues
    
    def _cue_spacing_local_default(self):
        return TWO_PI/self.N_cues_local

    def _cue_spacing_distal_default(self):
        return TWO_PI/self.N_cues_distal

    def _local_cue_std_default(self):
        return self.cue_std

    def _distal_cue_std_default(self):
        return self.cue_std
    
    def _gamma_local_default(self):
        return gamma(self.local_cue_std, self.epsilon) / TWO_PI

    def _gamma_distal_default(self):
        return gamma(self.distal_cue_std, self.epsilon) / TWO_PI

    def _F_local_default(self):
        return self._get_F_function(self.local_cue_std)
        
    def _F_distal_default(self):
        return self._get_F_function(self.distal_cue_std)
        
    def _cue_t_local_default(self):
        return np.zeros(self.N_cues_local, 'd')
        
    def _cue_t_distal_default(self):
        return np.zeros(self.N_cues_distal, 'd')
        
    def _get_F_function(self, cue_std):
        """Generate the interpolation function object representing the spatial 
        intensity function of external cue around the track.
        """
        alpha_ix = np.linspace(-pi, pi, 256)
        cue_intensity = np.exp((1/cue_std**2) * (np.cos(alpha_ix)-1))
        return interp1d(alpha_ix, cue_intensity)
    
    def _dtheta_t_default(self):
        return np.empty(self.N_theta, 'd')
    
    def _dtheta_local_default(self):
        """Phase difference from reference wave for local cues
        """
        # Initialized to -99 as a storage flag
        return np.zeros((self.N_cues_local, self.N_theta), 'd') - 99
    
    def _dtheta_distal_default(self):
        """Phase difference from reference wave for distal cues
        """
        # Initialized to -99 as a storage flag
        return np.zeros((self.N_cues_distal, self.N_theta), 'd') - 99
    
    def _I_default(self):
        return np.zeros(self.N_outputs, 'd')
        
    # Parameter cloning method 
    
    def clone_dict(self):
        """Gets a call dictionary that will exactly clone this model
        """
        if not self.done:
            self.out('Complete simulation before cloning!', error=True)
            return
        d = self.parameter_dict()
        d.update(done=False)
        theta_params = dict(
            W = self.W,
            phi = self.phi,
            scale = self.scale,
            theta_t0 = self.theta_t0,
            dtheta_local = self.dtheta_local,
            dtheta_distal = self.dtheta_distal,
            refresh_fixed_points = False
            )
        d.update(theta_params)
        return d
