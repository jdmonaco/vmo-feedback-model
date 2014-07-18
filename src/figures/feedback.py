#encoding: utf-8
"""
feedback.py -- Feedback figure showing phase-code retrieval due to single cues

Created by Joe Monaco on 2010-10-12.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pylab as plt
from enthought.traits.api import Int, Float, Array

# Package imports
from ..core.model import ModelPostMortem
from ..core.analysis import BaseAnalysis
from ..vmo import VMOModel
from ..error import alpha, gamma
from ..tools.images import array_to_image
from ..tools.stats import smooth_pdf
from ..tools.radians import circle_diff_vec


class VMOTrackFeedback(VMOModel):
    
    """
    Add tracking of p*M path integration signal across oscillators
    """
    
    label = "Tracking Model"
    
    omega = 0
    dt = 0.002

    local_phase_error = Array(track=True)
    distal_phase_error = Array(track=True)
    local_fdbk = Array(track=True)
    distal_fdbk = Array(track=True)
    theta = Array(track=True)
    
    def run_timestep(self):
        super(VMOTrackFeedback, self).run_timestep()
        self.local_phase_error[:] = \
            circle_diff_vec(
                self.dtheta_local[self.active_local_cue], self.dtheta_t)
        self.distal_phase_error[:] = \
            circle_diff_vec(
                self.dtheta_distal[self.active_distal_cue], self.dtheta_t)
        self.local_fdbk[:] = self.C_local * self.local_phase_error
        self.distal_fdbk[:] = self.C_distal * self.distal_phase_error

    def _local_phase_error_default(self):
        return np.zeros(self.N_theta, 'd')

    def _distal_phase_error_default(self):
        return np.zeros(self.N_theta, 'd')

    def _local_fdbk_default(self):
        return np.zeros(self.N_theta, 'd')
    
    def _distal_fdbk_default(self):
        return np.zeros(self.N_theta, 'd')
    
    def _theta_default(self):
        return np.zeros(self.N_theta, 'd')
    

class VMOToyModel(VMOTrackFeedback):
    
    label = "Ideal Traversal"
    
    N_theta = 1
    N_outputs = 1
    N_cues_local = 1
    N_cues_distal = 1
    N_errors = Int(5)
    max_error = Float(pi/2)
    cue_t0 = 16.776
    local_cue_std = pi/15
    cue_std_t = Float
    num_trials = 5
    theta_t0_test = Float(-pi/4)
    vbar = Float(10.0) # constant velocity
    
    def __init__(self, **traits):
        super(VMOToyModel, self).__init__(**traits)
        self.dx_t = lambda t: self.vbar
        self.dy_t = lambda t: 0.0
        self.gamma_distal = 0.0
        self.T = 25
    
    def trial_setup(self):
        super(VMOToyModel, self).trial_setup()
        if self.trial <= 2:
            self.gamma_local = 0.0
        else:
            self.gamma_local = self._gamma_local_default()
            # self.dtheta_local[0,0] = 1.84
    
    def _theta_t0_default(self):
        t0list = [[self.theta_t0_test]]*2
        t0tests = self.theta_t0_test + self.max_error*np.linspace(-1, 1, self.N_errors)
        t0list += [[error] for error in t0tests]
        return np.array(t0list, 'd')
    
    def _num_trials_default(self):
        return self.N_errors + 2
    
    def _scale_default(self):
        return np.array([200], 'd')
    
    def _phi_default(self):
        return np.array([pi/3])
    
    def _cue_std_t_default(self):
        """Convert from cue width in space to cue width in time
        """
        return self.local_cue_std*(9/pi) # radians * (180/pi) / 10
    
    def _F_local_default(self):
        """Get cue profile in time, not space
        """
        return lambda alpha: np.exp(-(self.t-self.cue_t0)**2/(2*self.cue_std_t**2))
    
    
class FeedbackFigure(BaseAnalysis):
    
    """
    Run simple feedback simulations across one cue for comparison with analytic
    solution. Tracking for *local_phase_error* must be set to True in 
    VMOTrackFeedback.
    """
    
    label = "feedback"
    
    def collect_data(self, cue_sizes=[10, 30], cue_offset=9*pi/8, **kwargs):
        """Run a series of feedback simulations with cues of varying sizes.

        Additional keyword arguments are passed on to VMOTrackFeedback.
        """
        # Save analysis parameters
        self.results['cue_sizes'] = cue_sizes = (pi/180)*np.array(cue_sizes)
        self.results['N_cues'] = N_cues = len(cue_sizes)
        
        # Set up model parameters
        pdict = dict(   N_outputs=1, 
                        N_theta=16,
                        N_cues_local=1, 
                        N_cues_distal=1, 
                        cue_offset=cue_offset,
                        C_W=0.05,
                        gamma_local=0,
                        gamma_distal=0, 
                        num_trials=2*N_cues+2,
                        monitoring=True  )
        pdict.update(kwargs)
        
        # Run the feedback model and save post-mortem data
        self.out('Running feedback simulations...')
        model = VMOTrackFeedback(**pdict)
        model.T = 25.0
        theta_t0 = model.theta_t0[0].copy()
        theta_t0_new = model.theta_t0[1].copy()
        model.theta_t0[1:N_cues+2] = theta_t0 # first cue set same as training
        model.theta_t0[N_cues+2:2*N_cues+2] = theta_t0_new # second cue set is random
        model.advance() # training
        model.advance() # path integration
        self.results['cue_t0'] = model.cue_t_local[0] # save cue crossing time
        for repeat in 'once', 'twice':
            for size in cue_sizes:
                model.local_cue_std = size
                model.gamma_local = model._gamma_local_default()
                model.advance()
        model.post_mortem().tofile(os.path.join(self.datadir, 'data'))
        
        # Run the idealize toy model simulation
        self.out('Running toy simulations...')
        toy = VMOToyModel(cue_t0=model.cue_t_local[0], local_cue_std=cue_sizes[0], 
            cue_offset=cue_offset)
        toy.advance_all()
        self.results['toy_phase'] = toy.dtheta_local[0,0]
        toy.post_mortem().tofile(os.path.join(self.datadir, 'toy'))        
        
        # Good-bye
        self.out('All done!')        
                
    def create_plots(self, time_window=8, draw_background=True, draw_lines=True):
        # Move into data directoary and start logging
        os.chdir(self.datadir)
        self.out.outfd = file('figure.log', 'w')
        
        # Set up main figure for plotting
        self.figure = {}
        figsize = 14, 10
        plt.rcParams['figure.figsize'] = figsize
        self.figure['feedback'] = f = plt.figure(figsize=figsize)
        f.suptitle(self.label.title())

        # Load data
        N_cues = self.results['N_cues']
        cue_t0 = self.results['cue_t0']
        cue_sizes = self.results['cue_sizes']
        data = ModelPostMortem.fromfile('data.tar.gz')
        toy = ModelPostMortem.fromfile('toy.tar.gz')
        target = self.results['toy_phase']
        
        # Log some data
        self.out('Cue sizes = %s'%repr(cue_sizes))
        self.out('Cue occurs at t = %.3f seconds'%cue_t0)
        self.out('Toy model phase target = %.3f'%target)
        
        # Cue traversal time window
        time_window = float(time_window)
        self.out('Displaying a %.1f-sec window around t=%.1f-sec'%(time_window,cue_t0))
        _model_dt = data.t[1]-data.t[0]
        _t0 = int(cue_t0/_model_dt)
        _htw = int(0.5*time_window/_model_dt)
        dt = slice(_t0-_htw, _t0+_htw)
        v = np.squeeze(data.vel[0])
        rs_bar = np.sqrt((v**2).sum(axis=1))[dt].mean()
        self.out('Average running speed across window = %.3f cm/s'%rs_bar)
        
        # Function for drawing cue as background intensity map
        def draw_cue_bg(ax, trial, ymin=None, ymax=None, bins=128, use_toy=False):
            if ymin is None:
                ymin = ax.get_ylim()[0]
            if ymax is None:
                ymax = ax.get_ylim()[1]
            if use_toy:
                cue_profile = toy.C_local[trial]
            else:
                cue_profile = data.C_local[trial]
            ct = np.linspace(-time_window/2, time_window/2, bins+1)
            for i in xrange(bins):
                r = mpl.patches.Rectangle((ct[i], ymin), ct[i+1]-ct[i], 
                    ymax-ymin, alpha=0.2, lw=0, fill=True, zorder=-1,
                    fc=mpl.cm.jet(cue_profile[
                        int((float(i)/bins)*(dt.stop-dt.start))+dt.start]/
                        cue_profile[dt].max()))
                ax.add_artist(r)
        
        # Panel grid metadata
        time = toy.t[dt] - cue_t0
        xlim = time[0], time[-1]
        panels = 4
        rows = 3
        row = 1

        # Idealized trajectory model of single oscillator
        inset = (-1.2, 1.4, 2.4, 1.0)
        ax = plt.subplot(rows, panels, 1)
        if draw_lines:
            ax.plot(time, toy.theta[1][dt,0], 'm--', lw=2, zorder=5)
            for trial in xrange(2, toy.ntrials):
                ax.plot(time, toy.theta[trial][dt,0], 'k-', aa=True, zorder=3)
        ax.set_xlim(xlim)
        if draw_lines:
            ax.add_artist(
                mpl.patches.Rectangle(inset[:2], inset[2], inset[3], 
                    fill=False, lw=1, ec='0.5', zorder=2))
        if draw_background:
            draw_cue_bg(ax, 2, use_toy=True)
        
        ax = plt.subplot(rows, panels, 2) # inset of first panel
        if draw_lines:
            ax.plot(time, toy.theta[1][dt,0], 'm--', lw=2, zorder=5)
            for trial in xrange(2, toy.ntrials):
                ax.plot(time, toy.theta[trial][dt,0], 'k-', aa=True, zorder=3)
            ax.plot([inset[0], inset[0]+inset[2]], [target]*2, '-', c='0.5', lw=2, zorder=2)
        ax.set_xlim(inset[0], inset[0]+inset[2])
        ax.set_ylim(inset[1], inset[1]+inset[3])
        if draw_background:
            draw_cue_bg(ax, 2, bins=256, use_toy=True)
        
        ax = plt.subplot(rows, panels, 3)
        zero_trial = 1+(toy.ntrials-2)/2
        if draw_lines:
            for trial in xrange(2, toy.ntrials):
                ax.plot(time, toy.theta[trial][dt,0]-toy.theta[-zero_trial][dt,0], 'k-')
        ax.set_xlim(xlim)
        if draw_background:
            draw_cue_bg(ax, 2, use_toy=True)
        
        ax = plt.subplot(rows, panels, 4)
        alpha_t = alpha(time, A=2*pi*data.C_local[2].max(), 
            sigma=(9/pi)*cue_sizes[0], r=1, s=1)
        if draw_lines:
            for trial in xrange(2, toy.ntrials):
                error = toy.theta[trial][dt,0]-toy.theta[-zero_trial][dt,0]
                ax.plot(time, error/error[0], 'k-', aa=True)
            ax.plot(time, alpha_t/alpha_t[0], 'r--', zorder=5, lw=2, aa=True)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.05)
        if draw_background:
            draw_cue_bg(ax, 2, 0, 1.05, use_toy=True)
        
        # Get time index from trajectory simulation        
        time = data.t[dt] - cue_t0
        
        # Show phase correction for different sized cues in circle simulation
        for i in xrange(N_cues):
            trial = i + 2 + N_cues
            row += 1
            
            # Cropped trajectory
            ax = plt.subplot(rows, panels, panels*(row-1)+1)
            ax.plot(data.x[trial], data.y[trial], 'k-', lw=0.5, zorder=-1)
            ax.scatter(x=data.x[trial][dt], y=data.y[trial][dt], 
                c=data.C_local[trial][dt], s=5, linewidths=0)
            ax.axis('equal')
            ax.set_axis_off()

            self.out('Cue %d, max C = %.3f'%(i+1, data.C_local[trial].max()))
            
            # Subtract path integration signal
            cuemod = data.theta[trial][dt] #- data.theta[1][dt]
            for j in xrange(cuemod.shape[1]):
                if cuemod[-1,j] < -pi:
                    cuemod[:,j] += 2*pi
                elif cuemod[-1,j] > pi:
                    cuemod[:,j] -= 2*pi
            ax = plt.subplot(rows, panels, panels*(row-1)+2)
            if draw_lines:
                ax.plot(time, cuemod, 'k-', aa=True)
            ax.set_xlim(xlim)
            # ax.set_ylim(-pi, pi)
            if draw_background:
                draw_cue_bg(ax, trial)
            
            # Subtract cue-evoked phase error
            error = data.theta[trial][dt] - data.theta[trial-N_cues][dt]
            for j in xrange(error.shape[1]):
                if error[-1,j] < -pi:
                    error[:,j] += 2*pi
                elif error[-1,j] > pi:
                    error[:,j] -= 2*pi
            ax = plt.subplot(rows, panels, panels*(row-1)+3)
            if draw_lines:
                ax.plot(time, error, 'k-', aa=True)
            ax.set_xlim(xlim)
            # ax.set_ylim(-pi, pi)
            if draw_background:
                draw_cue_bg(ax, trial)
            
            self.out('Cue %d, epsilon = %.3f'%(i+1, np.median(error[-1]/error[0])))

            # Normalized phase error over time with matched analytical curve
            fwhm = _model_dt * (data.C_local[trial]>data.C_local[trial].max()/2).sum()
            sigma_t = fwhm / (2*np.sqrt(2*np.log(2)))
            alpha_t = alpha(time, A=2*pi*data.C_local[trial].max(), sigma=sigma_t, r=1, s=1)
            ax = plt.subplot(rows, panels, panels*(row-1)+4)
            if draw_lines:
                ax.plot(time, error/error[0], 'k-', aa=True)
                ax.plot(data.t[dt]-cue_t0, alpha_t/alpha_t[0], 'r--', lw=2, aa=True)
            ax.set_xlim(xlim)
            ax.set_ylim(0, 1.05)
            if draw_background:
                draw_cue_bg(ax, trial, 0, 1.05)
        
        plt.draw()
        plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
        self.out.outfd.close()
