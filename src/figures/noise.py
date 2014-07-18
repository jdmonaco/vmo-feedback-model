#encoding: utf-8
"""
noise.py -- Continuous phase noise figure showing decorrelation due to phase 
    noise, plus rescue by cue

Created by Joe Monaco on 2010-10-12.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
from numpy import pi
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pylab as plt
from enthought.traits.api import Float

# Package imports
from ..core.analysis import BaseAnalysis
from ..vmo import VMOModel
from ..session import VMOSession
from ..compare import population_spatial_correlation


class VMONoiseModel(VMOModel):
    
    """
    Variant on VMOModel that injects phase noise into the theta oscillators
    """
    
    label = "Noise Model"
    sigma = Float
    dt = 0.010
    
    def run_timestep(self):
        super(VMONoiseModel, self).run_timestep()
        self.theta[:] += \
            (self.sigma * np.sqrt(self.dt)) * np.random.randn(self.N_theta) 


class PhaseNoiseFigure(BaseAnalysis):
    
    """
    Run a simple global Error experiment based on random initial reset
    """
    
    label = "noise analysis"
    
    def collect_data(self, cue_size=10, sigma=0.1, test_factors=[2], 
        cue_offset=9*pi/8, **kwargs):
        """Run VMONoiseModel experiment by testing different levels of stochastic 
        phase noise and assessing spatial correlation rescue by cue feedback.
        
        Keyword arguments:
        cue_size -- width (s.d.) of cue in degrees
        sigma -- baseline level of phase noise
        test_factors -- factors of sigma to test in addition to 0 and 1
        cue_offset -- track angle offset of cue position
        
        Additional keyword arguments are passed on to VMOModel.
        """
        self.results['cue_size'] = cue_size = (pi/180) * cue_size
        self.results['cue_offset'] = cue_offset
        self.results['test_factors'] = test_factors = [0, 1] + test_factors
        self.results['sigma'] = sigma
        
        # Set up model parameters
        pdict = dict(   N_outputs=500, 
                        N_theta=1000,
                        C_W=0.05,
                        N_cues_local=1, 
                        N_cues_distal=1, 
                        local_cue_std=cue_size,
                        cue_offset=cue_offset,
                        init_random=False,
                        gamma_distal=0, 
                        num_trials=2*len(test_factors),
                        monitoring=True  )
        pdict.update(kwargs)
        
        # Create the simulation object and save the cue peak (gamma)
        self.out('Running training simulation...')
        model = VMONoiseModel(**pdict)
        if 'T' in kwargs:
            model.T = kwargs['T']
        cue_gamma = model.gamma_local
    
        # Simulate the phase noise test trials without, then with, cue
        for gamma in 0.0, cue_gamma:
            model.gamma_local = gamma
            for factor in test_factors:
                model.sigma = sigma * factor
                model.advance()
        
        # Compute responses and save session data
        self.out('Computing and saving session data files...')
        sessions = VMOSession.get_session_list(model)
        VMOSession.save_session_list(sessions, os.path.join(self.datadir, 'sessions'))
        
        # Save raw simulation data file and clean up
        model.post_mortem().tofile(os.path.join(self.datadir, 'data'))
        
        # Compute population and population lap matrices and save to data directory
        self.out('Computing and saving population responses...')
        clusts = np.arange(pdict['N_outputs'])
        R = [SD.get_population_matrix(clusters=clusts, inplace=True) for SD in sessions]
        R_laps = [SD.get_population_lap_matrix(clusters=clusts, inplace=True) for SD in sessions]
        np.save(os.path.join(self.datadir, 'R_session'), np.asarray(R))
        np.save(os.path.join(self.datadir, 'R_laps'), np.asarray(R_laps))
        
        # All done!
        self.out('Good bye!')
        
    def create_plots(self, save_data=False):
        # Move into data directoary and start logging
        os.chdir(self.datadir)
        self.out.outfd = file('figure.log', 'w')
        
        # Set up analysis metadata
        factors = self.results['test_factors']
        nfactors = len(factors)
        sigma = self.results['sigma']
        self.out('Noise level sigma = %.5f'%sigma)
        cue_size = self.results['cue_size']
        cue_offset = self.results['cue_offset']
        self.out('Cue size = %.2f at track position %.2f'%(cue_size, cue_offset))
        labels = [str(int(f))+'s' for f in factors] + \
            [str(int(f))+'s+cue' for f in factors]
        self.out('Found phase noise analysis trials:\n%s'%repr(labels))
        
        # Load the stored data
        self.out("Loading population response matrices...")
        R = np.load('R_session.npy')
        PLM = np.load('R_laps.npy')
        ntrials, nunits, nbins, nlaps = PLM.shape
        
        # Set up main figure for plotting
        self.figure = {}
        figsize = 8.5, 13
        plt.rcParams['figure.figsize'] = figsize
        self.figure['phase_noise'] = f = plt.figure(figsize=figsize)
        f.suptitle(self.label.title())

        # Figure layout metadata
        rows = 4

        # Plot a bar graph of population whole-session spatial correlations
        ax = plt.subplot(rows, 2, 1)
        width = 0.5
        padding = 0.25
        curleft = 0
        ylim = [0, 1.05]
        xlim = [curleft-padding, -1]
        xticks = []
        lefts = []
        corrs = []
        col_nocue = '0.5'
        col_cue = 'k'
        cols = []

        for tr in xrange(nfactors):
            # Noise bar
            lefts.append(curleft)
            corrs.append(population_spatial_correlation(R[0], R[tr]))
            cols.append(col_nocue)
            curleft += width
            xticks.append(curleft)
            
            # Noise + cue bar
            lefts.append(curleft)
            corrs.append(population_spatial_correlation(R[0], R[tr+nfactors]))
            cols.append(col_cue)
            curleft += width + padding

        ax.bar(lefts, corrs, color=cols, width=width, linewidth=0, aa=False)
        xlim[1] = curleft
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels[:nfactors])
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Noise Factor')
        
        if save_data:
            np.save('full_corrs.npy', np.array(corrs))
        
        # Plot spatial correlations across laps
        ax1 = plt.subplot(rows, 2, 2)
        ax2 = plt.subplot(rows, 2, 3)
        lap_corrs = np.empty((ntrials, nlaps), 'd')
        col_seq = 'kbgrcm'

        for tr in xrange(ntrials):
            for lap in xrange(nlaps):
                lap_corrs[tr, lap] = \
                    population_spatial_correlation(R[0], PLM[tr, ..., lap])
        
        if save_data:
            np.save('lap_corrs.npy', lap_corrs)
        
        for tr in xrange(nfactors):
            ax1.plot(np.arange(nlaps)+1, lap_corrs[tr], c=col_seq[tr], 
                lw=2, label=labels[tr])
            ax2.plot(np.arange(nlaps)+1, lap_corrs[tr+nfactors], c=col_seq[tr], 
                lw=2)

        ax1.set_xlim(1, nlaps)
        ax1.set_ylim(min(lap_corrs.min(), 0), 1.05)
        ax1.set_xlabel('Laps')
        ax1.legend(loc='upper left', bbox_to_anchor=(0.7, -0.05))
            
        ax2.set_xlim(1, nlaps)
        ax2.set_ylim(min(lap_corrs.min(), 0), 1.05)
        ax2.set_xlabel('Laps')
            
        # Plot cue-triggered averages of spatial correlation
        ax = plt.subplot(rows, 2, 2*rows-1)
        lag_cue = pi # set lag in radians here
        lag = int(lag_cue * (180/pi)) # lag in degree-bins
        cue_bin = int(cue_offset * (180/pi)) # cue position
        
        min_corr = 0
        for tr in xrange(nfactors+1, 2*nfactors):
            cta = np.zeros((2*lag+1,), 'd')
            for b in xrange(cue_bin-lag, cue_bin+lag+1):
                b = int(np.fmod(b, 360))
                if b < 0:
                    b += 360
                lapbin_corr = 0
                for lap in xrange(nlaps):
                    lapbin_corr += pearsonr(R[0, :, b], PLM[tr, :, b, lap])[0]
                lapbin_corr /= nlaps
                cta[b-(cue_bin-lag)] = lapbin_corr
            
            # Handle trajectory zero-crossing artifact
            zero_ix = (cta == 0.0).nonzero()[0] 
            for ix in zero_ix:
                cta[ix] = (cta[ix-1] + cta[ix+1])/2
            
            # Flip average to match CW running direction, and plot
            ax.plot(np.arange(-lag, lag+1), cta[::-1], 
                c=col_seq[tr-nfactors], lw=2, label=labels[tr])
            if cta.min() < min_corr:
                min_corr = cta.min()
        
        ax.set_xlim(-lag, lag)
        ax.set_ylim(min_corr, 1.05)
        ax.set_xlabel('Track angle from cue (degrees)')
        ax.set_ylabel('Correlation')
        ax.legend(loc='lower left', bbox_to_anchor=(0.85, 1.02))        
        
        plt.draw()
        plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
        self.out.outfd.close()
