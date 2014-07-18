#encoding: utf-8
"""
remapping -- Remapping figure showing orthogonalization from initial phase reset

Created by Joe Monaco on 2010-10-12.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

# Package imports
from ..core.analysis import BaseAnalysis
from ..vmo import VMOModel
from ..session import VMOSession
from ..compare import (correlation_matrix, correlation_diagonals, 
    population_spatial_correlation)
from ..tools.images import array_to_image
from ..tools.radians import circle_diff_vec


class RemappingFigure(BaseAnalysis):
    
    """
    Run complete remapping experiment based on random initial reset
    """
    
    label = "remapping"
    
    def collect_data(self, N_samples=2, **kwargs):
        """Run basic VMOModel remapping experiment by randomly initializing 
        the phase code of a network of oscillators and place units.

        Keyword arguments:
        N_samples -- total number of simulations to run (N-1 remapped from 1st)
        
        Additional keyword arguments are passed on to VMOModel.
        """
        self.results['N_samples'] = N_samples
        
        # Set up model parameters
        pdict = dict(   N_outputs=500, 
                        N_theta=1000,
                        N_cues=1, 
                        C_W=0.05,
                        gamma_local=0, 
                        gamma_distal=0, 
                        num_trials=N_samples, 
                        refresh_fixed_points=False  )
        pdict.update(kwargs)
        
        # Set up and run the path integration model
        self.out('Running remapping simulations...')
        model = VMOModel(**pdict)
        model.advance_all()
        sessions = VMOSession.get_session_list(model)
        VMOSession.save_session_list(sessions, 
            os.path.join(self.datadir, 'samples'))
        
        # Get unit ordering based on first environment
        sortix = list(sessions[0].sortix)
        sortix += list(set(range(sessions[0].num_units)) - set(sortix))
        self.results['sortix'] = np.array(sortix)
        
        # Save multi-session population responses and activity patterns
        self.out('Computing and storing population responses...')
        R = [SD.get_population_matrix(clusters=sortix) for SD in sessions]
        np.save(os.path.join(self.datadir, 'R.npy'), np.asarray(R))
        
        # Good-bye
        self.out('All done!')
        
    def create_plots(self, N_examples=4, examples=None):
        """Create figure(s) with basic data panels 
        """
        # Change to data directoary and start logging
        os.chdir(self.datadir)
        self.out.outfd = file('figure.log', 'w')
        
        # Set up main figure for plotting
        self.figure = {}
        figsize = 9, 12
        plt.rcParams['figure.figsize'] = figsize
        self.figure['remapping'] = f = plt.figure(figsize=figsize)
        f.suptitle(self.label.title())

        # Load the data
        R = np.load(os.path.join(self.datadir, 'R.npy'))
        N = self.results['N_samples']
                
        # Example active unit responses across environments
        if examples is None:
            active = set()
            for j in xrange(N):
                active = active.union(set((R[j].max(axis=1)>=1).nonzero()[0]))
            active = list(active)
            active.sort()
            examples = np.random.permutation(len(active))[:N_examples]
            examples = np.array(active)[examples]
        self.out('Plotting example responses: %s'%repr(examples))
        for i,ex in enumerate(examples):
            self.out('Unit %d max response = %.2f Hz'%(ex, R[:,ex].max()))
            for j in xrange(N):
                ax = plt.subplot(2*N_examples, N, N*i+j+1)
                ax.plot(R[j,ex], c='k', lw=1.5)
                ax.set_xlim(0, 360)
                ax.set_ylim(-0.1*R[:,ex].max(), 1.1*R[:,ex].max())
                ax.set_axis_off()
            
        # Population responses
        for j in xrange(N):
            self.out('Environment %d population max = %.2f Hz'%(j+1, R[j].max()))
            ax = plt.subplot(2, N, j+1+N)
            ax.imshow(R[j], aspect='auto', interpolation='nearest')
            array_to_image(R[j], 'pop_env_%02d.png'%(j+1), cmap=mpl.cm.gray_r)
                        
        plt.draw()
        plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
        self.out.outfd.close()
