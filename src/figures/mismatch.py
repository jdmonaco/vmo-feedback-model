#encoding: utf-8
"""
Mismatch -- Double rotation figure for various mismatch angles

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

# Package imports
from ..core.analysis import BaseAnalysis
from ..session import VMOSession
from ..remapping.simulate import VMOExperiment
from ..remapping.mismatch import MismatchAnalysis
from ..remapping.trends import MismatchTrends, plot_category_chart
from ..tools.images import array_to_image


class MismatchFigure(BaseAnalysis):
    
    """
    Run symmetric and asymmetric example cue mismatch experiments
    """
    
    label = "mismatch"
    
    def collect_data(self, sym_cues=(4, 10), asym_cues=(4, 20), 
        mismatch=[45, 90, 135, 180], **kwargs):
        """Run VMOModel mismatch experiment by setting up two sets of cues 
        that counter-rotate to produce cue conflict
        
        Keyword arguments:
        sym_cues/asym_cues -- cue definitions for the two mismatch experiments
            that are simulated here: (cue number, cue size) tuples, where cue
            size is in degrees; sym_cues definition is used for both local and
            distal cue sets in the symmetric cue experiment, and asym_cues
            specifies the local cue configuration for the asymmetric experiment
            (distal configuration is as specified by sym_cues)
        mismatch -- list of mismatch angles to test

        Additional keyword arguments are passed on to VMOModel.
        """
        N_sym_cues, sym_cue_size = sym_cues
        sym_cue_size *= (pi/180)
        N_asym_cues, asym_cue_size = asym_cues
        asym_cue_size *= (pi/180)
        
        self.results['N_sym_cues'] = N_sym_cues
        self.results['sym_cue_size'] = sym_cue_size
        self.results['N_asym_cues'] = N_asym_cues
        self.results['asym_cue_size'] = asym_cue_size
        
        self.results['mismatch'] = mismatch
        
        pdict = dict(   N_outputs=500, 
                        N_theta=1000,
                        C_W=0.05,
                        N_cues=N_sym_cues,
                        cue_std=sym_cue_size,
                        monitoring=True  
                        )
        pdict.update(kwargs)
        
        sym_dir = os.path.join(self.datadir, 'symmetric')
        if os.path.isdir(sym_dir):
            self.out('Found symmetric mismatch experiment...')
            sym_exp = VMOExperiment.load_data(
                os.path.join(sym_dir, 'analysis.pickle'))
        else:
            self.out('Starting symmetric mismatch experiment...')
            sym_exp = VMOExperiment(datadir=sym_dir)
            sym_exp(mismatch=mismatch, **pdict)
            
        asym_dir = os.path.join(self.datadir, 'asymmetric')
        pdict['N_cues_local'] = N_asym_cues
        pdict['local_cue_std'] = asym_cue_size
        if os.path.isdir(asym_dir):
            self.out('Found asymmetric mismatch experiment...')
            asym_exp = VMOExperiment.load_data(
                os.path.join(sym_dir, 'analysis.pickle'))
        else:
            self.out('Starting asymmetric mismatch experiment...')
            asym_exp = VMOExperiment(datadir=asym_dir)
            asym_exp(mismatch=mismatch, **pdict)
        
        self.out('Recovering and saving symmetric cues...')
        sym_cues = VMOSession.fromfile(os.path.join(sym_dir, 'STD.tar.gz')).recover_cues()
        np.savez(os.path.join(self.datadir, 'sym_cues'), 
            t=sym_cues.t, x=sym_cues.x, y=sym_cues.y, 
            C_local=sym_cues.C_local, C_distal=sym_cues.C_distal)
        
        self.out('Recovering and saving asymmetric cues...')
        asym_cues = VMOSession.fromfile(os.path.join(asym_dir, 'STD.tar.gz')).recover_cues()
        np.savez(os.path.join(self.datadir, 'asym_cues'), 
            t=asym_cues.t, x=asym_cues.x, y=asym_cues.y, 
            C_local=asym_cues.C_local, C_distal=asym_cues.C_distal)
        
        self.out('Running mismatch analysis on symmetric cues...')
        sym_mma = []
        for angle in mismatch:
            sym_mma_dir = os.path.join(sym_dir, 'mismatch_%03d'%angle)
            if os.path.exists(sym_mma_dir):
                ana = MismatchAnalysis.load_data(os.path.join(sym_mma_dir, 'analysis.pickle'))
            else:
                ana = MismatchAnalysis(datadir=sym_mma_dir)
                ana(mismatch=angle, load_dir=sym_dir)
            sym_mma.append(ana)
        
        self.out('Running mismatch analysis on asymmetric cues...')
        asym_mma = []
        for angle in mismatch:
            asym_mma_dir = os.path.join(asym_dir, 'mismatch_%03d'%angle)
            if os.path.exists(asym_mma_dir):
                ana = MismatchAnalysis.load_data(os.path.join(asym_mma_dir, 'analysis.pickle'))
            else:
                ana = MismatchAnalysis(datadir=asym_mma_dir)
                ana(mismatch=angle, load_dir=asym_dir)
            asym_mma.append(ana)
        
        self.out('Computing symmetric mismatch trends...')
        trends_dir = os.path.join(sym_dir, 'trends')
        trends = MismatchTrends(datadir=trends_dir)
        trends(*sym_mma)
        
        self.out('Computing asymetric mismatch trends...')
        trends_dir = os.path.join(asym_dir, 'trends')
        trends = MismatchTrends(datadir=trends_dir)
        trends(*asym_mma)
        
        self.out("Good-bye!")
        
    def create_plots(self, data_trends=None):
        """Create cue plots and rotation/correlation histogram panels showing
        results of simulated mismatch experiments.
        
        data_trends -- MismatchTrends objects for hippocampal mismatch data
            (if None, this is not plotted)
        """
        # Move into data directoary and start logging
        os.chdir(self.datadir)
        self.out.outfd = file('figure.log', 'w')
        
        # Set up main figure for plotting
        self.figure = {}
        figsize = 9, 9
        plt.rcParams['figure.figsize'] = figsize
        self.figure['mismatch'] = f = plt.figure(figsize=figsize)
        f.suptitle(self.label.title())

        # Load the cue data
        sym_cue = np.load('sym_cues.npz')
        asym_cue = np.load('asym_cues.npz')
        self.out('Symmetric local cue max = %.4f'%sym_cue['C_local'].max())
        self.out('Symmetric distal cue max = %.4f'%sym_cue['C_distal'].max())
        self.out('Asymmetric local cue max = %.4f'%asym_cue['C_local'].max())
        self.out('Asymmetric distal cue max = %.4f'%asym_cue['C_distal'].max())
        
        # Load mismatch data
        mismatch = self.results['mismatch']
        nmismatch = len(mismatch)

        def get_mismatch_dict(mdir):
            mdict = {}
            for angle in mismatch:
                mma = MismatchAnalysis.load_data(
                    os.path.join(mdir, 'mismatch_%03d'%angle, 
                        'analysis.pickle'))
                rc = mma.results['rotcorr']
                _r, _c = rc
                _r[_r>=180] -= 360 # map [0, 360) angles to [-180, 180)
                mdict[angle] = rc
            return mdict
            
        sym_data = get_mismatch_dict('symmetric')
        asym_data = get_mismatch_dict('asymmetric')
            
        # Plot metadata
        nrows = 5
        ncols = int(np.ceil(nmismatch/2)) + 1
        row = 1
        
        # Plot the cues
        def plot_cue(ax, cue, cue_str, skip=4):
            ax.scatter(x=cue['x'][::skip], y=cue['y'][::skip], c=cue['C_'+cue_str][::skip], 
                s=2, linewidths=0)
            ax.axis('equal')
            ax.set_title(cue_str)
            ax.set_axis_off()
        
        plot_cue(plt.subplot(nrows, ncols, 0*ncols + 1), sym_cue, 'local')
        plot_cue(plt.subplot(nrows, ncols, 1*ncols + 1), sym_cue, 'distal')
        plot_cue(plt.subplot(nrows, ncols, 2*ncols + 1), asym_cue, 'local')
        plot_cue(plt.subplot(nrows, ncols, 3*ncols + 1), asym_cue, 'distal')
        
        # Plot the rotational correlations
        def plot_rot_corr(ax, data, angle):
            r, c = data[angle]
            H, x, y = np.histogram2d(r, c, bins=16, range=[[-180, 180], [0, 1]])
            ax.pcolor(H, cmap=mpl.cm.gray_r)
            ax.set_title(str(angle))
            ax.axis('tight')
            ax.axis('equal')
            ax.set_axis_off()
            return H
        
        spix = 2
        for angle in mismatch:
            if spix % ncols == 1:
                spix += 1
            H = plot_rot_corr(plt.subplot(nrows, ncols, spix), sym_data, angle)
            self.out('Symmetric mismatch %d H_max = %d'%(angle, H.max()))
            array_to_image(np.flipud(H), 'sym_rotcorr_%03d.png'%angle, cmap=mpl.cm.gray_r)
            spix += 1

        spix = 2*ncols + 2
        for angle in mismatch:
            if spix % ncols == 1:
                spix += 1
            H = plot_rot_corr(plt.subplot(nrows, ncols, spix), asym_data, angle)
            self.out('Asymmetric mismatch %d H_max = %d'%(angle, H.max()))
            array_to_image(np.flipud(H), 'asym_rotcorr_%03d.png'%angle, cmap=mpl.cm.gray_r)
            spix += 1

        # Plot stacked response category charts
        figsize = 14, 4
        plt.rcParams['figure.figsize'] = figsize
        self.figure['categories'] = f = plt.figure(figsize=figsize)
        f.suptitle('Response Categories')
        
        sym_trends = MismatchTrends.load_data(os.path.join('symmetric', 'trends'))
        asym_trends = MismatchTrends.load_data(os.path.join('asymmetric', 'trends'))
        
        plot_category_chart(plt.subplot(131), sym_trends.results, False)
        plot_category_chart(plt.subplot(132), asym_trends.results, False)
        if data_trends is not None:
            plot_category_chart(plt.subplot(133), data_trends.results, False)

        plt.draw()
        plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
        self.out.outfd.close()
