#encoding: utf-8
"""
pathint -- Path integration figure showing time-series data (without cues)

Created by Joe Monaco on 2010-10-07.

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
from ..vmo import VMOModel
from ..session import VMOSession
from ..placemap import CirclePlaceMap
from ..compare import correlation_matrix, correlation_diagonals
from ..tools.stats import smooth_pdf
from ..tools.filters import halfwave
from ..tools.images import array_to_image


class PathIntFigure(BaseAnalysis):
    
    """
    Create the example path integration dataset for producing the basic
    introductory figure to the path integration mechanism of the 
    VMOModel model
    """
    
    label = 'path integration'
    
    def collect_data(self, **kwargs):
        """Run basic VMOModel path integration simulation and save relevant
        data for the intro figure of the paper.
        
        Additional keyword arguments are passed on to VMOModel.
        """
        # Set up and run the path integration model
        self.out('Running simulation...')
        model = VMOModel(N_outputs=500, N_theta=1000, N_cues=1, C_W=0.05,
            gamma_local=0, gamma_distal=0, **kwargs)
        model.advance()
        pm = model.post_mortem()
        
        # Save session data object for visualization
        session_fn = 'session.tar.gz'
        self.out('Computing and saving responses to %s...'%repr(session_fn))
        SD = VMOSession(model, save_I=True, save_E=True)
        SD.tofile(os.path.join(self.datadir, session_fn))
        self.results['session_file'] = session_fn
        
        # Save other simulation data
        data_fn = 'data.npz'
        self.out('Saving other data to %s...'%repr(data_fn))
        np.savez(os.path.join(self.datadir, data_fn), vel=pm.vel)
        self.results['data_file'] = data_fn
        
        # Good-bye!
        self.out('All done!')
        
    def create_plots(self, examples=[0,1], lap_inset=[3,4]):
        """Create figure with basic data panels 
        """
        self.figure = {}
        figsize = 10, 12
        plt.rcParams['figure.figsize'] = figsize
        self.figure['pathint'] = f = plt.figure(figsize=figsize)
        f.suptitle(self.label.title())
        
        # Load the simulation data if necessary
        if 'SD' in self.results:
            SD = self.results['SD']
        else:
            self.results['SD'] = SD = VMOSession.fromfile(
                os.path.join(self.datadir, self.results['session_file']))
        if 'data' in self.results:
            data = self.results['data']
        else:
            self.results['data'] = data = np.load(
                os.path.join(self.datadir, self.results['data_file']))
        os.chdir(self.datadir)
        self.out.outfd = file('figure.log', 'w')
        
        # Track angle
        tlim = (SD.t[0], SD.t[-1])
        alpha_star = SD.alpha - SD.alpha[0]
        alpha_star[alpha_star<=0] += 2*pi
        ax = plt.subplot(12,1,1)
        ax.plot(SD.t, alpha_star, 'k-', lw=1.5)
        ax.set_xlim(tlim)
        ax.set_axis_off()
        
        # V_x
        vlim = (-40,40)
        bar_x = 100
        bar_y = vlim[0]+4
        ax = plt.subplot(12,1,2)
        ax.plot(SD.t, data['vel'][:,0,0], 'k-', lw=0.5, zorder=5)
        ax.plot(tlim, [0,0], '-', c='0.6', lw=1, zorder=2)
        ax.plot([bar_x,bar_x+30], [bar_y]*2, 'k-', lw=1.5, zorder=3) # scale bar: 30 s
        ax.plot([bar_x+30]*2, [bar_y,bar_y+10], 'k-', lw=1.5, zorder=3) # 10 cm/s
        self.out('Velocity trace scale bar = 30s x 10cm/s')
        ax.set_xlim(tlim)
        ax.set_ylim(vlim)
        ax.set_axis_off()

        # V_y
        ax = plt.subplot(12,1,3)
        ax.plot(SD.t, data['vel'][:,0,1], 'k-', lw=0.5, zorder=5)
        ax.plot(tlim, [0,0], '-', c='0.6', lw=1, zorder=2)
        ax.set_xlim(tlim)
        ax.set_ylim(vlim)
        ax.set_axis_off()     
        
        # Ex 1
        ax1 = plt.subplot(12,1,4)
        ax1.plot(SD.t, SD.I_cache[examples[0]], 'b-', lw=0.5, zorder=5)
        # ax1.plot(tlim, [0,0], 'k-', lw=1, zorder=7)
        ax1.axis('tight')
        ax1.set_xlim(tlim)
        ax1.set_axis_off()
        
        # Ex 2
        ax2 = plt.subplot(12,1,5)
        ax2.plot(SD.t, SD.I_cache[examples[1]], 'b-', lw=0.5, zorder=5)
        # ax2.plot(tlim, [0,0], 'k-', lw=1, zorder=7)
        ax2.axis('tight')
        ax2.set_xlim(tlim)
        ax2.set_axis_off()
        
        # Set y limits for examples
        max_I = 1.03*np.max(np.absolute(SD.I_cache[examples]))
        ax1.set_ylim(-max_I, max_I)
        ax2.set_ylim(-max_I, max_I)
        
        # Lap indicators
        rect_kw = dict(fill=True, linewidth=0, zorder=0, alpha=0.2)
        cols = ['b', 'g']
        for sp in 2, 3, 4, 5:
            ax = plt.subplot(12,1,sp)
            for lap in xrange(len(SD.laps) - 1):
                lap_rect = mpl.patches.Rectangle((SD.laps[lap], vlim[0]), 
                    SD.laps[lap+1]-SD.laps[lap], vlim[1]-vlim[0], 
                    fc=cols[lap%2], **rect_kw)
                ax.add_artist(lap_rect)
        
        # Example insets
        crop = slice((SD.t >= SD.laps[lap_inset[0]]).nonzero()[0][0], 
            (SD.t >= SD.laps[lap_inset[1]+1]).nonzero()[0][0])
        examples = np.asarray(examples)
        E_max = 1.05*np.absolute(SD.E_cache[examples,crop]).max()
        for subp in ((12,1,6), (12,1,7)), ((12,2,15), (12,2,17)):
            ax = plt.subplot(*subp[0])
            ax.plot(SD.t[crop], SD.I_cache[examples[0],crop], 'b-', lw=0.5)
            ax.plot(SD.t[crop], SD.E_cache[examples[0],crop], 'k-', lw=1.5, aa=True)
            ax.plot([SD.t[crop.start], SD.t[crop.stop]], [SD.thresh]*2, 'k--')
            ax.axis('tight')
            ax.set_ylim(-E_max, E_max)
            ax.set_axis_off()        
            ax = plt.subplot(*subp[1])
            ax.plot(SD.t[crop], SD.I_cache[examples[1],crop], 'b-', lw=0.5)
            ax.plot(SD.t[crop], SD.E_cache[examples[1],crop], 'k-', lw=1.5, aa=True)
            ax.plot([SD.t[crop.start], SD.t[crop.stop]], [SD.thresh]*2, 'k--')
            ax.axis('tight')
            ax.set_ylim(-E_max, E_max)
            ax.set_axis_off()
            bar_x = SD.laps[lap_inset[0]]
            bar_y = -E_max+2
            ax.plot([bar_x,bar_x+5], [bar_y]*2, 'k-', lw=1.5, zorder=3) # scale bar: 5 s
            ax.plot([bar_x+5]*2, [bar_y,bar_y+5], 'k-', lw=1.5, zorder=3) # 5 Hz
        self.out('Input scale bar = 5s x 5Hz')


        # Start new figure
        self.figure['responses'] = f = plt.figure(figsize=figsize)
        f.suptitle('Output Responses')
        
        # Input peak distro
        E_max = SD.E_cache.max()
        ax = plt.subplot(421)
        ax.plot(*smooth_pdf(SD.E_cache.max(axis=1)), c='k', lw=1.5, aa=True)
        ax.plot([SD.thresh]*2, [0, 1.05*ax.get_ylim()[1]], 'r-', zorder=5)
        ax.axis('tight')
        ax.set_xlim(0, 1.05*E_max)
        
        # Input/output trajectories
        # ax = plt.subplot(8,6,4)
        # ax.scatter(x=SD.x[::4], y=SD.y[::4], c=SD.E_cache[examples[0],::4], 
        #     s=1, lw=0, marker='o')
        # ax.axis('equal')
        # ax.set_axis_off()
        
        ax = plt.subplot(843)
        ax.scatter(x=SD.x[::4], y=SD.y[::4], 
            c=halfwave(SD.E_cache[examples[0],::4]-SD.thresh), 
            s=1, lw=0, marker='o')
        ax.axis('equal')
        ax.set_axis_off()
        
        ax = plt.subplot(844)
        ax.pcolor(SD.R_laps[examples[0]].T, cmap=mpl.cm.jet)
        self.out('Example 1, lap pcolor max = %.2f'%SD.R_laps[examples[0]].max())
        ax.axis('tight')
        ax.set_axis_off()
        array_to_image(np.flipud(SD.R_laps[examples[0]].T), 'ex1_pcolor.png', 
            cmap=mpl.cm.jet)
        
        # ax = plt.subplot(8,6,10)
        # ax.scatter(x=SD.x[::4], y=SD.y[::4], c=SD.E_cache[examples[1],::4], 
        #     s=1, lw=0, marker='o')
        # ax.axis('equal')
        # ax.set_axis_off()
        
        ax = plt.subplot(847)
        ax.scatter(x=SD.x[::4], y=SD.y[::4], 
            c=halfwave(SD.E_cache[examples[1],::4]-SD.thresh), 
            s=1, lw=0, marker='o')
        ax.axis('equal')
        ax.set_axis_off()
        
        ax = plt.subplot(848)
        ax.pcolor(SD.R_laps[examples[1]].T, cmap=mpl.cm.jet)
        self.out('Example 2, lap pcolor max = %.2f'%SD.R_laps[examples[1]].max())
        ax.axis('tight')
        ax.set_axis_off()
        array_to_image(np.flipud(SD.R_laps[examples[1]].T), 'ex2_pcolor.png', 
            cmap=mpl.cm.jet)
        
        # Population responses
        # ax = plt.subplot(434)
        R = SD.get_population_matrix()
        # ax.imshow(R, aspect='auto', interpolation='nearest')
        # ax.set_axis_off()
        array_to_image(R, 'pop_response.png', 
            cmap=mpl.cm.jet)
        self.out('Pop matrix max = %.2f'%SD.R.max())        
        
        # ax = plt.subplot(435)
        PM = CirclePlaceMap(SD)
        # ax.imshow(PM.coverage_maps, cmap=mpl.cm.gray_r, aspect='auto', 
        #     interpolation='nearest')
        # ax.set_axis_off()
        array_to_image(PM.coverage_maps, 'field_extents.png', 
            cmap=mpl.cm.gray_r)
        self.out('Num active units = %d'%PM.num_active)
        
        # ax = plt.subplot(8, 3, 9)
        C = correlation_matrix(SD)
        # ax.imshow(np.flipud(C))
        # ax.axis('equal')
        # ax.set_axis_off()
        array_to_image(np.flipud(C), 'corr_matrix.png', 
            cmap=mpl.cm.jet)
        
        # ax = plt.subplot(8, 3, 12)
        D = correlation_diagonals(C, centered=True)
        # ax.plot(D[0], D[1], 'k-', lw=1.5, aa=True)
        # ax.axis('tight')
        # ax.set_ylim(0, 1.05)
        width = (D[0,1]-D[0,0]) * (D[1]>=D[1].max()/2).sum()
        self.out('Corr width at half max = %.1f degrees'%width)
        
        # Population distros
        # ax = plt.subplot(8, 3, 13)
        # ax.plot(*smooth_pdf(PM.field_size, sd=10), c='k', lw=1.5)
        # ax.axis('tight')
        # newy = 1.1*ax.get_ylim()[1]
        # ax.set_xlim(-10, 200)
        # ax.set_ylim(ymax=newy)
        med_size = np.median(PM.field_size)
        # ax.plot([med_size]*2, [0, newy], 'r-', zorder=1)
        self.out('Number of fields = %d'%PM.num_fields.sum())
        self.out('Median field size = %.1f degrees'%med_size)
        
        # ax = plt.subplot(8, 3, 14)
        # ax.plot(*smooth_pdf(PM.maxima[:,1]), c='k', lw=1.5)
        # ax.axis('tight')
        # newy = 1.1*ax.get_ylim()[1]
        # ax.set_xlim(-1, 12)
        # ax.set_ylim(ymax=newy)
        med_rate = np.median(PM.maxima[:,1])
        # ax.plot([med_rate]*2, [0, newy], 'r-', zorder=1)
        self.out('Median firing rate = %.2f Hz'%med_rate)
        
        # ax = plt.subplot(8, 3, 15)
        # ax.plot(*smooth_pdf(SD.I_rate[SD.sortix], sd=0.25), c='k', lw=1.5)
        # ax.axis('tight')
        # newy = 1.1*ax.get_ylim()[1]
        # ax.set_xlim(0, 12)
        # ax.set_ylim(ymax=newy)
        med_info = np.median(SD.I_rate[SD.sortix])
        # ax.plot([med_info]*2, [0, newy], 'r-', zorder=1)
        self.out('Median spatial info = %.1f bits/spike'%med_info)
        
        plt.draw()
        plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
        self.out.outfd.close()
