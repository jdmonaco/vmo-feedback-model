# encoding: utf-8
"""
trends.py -- Analysis of trends in response changes across mismatch angle

Exported namespace: MismatchTrends

Created by Joe Monaco on 2010-02-17.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import numpy as np

# Package imports
from . import CL_LABELS, CL_COLORS
from ..core.analysis import BaseAnalysis
from ..tools.stats import smooth_pdf
from ..tools import circstat


class MismatchTrends(BaseAnalysis):

    """
    Collate data from multiple MismatchAnalysis results to show trends in 
    response changes across mismatch angles.
    """
    
    label = "mismatch trends"
    
    def collect_data(self, *args):
        """Collate data from previous results for analysis and visualization
        """
        # Sort analysis results data and get list of mismatch angles
        self.out('Sorting results data according to mismatch...')
        sort_data = []
        for session in args:
            data = session.results
            angle = data['mismatch']
            if type(angle) is tuple:
                label = ', '.join([str(a) for a in angle])
                angle = min(angle)
            elif type(angle) is int:
                label = str(angle)
            else:
                raise TypeError, \
                    'bad mismatch angle type (%s)'%str(type(angle))
            sort_data.append((angle, label, data))
        sort_data.sort()
        mismatch, angle_labels, data_list = np.array(sort_data, 'O').T
        self.results['mismatch_labels'] = angle_labels
        self.results['N_mismatch'] = N = len(data_list)
        
        # Mean and SEM of rotation angles and peak correlations
        self.out('Computing statistics of rotations and correlations...')
        self.results['rotations_mean'] = rot_mean = np.empty(N, 'd')
        self.results['rotations_sem'] = rot_sem = np.empty(N, 'd')
        self.results['correlations_mean'] = corr_mean = np.empty(N, 'd')
        self.results['correlations_sem'] = corr_sem = np.empty(N, 'd')
        self.results['N_common'] = N_common = np.empty(N, 'd')
        for i, data in enumerate(data_list):
            rots, corrs = data['rotcorr']
            rots = (np.pi/180)*rots
            N_common[i] = rots.shape[0]
            rot_mean[i] = circstat.mean(rots)
            rot_sem[i] = circstat.std(rots) / np.sqrt(N_common[i])
            corr_mean[i] = np.mean(corrs)
            corr_sem[i] = np.std(corrs) / np.sqrt(N_common[i])
        rot_mean[rot_mean>np.pi] -= 2*np.pi # make distal rots negative
        rot_mean *= 180/np.pi # convert back to degrees
        
        # Smoothed density distributions
        self.out('Computing smoothed density estimates...')
        rot_pdf = []
        corr_pdf = []
        for data in data_list:
            rots, corrs = data['rotcorr'].copy()
            rots[rots>180] -= 360 # make distal rots negative
            rot_pdf.append(smooth_pdf(rots))
            corr_pdf.append(smooth_pdf(corrs))
        self.results['rotations_pdf'] = np.array(rot_pdf, 'O')
        self.results['correlations_pdf'] = np.array(corr_pdf, 'O')
        
        # Population code rotation via correlation diagonals
        self.out('Collating correlation diagonals...')
        diags = [data['diags_MIS'] for data in data_list]
        self.results['diagonals'] = np.array(diags, 'O')
        
        # Response category distribution
        self.out('Collating categorical response distributions...')
        self.results['categories'] = categories = {}
        N_total = []
        for data in data_list:
            N_total.append(sum([data['response_tallies'][key] for key in 
                CL_LABELS]))
        for key in CL_LABELS:
            categories[key] = \
                np.array(
                    [data_list[i]['response_tallies'][key]/float(N_total[i])
                    for i in xrange(N)])
        self.results['N_total'] = np.array(N_total)
        
        # Good-bye!
        self.out('All done!')
        
    def create_plots(self):
        """Create trends plots for rotations, peak correlations and categorical
        remapping statistics.
        """
        from pylab import figure, subplot, rcParams, draw
        from ..tools.images import tiling_dims
        self.figure = {}
        
        res = self.results
        labels = res['mismatch_labels']
        N = res['N_mismatch']
        N_total = res['N_total']
        diagonals_figsize = 10, 5
        rotation_figsize = 13, 9
        category_figsize = 10, 7
        category_pie_figsize = 10, 10
        
        # Population mismatch correlation diagonals
        rcParams['figure.figsize'] = diagonals_figsize
        self.figure['diagonals_trends'] = f = figure()
        f.set_size_inches(diagonals_figsize)
        f.suptitle('Population Mismatch Correlations', fontsize=16)
        line_kwargs = dict(lw=2, aa=True)
        ax = subplot(111)
        ax.hold(True)
        for i in xrange(N):
            ax.plot(*res['diagonals'][i], label=labels[i], **line_kwargs)
        ax.axis('tight')
        ax.set_ylim(0,1)
        ax.set_xlabel('Rotation (degrees)')
        ax.set_ylabel('Diagonal Correlation')
        ax.legend(fancybox=True, loc=2)
        
        # Population and cluster rotations and correlations
        rcParams['figure.figsize'] = rotation_figsize
        self.figure['rotation_trends'] = f = figure()
        f.set_size_inches(rotation_figsize)
        f.suptitle('Cluster Rotation and Peak Correlation', fontsize=16)
        ax = subplot(221)
        ax.hold(True)
        for i in xrange(N):
            ax.plot(*res['rotations_pdf'][i], label=labels[i], **line_kwargs)
        ax.set_xlim(-180, 180)
        ax.set_xlabel('Rotation (degrees)')
        ax.set_ylabel('Pr[Rotation]')
        ax.legend(fancybox=True, loc=2)
        
        ax = subplot(223)
        ax.hold(True)
        for i in xrange(N):
            ax.plot(*res['correlations_pdf'][i], label=labels[i], **line_kwargs)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Peak Correlation')
        ax.set_ylabel('Pr[Peak Correlation]')
        
        err_kwargs = dict(fmt='-', ecolor='k', capsize=5, ms=3, elinewidth=1, 
            lw=2, aa=True)
        ax1 = subplot(122)
        ax2 = ax1.twinx()
        x = np.arange(1, N+1)
        rh = ax1.errorbar(x, res['rotations_mean'], yerr=res['rotations_sem'], 
            c=(0.2, 0.0, 0.8), **err_kwargs)
        ch = ax2.errorbar(x, res['correlations_mean'], yerr=res['correlations_sem'], 
            c=(0.8, 0.0, 0.2), **err_kwargs)
        ax2.set_ylabel('Peak Correlation', rotation=270)
        ax2.set_ylim(0, 1)
        ax1.set_ylabel('Rotation (degrees)')
        ax1.set_xlabel('Mismatch')
        ax1.set_xlim(0.5, N+0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend((rh[0], ch[0]), ('Rotation', 'Peak Correlation'), 
            fancybox=True, loc=0)
        
        # Category distribution stacked bar chart
        rcParams['figure.figsize'] = category_figsize
        self.figure['response_trends'] = f = figure()
        f.set_size_inches(category_figsize)
        f.suptitle('Response Changes: Trends', fontsize=16)
        plot_category_chart(subplot(111), res)
        
        # Category distribution pie charts
        rcParams['figure.figsize'] = category_pie_figsize
        self.figure['response_trends_pie'] = f = figure()
        f.set_size_inches(category_pie_figsize)
        f.suptitle('Response Changes', fontsize=16)
        r, c = tiling_dims(N)
        pie_kwargs = dict(labels=CL_LABELS, colors=CL_COLORS, autopct='%.1f', 
            pctdistance=0.8, labeldistance=100)
        for i in xrange(N):
            ax = subplot(r, c, i+1)
            tally = [cats[k][i] for k in CL_LABELS]
            ax.pie(tally, **pie_kwargs)
            ax.axis('equal')
            ax.set_xlim(-1.04, 1.04) # fixes weird edge clipping
            ax.set_title('%s (%d total)'%(labels[i], N_total[i]))
            if i == 0:
                ax.legend(fancybox=True, loc='right', 
                    bbox_to_anchor=(0.1, -0.05))

        # Reset figure size
        draw()
        rcParams['figure.figsize'] = 9, 9


# Plot helper functions

def plot_category_chart(ax, res, show_legend=True):
    """Plot stacked bar chart of remapping response categories
    
    Arguments:
    ax -- axis to plot into
    res -- trends results dictionary
    
    Keyword arguments:
    show_legend -- whether to draw legend
    """
    from matplotlib.patches import Polygon
    labels = res['mismatch_labels']
    N = res['N_mismatch']
    x = np.arange(1, N+1)
    
    stacked = np.empty((len(CL_LABELS)+1, N), 'd')
    stacked[0] = 0.0
    stacked[-1] = 1.0
    cats = res['categories']
    for i in xrange(1,len(CL_LABELS)):
        stacked[i] = stacked[i-1] + cats[CL_LABELS[i-1]]
    poly_kwargs = dict(aa=True, lw=1.5, alpha=0.85)
    xloop = np.concatenate((x, x[::-1]))
    for i in xrange(len(CL_LABELS)):
        yloop = np.concatenate((stacked[i], stacked[i+1,::-1]))
        p = Polygon(np.c_[xloop, yloop], fc=CL_COLORS[i], 
            ec=CL_COLORS[i], label=CL_LABELS[i], **poly_kwargs)
        ax.add_artist(p)
        p.set_clip_box(ax.bbox)
    ax.axis([1, N, 0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Mismatch')
    ax.set_ylabel('Response Fraction')
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return ax
