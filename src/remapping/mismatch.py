# encoding: utf-8
"""
mismatch.py -- Analysis for double-rotation data, single mismatch angle

Exported namespace: MismatchAnalysis

Created by Joe Monaco on 2010-02-15.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
import os
import numpy as np
from matplotlib import cm
from enthought.traits.api import Float

# Package imports
from . import CL_COLORS, CL_LABELS
from ..session import VMOSession
from ..core.analysis import BaseAnalysis
from ..tools.images import array_to_image
from ..compare import (common_units, comparison_matrices, 
    correlation_matrix, correlation_diagonals, mismatch_rotation, 
    mismatch_response_tally)


class MismatchAnalysis(BaseAnalysis):
    
    """
    BaseMismatchAnalysis subclass for VMODoubleRotation experiment simulations
    
    Keyword arguments:
    mismatch -- angle or tuple of angles specifying which mismatch sessions
        should be loaded
    load_dir -- name of the directory containing subject subdirectories (which
        must be of the form RatXX/), which contain the MIS_XXX.tar.gz archives
        of VMOSession objects for various mismatch angles
    cluster_criteria -- dictionary of cluster criteria for VMOSession;
        valid keys include min_spike_count and min_info_rate
    """
    
    label = "Mismatch Analysis"
    blur = Float(4.3)
    
    def load_dataset(self, mismatch=45, load_dir=None, cluster_criteria={}, 
        **kwargs):
        """Load session data, compute population matrices and return all the 
        info needed to run the analysis.
        """
        from glob import glob
        if not os.path.isdir(load_dir):
            raise ValueError, 'not a valid data directory!'
        
        if type(mismatch) is int:
            mismatch = (mismatch,)
        
        rat_dirs = []
        for rat_dir in glob(os.path.join(load_dir, 'Rat*')):
            if os.path.isdir(rat_dir):
                rat_dirs.append(os.path.abspath(rat_dir))
        if not rat_dirs:
            rat_dirs = [load_dir]
        
        rat = []
        angle = []
        session_pairs = []
        R_pairs = []
        total_clusts = 0
        clusters_included = {}
        for rat_dir in rat_dirs:
            load_files = []
            files = \
                [os.path.join(rat_dir, 'MIS_%03d.tar.gz'%m) for m in mismatch]
            for afile in files:
                if os.path.exists(afile):
                    load_files.append(afile)
            if not len(load_files):
                continue
            if len(rat_dirs) > 1:
                cur_rat = int(rat_dir[-2:])
            else:
                cur_rat = 0
            STDfile = os.path.join(rat_dir, 'STD.tar.gz')
            STD = VMOSession.fromfile(STDfile)
            for key in cluster_criteria:
                if hasattr(STD, key):
                    setattr(STD, key, cluster_criteria[key])
            for MISfile in load_files:
                MIS = VMOSession.fromfile(MISfile)
                for key in cluster_criteria:
                    if hasattr(MIS, key):
                        setattr(MIS, key, cluster_criteria[key])
                pair = STD, MIS
                common = common_units(*pair)
                if len(common):
                    self.out('Loading mismatch data:\n%s'%MISfile)
                    session_pairs.append(pair)
                    R_pairs.append(comparison_matrices(*pair))
                    total_clusts += len(common)
                    rat.append(cur_rat)
                    MISangle = int(MISfile.split('.')[0][-3:])
                    angle.append(MISangle)
                    clusters_included[(cur_rat, 1, 2, MISangle)] = common
        session_pairs = np.array(session_pairs, 'O')
        sessions = \
            np.core.records.fromarrays([np.array(rat), np.ones(len(rat)), 
            1+np.ones(len(rat)), np.array(angle)], 
            names='rat,day,session,angle', formats='i,i,i,i')
        return \
            sessions, session_pairs, R_pairs, total_clusts, clusters_included
            
    def collect_data(self, category_criteria={}, **kwargs):
        """Load filtered sets of session data and perform a series of analyses
        
        Keyword arguments:
        category_criteria -- dictionary of kwargs for mismatch_response_tally
            to specify category thresholds (e.g., angle_tol, min_corr)

        See subclass docstrings for keyword arguments to control session 
        filtering criteria.

        Analysis is performed both per-rat and globally, with saved results
        data and image files for both scopes.
        """
        # Create images directory
        save_dir = os.path.join(self.datadir, 'images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Load session data (this is subclass-specific, a required override)
        self.results['mismatch'] = kwargs['mismatch']
        sessions, session_pairs, R_pairs, total_clusts, clusters_included = \
            self.load_dataset(**kwargs)
        self.blur = session_pairs[0,0].default_blur_width
        self.results['N_clusters'] = total_clusts
        self.results['clusters'] = clusters_included
        self.out('Loaded %d mismatch sessions with %d clusters!'%
            (len(sessions), total_clusts))
        
        # Collapse across days within rats for per-animal responses
        rats = set(sessions.rat)
        self.results['rats'] = list(rats)
        self.out('Collating response matrices for %d rat(s)...'%len(rats))
        R_rats = {}
        for rat in rats:
            rat_pairs = [R_pairs[i] for i in np.nonzero(sessions.rat==rat)[0]]
            for pair in rat_pairs:
                if rat not in R_rats:
                    R_rats[rat] = pair[0].copy(), pair[1].copy()
                    continue
                R_rats[rat] = np.concatenate((R_rats[rat][0], pair[0])), \
                              np.concatenate((R_rats[rat][1], pair[1]))
            self.out('Finished collating responses for rat %d'%rat)
        
        # Collapse across all sessions to get global response matrices
        self.out('Collapsing all sessions into global response...')
        R_STD = R_MIS = None
        for pair in R_pairs:
            if R_STD is None:
                R_STD, R_MIS = pair
                continue
            R_STD = np.concatenate((R_STD, pair[0]))
            R_MIS = np.concatenate((R_MIS, pair[1]))
        
        # Auto- and cross-correlation matrices and diagonals
        self.out('Computing per-rat correlation matrices...')
        for rat in rats:
            # Compute and store the data
            self.results['C_STD_rat%d'%rat] = C_STD = \
                correlation_matrix(R_rats[rat][0])
            self.results['diags_STD_rat%d'%rat] = \
                correlation_diagonals(C_STD, centered=True, blur=self.blur)
            self.results['C_MIS_rat%d'%rat] = C_MIS = \
                correlation_matrix(*R_rats[rat])
            self.results['diags_MIS_rat%d'%rat] = D_MIS = \
                correlation_diagonals(C_MIS, centered=True, blur=self.blur)
            self.results['popshift_rat%d'%rat] = \
                np.array([D_MIS[0, np.argmax(D_MIS[1])], D_MIS[1].max()])
            
            # Save image files of correlation matrices
            array_to_image(np.flipud(C_STD), os.path.join(save_dir, 
                'C_STD_rat%d.png'%rat), cmap=cm.jet, norm=False)
            array_to_image(np.flipud(C_MIS), os.path.join(save_dir, 
                'C_MIS_rat%d.png'%rat), cmap=cm.jet, norm=False)
        
        # Global correlation matrices
        self.out('Computing global correlation matrices...')
        self.results['C_STD'] = C_STD = correlation_matrix(R_STD)
        self.results['diags_STD'] = \
            correlation_diagonals(C_STD, centered=True, blur=self.blur)
        self.results['C_MIS'] = C_MIS = correlation_matrix(R_STD, R_MIS)
        self.results['diags_MIS'] = D_MIS = \
            correlation_diagonals(C_MIS, centered=True, blur=self.blur)
        self.results['popshift'] = np.array([D_MIS[0, np.argmax(D_MIS[1])], 
            D_MIS[1].max()])
        
        # Save image files of global correlation matrices
        array_to_image(np.flipud(C_STD), os.path.join(save_dir, 'C_STD.png'), 
            cmap=cm.jet, norm=False)
        array_to_image(np.flipud(C_MIS), os.path.join(save_dir, 'C_MIS.png'), 
            cmap=cm.jet, norm=False)
        
        # Compute actual rotation angles via maximal correlation per-rat
        self.out('Computing rotation angles and correlations...')
        for rat in rats:
            self.results['rotcorr_rat%d'%rat] = mismatch_rotation(*R_rats[rat])
        self.results['rotcorr'] = mismatch_rotation(R_STD, R_MIS)
        
        # Tally up per-session categorical response changes
        self.out('Categorizing response changes across sessions...')
        rat_tallies = {}
        tallies = {}
        for rat in rats:
            rat_sessions = (sessions.rat == rat).nonzero()[0]
            for ix in rat_sessions:
                angle = sessions[ix][3]
                pair = session_pairs[ix]
                if rat not in rat_tallies:
                    rat_tallies[rat] = \
                        mismatch_response_tally(pair[0], pair[1], angle, 
                            **category_criteria)
                    continue
                new_tally = mismatch_response_tally(pair[0], pair[1], angle,
                    **category_criteria)
                for category in new_tally:
                    rat_tallies[rat][category] += new_tally[category]
            self.out('Finished response counts for rat %d'%rat)
            for category in rat_tallies[rat]:
                if category not in tallies:
                    tallies[category] = rat_tallies[rat][category]
                    continue
                tallies[category] += rat_tallies[rat][category]
            self.results['response_tallies_rat%d'%rat] = rat_tallies[rat]
        self.results['response_tallies'] = tallies
        
        # Good-bye
        self.out('All done!')
        
    def create_plots(self):
        """Create per-rat and global figures of rotations, correlations, 
        correlation diagonals, and response change pie charts
        """
        from pylab import figure, axes, axis, subplot, polar, pie, rcParams
        from ..tools.images import tiling_dims
        
        self.figure = {}
        res = self.results
        rats = res['rats']
        rats.sort()
        r, c = tiling_dims(len(rats))
        rat_plots_size = 13, 9
        global_plots_size = 9, 9
        
        # Per-rat rotations/correlations polar cluster plots
        polar_kwargs = dict(marker='o', ms=10, mfc='b', mew=0, alpha=0.6,
            ls='', aa=True)
        deg_labels = [u'%d\xb0'%d for d in [-90,-45,0,45,90,135,180,-135]]
        if len(rats) > 1:
            rcParams['figure.figsize'] = rat_plots_size
            self.figure['rotcorr_rats'] = f = figure()
            f.set_size_inches(rat_plots_size)
            f.suptitle('Cluster Rotation and Peak Correlation', fontsize=16)
            for i,rat in enumerate(rats):
                ax = subplot(r, c, i+1, polar=True)
                rots, corrs = res['rotcorr_rat%d'%rat]
                polar((np.pi/180)*rots + np.pi/2, corrs, **polar_kwargs)
                ax.set_rmax(1.0)
                ax.set_xticklabels(deg_labels)
                ax.set_title('Rat %d'%rat)
            
        # Global polar cluster plot
        rcParams['figure.figsize'] = global_plots_size
        self.figure['rotcorr'] = f = figure()
        f.set_size_inches(global_plots_size)
        f.suptitle('Cluster Rotation and Peak Correlation (%d rats)'%len(rats), 
            fontsize=16)
        rots, corrs = res['rotcorr']
        polar_kwargs.update(ms=12)
        ax = subplot(111, polar=True)
        polar((np.pi/180)*rots + np.pi/2, corrs, **polar_kwargs)
        ax.set_rmax(1.0)
        ax.set_xticklabels(deg_labels)
        
        # Per-rat correlation diagonals plots
        diags_kwargs = dict(c='k', ls='-', aa=True, marker='')
        if len(rats) > 1:
            rcParams['figure.figsize'] = rat_plots_size
            self.figure['diags_rats'] = f = figure()
            f.set_size_inches(rat_plots_size)
            f.suptitle('Correlation Diagonals', fontsize=16)
            for i,rat in enumerate(rats):
                ax = subplot(r, c, i+1)
                d_STD = res['diags_STD_rat%d'%rat]
                d_MIS = res['diags_MIS_rat%d'%rat]
                ax.plot(*d_STD, lw=1, label='STD', **diags_kwargs)
                ax.plot(*d_MIS, lw=2, label='MIS', **diags_kwargs)
                ax.set_title('Rat %d'%rat)
                ax.set_xlim(d_STD[0][0], d_STD[0][-1])
                ax.set_ylim(0, 1)
                if i == 0:
                    ax.legend(fancybox=True)
                ax.text(0.03, 0.92, 'Angle = %.1f'%res['popshift_rat%d'%rat][0],
                    transform=ax.transAxes)
                ax.text(0.03, 0.87, 'Peak = %.3f'%res['popshift_rat%d'%rat][1],
                    transform=ax.transAxes)

        # Global diagonals plot
        rcParams['figure.figsize'] = global_plots_size
        self.figure['diags'] = f = figure()
        f.set_size_inches(global_plots_size)
        f.suptitle('Population Correlations (%d rats)'%len(rats), fontsize=16)
        ax = axes()
        d_STD = res['diags_STD']
        d_MIS = res['diags_MIS']
        ax.plot(*d_STD, lw=1.5, label='STD', **diags_kwargs)
        ax.plot(*d_MIS, lw=3, label='MIS', **diags_kwargs)
        ax.set_xlim(d_STD[0][0], d_STD[0][-1])
        ax.set_ylim(0, 1)
        ax.legend(fancybox=True)
        ax.text(0.03, 0.95, 'Angle = %.1f'%res['popshift'][0],
            transform=ax.transAxes)
        ax.text(0.03, 0.92, 'Peak = %.3f'%res['popshift'][1],
            transform=ax.transAxes)
        
        # Per-rat response change tally pie charts
        pie_kwargs = dict(labels=CL_LABELS, colors=CL_COLORS, autopct='%.1f', 
            pctdistance=0.8, labeldistance=100)
        if len(rats) > 1:
            rcParams['figure.figsize'] = rat_plots_size
            self.figure['response_tally_rats'] = f = figure()
            f.set_size_inches(rat_plots_size)
            f.suptitle('Response Changes', fontsize=16)
            for i,rat in enumerate(rats):
                ax = subplot(r, c, i+1)
                tally = [res['response_tallies_rat%d'%rat][k] for k in CL_LABELS]
                pie(tally, **pie_kwargs)
                axis('equal')
                ax.set_xlim(-1.04, 1.04) # fixes weird edge clipping
                ax.set_title('Rat %d (%d total)'%(rat, sum(tally)))
                if i == 0:
                    ax.legend(fancybox=True, loc='right', 
                        bbox_to_anchor=(0.0, 0.5))
        
        # Global response change tally pie chart
        rcParams['figure.figsize'] = global_plots_size
        self.figure['response_tally'] = f = figure()
        tally = [res['response_tallies'][k] for k in CL_LABELS]
        f.set_size_inches(global_plots_size)
        f.suptitle('Response Changes (%d total)'%sum(tally), fontsize=16)
        pie_kwargs.update(pctdistance=0.9)
        ax = axes()
        pie(tally, **pie_kwargs)
        axis('equal')
        ax.set_xlim(-1.04, 1.04) # fixes weird edge clipping
        ax.legend(fancybox=True, loc='upper left', bbox_to_anchor=(-0.11, 1.05))
        
        # Reset figure sizes
        rcParams['figure.figsize'] = 9, 9    
