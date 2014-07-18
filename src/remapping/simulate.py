# encoding: utf-8
"""
simulate.py -- Simulate double rotation experiments using VMOModel

Exported namespace: VMOExperiment

Created by Joe Monaco on 2010-02-03.

Copyright (c) 2009-2011 Johns Hopkins University. All rights reserved.

This software is provided AS IS under the terms of the Open Source MIT License. 
See http://www.opensource.org/licenses/mit-license.php.
"""

# Library imports
from IPython.kernel.client import MapTask
import numpy as np
import os

# Package imports
from ..core.analysis import BaseAnalysis
from ..double_rotation import VMODoubleRotation
from ..session import VMOSession

# Directory constants
RAT_DIR = "Rat%02d"


def run_session(model_dict, save_dir, get_clone=False):
    """Run a session as part of a double-rotation experiment
    """
    success = False
    try:
        model = VMODoubleRotation(**model_dict)
        model.advance()
        data = VMOSession(model)
        VMOSession.save_session_list([data], save_dir)
    except:
        raise
    else:
        success = True
        if get_clone:
            success = model.clone_dict()
    return success


class VMOExperiment(BaseAnalysis):

    """
    Run double-rotation experiments using the VMODoubleRotation model class
    
    Convenience methods:
    run_mismatch_analyses -- Run mismatch analysis on each simulated mismatch
        angle followed by a remapping trends analysis. All data is saved to
        the analysis data directory.
    """
    
    label = "Cue Experiment"
    
    def collect_data(self, rats=1, mismatch=None, **kwargs):
        """Run the simulations and collect results data
        
        Simulated experimental data is saved in per-rat directories containing
        MIS_XXX.tar.gz archive files of VMOSession objects.
        
        Keyword arguments:
        rats -- number of experiments to run with different random networks
        mismatch -- list of mismatch angles (in degrees; don't include 0 for
            standard session, this is done automatically)
        
        Additional keywords are passed in as model parameters.
        """
        # Set the mismatch angles and convert to radians         
        if mismatch is None:
            mismatch = [45, 90, 135, 180]
        self.results['mismatch'] = mismatch
        self.results['rats'] = rats
        mismatch = [(np.pi/180) * angle for angle in mismatch]
            
        # Set up parameter dictionary
        pdict = dict(
            N_theta=1000, 
            N_outputs=500, 
            C_W=0.05, 
            cue_std=np.pi/24
            )
        pdict.update(kwargs)
        
        # Set up IPython engines
        mec = self.get_multiengine_client()
        tc = self.get_task_client()
        mec.execute('from vmo_feedback import VMODoubleRotation, VMOSession')
        mec.clear_queue()
        tc.clear()        

        # Run STD sessions and save model states
        mis_args = []
        for rat in xrange(rats):
            # Run the standard session to get network, fixed points
            self.out('Running standard session for rat %d...'%rat)
            if rats == 1:
                rat_dir = self.datadir
            else:
                rat_dir = os.path.join(self.datadir, RAT_DIR%rat)
            pdict.update(mismatch=[0])
            clone_dict = run_session(pdict, rat_dir, get_clone=True)
            if not clone_dict:
                self.out('STD session failed', error=True)
                continue
            mis_args.append((rat_dir, clone_dict))
        
        # Farm out mismatch sessions to task controller
        self.out('Now task-farming the mismatch sessions...')
        for rat in xrange(rats):
            rat_dir, clone_dict = mis_args[rat]
            tasks = []
            for angle in mismatch:
                clone_dict.update(mismatch=[angle])
                tasks.append(
                    tc.run(
                        MapTask(run_session, 
                            args=(clone_dict, rat_dir), 
                            kwargs={'get_clone':False})))
        tc.barrier(tasks)
        success = np.all([tc.get_task_result(t_id) for t_id in tasks])
        tc.clear()
        
        if success:
            self.out('Successfully completed mismatch sessions!')
        else:
            self.out('Error(s) detected during mismatch sessions', 
                error=True)

        # Good-bye
        self.out('All done!')
    
    def run_mismatch_analyses(self):
        """Perform MismatchAnalysis for each mismatch angle in this experiment
        and then perform MismatchTrends on those results, saving all the data
        and figures in this experiment's data directory.
        """
        if not self.finished:
            self.out('Analysis has not been completed yet!', error=True)
            return
            
        from mismatch import MismatchAnalysis
        from trends import MismatchTrends
        from pylab import close
        
        self.out('Running mismatch analysis for each angle...')
        mismatch = self.results['mismatch']
        MA_list = []
        for angle in mismatch:
            MA_dir = os.path.join(self.datadir, 'mismatch_%03d'%angle)
            MA = MismatchAnalysis(desc='mismatch', datadir=MA_dir)
            MA(load_dir=self.datadir, mismatch=angle)
            MA.view()
            MA.save_plots()
            close('all')
            MA_list.append(MA)
        
        self.out('Running mismatch trends analysis across angles...')
        trends_dir = os.path.join(self.datadir, 'trends')
        trends = MismatchTrends(desc='experiment', datadir=trends_dir)
        trends(*MA_list)
        trends.save_data()
        trends.view()
        trends.save_plots()
        close('all')
        