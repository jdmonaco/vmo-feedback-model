===============================================================================
VMO Feedback Model on ModelDB
===============================================================================
-------------------------------------------------------------------------------
Sensory feedback in a multiple oscillator model of place cell activity 
-------------------------------------------------------------------------------

:Authors: Joseph D. Monaco [1], 
          James J. Knierim [1], 
          Kechen Zhang [2]

:Contact: jmonaco@jhu.edu

:Organization: [1] Zanvyl Krieger Mind/Brain Institute, Department of
    Neuroscience, Johns Hopkins University, Baltimore, MD, USA; [2] Department 
    of Biomedical Engineering, Johns Hopkins School of Medicine, Baltimore, MD, 
    USA

:Abstract: Mammals navigate by integrating self-motion signals ('path
    integration') and occasionally fixing on familiar environmental landmarks.
    The rat hippocampus is a model system of spatial representation in which
    place cells are thought to integrate both sensory and spatial information
    from entorhinal cortex. The localized firing fields of hippocampal place
    cells and entorhinal grid cells demonstrate a phase relationship with the
    local theta (6-10 Hz) rhythm that may be a temporal signature of path
    integration. However, encoding self-motion in the phase of theta
    oscillations requires high temporal precision and is susceptible to
    idiothetic noise, neuronal variability, and a changing environment. We
    present a model based on oscillatory interference theory, previously studied
    in the context of grid cells, in which transient temporal synchronization
    among a pool of path-integrating theta oscillators produces hippocampal-like
    place fields. We hypothesize that a spatiotemporally extended sensory
    interaction with external cues modulates feedback to the theta oscillators.
    We implement a form of this cue-driven feedback and show that it can restore
    learned fixed-points in the phase code of position. A single cue can
    smoothly reset oscillator phases to correct for both systematic errors and
    continuous noise in path integration. Further, simulations in which local
    and global cues are rotated against each other reveal a phase-code mechanism
    in which conflicting cue arrangements can reproduce experimentally observed
    distributions of 'partial remapping' responses. This abstract model
    demonstrates that phase-code feedback can provide stability to the temporal
    coding of position during navigation and may contribute to the
    context-dependence of hippocampal spatial representations. While the
    anatomical substrates of these processes have not been fully characterized,
    our findings suggest several signatures that can be evaluated in future
    experiments.



Installation
------------

Please see the ``INSTALL`` file for details, but you essentially need to have the
Enthought EPD python distribution installed. Then you unzip this archive, go into
the new directory and run ``sudo python setup.py install``. The model can then be
run interactively in an IPython session.


Libraries
---------

Here is a brief description of the main modules and classes:

Top-level Modules
~~~~~~~~~~~~~~~~~

``vmo``
    ``VMOModel``: main model simulation class
``double_rotation``
    ``VMODoubleRotation``: subclass that performs double rotation
``session``
    ``VMOSession``: container that computes and stores simulation results
``compare``
    Functions for comparing remapping responses
``placemap``
    ``CirclePlaceMap``: class that computes place fields
``error``
    Function definitions for solutions to the phase error integral
    
Subpackages
~~~~~~~~~~~

``core`` 
    - Base classes for models, analyses, and time-series data
``figures``
    - ``PathIntFigure``: basis for Figure 2
    - ``RemappingFigure``: basis for Figure 3
    - ``FeedbackFigure``: basis for Figure 5
        * ``VMOTrackModel``: model subclass that tracks phase information
        * ``VMOToyModel``: implements idealized linear trajectory
    - ``PhaseNoiseFigure``: basis for Figure 6
    - ``MismatchFigure`` [#ip]_: basis for Figures 7 and 8
``remapping``
    - ``VMOExperiment`` [#ip]_: parallelized double rotation simulations
    - ``MismatchAnalysis``: comprehensive remapping comparisons
    - ``MismatchTrends``: remapping trends across mismatch angles
``tools``
    - A collection of supporting utility functions
``trajectory``
    - ``CircleTrackData``: loads trajectory data, computes laps, etc.
    
.. [#ip] 
   These classes farm simulations out to IPython ipengine instances running
   on your machine. You must first start them in another terminal::

      $ ipcluster local -n C

   Set ``C`` to the number of cores available on your machine.
    

Example Usage
-------------

You can run the model itself, specifying various parameters, or you can run
pre-cooked analyses that were used as the basis of figures in the paper.

Running the model
~~~~~~~~~~~~~~~~~

Start IPython in ``-pylab`` mode::

    $ ipython -pylab

Then, import the libraries and create a model instance::

    In [0]: from vmo_feedback import *
    In [1]: model = VMOModel()

It will automatically load the trajectory data. To see all the user-settable
parameters, you can print the model::

    In [2]: print model
    VMOModel(Model) object
    --------------------------------
    Parameters:
            C_W : 0.050000000000000003
            N_cues : 3
            N_cues_distal : 3
            N_cues_local : 3
            N_outputs : 500
            N_theta : 1000
            cue_offset : 0.0
            cue_std : 0.15707963267948966
            distal_cue_std : 0.15707963267948966
            distal_offset : 0.0
            done : False
            epsilon : 0.050000000000000003
            gamma_distal : 0.46101227797811462
            gamma_local : 0.46101227797811462
            init_random : True
            lambda_range : (100, 200)
            local_cue_std : 0.15707963267948966
            local_offset : 0.0
            monitoring : True
            omega : 7
            pause : False
            refresh_fixed_points : True
    Tracking:
            C_distal, C_local, I, active_distal_cue, x, y, alpha, 
            active_local_cue, vel

In parameter names above, parameters with ``local`` and ``distal`` control one of
the two cue sets (local rotate CCW, distal CW) to allow for independent
manipulation::

    C_W            feedforward connectivity and 
    N_theta        number of theta oscillators
    N_outputs      the number of output units; each receives input from 
                       C_W*N_theta oscillators
    N_cues         number of cues per cue set
    cue_offset     starting point on the track for laying out cues
    cue_std        cue size (s.d.) in radians
    epsilon        error tolerance
    lambda_range   the range of spatial scales (multiplied by 2PI)
    gamma          peak cue gains; if not specified, these gains are set
                       automatically based on epsilon
    init_random    whether each trial gets randomized initial phases
    omega          frequency of the carrier signal.

The values shown are the defaults used for the paper (aside from cue
manipulations, etc.). The tracked variables are time-series data that are saved
from the simulation::

    C_*            cue interaction coefficients
    I              summed oscillatory inputs for each output
    x, y           trajectory position
    vel            instantaneous velocity vector
    alpha          track angle
    active_*_cue   index number of the currently active cue

Parameters can be changed by passing them as keyword arguments to the
constructor. To simulate only 200 oscillators, you would call
``VMOModel(N_theta=200)``.

Run the simulation::

    In [3]: model.advance()

Look at the tracked data::

    In [4]: data = model.post_mortem()
    In [5]: print data.tracking
    In [6]: subplot(221)
    In [7]: plot(data.x, data.y, 'k-')

Plot cue coefficients for the local and distal cue sets::

    In [8]: subplot(222)
    In [9]: scatter(x=data.x, y=data.y, c=data.C_local, s=1, linewidths=0)
    In [10]: subplot(224)
    In [11]: scatter(x=data.x, y=data.y, c=data.C_distal, s=1, linewidths=0)

Now compute the output responses based on the amplitude envelope of excitation.
These are computed and stored in ``VMOSession`` objects based on the simulation
results::

    In [12]: session = VMOSession(model)

Get the population response matrix and plot the place fields::

    In [13]: subplot(223)
    In [14]: R = session.get_population_matrix()
    In [15]: plot(R.T, 'k-')
    In [16]: xlim(0, 360)

The ``VMODoubleRotation`` is a subclass that automatically performs double
rotations across a series of mismatch angles. Running double rotation and
computing the responses is as easy as::

    In [17]: drmodel = VMODoubleRotation()
    In [18]: drmodel.advance_all()
    In [19]: drsessions = VMOSession.get_session_list(drmodel)

The analysis subclasses in the remapping subpackage encapsulate double rotation
experiment simulations and various remapping analyses. 


Running figure analyses
~~~~~~~~~~~~~~~~~~~~~~~

To run the figure analyses, you simply create an analysis object and run it by
calling it with analysis parameters. The single-cue feedback analysis is
performed by the ``FeedbackFigure`` analysis class::

    In [20]: fig = FeedbackFigure(desc='test')
    In [21]: fig.collect_data?

The first command creates an analysis object with the description 'test'. The
second command (with the ``?``) tells IPython to print out meta-data about the
``collect_data`` method. This is the method that actually performs the analysis
when you call the object, so this tells you the available parameters along with
their descriptions. We could run the analysis with different cue sizes::

    In [22]: fig(cue_sizes=[5, 25])

This performs the simulations, collects data for the figures, and stores data,
statistics, and an *analysis.log* file in the analysis directory. When that
completes, you can bring up the resulting figure and save it::

    In [23]: fig.view()
    In [24]: fig.save_plots()

Running the ``view`` method renders the figures, outputs RGB image files, and
saves a *figure.log* file in the analysis directory. Some of the figures have
parameter arguments to change the figure. You will have to use the
``create_plots`` method, as this is what the ``view`` method actually calls. To see
the figure parameters and make changes::

    In [25]: fig.create_plots?
    In [26]: fig.create_plots(...)

The same process can be used for the other figure analysis classes. You can
create your own analyses by subclassing from ``core.analysis.BaseAnalysis`` and
implementing the ``collect_data`` and ``create_plots`` methods.

----

Please explore the code, and let me know at `jmonaco@jhu.edu
<mailto:jmonaco@jhu.edu>`_ if there are any major issues. There are no guarantees
that this code will work perfectly everywhere.

Enjoy.
