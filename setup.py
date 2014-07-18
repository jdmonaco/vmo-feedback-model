#!/usr/bin/env python

from distutils.core import setup
    
setup(      
            name="vmo_feedback",
            description="Oscillatory Interference Model with Sensory Feedback",
            long_description=
"""Many animals use a form of dead reckoning known as 'path integration' to
maintain a sense of their location as they explore the world. However, internal
motion signals and the neural activity that integrates them can be noisy,
leading inevitably to inaccurate position estimates. The rat hippocampus and
entorhinal cortex support a flexible system of spatial representation that is
critical to spatial learning and memory. The position signal encoded by this
system is thought to rely on path integration, but it must be recalibrated by
familiar landmarks to maintain accuracy. To explore the interaction between
path integration and external landmarks, we present a model of hippocampal
activity based on the interference of theta-frequency oscillations that are
modulated by realistic animal movements around a track. We show that spatial
activity degrades with noise, but introducing external cues based on direct
sensory feedback can prevent this degradation. When these cues are put into
conflict with each other, their interaction produces a diverse array of
response changes that resembles experimental observations. Feedback driven by
attending to landmarks may be critical to navigation and spatial memory in
mammals. """,
            version="1.0",
            author="Joe Monaco",
            author_email="jmonaco@jhu.edu",
            url="http://jdmonaco.com/vmo-feedback/",
            platforms=['Mac OSX', 'Linux', 'Windows'],
            license='The MIT License',
            package_dir={   'vmo_feedback': 'src'},
            packages=[  'vmo_feedback', 
                        'vmo_feedback.core', 
                        'vmo_feedback.figures', 
                        'vmo_feedback.remapping', 
                        'vmo_feedback.tools', 
                        'vmo_feedback.trajectory'   ],
            package_data={  'vmo_feedback.trajectory':  ['center.xy',
                                                         'circle_track.csv'  ]}
)
