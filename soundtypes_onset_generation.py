
# -*- coding: utf-8 -*-
"""
@author: Carmine-Emanuele Cella, 2016
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from st_tools import get_segments, make_soundtypes

N_COEFF = 20
ST_RATIO = .7
INPUT_FILE = 'samples/Jarrett_Vienna_cut.wav'
N_FRAMES = 100
FRAME_SIZE = 1024
HOP_SIZE = 1024
MAX_LOOPS = 3
WIDTH = 16
FADE_MS = 10
SR = 44100
        
if __name__ == "__main__":
    print ('[soundtypes - probabilistic generation on onsets]\n')

    print ('computing segments...')
    [y, sr] = librosa.core.load(INPUT_FILE, SR)
    
    (segments, onsets, flux) = get_segments (y, sr, FRAME_SIZE, HOP_SIZE, \
        FADE_MS, WIDTH)
        
    plt.close('all')
    plt.plot (flux)
    locations = np.zeros(flux.shape)
    locations[onsets] = np.max(flux)
    plt.stem(locations, 'r')
    plt.show()

    print ('computing features...')   
    features = []
    for i in range (len(segments)):
        C = librosa.feature.mfcc(y=segments[i], sr=sr, n_mfcc=N_COEFF, 
                                 n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        features.append(np.mean (C, axis=1))                             

    C = np.vstack(features)
    
    print ('multidimensional scaling...')
    mds = MDS(2)
    C_scaled = mds.fit_transform (C)

    print ('computing soundtypes...')
    (dictionary, markov, centroids, labels) = \
        make_soundtypes(C_scaled, ST_RATIO)
    n_clusters = centroids.shape[0]
    print (markov)
    
    print ('generate new sequence...')
    w1 = np.random.randint (n_clusters)
    prev_w1 = 0
    loops = 0
    gen_sequence = []
    gen_sound = []
    for i in range(N_FRAMES):
        l = markov[(w1)]
        if len(l) == 0:
            w1 = np.random.randint(n_clusters)
        else:
            w1 = l[np.random.randint(len(l))]
        if prev_w1 == w1:
            loops += 1
            
        if loops > MAX_LOOPS:
            w1 = np.random.randint(n_clusters)
            loops = 0
            
        gen_sequence.append(w1)
        p = dictionary[(w1)]
        atom = p[np.random.randint(len(p))]

        gen_sound.append (segments[atom])
        
    gen_sound = np.hstack (gen_sound)
    
    print ('saving audio data...')
    librosa.output.write_wav('generated_sound.wav', gen_sound, sr)

    plt.figure ()
    plt.plot (C_scaled[:, 0], C_scaled[:, 1], 'go')
    plt.plot (centroids[:, 0], centroids[:, 1], 'ro')
    plt.title('points and centroids')

    plt.figure()
    plt.plot(gen_sequence)
    plt.title('generated sequence')




