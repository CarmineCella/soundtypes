
# -*- coding: utf-8 -*-
"""
@author: Carmine-Emanuele Cella, 2016
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from st_tools import make_soundtypes

N_COEFF = 14
ST_RATIO = .9
INPUT_FILE = 'samples/bass.wav'
N_FRAMES = 500
FRAME_SIZE = 1024
HOP_SIZE = 512
MAX_LOOPS = 3 
SR = 44100

if __name__ == "__main__":
    print ('[soundtypes - probabilistic generation]\n')
    print ('computing features...')
    [y, sr] = librosa.core.load(INPUT_FILE, SR)
    y_pad = np.zeros(len(y) + FRAME_SIZE)
    y_pad[1:len(y)+1] = y
    C = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_COEFF, n_fft=FRAME_SIZE, 
                             hop_length=HOP_SIZE)

    print ('multidimensional scaling...')
    mds = MDS(2)
    C_scaled = mds.fit_transform (C.T)

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
    gen_sound = np.zeros(N_FRAMES * HOP_SIZE + FRAME_SIZE)
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

        chunk = y_pad[atom*HOP_SIZE:atom*HOP_SIZE+FRAME_SIZE] \
            * np.hanning(FRAME_SIZE)
        gen_sound[i*HOP_SIZE:i*HOP_SIZE+FRAME_SIZE] += chunk

    print ('saving audio data...')
    librosa.output.write_wav('generated_sound.wav', gen_sound, sr)

    plt.close ('all')

    plt.figure ()
    plt.plot (C_scaled[:, 0], C_scaled[:, 1], 'go')
    plt.plot (centroids[:, 0], centroids[:, 1], 'ro')
    plt.title('points and centroids')

    plt.figure()
    plt.plot(gen_sequence)
    plt.title('generated sequence')




