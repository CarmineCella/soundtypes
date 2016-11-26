
# -*- coding: utf-8 -*-
"""
@author: Carmine-Emanuele Cella, 2016
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import MDS


N_COEFF = 20
ST_RATIO = .7
INPUT_FILE = 'samples/Jarrett_Vienna_cut.wav'
N_FRAMES = 100
FRAME_SIZE = 1024
HOP_SIZE = 1024
MAX_LOOPS = 3
CLUSTERING_ALGO = KMeans
WIDTH = 16
FADE_MS = 10

def spectral_flux (M):
    flux = []
    for i in range (M.shape[1] - 1):
        flux.append (np.sum(np.abs (M[:, i] -  M[:, i + 1])))                                                

    if (np.max(flux)):
        flux /= np.max(flux)                                   
    return np.array(flux)
    
def find_peaks (data, width):
    peaks = [0]
    prev  = 0
    delay = ((width) // 2) + 1
    data = np.convolve(data, np.hanning (width))[:-width]    
    for i in range (1, len(data)-1):
        if data[i - 1] < data [i] and data [i + 1] < data[i]:
            pos = max (0, i - delay)
            if (pos != prev):
                peaks.append(pos)
                prev = pos
            
    return data[delay:], np.array(peaks)
    
def fade_segment (segment, ms, sr):
    samples = int ((ms / 1000) * sr)
    if (samples * 2 >= len(segment)):
        return segment
    ramp_up = np.linspace (0, 1, samples)
    ramp_dw = np.linspace (1, 0, samples)
    segment[0:samples] *= ramp_up
    segment[-samples:] *= ramp_dw
    return segment
        
if __name__ == "__main__":
    print ('[soundtypes - probabilistic generation on onsets]\n')

    print ('computing onsets...')
    [y, sr] = librosa.core.load(INPUT_FILE)
    y_pad = np.zeros(len(y) + FRAME_SIZE)
    y_pad[1:len(y)+1] = y
    
    M = np.abs (librosa.spectrum.stft(y=y, n_fft=FRAME_SIZE, 
                                        hop_length=HOP_SIZE))

    flux = spectral_flux (M)
    flux, onsets = find_peaks (flux, WIDTH)                              

    plt.close('all')
    plt.plot (flux)
    locations = np.zeros(flux.shape)
    locations[onsets] = np.max(flux)
    plt.stem(locations, 'r')
    plt.show()

    segments = []
    for i in range (1, len(onsets)):
        chunk = y_pad[onsets[i - 1] * HOP_SIZE : onsets[i] * HOP_SIZE]
        chunk = fade_segment (chunk, FADE_MS, sr)
        segments.append (chunk)

    print ('computing features...')   
    features = []
    for i in range (len(segments)):
        C = librosa.feature.mfcc(y=segments[i], sr=sr, n_mfcc=N_COEFF, 
                                 n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        features.append(np.mean (C, axis=1))                             

    C = np.vstack(features)
    
    print ('computing multidimensional scaling...')
    mds = MDS(2)
    C_scaled = mds.fit_transform (C)

    print ('computing clusters...')
    n_clusters = int(C_scaled.shape[0] * ST_RATIO)
    cl_algo = CLUSTERING_ALGO (n_clusters).fit (C_scaled)
    centroids = cl_algo.cluster_centers_
    labels = cl_algo.predict(C_scaled)

    print ('computing probabilities...')
    markov = {i:[] for i in labels}

    pos = 0
    while pos < len(labels) - 1:
        if labels[pos+1] != labels[pos]:
            markov[labels[pos]].append(labels[pos+1])
        pos += 1

    print (markov)
    print ('generate new sequence...')

    soundtypes = {i:[] for i in range(n_clusters)}
    for i in range(n_clusters):
        for j in range(len(labels)):
            if labels[j] == i:
                soundtypes[i].append(j)

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
        p = soundtypes[(w1)]
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




